package com.docshot.cv

import android.util.Log
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

private const val TAG = "DocShot:Detector"

/** Minimum confidence to accept a detection. Below this, detectDocument() returns null. */
const val MIN_CONFIDENCE_THRESHOLD = 0.35

/**
 * Result of document detection (corners only, no rectification).
 * @param corners Detected corner positions [TL, TR, BR, BL] in source image coordinates.
 * @param detectionMs Detection processing time in milliseconds.
 */
data class DocumentCorners(
    val corners: List<Point>,
    val detectionMs: Double,
    val confidence: Double = 0.0
)

/**
 * Result of document detection and rectification.
 * @param rectified The perspective-corrected document image (caller must release).
 * @param corners Detected corner positions [TL, TR, BR, BL] in source image coordinates.
 * @param pipelineMs Total processing time in milliseconds.
 */
data class DetectionResult(
    val rectified: Mat,
    val corners: List<Point>,
    val pipelineMs: Double,
    val confidence: Double = 0.0
)

/**
 * Runs detection only: finds the best document quadrilateral without rectifying.
 * Used by real-time frame analysis where only corners are needed for the overlay.
 *
 * Pipeline stages:
 * 1. Preprocess (grayscale + blur)
 * 2. Canny edge detection (auto-threshold)
 * 3. Contour finding + polygon approximation
 * 4. Quadrilateral scoring and ranking
 * 5. Edge density validation + confidence scoring
 *
 * All intermediate Mats are released. Detections with confidence below
 * [MIN_CONFIDENCE_THRESHOLD] are suppressed (returns null).
 *
 * @param input BGR, RGBA, or grayscale image (not modified).
 */
fun detectDocument(input: Mat): DocumentCorners? {
    val start = System.nanoTime()

    val imageSize = Size(input.cols().toDouble(), input.rows().toDouble())
    val imageArea = imageSize.width * imageSize.height

    val gray = preprocess(input)
    val edges = detectEdges(gray)
    gray.release()

    val quads = findQuadrilaterals(edges, imageSize)
    // NOTE: edges Mat is kept alive until after confidence computation
    // so QuadValidator can measure edge support along the detected quad.

    if (quads.isEmpty()) {
        edges.release()
        val ms = (System.nanoTime() - start) / 1_000_000.0
        Log.d(TAG, "detectDocument: %.1f ms — no quads found".format(ms))
        return null
    }

    val corners = bestQuad(quads, imageArea)
    if (corners == null) {
        edges.release()
        val ms = (System.nanoTime() - start) / 1_000_000.0
        Log.d(TAG, "detectDocument: %.1f ms — no valid quad after scoring".format(ms))
        return null
    }

    // Measure how well the detected quad aligns with actual Canny edge pixels.
    // Both `edges` and `corners` are at the same resolution (input image size,
    // no internal downscaling in this function), so coordinates match directly.
    val edgeDensity = QuadValidator.edgeDensityScore(edges, corners)
    edges.release()
    Log.d(TAG, "Edge density: %.2f".format(edgeDensity))

    // Confidence = weighted combination of three complementary signals:
    //   1. Quad score (60%) — geometric quality: area coverage + angle regularity.
    //      Highest weight because a well-shaped, large quad is the strongest
    //      indicator of a real document.
    //   2. Area ratio (20%) — quad area relative to image area. Penalizes tiny
    //      detections that are likely noise or distant objects.
    //   3. Edge density (20%) — fraction of the quad perimeter supported by
    //      Canny edge pixels. Validates that the quad boundary corresponds to
    //      real image edges, not an artifact of contour approximation.
    val quadScore = scoreQuad(corners, imageArea)
    val areaRatio = (quadArea(corners) / imageArea).coerceIn(0.0, 1.0)
    val confidence = 0.6 * quadScore + 0.2 * areaRatio + 0.2 * edgeDensity

    // Suppress low-confidence detections to reduce false positives.
    // Threshold of 0.35 is slightly permissive — will be tuned with the
    // Phase 7 Group G test dataset.
    if (confidence < MIN_CONFIDENCE_THRESHOLD) {
        val ms = (System.nanoTime() - start) / 1_000_000.0
        Log.d(TAG, "Detection suppressed: confidence %.2f < %.2f threshold".format(confidence, MIN_CONFIDENCE_THRESHOLD))
        return null
    }

    val ms = (System.nanoTime() - start) / 1_000_000.0
    Log.d(TAG, "detectDocument: %.1f ms, confidence: %.2f".format(ms, confidence))
    return DocumentCorners(
        corners = corners,
        detectionMs = ms,
        confidence = confidence
    )
}

/**
 * Runs the full detection → rectification pipeline on a single image.
 * Returns null if no document-like quadrilateral was found.
 *
 * All intermediate Mats are released. Caller must release the Mat inside [DetectionResult].
 *
 * @param input BGR or RGBA image (not modified).
 * @param interpolation Warp interpolation mode. Use INTER_CUBIC for captures,
 *   INTER_LINEAR for preview.
 */
fun detectAndRectify(
    input: Mat,
    interpolation: Int = Imgproc.INTER_CUBIC
): DetectionResult? {
    val start = System.nanoTime()

    val detection = detectDocument(input) ?: return null
    val rectified = rectify(input, detection.corners, interpolation)

    val ms = (System.nanoTime() - start) / 1_000_000.0
    Log.d(TAG, "detectAndRectify: %.1f ms total".format(ms))

    return DetectionResult(
        rectified = rectified,
        corners = detection.corners,
        pipelineMs = ms,
        confidence = detection.confidence
    )
}
