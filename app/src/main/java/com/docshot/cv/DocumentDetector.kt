package com.docshot.cv

import android.util.Log
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

private const val TAG = "DocShot:Detector"

/**
 * Result of document detection (corners only, no rectification).
 * @param corners Detected corner positions [TL, TR, BR, BL] in source image coordinates.
 * @param detectionMs Detection processing time in milliseconds.
 */
data class DocumentCorners(
    val corners: List<Point>,
    val detectionMs: Double
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
    val pipelineMs: Double
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
 *
 * All intermediate Mats are released.
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
    edges.release()

    if (quads.isEmpty()) {
        val ms = (System.nanoTime() - start) / 1_000_000.0
        Log.d(TAG, "detectDocument: %.1f ms — no quads found".format(ms))
        return null
    }

    val corners = bestQuad(quads, imageArea)
    if (corners == null) {
        val ms = (System.nanoTime() - start) / 1_000_000.0
        Log.d(TAG, "detectDocument: %.1f ms — no valid quad after scoring".format(ms))
        return null
    }

    val ms = (System.nanoTime() - start) / 1_000_000.0
    Log.d(TAG, "detectDocument: %.1f ms".format(ms))
    return DocumentCorners(corners, ms)
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
        pipelineMs = ms
    )
}
