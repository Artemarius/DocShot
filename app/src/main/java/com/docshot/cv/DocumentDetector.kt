package com.docshot.cv

import android.util.Log
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

private const val TAG = "DocShot:Detector"

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
 * Runs the full detection → rectification pipeline on a single image.
 * Returns null if no document-like quadrilateral was found.
 *
 * Pipeline stages:
 * 1. Preprocess (grayscale + blur)
 * 2. Canny edge detection (auto-threshold)
 * 3. Contour finding + polygon approximation
 * 4. Quadrilateral scoring and ranking
 * 5. Perspective warp
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

    val imageSize = Size(input.cols().toDouble(), input.rows().toDouble())
    val imageArea = imageSize.width * imageSize.height

    // 1. Preprocess
    val gray = preprocess(input)

    // 2. Edge detection
    val edges = detectEdges(gray)
    gray.release()

    // 3. Find quadrilateral candidates
    val quads = findQuadrilaterals(edges, imageSize)
    edges.release()

    if (quads.isEmpty()) {
        val ms = (System.nanoTime() - start) / 1_000_000.0
        Log.d(TAG, "detectAndRectify: %.1f ms — no quads found".format(ms))
        return null
    }

    // 4. Score and pick best quad (returns ordered corners or null)
    val corners = bestQuad(quads, imageArea)
    if (corners == null) {
        val ms = (System.nanoTime() - start) / 1_000_000.0
        Log.d(TAG, "detectAndRectify: %.1f ms — no valid quad after scoring".format(ms))
        return null
    }

    // 5. Rectify
    val rectified = rectify(input, corners, interpolation)

    val ms = (System.nanoTime() - start) / 1_000_000.0
    Log.d(TAG, "detectAndRectify: %.1f ms total".format(ms))

    return DetectionResult(
        rectified = rectified,
        corners = corners,
        pipelineMs = ms
    )
}
