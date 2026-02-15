package com.docshot.cv

import android.util.Log
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.Core
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

private const val TAG = "DocShot:Edge"

// Cached structuring kernels — immutable, created once and reused across calls.
// Avoids per-frame allocation of these tiny Mats.
private val kernel3x3: Mat by lazy {
    Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
}
private val kernel5x5: Mat by lazy {
    Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
}

/**
 * Runs Canny edge detection with automatic or explicit threshold selection.
 * When [thresholdLow] and [thresholdHigh] are <= 0 (default), thresholds are
 * derived from the median intensity: low = 0.67 * median, high = 1.33 * median.
 * Explicit thresholds are useful for CLAHE-preprocessed images where the
 * auto-threshold formula overestimates edge gradients.
 * Caller must release the returned Mat.
 */
fun detectEdges(
    grayscale: Mat,
    thresholdLow: Double = -1.0,
    thresholdHigh: Double = -1.0
): Mat {
    val start = System.nanoTime()

    val median = computeMedian(grayscale)
    val low = if (thresholdLow > 0) thresholdLow else (0.67 * median).coerceIn(10.0, 200.0)
    val high = if (thresholdHigh > 0) thresholdHigh else (1.33 * median).coerceIn(30.0, 250.0)

    val edges = Mat()
    Imgproc.Canny(grayscale, edges, low, high)

    // Morphological close (dilate then erode) bridges gaps in document edges
    // without bloating them — dilate-only tends to thicken text edges too much
    Imgproc.morphologyEx(edges, edges, Imgproc.MORPH_CLOSE, kernel3x3)

    val ms = (System.nanoTime() - start) / 1_000_000.0
    Log.d(TAG, "detectEdges: %.1f ms (median=%.0f, low=%.0f, high=%.0f)".format(ms, median, low, high))
    return edges
}

/**
 * Runs Canny edge detection with a heavier morphological close (5x5 kernel).
 * The larger kernel bridges wider gaps caused by patterned/textured surfaces
 * where document edges get fragmented by background texture.
 * Caller must release the returned Mat.
 */
fun detectEdgesHeavyMorph(grayscale: Mat): Mat {
    val start = System.nanoTime()

    val median = computeMedian(grayscale)
    val low = (0.67 * median).coerceIn(10.0, 200.0)
    val high = (1.33 * median).coerceIn(30.0, 250.0)

    val edges = Mat()
    Imgproc.Canny(grayscale, edges, low, high)

    // 5x5 morph close: bridges wider gaps in edges caused by texture interference
    Imgproc.morphologyEx(edges, edges, Imgproc.MORPH_CLOSE, kernel5x5)

    val ms = (System.nanoTime() - start) / 1_000_000.0
    Log.d(TAG, "detectEdgesHeavyMorph: %.1f ms (median=%.0f, low=%.0f, high=%.0f)".format(ms, median, low, high))
    return edges
}

internal fun computeMedian(gray: Mat): Double {
    val mean = MatOfDouble()
    val stddev = MatOfDouble()
    Core.meanStdDev(gray, mean, stddev)
    // Use mean as a fast proxy for median — close enough for threshold selection
    val result = mean.get(0, 0)[0]
    mean.release()
    stddev.release()
    return result
}
