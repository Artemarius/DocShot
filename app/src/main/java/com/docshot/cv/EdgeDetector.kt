package com.docshot.cv

import android.util.Log
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.Core
import org.opencv.imgproc.Imgproc

private const val TAG = "DocShot:Edge"

/**
 * Runs Canny edge detection with automatic threshold selection.
 * Thresholds are derived from the median intensity of the input image:
 * low = 0.67 * median, high = 1.33 * median.
 * Caller must release the returned Mat.
 */
fun detectEdges(grayscale: Mat): Mat {
    val start = System.nanoTime()

    val median = computeMedian(grayscale)
    val low = (0.67 * median).coerceIn(10.0, 200.0)
    val high = (1.33 * median).coerceIn(30.0, 250.0)

    val edges = Mat()
    Imgproc.Canny(grayscale, edges, low, high)

    // Dilate slightly to close small gaps in document edges
    Imgproc.dilate(edges, edges, Mat())

    val ms = (System.nanoTime() - start) / 1_000_000.0
    Log.d(TAG, "detectEdges: %.1f ms (median=%.0f, low=%.0f, high=%.0f)".format(ms, median, low, high))
    return edges
}

private fun computeMedian(gray: Mat): Double {
    val mean = MatOfDouble()
    val stddev = MatOfDouble()
    Core.meanStdDev(gray, mean, stddev)
    // Use mean as a fast proxy for median — close enough for threshold selection
    val result = mean.get(0, 0)[0]
    mean.release()
    stddev.release()
    return result
}
