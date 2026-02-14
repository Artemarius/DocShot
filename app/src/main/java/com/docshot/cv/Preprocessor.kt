package com.docshot.cv

import android.util.Log
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

private const val TAG = "DocShot:Preprocess"

/**
 * Converts a BGR/RGBA input to a blurred grayscale image suitable for edge detection.
 * Caller must release the returned Mat.
 */
fun preprocess(input: Mat): Mat {
    val start = System.nanoTime()

    val gray = Mat()
    val channels = input.channels()
    when (channels) {
        4 -> Imgproc.cvtColor(input, gray, Imgproc.COLOR_RGBA2GRAY)
        3 -> Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY)
        1 -> input.copyTo(gray)
        else -> error("Unexpected channel count: $channels")
    }

    val blurred = Mat()
    // 5x5 Gaussian removes sensor noise without destroying document edges
    Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
    gray.release()

    val ms = (System.nanoTime() - start) / 1_000_000.0
    Log.d(TAG, "preprocess: %.1f ms".format(ms))
    return blurred
}
