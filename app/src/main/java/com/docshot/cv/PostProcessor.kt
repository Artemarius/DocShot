package com.docshot.cv

import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

private const val TAG = "DocShot:PostProcess"

/**
 * Available post-processing filters for rectified document images.
 */
enum class PostProcessFilter {
    NONE,           // No processing, return a copy
    BLACK_WHITE,    // Adaptive threshold for clean B&W document scan
    CONTRAST,       // CLAHE contrast + brightness enhancement
    COLOR_CORRECT   // Gray-world white balance correction
}

/**
 * Applies a post-processing filter to the given bitmap.
 * Returns a NEW bitmap — the caller is responsible for recycling both
 * the source and the returned bitmap when done.
 */
fun applyFilter(source: Bitmap, filter: PostProcessFilter): Bitmap {
    return when (filter) {
        PostProcessFilter.NONE -> source.copy(source.config, false)
        PostProcessFilter.BLACK_WHITE -> adaptiveThresholdBW(source)
        PostProcessFilter.CONTRAST -> enhanceContrast(source)
        PostProcessFilter.COLOR_CORRECT -> correctColor(source)
    }
}

/**
 * Converts the document to a clean black-and-white scan using adaptive thresholding.
 * Tuned for typical printed-text documents.
 */
private fun adaptiveThresholdBW(source: Bitmap): Bitmap {
    val start = System.nanoTime()

    val rgba = Mat()
    Utils.bitmapToMat(source, rgba)

    // Convert RGBA → gray directly (skip BGR intermediate)
    val gray = Mat()
    Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
    rgba.release()

    val binary = Mat()
    // blockSize=15, C=10: good balance for typical printed documents —
    // large enough block to handle gradual illumination changes,
    // C=10 subtracts enough to keep thin strokes clean without noise
    Imgproc.adaptiveThreshold(
        gray,
        binary,
        /* maxValue = */ 255.0,
        /* adaptiveMethod = */ Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
        /* thresholdType = */ Imgproc.THRESH_BINARY,
        /* blockSize = */ 15,
        /* C = */ 10.0
    )
    gray.release()

    val result = Bitmap.createBitmap(source.width, source.height, Bitmap.Config.ARGB_8888)
    Utils.matToBitmap(binary, result)
    binary.release()

    val ms = (System.nanoTime() - start) / 1_000_000.0
    Log.d(TAG, "adaptiveThresholdBW: %.1f ms".format(ms))
    return result
}

/**
 * Enhances contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
 * applied to the L channel in LAB color space.
 */
private fun enhanceContrast(source: Bitmap): Bitmap {
    val start = System.nanoTime()

    val rgba = Mat()
    Utils.bitmapToMat(source, rgba)
    val bgr = Mat()
    Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
    rgba.release()

    val lab = Mat()
    Imgproc.cvtColor(bgr, lab, Imgproc.COLOR_BGR2Lab)
    bgr.release()

    val channels = mutableListOf<Mat>()
    Core.split(lab, channels)
    lab.release()

    // CLAHE on L channel: clipLimit=2.0 prevents over-amplification of noise,
    // 8x8 tile grid provides locally adaptive enhancement
    val clahe = Imgproc.createCLAHE(/* clipLimit = */ 2.0, /* tileGridSize = */ Size(8.0, 8.0))
    val enhancedL = Mat()
    clahe.apply(channels[0], enhancedL)
    channels[0].release()
    channels[0] = enhancedL

    val mergedLab = Mat()
    Core.merge(channels, mergedLab)
    channels.forEach { it.release() }

    val resultBgr = Mat()
    Imgproc.cvtColor(mergedLab, resultBgr, Imgproc.COLOR_Lab2BGR)
    mergedLab.release()

    val resultRgba = Mat()
    Imgproc.cvtColor(resultBgr, resultRgba, Imgproc.COLOR_BGR2RGBA)
    resultBgr.release()

    val result = Bitmap.createBitmap(source.width, source.height, Bitmap.Config.ARGB_8888)
    Utils.matToBitmap(resultRgba, result)
    resultRgba.release()

    val ms = (System.nanoTime() - start) / 1_000_000.0
    Log.d(TAG, "enhanceContrast (CLAHE): %.1f ms".format(ms))
    return result
}

/**
 * Applies gray-world white balance correction.
 * Scales each BGR channel so that the per-channel means converge to the
 * global mean, compensating for color casts from uneven lighting.
 *
 * Optimized: uses Core.normalize for combined clip+clamp in one pass,
 * reducing intermediate Mat count.
 */
private fun correctColor(source: Bitmap): Bitmap {
    val start = System.nanoTime()

    val rgba = Mat()
    Utils.bitmapToMat(source, rgba)
    val bgr = Mat()
    Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
    rgba.release()

    // Convert to float for precise scaling without integer truncation
    val bgrFloat = Mat()
    bgr.convertTo(bgrFloat, CvType.CV_32FC3)
    bgr.release()

    val channels = mutableListOf<Mat>()
    Core.split(bgrFloat, channels)
    bgrFloat.release()

    // Gray-world assumption: the average color of the scene should be neutral gray
    val meanB = Core.mean(channels[0]).`val`[0]
    val meanG = Core.mean(channels[1]).`val`[0]
    val meanR = Core.mean(channels[2]).`val`[0]
    val target = (meanB + meanG + meanR) / 3.0

    // Scale each channel to match the target mean; avoid division by zero
    if (meanB > 1e-6) Core.multiply(channels[0], Scalar(target / meanB), channels[0])
    if (meanG > 1e-6) Core.multiply(channels[1], Scalar(target / meanG), channels[1])
    if (meanR > 1e-6) Core.multiply(channels[2], Scalar(target / meanR), channels[2])

    val merged = Mat()
    Core.merge(channels, merged)
    channels.forEach { it.release() }

    // Clip to [0, 255] and convert to 8-bit in one step
    val resultBgr = Mat()
    merged.convertTo(resultBgr, CvType.CV_8UC3, 1.0, 0.0)
    merged.release()
    // convertTo with CV_8UC3 automatically saturates (clamps to [0,255])

    val resultRgba = Mat()
    Imgproc.cvtColor(resultBgr, resultRgba, Imgproc.COLOR_BGR2RGBA)
    resultBgr.release()

    val result = Bitmap.createBitmap(source.width, source.height, Bitmap.Config.ARGB_8888)
    Utils.matToBitmap(resultRgba, result)
    resultRgba.release()

    val ms = (System.nanoTime() - start) / 1_000_000.0
    Log.d(TAG, "correctColor (gray-world): %.1f ms (means B=%.1f G=%.1f R=%.1f target=%.1f)".format(
        ms, meanB, meanG, meanR, target
    ))
    return result
}
