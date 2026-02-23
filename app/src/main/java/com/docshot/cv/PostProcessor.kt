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

/** Downsample factor for illumination estimation — at 3000px wide, yields ~375px;
 *  text is fully blurred away at this scale. */
private const val DOWNSAMPLE_FACTOR = 8

/** Gaussian blur kernel size on the downsampled image — equivalent to ~400px on
 *  full-res; large enough to capture smooth illumination gradients, small enough
 *  to handle nonlinear falloff. Must be odd. */
private const val BLUR_KERNEL_SIZE = 51

/**
 * Available post-processing filters for rectified document images.
 */
enum class PostProcessFilter {
    NONE,           // No processing, return a copy
    BLACK_WHITE,    // Adaptive threshold for clean B&W document scan
    CONTRAST,       // CLAHE contrast + brightness enhancement
    COLOR_CORRECT   // Lighting gradient correction (even illumination)
}

/**
 * Applies a post-processing filter to the given bitmap, optionally preceded
 * by gray world white balance correction.
 * Returns a NEW bitmap — the caller is responsible for recycling both
 * the source and the returned bitmap when done.
 */
fun applyFilter(source: Bitmap, filter: PostProcessFilter, applyWhiteBalance: Boolean = false): Bitmap {
    // Step 1: optionally apply white balance as base correction
    val wbBitmap = if (applyWhiteBalance) applyWhiteBalance(source) else null
    val effectiveSource = wbBitmap ?: source

    // Step 2: apply the selected filter
    val result = when (filter) {
        PostProcessFilter.NONE -> effectiveSource.copy(effectiveSource.config ?: Bitmap.Config.ARGB_8888, false)
        PostProcessFilter.BLACK_WHITE -> adaptiveThresholdBW(effectiveSource)
        PostProcessFilter.CONTRAST -> enhanceContrast(effectiveSource)
        PostProcessFilter.COLOR_CORRECT -> correctColor(effectiveSource)
    }

    // Recycle the intermediate WB bitmap (filter functions create their own output)
    wbBitmap?.recycle()

    return result
}

/**
 * Applies gray world white balance correction.
 * Normalizes per-channel means so R, G, B averages converge to a common gray.
 * Effective for outdoor blue tint from daylight color temperature.
 */
fun applyWhiteBalance(source: Bitmap): Bitmap {
    val start = System.nanoTime()

    val rgba = Mat()
    Utils.bitmapToMat(source, rgba)
    val bgr = Mat()
    Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
    rgba.release()

    val channels = mutableListOf<Mat>()
    Core.split(bgr, channels)
    bgr.release()

    // Compute per-channel means
    val meanB = Core.mean(channels[0]).`val`[0]
    val meanG = Core.mean(channels[1]).`val`[0]
    val meanR = Core.mean(channels[2]).`val`[0]
    val avgMean = (meanB + meanG + meanR) / 3.0

    // Scale each channel to match the average mean
    val scaleB = avgMean / maxOf(meanB, 1.0)
    val scaleG = avgMean / maxOf(meanG, 1.0)
    val scaleR = avgMean / maxOf(meanR, 1.0)

    channels[0].convertTo(channels[0], -1, scaleB, 0.0)
    channels[1].convertTo(channels[1], -1, scaleG, 0.0)
    channels[2].convertTo(channels[2], -1, scaleR, 0.0)

    val corrected = Mat()
    Core.merge(channels, corrected)
    channels.forEach { it.release() }

    val resultRgba = Mat()
    Imgproc.cvtColor(corrected, resultRgba, Imgproc.COLOR_BGR2RGBA)
    corrected.release()

    val result = Bitmap.createBitmap(source.width, source.height, Bitmap.Config.ARGB_8888)
    Utils.matToBitmap(resultRgba, result)
    resultRgba.release()

    val ms = (System.nanoTime() - start) / 1_000_000.0
    Log.d(TAG, "whiteBalance (gray world): %.1f ms (B=%.1f G=%.1f R=%.1f → avg=%.1f, scales=%.2f/%.2f/%.2f)".format(
        ms, meanB, meanG, meanR, avgMean, scaleB, scaleG, scaleR))
    return result
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
 * Corrects uneven lighting (brightness gradients) across the document.
 *
 * Uses low-frequency illumination estimation: downsample the L channel,
 * heavy Gaussian blur to remove all text/detail, then divide the original
 * luminance by this estimated illumination field. Works in LAB color space
 * so chrominance (A/B channels) is untouched.
 *
 * Effective for side lighting, angled shots, or any scenario where one edge
 * of the document is noticeably darker than the opposite.
 */
private fun correctColor(source: Bitmap): Bitmap {
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

    val lChannel = channels[0] // 8-bit L channel [0..255]

    // Convert L to float32 for precise division
    val lFloat = Mat()
    lChannel.convertTo(lFloat, CvType.CV_32F)

    // Downsample to isolate low-frequency illumination
    val smallW = maxOf(lFloat.cols() / DOWNSAMPLE_FACTOR, 1)
    val smallH = maxOf(lFloat.rows() / DOWNSAMPLE_FACTOR, 1)
    val smallL = Mat()
    Imgproc.resize(lFloat, smallL, Size(smallW.toDouble(), smallH.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)

    // Heavy blur on the small image — removes all text, leaves only illumination
    // Clamp kernel to image dimensions (must be odd and <= image size)
    val kw = minOf(BLUR_KERNEL_SIZE, if (smallW % 2 == 0) smallW - 1 else smallW)
    val kh = minOf(BLUR_KERNEL_SIZE, if (smallH % 2 == 0) smallH - 1 else smallH)
    val kernelSize = Size(maxOf(kw, 1).toDouble(), maxOf(kh, 1).toDouble())
    Imgproc.GaussianBlur(smallL, smallL, kernelSize, 0.0)

    // Upsample back to full resolution
    val illumination = Mat()
    Imgproc.resize(smallL, illumination, lFloat.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
    smallL.release()

    // Compute target mean brightness (preserve overall document brightness)
    val targetMean = Core.mean(lFloat).`val`[0]

    // Normalize: L_corrected = L_original / illumination * targetMean
    // Floor illumination at 1.0 to avoid division by near-zero in very dark regions
    val ones = Mat(illumination.size(), CvType.CV_32F, Scalar(1.0))
    Core.max(illumination, ones, illumination)
    ones.release()

    val correctedL = Mat()
    Core.divide(lFloat, illumination, correctedL, targetMean)
    lFloat.release()
    illumination.release()

    // Clamp to [0, 255] and convert back to 8-bit
    val correctedL8 = Mat()
    correctedL.convertTo(correctedL8, CvType.CV_8U)
    correctedL.release()
    // convertTo with CV_8U automatically saturates (clamps to [0,255])

    // Replace L channel, keep original A and B
    lChannel.release()
    channels[0] = correctedL8

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
    Log.d(TAG, "correctColor (gradient): %.1f ms (targetMean=%.1f)".format(ms, targetMean))
    return result
}
