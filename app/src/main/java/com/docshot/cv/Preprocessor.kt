package com.docshot.cv

import android.util.Log
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

private const val TAG = "DocShot:Preprocess"

/**
 * Preprocessing strategies for document edge detection.
 * Each strategy targets a specific scene condition.
 */
enum class PreprocessStrategy {
    /** Grayscale + 9x9 Gaussian blur. Default for high-contrast scenes. */
    STANDARD,

    /** CLAHE contrast enhancement + Gaussian blur. For low light / low contrast. */
    CLAHE_ENHANCED,

    /** Inverted HSV saturation channel + Gaussian blur. For colored backgrounds. */
    SATURATION_CHANNEL,

    /** Bilateral filter (edge-preserving denoise). For textured/patterned surfaces. */
    BILATERAL,

    /** Standard preprocess + heavier morphological close in EdgeDetector. For patterned surfaces. */
    HEAVY_MORPH
}

/**
 * Scene analysis result used to determine which preprocessing strategies to try.
 */
data class SceneAnalysis(
    val meanIntensity: Double,
    val stddevIntensity: Double,
    val strategies: List<PreprocessStrategy>
)

/**
 * Converts a BGR/RGBA input to a blurred grayscale image suitable for edge detection.
 * Caller must release the returned Mat.
 */
fun preprocess(input: Mat): Mat {
    return preprocessWithStrategy(input, PreprocessStrategy.STANDARD)
}

/**
 * Preprocesses an image using the specified strategy.
 * All strategies return a single-channel (grayscale) Mat ready for edge detection.
 * Caller must release the returned Mat.
 *
 * @param input BGR, RGBA, or grayscale image (not modified).
 * @param strategy The preprocessing approach to use.
 */
fun preprocessWithStrategy(input: Mat, strategy: PreprocessStrategy): Mat {
    val start = System.nanoTime()

    val result = when (strategy) {
        PreprocessStrategy.STANDARD -> preprocessStandard(input)
        PreprocessStrategy.CLAHE_ENHANCED -> preprocessClahe(input)
        PreprocessStrategy.SATURATION_CHANNEL -> preprocessSaturation(input)
        PreprocessStrategy.BILATERAL -> preprocessBilateral(input)
        PreprocessStrategy.HEAVY_MORPH -> preprocessStandard(input) // edge detector applies heavier morph
    }

    val ms = (System.nanoTime() - start) / 1_000_000.0
    Log.d(TAG, "preprocess($strategy): %.1f ms".format(ms))
    return result
}

/**
 * Fast scene analysis to determine which preprocessing strategies are worth trying.
 * Runs in <2ms by computing only mean and stddev of a grayscale version.
 *
 * Strategy selection logic:
 * - Low light (mean < 80) or low contrast (stddev < 30): try CLAHE first
 * - Color input (3+ channels): include SATURATION_CHANNEL
 * - Always include STANDARD as baseline
 * - BILATERAL and HEAVY_MORPH as last resorts
 *
 * @param input BGR, RGBA, or grayscale image.
 * @return Ordered list of strategies to try, most promising first.
 */
fun analyzeScene(input: Mat): SceneAnalysis {
    val start = System.nanoTime()

    // Convert to gray for stats (reuse if already gray)
    val gray = if (input.channels() == 1) {
        input
    } else {
        val g = Mat()
        val code = if (input.channels() == 4) Imgproc.COLOR_RGBA2GRAY else Imgproc.COLOR_BGR2GRAY
        Imgproc.cvtColor(input, g, code)
        g
    }

    val mean = MatOfDouble()
    val stddev = MatOfDouble()
    Core.meanStdDev(gray, mean, stddev)
    val meanVal = mean.get(0, 0)[0]
    val stddevVal = stddev.get(0, 0)[0]
    mean.release()
    stddev.release()
    if (gray !== input) gray.release()

    val strategies = mutableListOf<PreprocessStrategy>()

    // Low light or low contrast → CLAHE is most likely to help
    val isLowLight = meanVal < 80.0
    val isLowContrast = stddevVal < 30.0
    if (isLowLight || isLowContrast) {
        strategies.add(PreprocessStrategy.CLAHE_ENHANCED)
    }

    // Always try standard
    strategies.add(PreprocessStrategy.STANDARD)

    // Color input → saturation channel may isolate white document from colored bg
    if (input.channels() >= 3) {
        strategies.add(PreprocessStrategy.SATURATION_CHANNEL)
    }

    // Bilateral and heavy morph as fallbacks for textured surfaces
    strategies.add(PreprocessStrategy.BILATERAL)
    strategies.add(PreprocessStrategy.HEAVY_MORPH)

    val ms = (System.nanoTime() - start) / 1_000_000.0
    Log.d(TAG, "analyzeScene: %.1f ms (mean=%.0f, stddev=%.0f) -> %s".format(
        ms, meanVal, stddevVal, strategies))

    return SceneAnalysis(
        meanIntensity = meanVal,
        stddevIntensity = stddevVal,
        strategies = strategies
    )
}

// ----------------------------------------------------------------
// Private strategy implementations
// ----------------------------------------------------------------

private fun toGray(input: Mat): Mat {
    val gray = Mat()
    when (input.channels()) {
        4 -> Imgproc.cvtColor(input, gray, Imgproc.COLOR_RGBA2GRAY)
        3 -> Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY)
        1 -> input.copyTo(gray)
        else -> error("Unexpected channel count: ${input.channels()}")
    }
    return gray
}

/** Original pipeline: grayscale + 9x9 Gaussian. */
private fun preprocessStandard(input: Mat): Mat {
    val gray = toGray(input)
    val blurred = Mat()
    // 9x9 Gaussian suppresses text/table edges so document boundary dominates
    Imgproc.GaussianBlur(gray, blurred, Size(9.0, 9.0), 0.0)
    gray.release()
    return blurred
}

/**
 * CLAHE (Contrast Limited Adaptive Histogram Equalization) + Gaussian blur.
 * Enhances local contrast in dark or washed-out images so that subtle
 * document boundaries become visible to Canny.
 */
private fun preprocessClahe(input: Mat): Mat {
    val gray = toGray(input)
    val clahe = Imgproc.createCLAHE(3.0, Size(4.0, 4.0))
    val enhanced = Mat()
    clahe.apply(gray, enhanced)
    gray.release()

    val blurred = Mat()
    Imgproc.GaussianBlur(enhanced, blurred, Size(9.0, 9.0), 0.0)
    enhanced.release()
    return blurred
}

/**
 * Uses the inverted saturation channel from HSV color space.
 * White documents have near-zero saturation while colored backgrounds
 * have high saturation. Inverting makes the document bright (high intensity)
 * regardless of background color, producing strong edges at the boundary.
 */
private fun preprocessSaturation(input: Mat): Mat {
    if (input.channels() < 3) {
        // Fall back to standard for grayscale input
        return preprocessStandard(input)
    }

    val bgr = if (input.channels() == 4) {
        val b = Mat()
        Imgproc.cvtColor(input, b, Imgproc.COLOR_RGBA2BGR)
        b
    } else {
        input
    }

    val hsv = Mat()
    Imgproc.cvtColor(bgr, hsv, Imgproc.COLOR_BGR2HSV)
    if (bgr !== input) bgr.release()

    // Extract saturation channel and invert it
    val channels = mutableListOf<Mat>()
    Core.split(hsv, channels)
    hsv.release()

    val saturation = channels[1]
    val inverted = Mat()
    Core.bitwise_not(saturation, inverted)

    // Release all HSV channels
    channels.forEach { it.release() }

    val blurred = Mat()
    Imgproc.GaussianBlur(inverted, blurred, Size(9.0, 9.0), 0.0)
    inverted.release()
    return blurred
}

/**
 * Bilateral filter preserves edges while smoothing texture/patterns.
 * More expensive than Gaussian but suppresses wood grain, fabric weave,
 * etc. without blurring document boundaries.
 */
private fun preprocessBilateral(input: Mat): Mat {
    val gray = toGray(input)
    val filtered = Mat()
    // d=9: neighborhood diameter
    // sigmaColor=75: color space filter sigma (higher = more colors mixed)
    // sigmaSpace=75: coordinate space filter sigma (higher = wider mixing)
    Imgproc.bilateralFilter(gray, filtered, 9, 75.0, 75.0)
    gray.release()
    return filtered
}
