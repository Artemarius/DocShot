package com.docshot.cv

import android.util.Log
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin

private const val TAG = "DocShot:Preprocess"

// Cached structuring kernels for morphological ops in preprocessing strategies.
private val preprocessMorphKernel3x3: Mat by lazy {
    Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
}
private val preprocessMorphKernel5x5: Mat by lazy {
    Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
}

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
    HEAVY_MORPH,

    /** Adaptive threshold binarization (blockSize=51, C=5). Binary output — bypasses Canny. */
    ADAPTIVE_THRESHOLD,

    /** LAB L-channel + aggressive CLAHE (clipLimit=6.0, tileSize=2x2). For warm/cool surface differences. */
    LAB_CLAHE,

    /** Sobel gradient magnitude thresholded at 95th percentile. Binary output — bypasses Canny. */
    GRADIENT_MAGNITUDE,

    /** Difference of Gaussians (3x3 - 21x21). Feeds Canny at very low thresholds (10/30). */
    DOG,

    /** Per-channel Canny (20/50) + bitwise OR. Binary output — bypasses Canny. Expensive (~8-12ms). */
    MULTICHANNEL_FUSION,

    /**
     * Directional gradient smoothing: Sobel Gx/Gy smoothed along 5 tilted 1D kernels
     * ([-10, -5, 0, +5, +10] degrees) to accumulate coherent edge gradients while
     * cancelling random noise/texture. Binary output — bypasses Canny.
     * Targets ultra-low-contrast white-on-white scenes (gradients down to ~3 units).
     */
    DIRECTIONAL_GRADIENT;

    /** True if this strategy produces a binary (0/255) edge map that bypasses Canny in the detector. */
    val isBinaryOutput: Boolean
        get() = this == ADAPTIVE_THRESHOLD || this == GRADIENT_MAGNITUDE
                || this == MULTICHANNEL_FUSION || this == DIRECTIONAL_GRADIENT
}

/**
 * Scene analysis result used to determine which preprocessing strategies to try.
 */
data class SceneAnalysis(
    val meanIntensity: Double,
    val stddevIntensity: Double,
    val strategies: List<PreprocessStrategy>,
    /** Grayscale Mat produced during analysis. Caller must release when done, or pass to preprocessing. */
    val grayMat: Mat? = null,
    /** True when both surfaces are uniformly bright (mean > 180, stddev < 35). */
    val isWhiteOnWhite: Boolean = false
)

/** Maximum number of frames to cache a SceneAnalysis result. */
private const val SCENE_CACHE_FRAMES = 10

// Scene analysis cache — avoids recomputing mean/stddev/strategy list every frame.
// Scene lighting rarely changes faster than ~333ms (10 frames at 30fps).
private var cachedAnalysis: SceneAnalysis? = null
private var cacheFrameCounter = 0

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
 * @param pool Optional Mat pool for intermediate allocations.
 * @param sharedGray Optional pre-computed grayscale Mat (from [analyzeScene]). Not released by this function.
 */
fun preprocessWithStrategy(
    input: Mat,
    strategy: PreprocessStrategy,
    pool: MatPool? = null,
    sharedGray: Mat? = null
): Mat {
    val start = System.nanoTime()

    val result = when (strategy) {
        PreprocessStrategy.STANDARD -> preprocessStandard(input, pool, sharedGray)
        PreprocessStrategy.CLAHE_ENHANCED -> preprocessClahe(input, pool, sharedGray)
        PreprocessStrategy.SATURATION_CHANNEL -> preprocessSaturation(input, pool)
        PreprocessStrategy.BILATERAL -> preprocessBilateral(input, pool, sharedGray)
        PreprocessStrategy.HEAVY_MORPH -> preprocessStandard(input, pool, sharedGray) // edge detector applies heavier morph
        PreprocessStrategy.ADAPTIVE_THRESHOLD -> preprocessAdaptiveThreshold(input, pool, sharedGray)
        PreprocessStrategy.LAB_CLAHE -> preprocessLabClahe(input, pool, sharedGray)
        PreprocessStrategy.GRADIENT_MAGNITUDE -> preprocessGradientMagnitude(input, pool, sharedGray)
        PreprocessStrategy.DOG -> preprocessDoG(input, pool, sharedGray)
        PreprocessStrategy.MULTICHANNEL_FUSION -> preprocessMultichannelFusion(input)
        PreprocessStrategy.DIRECTIONAL_GRADIENT -> preprocessDirectionalGradient(input, pool, sharedGray)
    }

    val ms = (System.nanoTime() - start) / 1_000_000.0
    Log.d(TAG, "preprocess($strategy): %.1f ms".format(ms))
    return result
}

/**
 * Fast scene analysis to determine which preprocessing strategies are worth trying.
 * Runs in <2ms by computing only mean and stddev of a grayscale version.
 *
 * Results are cached for [SCENE_CACHE_FRAMES] frames to avoid redundant computation.
 * The returned [SceneAnalysis.grayMat] can be passed to [preprocessWithStrategy] as
 * `sharedGray` to avoid redundant grayscale conversion.
 *
 * Strategy selection logic:
 * - Low light (mean < 80) or low contrast (stddev < 30): try CLAHE first
 * - Color input (3+ channels): include SATURATION_CHANNEL
 * - Always include STANDARD as baseline
 * - BILATERAL and HEAVY_MORPH as last resorts
 *
 * @param input BGR, RGBA, or grayscale image.
 * @param useCache Whether to use/update the frame-based cache. Disable for one-shot capture.
 * @return Ordered list of strategies to try, most promising first.
 */
fun analyzeScene(input: Mat, useCache: Boolean = true): SceneAnalysis {
    // Return cached result if still fresh (strategies depend on lighting, not content)
    if (useCache) {
        val cached = cachedAnalysis
        if (cached != null && cacheFrameCounter < SCENE_CACHE_FRAMES) {
            cacheFrameCounter++
            // Don't return a grayMat from cache — it may have been released
            return cached.copy(grayMat = null)
        }
    }

    val start = System.nanoTime()

    // Convert to gray for stats — keep it for sharing with preprocessing
    val grayOwned: Mat?
    val gray: Mat
    if (input.channels() == 1) {
        gray = input
        grayOwned = null
    } else {
        val g = Mat()
        val code = if (input.channels() == 4) Imgproc.COLOR_RGBA2GRAY else Imgproc.COLOR_BGR2GRAY
        Imgproc.cvtColor(input, g, code)
        gray = g
        grayOwned = g
    }

    val mean = MatOfDouble()
    val stddev = MatOfDouble()
    Core.meanStdDev(gray, mean, stddev)
    val meanVal = mean.get(0, 0)[0]
    val stddevVal = stddev.get(0, 0)[0]
    mean.release()
    stddev.release()

    val isLowLight = meanVal < 80.0
    val isLowContrast = stddevVal < 30.0
    // White-on-white: both surfaces uniformly bright, subtle boundary gradients.
    // Auto-Canny thresholds saturate (0.67*220 ≈ 147), missing 5-15 unit edges.
    val isWhiteOnWhite = meanVal > 180.0 && stddevVal < 35.0

    val strategies = mutableListOf<PreprocessStrategy>()

    if (isWhiteOnWhite) {
        // Specialized low-contrast strategies for white-on-white scenes,
        // ordered by benchmark results (S21). Short-circuit at 0.65 confidence
        // means typically only the first strategy runs (~3ms).
        //
        // Benchmark (6 synthetic images, 800x600):
        //   DOG:             6/6 detected, 2.9-3.8ms, 0.0px median error
        //   GRADIENT_MAG:    5/6 detected, 4.6-10.8ms (missed textured)
        //   LAB_CLAHE:       5/6 detected, 3.7-6.3ms  (missed textured)
        //   CLAHE_ENHANCED:  6/6 detected, 2.8-7.0ms  (existing fallback)
        //   MULTI_FUSION:    5/6 detected, 2.6-4.5ms  (missed textured)
        //   ADAPTIVE_THRESH: 3/6 detected, 7.6-10.8ms (missed <=20 gradient)
        //   DIRECTIONAL_GRADIENT: TBD (v1.2.5, targets ~3 unit gradients)
        strategies.add(PreprocessStrategy.DOG)
        strategies.add(PreprocessStrategy.DIRECTIONAL_GRADIENT)
        strategies.add(PreprocessStrategy.GRADIENT_MAGNITUDE)
        strategies.add(PreprocessStrategy.LAB_CLAHE)
        strategies.add(PreprocessStrategy.CLAHE_ENHANCED) // existing, reliable fallback
        strategies.add(PreprocessStrategy.MULTICHANNEL_FUSION)
        strategies.add(PreprocessStrategy.ADAPTIVE_THRESHOLD) // weakest, last resort
    } else {
        // Low light or low contrast → CLAHE is most likely to help, try it first
        if (isLowLight || isLowContrast) {
            strategies.add(PreprocessStrategy.CLAHE_ENHANCED)
        }

        // Always try standard
        strategies.add(PreprocessStrategy.STANDARD)

        // CLAHE as fallback when not already prioritized: scenes with moderate
        // contrast (stddev 30-60) may have document/background differences that
        // are too subtle for auto-Canny after Gaussian blur. CLAHE + lower
        // thresholds catches these cases. Short-circuit prevents wasted work
        // when STANDARD already succeeds.
        if (!isLowLight && !isLowContrast) {
            strategies.add(PreprocessStrategy.CLAHE_ENHANCED)
        }

        // Color input → saturation channel may isolate white document from colored bg
        if (input.channels() >= 3) {
            strategies.add(PreprocessStrategy.SATURATION_CHANNEL)
        }

        // Bilateral and heavy morph as fallbacks for textured surfaces
        strategies.add(PreprocessStrategy.BILATERAL)
        strategies.add(PreprocessStrategy.HEAVY_MORPH)
    }

    val ms = (System.nanoTime() - start) / 1_000_000.0
    Log.d(TAG, "analyzeScene: %.1f ms (mean=%.0f, stddev=%.0f, whiteOnWhite=%s) -> %s".format(
        ms, meanVal, stddevVal, isWhiteOnWhite, strategies))

    val result = SceneAnalysis(
        meanIntensity = meanVal,
        stddevIntensity = stddevVal,
        strategies = strategies,
        grayMat = grayOwned,  // null if input was already gray
        isWhiteOnWhite = isWhiteOnWhite
    )

    // Update cache (without grayMat — each frame gets its own)
    if (useCache) {
        cachedAnalysis = result.copy(grayMat = null)
        cacheFrameCounter = 0
    }

    return result
}

/**
 * Invalidates the scene analysis cache. Call when detection context changes
 * (e.g., transitioning from frame analysis to one-shot capture).
 */
fun invalidateSceneCache() {
    cachedAnalysis = null
    cacheFrameCounter = 0
}

// ----------------------------------------------------------------
// Private strategy implementations
// ----------------------------------------------------------------

private fun toGray(input: Mat, pool: MatPool? = null, sharedGray: Mat? = null): Mat {
    // Reuse shared gray if available (caller-owned, we copy to avoid aliasing)
    if (sharedGray != null) {
        val copy = Mat()
        sharedGray.copyTo(copy)
        return copy
    }
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
private fun preprocessStandard(input: Mat, pool: MatPool? = null, sharedGray: Mat? = null): Mat {
    val gray = toGray(input, pool, sharedGray)
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
private fun preprocessClahe(input: Mat, pool: MatPool? = null, sharedGray: Mat? = null): Mat {
    val gray = toGray(input, pool, sharedGray)
    val clahe = Imgproc.createCLAHE(3.0, Size(4.0, 4.0))
    val enhanced = Mat()
    clahe.apply(gray, enhanced)
    gray.release()

    val blurred = Mat()
    // 5x5 kernel (smaller than standard 9x9) preserves more edge gradient after CLAHE,
    // critical for low-contrast scenes where gradients are already subtle.
    Imgproc.GaussianBlur(enhanced, blurred, Size(5.0, 5.0), 0.0)
    enhanced.release()
    return blurred
}

/**
 * Uses the inverted saturation channel from HSV color space.
 * White documents have near-zero saturation while colored backgrounds
 * have high saturation. Inverting makes the document bright (high intensity)
 * regardless of background color, producing strong edges at the boundary.
 */
private fun preprocessSaturation(input: Mat, pool: MatPool? = null): Mat {
    if (input.channels() < 3) {
        // Fall back to standard for grayscale input
        return preprocessStandard(input, pool)
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
private fun preprocessBilateral(input: Mat, pool: MatPool? = null, sharedGray: Mat? = null): Mat {
    val gray = toGray(input, pool, sharedGray)
    val filtered = Mat()
    // d=9: neighborhood diameter
    // sigmaColor=75: color space filter sigma (higher = more colors mixed)
    // sigmaSpace=75: coordinate space filter sigma (higher = wider mixing)
    Imgproc.bilateralFilter(gray, filtered, 9, 75.0, 75.0)
    gray.release()
    return filtered
}

// ----------------------------------------------------------------
// WP-1..5: Low-contrast / white-on-white strategy implementations
// ----------------------------------------------------------------

/**
 * Adaptive threshold binarization for white-on-white scenes.
 * Large block size (51) captures document boundary scale, not text lines.
 * Returns a binary edge image (0/255) — bypasses Canny in the detector.
 *
 * Pipeline: grayscale → blur → adaptive threshold → morph gradient (boundary extraction) → morph close.
 */
private fun preprocessAdaptiveThreshold(input: Mat, pool: MatPool? = null, sharedGray: Mat? = null): Mat {
    val gray = toGray(input, pool, sharedGray)

    // Light blur to reduce noise before thresholding
    val blurred = Mat()
    Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
    gray.release()

    val binary = Mat()
    // blockSize=51: large enough to capture document boundary, not text lines.
    // C=5: pixels must be noticeably darker than their local neighborhood
    // to be classified as background (0). The document (brighter) stays white (255).
    Imgproc.adaptiveThreshold(
        blurred, binary, 255.0,
        Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
        Imgproc.THRESH_BINARY,
        51, 5.0
    )
    blurred.release()

    // Morphological gradient (dilate - erode) extracts the boundary of the
    // binary segmentation, producing thin edge lines suitable for findContours.
    val edges = Mat()
    Imgproc.morphologyEx(binary, edges, Imgproc.MORPH_GRADIENT, preprocessMorphKernel3x3)
    binary.release()

    // Close to bridge small gaps at document boundary
    Imgproc.morphologyEx(edges, edges, Imgproc.MORPH_CLOSE, preprocessMorphKernel3x3)

    return edges
}

/**
 * LAB L-channel with aggressive CLAHE for white-on-white scenes.
 * LAB separates luminance from chrominance — surfaces that differ in warmth/coolness
 * (e.g., cream tablecloth vs white paper) show micro-contrast on the L channel that
 * is invisible in grayscale. Aggressive CLAHE (clipLimit=6.0, tileSize=2x2) amplifies
 * these subtle differences.
 *
 * Returns a grayscale image — feeds through Canny at 30/60 thresholds.
 */
private fun preprocessLabClahe(input: Mat, pool: MatPool? = null, sharedGray: Mat? = null): Mat {
    if (input.channels() < 3) {
        // Grayscale input: apply aggressive CLAHE directly (no LAB conversion)
        val gray = toGray(input, pool, sharedGray)
        val clahe = Imgproc.createCLAHE(6.0, Size(2.0, 2.0))
        val enhanced = Mat()
        clahe.apply(gray, enhanced)
        gray.release()
        val blurred = Mat()
        Imgproc.GaussianBlur(enhanced, blurred, Size(5.0, 5.0), 0.0)
        enhanced.release()
        return blurred
    }

    val bgr = if (input.channels() == 4) {
        val b = Mat()
        Imgproc.cvtColor(input, b, Imgproc.COLOR_RGBA2BGR)
        b
    } else {
        input
    }

    val lab = Mat()
    Imgproc.cvtColor(bgr, lab, Imgproc.COLOR_BGR2Lab)
    if (bgr !== input) bgr.release()

    val channels = mutableListOf<Mat>()
    Core.split(lab, channels)
    lab.release()

    val lChannel = channels[0]
    for (i in 1 until channels.size) channels[i].release()

    // Aggressive CLAHE: higher clip limit and smaller tiles amplify micro-contrast
    // between surfaces that differ in warmth/coolness
    val clahe = Imgproc.createCLAHE(6.0, Size(2.0, 2.0))
    val enhanced = Mat()
    clahe.apply(lChannel, enhanced)
    lChannel.release()

    val blurred = Mat()
    Imgproc.GaussianBlur(enhanced, blurred, Size(5.0, 5.0), 0.0)
    enhanced.release()

    return blurred
}

/**
 * Sobel gradient magnitude thresholded at the 95th percentile.
 * The document boundary is the strongest gradient in a white-on-white scene
 * regardless of absolute intensity — relative strength is what matters.
 * Returns a binary edge image (0/255) — bypasses Canny in the detector.
 *
 * Pipeline: grayscale → Sobel X/Y → magnitude → normalize → 95th percentile threshold → morph close.
 */
private fun preprocessGradientMagnitude(input: Mat, pool: MatPool? = null, sharedGray: Mat? = null): Mat {
    val gray = toGray(input, pool, sharedGray)

    val gradX = Mat()
    val gradY = Mat()
    Imgproc.Sobel(gray, gradX, CvType.CV_32F, 1, 0)
    Imgproc.Sobel(gray, gradY, CvType.CV_32F, 0, 1)
    gray.release()

    val magnitude = Mat()
    Core.magnitude(gradX, gradY, magnitude)
    gradX.release()
    gradY.release()

    // Normalize to 0-255 for percentile computation
    val magnitudeU8 = Mat()
    Core.normalize(magnitude, magnitudeU8, 0.0, 255.0, Core.NORM_MINMAX)
    magnitude.release()
    magnitudeU8.convertTo(magnitudeU8, CvType.CV_8UC1)

    // Find 95th percentile threshold via histogram on raw byte data
    val data = ByteArray(magnitudeU8.total().toInt() * magnitudeU8.channels())
    magnitudeU8.get(0, 0, data)

    val histogram = IntArray(256)
    for (b in data) {
        histogram[b.toInt() and 0xFF]++
    }

    val totalPixels = data.size
    val target = (totalPixels * 0.95).toInt()
    var cumSum = 0
    var thresholdVal = 255
    for (i in 0 until 256) {
        cumSum += histogram[i]
        if (cumSum >= target) {
            thresholdVal = i
            break
        }
    }

    val binary = Mat()
    Imgproc.threshold(magnitudeU8, binary, thresholdVal.toDouble(), 255.0, Imgproc.THRESH_BINARY)
    magnitudeU8.release()

    // Morph close (5x5) to connect nearby gradient pixels into continuous edges
    Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_CLOSE, preprocessMorphKernel5x5)

    return binary
}

/**
 * Difference of Gaussians: GaussianBlur(3x3) - GaussianBlur(21x21).
 * Isolates edge-scale features while suppressing both fine texture (text, grain)
 * and broad illumination gradients. For white-on-white scenes, the document
 * boundary is the only feature at the right spatial scale.
 *
 * Returns a grayscale image — feeds through Canny at very low thresholds (10/30).
 */
private fun preprocessDoG(input: Mat, pool: MatPool? = null, sharedGray: Mat? = null): Mat {
    val gray = toGray(input, pool, sharedGray)

    val blurSmall = Mat()
    val blurLarge = Mat()
    Imgproc.GaussianBlur(gray, blurSmall, Size(3.0, 3.0), 0.0)
    Imgproc.GaussianBlur(gray, blurLarge, Size(21.0, 21.0), 0.0)
    gray.release()

    // DoG = small blur - large blur: isolates edge-scale features
    val dog = Mat()
    Core.subtract(blurSmall, blurLarge, dog)
    blurSmall.release()
    blurLarge.release()

    // Subtraction can produce negative values — take absolute value for edge strength
    Core.convertScaleAbs(dog, dog)

    return dog
}

/**
 * Multi-channel edge fusion: runs Canny(20/50) on each BGR channel independently,
 * then combines with bitwise OR. Captures per-channel gradients invisible in
 * grayscale (e.g., blue channel difference between white paper and cream surface).
 *
 * Returns a binary edge image (0/255) — bypasses Canny in the detector.
 * Most expensive strategy (~8-12ms), used as last resort.
 */
private fun preprocessMultichannelFusion(input: Mat): Mat {
    if (input.channels() < 3) {
        // Grayscale: fall back to single-channel Canny with low thresholds
        val edges = Mat()
        Imgproc.Canny(input, edges, 20.0, 50.0)
        Imgproc.morphologyEx(edges, edges, Imgproc.MORPH_CLOSE, preprocessMorphKernel3x3)
        return edges
    }

    val bgr = if (input.channels() == 4) {
        val b = Mat()
        Imgproc.cvtColor(input, b, Imgproc.COLOR_RGBA2BGR)
        b
    } else {
        input
    }

    // Split into B, G, R channels
    val channels = mutableListOf<Mat>()
    Core.split(bgr, channels)
    if (bgr !== input) bgr.release()

    // Run Canny on each channel with low thresholds, OR results.
    // Per-channel gradients capture color differences invisible in grayscale.
    var combined: Mat? = null
    for (ch in channels) {
        val edges = Mat()
        Imgproc.Canny(ch, edges, 20.0, 50.0)
        ch.release()
        if (combined == null) {
            combined = edges
        } else {
            Core.bitwise_or(combined, edges, combined)
            edges.release()
        }
    }

    // Morph close to bridge gaps across channels
    Imgproc.morphologyEx(combined!!, combined, Imgproc.MORPH_CLOSE, preprocessMorphKernel3x3)

    return combined
}

// ----------------------------------------------------------------
// Directional gradient smoothing (v1.2.5)
// ----------------------------------------------------------------

/**
 * Tilt angles (degrees) for directional gradient smoothing kernels.
 * +-10 degrees covers typical camera-to-document alignment with <0.2dB worst-case loss.
 * 5 angles x 2 directions (H/V) = 10 filter2D calls total.
 */
private val TILT_ANGLES_DEG = doubleArrayOf(-10.0, -5.0, 0.0, 5.0, 10.0)

/** Length of the 1D smoothing kernel in pixels. 21px at 640x480 ≈ 3.3% of width. */
private const val KERNEL_LENGTH = 21

/**
 * Cached tilted kernels for horizontal edge detection (applied to |Gy|).
 * Each kernel is a 2D Mat where non-zero elements follow a line tilted at the
 * corresponding angle from horizontal, normalized to sum to 1.
 * Lazy initialization — created once, reused across all frames.
 */
private val horizontalKernels: Array<Mat> by lazy {
    Array(TILT_ANGLES_DEG.size) { i ->
        createTiltedKernel(angleDeg = TILT_ANGLES_DEG[i], isHorizontal = true)
    }
}

/**
 * Cached tilted kernels for vertical edge detection (applied to |Gx|).
 * Each kernel is the transpose of the horizontal kernel at the same angle.
 */
private val verticalKernels: Array<Mat> by lazy {
    Array(TILT_ANGLES_DEG.size) { i ->
        createTiltedKernel(angleDeg = TILT_ANGLES_DEG[i], isHorizontal = false)
    }
}

/**
 * Creates a tilted 1D averaging kernel as a 2D Mat.
 *
 * For horizontal smoothing (detecting horizontal edges via |Gy|): the base direction
 * is along the X axis, tilted by [angleDeg]. The kernel Mat is sized to fit the
 * rotated line, with non-zero elements along the tilted line, each = 1/KERNEL_LENGTH.
 *
 * For vertical smoothing (detecting vertical edges via |Gx|): same but the base
 * direction is along the Y axis.
 *
 * The kernel is a sparse 2D Mat where only pixels along the tilted line are non-zero.
 * Using Bresenham-style integer rounding to place kernel elements on the grid.
 */
private fun createTiltedKernel(angleDeg: Double, isHorizontal: Boolean): Mat {
    val angleRad = Math.toRadians(angleDeg)

    // Direction vector for the kernel line
    val dx: Double
    val dy: Double
    if (isHorizontal) {
        // Base direction horizontal (1, 0), tilted by angleDeg
        dx = cos(angleRad)
        dy = sin(angleRad)
    } else {
        // Base direction vertical (0, 1), tilted by angleDeg
        dx = -sin(angleRad)
        dy = cos(angleRad)
    }

    // Compute all sample positions centered at origin
    val halfLen = KERNEL_LENGTH / 2
    val positions = Array(KERNEL_LENGTH) { i ->
        val t = (i - halfLen).toDouble()
        Pair((t * dx).roundToInt(), (t * dy).roundToInt())
    }

    // Find bounding box
    val minX = positions.minOf { it.first }
    val maxX = positions.maxOf { it.first }
    val minY = positions.minOf { it.second }
    val maxY = positions.maxOf { it.second }

    val kernelWidth = maxX - minX + 1
    val kernelHeight = maxY - minY + 1

    // Create kernel Mat (CV_32F) initialized to zero
    val kernel = Mat.zeros(kernelHeight, kernelWidth, CvType.CV_32F)
    val weight = 1.0f / KERNEL_LENGTH

    // Place kernel elements along the tilted line
    for ((px, py) in positions) {
        val col = px - minX
        val row = py - minY
        kernel.put(row, col, floatArrayOf(weight))
    }

    return kernel
}

/**
 * Directional gradient smoothing for ultra-low-contrast white-on-white scenes.
 *
 * Smooths Sobel |Gy| along near-horizontal directions and |Gx| along near-vertical
 * directions using 5 tilted 1D kernels at [-10, -5, 0, +5, +10] degrees.
 * Coherent edge gradients accumulate (5x SNR improvement for a perfectly aligned edge)
 * while random noise and texture cancel. Mathematically equivalent to a local
 * restricted Radon transform.
 *
 * Returns a binary edge image (0/255) — bypasses Canny in the detector.
 *
 * Pipeline:
 * 1. Pre-blur (5x5 Gaussian, sigma=1.4) to widen gradient response
 * 2. Sobel Gx and Gy (CV_16S, ksize=3) → absolute values (CV_8U)
 * 3. For each of 5 tilt angles:
 *    - filter2D |Gy| with tilted horizontal kernel → horizontal edge response
 *    - filter2D |Gx| with tilted vertical kernel → vertical edge response
 * 4. Per-pixel max across 5 angles for H and V separately
 * 5. Combined response = max(H, V)
 * 6. Normalize to 0-255, threshold at 90th percentile
 * 7. Morphological close (3x3) to connect nearby edge fragments
 */
private fun preprocessDirectionalGradient(input: Mat, pool: MatPool? = null, sharedGray: Mat? = null): Mat {
    val gray = toGray(input, pool, sharedGray)

    // Step 1: Pre-blur to widen gradient response, improving tilt tolerance.
    // sigma=1.4 matches 5x5 kernel's natural sigma; larger would blur away the
    // subtle gradients we're trying to detect.
    val blurred = Mat()
    Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 1.4)
    gray.release()

    // Step 2: Sobel gradients in CV_16S (signed, avoids clipping at 255)
    val gradX16 = Mat()
    val gradY16 = Mat()
    Imgproc.Sobel(blurred, gradX16, CvType.CV_16S, 1, 0, /* ksize= */ 3)
    Imgproc.Sobel(blurred, gradY16, CvType.CV_16S, 0, 1, /* ksize= */ 3)
    blurred.release()

    // Convert to absolute values in CV_8U for filter2D input
    val absGx = Mat()
    val absGy = Mat()
    Core.convertScaleAbs(gradX16, absGx)
    Core.convertScaleAbs(gradY16, absGy)
    gradX16.release()
    gradY16.release()

    // Step 3-4: Apply tilted kernels and compute per-pixel max across angles.
    // H response: smooth |Gy| along near-horizontal directions (detects horizontal edges)
    // V response: smooth |Gx| along near-vertical directions (detects vertical edges)
    var hResponse: Mat? = null
    var vResponse: Mat? = null
    val tempH = Mat()
    val tempV = Mat()

    try {
        for (i in TILT_ANGLES_DEG.indices) {
            // Horizontal edge response: filter |Gy| with tilted horizontal kernel
            Imgproc.filter2D(absGy, tempH, CvType.CV_8U, horizontalKernels[i])
            if (hResponse == null) {
                hResponse = tempH.clone()
            } else {
                Core.max(hResponse, tempH, hResponse)
            }

            // Vertical edge response: filter |Gx| with tilted vertical kernel
            Imgproc.filter2D(absGx, tempV, CvType.CV_8U, verticalKernels[i])
            if (vResponse == null) {
                vResponse = tempV.clone()
            } else {
                Core.max(vResponse, tempV, vResponse)
            }
        }
    } finally {
        tempH.release()
        tempV.release()
        absGx.release()
        absGy.release()
    }

    // Step 5: Combine H and V responses — max captures both edge orientations
    val combined = Mat()
    Core.max(hResponse!!, vResponse!!, combined)
    hResponse.release()
    vResponse.release()

    // Step 6: Normalize to 0-255 range, then threshold at 90th percentile.
    // Normalization ensures the threshold percentile is meaningful regardless
    // of absolute gradient magnitudes (which vary with scene contrast).
    Core.normalize(combined, combined, 0.0, 255.0, Core.NORM_MINMAX)
    combined.convertTo(combined, CvType.CV_8UC1)

    // Find 90th percentile via histogram (O(n) scan, no sorting needed).
    // 90th percentile keeps the top 10% of gradient accumulations — document
    // boundary edges after directional smoothing are well above this.
    val data = ByteArray(combined.total().toInt() * combined.channels())
    combined.get(0, 0, data)

    val histogram = IntArray(256)
    for (b in data) {
        histogram[b.toInt() and 0xFF]++
    }

    val totalPixels = data.size
    val target = (totalPixels * 0.90).toInt()
    var cumSum = 0
    var thresholdVal = 255
    for (i in 0 until 256) {
        cumSum += histogram[i]
        if (cumSum >= target) {
            thresholdVal = i
            break
        }
    }

    val binary = Mat()
    Imgproc.threshold(combined, binary, thresholdVal.toDouble(), 255.0, Imgproc.THRESH_BINARY)
    combined.release()

    // Step 7: Morph close (3x3) to connect nearby edge fragments into continuous contours
    Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_CLOSE, preprocessMorphKernel3x3)

    return binary
}
