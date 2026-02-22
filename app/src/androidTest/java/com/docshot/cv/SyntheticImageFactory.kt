package com.docshot.cv

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * Generates synthetic test images via OpenCV draw primitives for
 * regression testing of the document detection pipeline.
 *
 * All factory methods return a BGR Mat that the caller must release.
 * Default image size is 800x600 (landscape).
 */
object SyntheticImageFactory {

    private const val DEFAULT_WIDTH = 800
    private const val DEFAULT_HEIGHT = 600

    // ----------------------------------------------------------------
    // Corner presets (in pixel coordinates for 800x600 images)
    // ----------------------------------------------------------------

    /** A4-proportioned document covering ~55% of the image area. */
    fun defaultA4Corners(): List<Point> = listOf(
        Point(150.0, 50.0),   // TL
        Point(650.0, 50.0),   // TR
        Point(650.0, 550.0),  // BR
        Point(150.0, 550.0)   // BL
    )

    /** Receipt-shaped document (~5% area, tall and narrow). */
    fun defaultReceiptCorners(): List<Point> = listOf(
        Point(320.0, 100.0),  // TL
        Point(480.0, 100.0),  // TR
        Point(480.0, 500.0),  // BR
        Point(320.0, 500.0)   // BL
    )

    /** Business card sized document (~8% area). */
    fun defaultBusinessCardCorners(): List<Point> = listOf(
        Point(280.0, 200.0),  // TL
        Point(520.0, 200.0),  // TR
        Point(520.0, 400.0),  // BR
        Point(280.0, 400.0)   // BL
    )

    /** US Letter-proportioned document (~15% area, 1:1.294 ratio). */
    fun defaultLetterCorners(): List<Point> = listOf(
        Point(200.0, 60.0),   // TL
        Point(600.0, 60.0),   // TR
        Point(600.0, 578.0),  // BR  (400w x 518h ≈ 1:1.295)
        Point(200.0, 578.0)   // BL
    )

    /** CR-80 ID card-sized document (~4% area, 1:1.586 ratio). */
    fun defaultIdCardCorners(): List<Point> = listOf(
        Point(310.0, 210.0),  // TL
        Point(490.0, 210.0),  // TR
        Point(490.0, 495.0),  // BR  (180w x 285h ≈ 1:1.583)
        Point(310.0, 495.0)   // BL
    )

    // ----------------------------------------------------------------
    // Image generators
    // ----------------------------------------------------------------

    /**
     * Baseline: white document on a solid-colored background.
     * High contrast, easy detection.
     */
    fun whiteDocOnSolidBg(
        bgColor: Scalar = Scalar(60.0, 60.0, 60.0),
        docColor: Scalar = Scalar(240.0, 240.0, 240.0),
        corners: List<Point> = defaultA4Corners(),
        width: Int = DEFAULT_WIDTH,
        height: Int = DEFAULT_HEIGHT
    ): Mat {
        val image = Mat(height, width, CvType.CV_8UC3, bgColor)
        fillQuad(image, corners, docColor)
        return image
    }

    /**
     * Low-contrast scene: beige background with white document.
     * Tests CLAHE strategy.
     */
    fun lowContrastDoc(
        bgColor: Scalar = Scalar(130.0, 135.0, 140.0),
        docColor: Scalar = Scalar(210.0, 210.0, 215.0),
        corners: List<Point> = defaultA4Corners(),
        width: Int = DEFAULT_WIDTH,
        height: Int = DEFAULT_HEIGHT
    ): Mat {
        val image = Mat(height, width, CvType.CV_8UC3, bgColor)
        fillQuad(image, corners, docColor)
        return image
    }

    /**
     * Low-light scene: all pixel values scaled by [brightness] factor.
     * @param brightness 0.0-1.0 where 0.25 = very dim.
     */
    fun lowLightDoc(
        brightness: Double = 0.25,
        corners: List<Point> = defaultA4Corners(),
        width: Int = DEFAULT_WIDTH,
        height: Int = DEFAULT_HEIGHT
    ): Mat {
        val image = whiteDocOnSolidBg(corners = corners, width = width, height = height)
        Core.multiply(image, Scalar(brightness, brightness, brightness), image)
        image.convertTo(image, CvType.CV_8UC3)
        return image
    }

    /**
     * Directional shadow across the document simulated as a gradient overlay.
     * @param shadowIntensity 0.0 = no shadow, 1.0 = fully black shadow on one side.
     */
    fun shadowedDoc(
        shadowIntensity: Double = 0.6,
        corners: List<Point> = defaultA4Corners(),
        width: Int = DEFAULT_WIDTH,
        height: Int = DEFAULT_HEIGHT
    ): Mat {
        val image = whiteDocOnSolidBg(corners = corners, width = width, height = height)

        // Create a horizontal gradient shadow from left (dark) to right (light)
        val shadow = Mat(height, width, CvType.CV_32FC3)
        for (col in 0 until width) {
            val factor = 1.0 - shadowIntensity * (1.0 - col.toDouble() / width)
            for (row in 0 until height) {
                shadow.put(row, col, factor, factor, factor)
            }
        }

        val imageFloat = Mat()
        image.convertTo(imageFloat, CvType.CV_32FC3)
        Core.multiply(imageFloat, shadow, imageFloat)
        imageFloat.convertTo(image, CvType.CV_8UC3)
        imageFloat.release()
        shadow.release()
        return image
    }

    /**
     * White document on a colored background (e.g., blue desk).
     * Tests saturation-channel strategy.
     */
    fun coloredBgDoc(
        bgColor: Scalar = Scalar(200.0, 100.0, 50.0),  // blue desk in BGR
        corners: List<Point> = defaultA4Corners(),
        width: Int = DEFAULT_WIDTH,
        height: Int = DEFAULT_HEIGHT
    ): Mat {
        val docColor = Scalar(240.0, 240.0, 240.0)
        val image = Mat(height, width, CvType.CV_8UC3, bgColor)
        fillQuad(image, corners, docColor)
        return image
    }

    /**
     * Document on a patterned background (parallel lines simulating wood grain).
     * Tests bilateral filtering strategy.
     */
    fun patternedBgDoc(
        patternType: String = "lines",
        corners: List<Point> = defaultA4Corners(),
        width: Int = DEFAULT_WIDTH,
        height: Int = DEFAULT_HEIGHT
    ): Mat {
        val image = Mat(height, width, CvType.CV_8UC3, Scalar(140.0, 120.0, 100.0))

        when (patternType) {
            "lines" -> {
                // Horizontal lines simulating wood grain
                for (y in 0 until height step 8) {
                    Imgproc.line(
                        image,
                        Point(0.0, y.toDouble()),
                        Point(width.toDouble(), y.toDouble()),
                        Scalar(100.0, 80.0, 60.0),
                        1
                    )
                }
            }
            "grid" -> {
                // Grid pattern
                for (y in 0 until height step 15) {
                    Imgproc.line(image, Point(0.0, y.toDouble()), Point(width.toDouble(), y.toDouble()), Scalar(110.0, 90.0, 70.0), 1)
                }
                for (x in 0 until width step 15) {
                    Imgproc.line(image, Point(x.toDouble(), 0.0), Point(x.toDouble(), height.toDouble()), Scalar(110.0, 90.0, 70.0), 1)
                }
            }
        }

        // Draw white document on top
        fillQuad(image, corners, Scalar(240.0, 240.0, 240.0))
        return image
    }

    /**
     * Small document (receipt or card sized) using given corners.
     * Typically 2-8% of image area.
     */
    fun smallDoc(
        corners: List<Point> = defaultReceiptCorners(),
        bgColor: Scalar = Scalar(60.0, 60.0, 60.0),
        docColor: Scalar = Scalar(240.0, 240.0, 240.0),
        width: Int = DEFAULT_WIDTH,
        height: Int = DEFAULT_HEIGHT
    ): Mat {
        val image = Mat(height, width, CvType.CV_8UC3, bgColor)
        fillQuad(image, corners, docColor)
        return image
    }

    /**
     * Document simulating partial visibility (extends to frame edges).
     *
     * Creates a large white rectangle with corners placed very close to
     * 2+ frame edges (within [EDGE_PROXIMITY_PX]). This simulates what
     * the camera sees when a document extends beyond the visible frame:
     * the detected quad touches multiple edges.
     *
     * [analyzeContours] flags such quads as partial documents.
     *
     * @param visibleCorners Controls how many edges the quad touches (2 or 3+).
     */
    fun partialDoc(
        visibleCorners: Int = 3,
        width: Int = DEFAULT_WIDTH,
        height: Int = DEFAULT_HEIGHT
    ): Mat {
        val image = Mat(height, width, CvType.CV_8UC3, Scalar(60.0, 60.0, 60.0))

        // All corners inside frame but touching edges (within 5px proximity).
        // Detected as a valid quad AND flagged as partial (touches 2+ edges).
        val corners = when (visibleCorners) {
            3 -> listOf(
                Point(100.0, 3.0),             // TL - near top edge
                Point(width - 3.0, 3.0),       // TR - near top + right edges
                Point(width - 3.0, 550.0),     // BR - near right edge
                Point(100.0, 550.0)            // BL - interior
            )
            else -> listOf(
                Point(100.0, 3.0),             // TL - near top edge
                Point(width - 3.0, 3.0),       // TR - near top + right edges
                Point(width - 3.0, height - 3.0), // BR - near right + bottom edges
                Point(100.0, height - 3.0)     // BL - near bottom edge
            )
        }

        fillQuad(image, corners, Scalar(240.0, 240.0, 240.0))
        return image
    }

    /**
     * Adds Gaussian sensor noise to an image.
     * @param stddev Standard deviation of the noise (10-30 typical for sensor noise).
     */
    fun addNoise(image: Mat, stddev: Double = 15.0): Mat {
        // Generate noise as 16-bit signed to allow negative values
        val noise = Mat(image.size(), CvType.CV_16SC3)
        Core.randn(noise, 0.0, stddev)
        // Convert image to 16-bit signed for safe addition (avoids overflow clipping)
        val image16 = Mat()
        image.convertTo(image16, CvType.CV_16SC3)
        val sumMat = Mat()
        Core.add(image16, noise, sumMat)
        image16.release()
        noise.release()
        // Convert back to 8-bit with saturation
        val result = Mat()
        sumMat.convertTo(result, CvType.CV_8UC3)
        sumMat.release()
        return result
    }

    /**
     * Vertical shadow across the document (top-to-bottom gradient).
     * Complements the horizontal shadow in [shadowedDoc].
     * @param shadowIntensity 0.0 = no shadow, 1.0 = fully black shadow at top.
     */
    fun verticalShadowDoc(
        shadowIntensity: Double = 0.6,
        corners: List<Point> = defaultA4Corners(),
        width: Int = DEFAULT_WIDTH,
        height: Int = DEFAULT_HEIGHT
    ): Mat {
        val image = whiteDocOnSolidBg(corners = corners, width = width, height = height)

        // Create a vertical gradient shadow from top (dark) to bottom (light)
        val shadow = Mat(height, width, CvType.CV_32FC3)
        for (row in 0 until height) {
            val factor = 1.0 - shadowIntensity * (1.0 - row.toDouble() / height)
            for (col in 0 until width) {
                shadow.put(row, col, factor, factor, factor)
            }
        }

        val imageFloat = Mat()
        image.convertTo(imageFloat, CvType.CV_32FC3)
        Core.multiply(imageFloat, shadow, imageFloat)
        imageFloat.convertTo(image, CvType.CV_8UC3)
        imageFloat.release()
        shadow.release()
        return image
    }

    /**
     * Overexposed / washed-out scene: lighter background with document
     * values pushed toward saturation via an exposure multiplier.
     * @param exposure Multiplier applied to all pixel values (1.5-2.0 typical).
     */
    fun overexposedDoc(
        exposure: Double = 1.8,
        corners: List<Point> = defaultA4Corners(),
        width: Int = DEFAULT_WIDTH,
        height: Int = DEFAULT_HEIGHT
    ): Mat {
        // Darker background so contrast survives the exposure boost
        // (100 * 1.8 = 180 bg vs 240 * 1.8 = 255 doc → still detectable)
        val image = whiteDocOnSolidBg(
            bgColor = Scalar(100.0, 100.0, 100.0),
            corners = corners,
            width = width,
            height = height
        )
        // Apply exposure multiplier — convertTo with alpha param clips to 0-255
        image.convertTo(image, CvType.CV_8UC3, exposure)
        return image
    }

    // ----------------------------------------------------------------
    // Low-contrast / white-on-white generators (WP-0)
    // ----------------------------------------------------------------

    /**
     * White document on near-white background (~30 unit gradient).
     * Tests the easiest white-on-white scenario.
     */
    fun whiteOnNearWhite(
        corners: List<Point> = defaultA4Corners(),
        width: Int = DEFAULT_WIDTH,
        height: Int = DEFAULT_HEIGHT
    ): Mat {
        val bgColor = Scalar(210.0, 210.0, 210.0)
        val docColor = Scalar(240.0, 240.0, 240.0)
        val image = Mat(height, width, CvType.CV_8UC3, bgColor)
        fillQuad(image, corners, docColor)
        return image
    }

    /**
     * White document on white background (~20 unit gradient).
     * Hardest uniform case — nearly invisible boundary.
     */
    fun whiteOnWhite(
        corners: List<Point> = defaultA4Corners(),
        width: Int = DEFAULT_WIDTH,
        height: Int = DEFAULT_HEIGHT
    ): Mat {
        val bgColor = Scalar(225.0, 225.0, 225.0)
        val docColor = Scalar(245.0, 245.0, 245.0)
        val image = Mat(height, width, CvType.CV_8UC3, bgColor)
        fillQuad(image, corners, docColor)
        return image
    }

    /**
     * White document on cream/warm background (~25 unit gradient with warm tone).
     * Cream tablecloth scenario — background has warm tint (more R than B in BGR).
     */
    fun whiteOnCream(
        corners: List<Point> = defaultA4Corners(),
        width: Int = DEFAULT_WIDTH,
        height: Int = DEFAULT_HEIGHT
    ): Mat {
        // BGR: B=200, G=210, R=220 → warm (cream) background
        val bgColor = Scalar(200.0, 210.0, 220.0)
        val docColor = Scalar(240.0, 240.0, 240.0)
        val image = Mat(height, width, CvType.CV_8UC3, bgColor)
        fillQuad(image, corners, docColor)
        return image
    }

    /**
     * White document on light wood background (~35 unit gradient with warm tone).
     * Light wood desk — larger gradient than cream, with noticeable warm tint.
     */
    fun whiteOnLightWood(
        corners: List<Point> = defaultA4Corners(),
        width: Int = DEFAULT_WIDTH,
        height: Int = DEFAULT_HEIGHT
    ): Mat {
        // BGR: B=180, G=200, R=215 → warm wood tone
        val bgColor = Scalar(180.0, 200.0, 215.0)
        val docColor = Scalar(240.0, 240.0, 240.0)
        val image = Mat(height, width, CvType.CV_8UC3, bgColor)
        fillQuad(image, corners, docColor)
        return image
    }

    /**
     * White document on textured white background (~15 unit gradient + noise).
     * Background has Gaussian noise (stddev=5) simulating fabric texture.
     * Document drawn on top is clean (solid color).
     */
    fun whiteOnWhiteTextured(
        corners: List<Point> = defaultA4Corners(),
        width: Int = DEFAULT_WIDTH,
        height: Int = DEFAULT_HEIGHT
    ): Mat {
        val bgColor = Scalar(225.0, 225.0, 225.0)
        val docColor = Scalar(240.0, 240.0, 240.0)
        val image = Mat(height, width, CvType.CV_8UC3, bgColor)

        // Add texture noise to background before drawing document
        val noise = Mat(height, width, CvType.CV_16SC3)
        Core.randn(noise, 0.0, 5.0)
        val image16 = Mat()
        image.convertTo(image16, CvType.CV_16SC3)
        Core.add(image16, noise, image16)
        image16.convertTo(image, CvType.CV_8UC3)
        image16.release()
        noise.release()

        // Document drawn on top overwrites noise in its region
        fillQuad(image, corners, docColor)
        return image
    }

    /**
     * White document on a glossy/gradient background (200→240 horizontal gradient).
     * Simulates reflective surfaces where the gradient varies across the background.
     * Document boundary contrast varies from ~40 (left) to ~0 (right).
     */
    fun glossyPaper(
        corners: List<Point> = defaultA4Corners(),
        width: Int = DEFAULT_WIDTH,
        height: Int = DEFAULT_HEIGHT
    ): Mat {
        // Create horizontal brightness gradient (200 at left, 240 at right)
        val gray = Mat(height, width, CvType.CV_8UC1)
        val rowData = ByteArray(width)
        for (col in 0 until width) {
            rowData[col] = (200 + 40 * col / width).toByte()
        }
        for (row in 0 until height) {
            gray.put(row, 0, rowData)
        }
        val image = Mat()
        Imgproc.cvtColor(gray, image, Imgproc.COLOR_GRAY2BGR)
        gray.release()

        fillQuad(image, corners, Scalar(240.0, 240.0, 240.0))
        return image
    }

    // ----------------------------------------------------------------
    // Ultra-low-contrast generators (v1.2.5 — 3-unit and 5-unit gradients)
    // ----------------------------------------------------------------

    /** Centered document covering ~65% of image area — used by ultra-low-contrast generators. */
    fun defaultUltraLowContrastCorners(
        width: Int = DEFAULT_WIDTH,
        height: Int = DEFAULT_HEIGHT
    ): List<Point> {
        // ~65% area: 80% of each dimension, centered
        val marginX = width * 0.10
        val marginY = height * 0.10
        return listOf(
            Point(marginX, marginY),                               // TL
            Point(width - marginX, marginY),                       // TR
            Point(width - marginX, height - marginY),              // BR
            Point(marginX, height - marginY)                       // BL
        )
    }

    /**
     * Ultra-low-contrast: only 3 units of gradient at the boundary.
     * Document RGB(230,230,230) on background RGB(233,233,233).
     * Theoretical floor for LSD (quant=1.0) detection. Pure signal — no texture, no noise.
     * Scene analysis: mean ~232, stddev ~1.3 → white-on-white (mean>180, stddev<35).
     */
    fun ultraLowContrast3Unit(
        width: Int = DEFAULT_WIDTH,
        height: Int = DEFAULT_HEIGHT
    ): Pair<Mat, List<Point>> {
        val corners = defaultUltraLowContrastCorners(width = width, height = height)
        val bgColor = Scalar(233.0, 233.0, 233.0)
        val docColor = Scalar(230.0, 230.0, 230.0)
        val image = Mat(height, width, CvType.CV_8UC3, bgColor)
        fillQuad(image, corners, docColor)
        return Pair(image, corners)
    }

    /**
     * Ultra-low-contrast: 5 units of gradient at the boundary.
     * Document RGB(225,225,225) on background RGB(230,230,230).
     * Should be detectable by DIRECTIONAL_GRADIENT strategy.
     * Scene analysis: mean ~228, stddev ~2.2 → white-on-white.
     */
    fun ultraLowContrast5Unit(
        width: Int = DEFAULT_WIDTH,
        height: Int = DEFAULT_HEIGHT
    ): Pair<Mat, List<Point>> {
        val corners = defaultUltraLowContrastCorners(width = width, height = height)
        val bgColor = Scalar(230.0, 230.0, 230.0)
        val docColor = Scalar(225.0, 225.0, 225.0)
        val image = Mat(height, width, CvType.CV_8UC3, bgColor)
        fillQuad(image, corners, docColor)
        return Pair(image, corners)
    }

    /**
     * Ultra-low-contrast: 5-unit gradient with Gaussian noise (stddev=3).
     * Tests whether directional smoothing can recover signal from noise.
     * SNR ~1.7 per pixel, ~7.7 after 21-pixel directional averaging.
     * Scene analysis: mean ~228, stddev ~3.5 → white-on-white.
     */
    fun ultraLowContrast5UnitNoisy(
        width: Int = DEFAULT_WIDTH,
        height: Int = DEFAULT_HEIGHT
    ): Pair<Mat, List<Point>> {
        val (cleanImage, corners) = ultraLowContrast5Unit(width = width, height = height)
        val noisyImage = addNoise(cleanImage, stddev = 3.0)
        cleanImage.release()
        return Pair(noisyImage, corners)
    }

    /**
     * Ultra-low-contrast: 5-unit gradient document tilted ~8 degrees.
     * Tests angle tolerance of DIRECTIONAL_GRADIENT (5-deg spacing, ±12.5 deg coverage)
     * and LSD (ang_th=22.5 deg). Corners computed via rotation about image center.
     * Scene analysis: mean ~228, stddev ~2.2 → white-on-white.
     */
    fun ultraLowContrastTilted8deg(
        width: Int = DEFAULT_WIDTH,
        height: Int = DEFAULT_HEIGHT
    ): Pair<Mat, List<Point>> {
        val bgColor = Scalar(230.0, 230.0, 230.0)
        val docColor = Scalar(225.0, 225.0, 225.0)
        val image = Mat(height, width, CvType.CV_8UC3, bgColor)

        // Compute axis-aligned document corners (~65% area, centered)
        val marginX = width * 0.15
        val marginY = height * 0.15
        val axisCorners = listOf(
            Point(marginX, marginY),
            Point(width - marginX, marginY),
            Point(width - marginX, height - marginY),
            Point(marginX, height - marginY)
        )

        // Rotate each corner ~8 degrees about image center
        val cx = width / 2.0
        val cy = height / 2.0
        val angleRad = Math.toRadians(8.0)
        val cosA = Math.cos(angleRad)
        val sinA = Math.sin(angleRad)

        val rotatedCorners = axisCorners.map { p ->
            val dx = p.x - cx
            val dy = p.y - cy
            Point(
                cx + dx * cosA - dy * sinA,
                cy + dx * sinA + dy * cosA
            )
        }

        fillQuad(image, rotatedCorners, docColor)
        return Pair(image, rotatedCorners)
    }

    /**
     * Ultra-low-contrast: 3-unit gradient with warm color difference.
     * Document RGB(230,228,225) on background RGB(233,231,228) (BGR order: 225,228,230 / 228,231,233).
     * Per-channel gradient is ~3 units but with slight warm/cool tone difference.
     * Tests whether multichannel approaches gain any advantage over grayscale.
     * Scene analysis: mean ~230, stddev ~2.2 → white-on-white.
     */
    fun ultraLowContrast3UnitWarm(
        width: Int = DEFAULT_WIDTH,
        height: Int = DEFAULT_HEIGHT
    ): Pair<Mat, List<Point>> {
        val corners = defaultUltraLowContrastCorners(width = width, height = height)
        // BGR order: B, G, R — warm means more red, less blue
        val bgColor = Scalar(228.0, 231.0, 233.0)   // RGB(233,231,228)
        val docColor = Scalar(225.0, 228.0, 230.0)   // RGB(230,228,225)
        val image = Mat(height, width, CvType.CV_8UC3, bgColor)
        fillQuad(image, corners, docColor)
        return Pair(image, corners)
    }

    // ----------------------------------------------------------------
    // Low-contrast non-white / grout line generators (v1.2.5 WP-E)
    // ----------------------------------------------------------------

    /**
     * White document on beige surface (~50 unit gradient).
     * Document RGB(240,240,240) on beige RGB(190,180,160) — warm surface with color tint.
     * Scene analysis: mean ~200, stddev ~25-30 → isLowContrast but NOT isWhiteOnWhite
     * (mean borderline ~200, stddev ~25 exceeds the tight <35 check only because
     * the contrast is moderate). Tests strategy broadening for non-white low-contrast scenes.
     */
    fun docOnBeigeSurface(
        width: Int = DEFAULT_WIDTH,
        height: Int = DEFAULT_HEIGHT
    ): Pair<Mat, List<Point>> {
        val corners = defaultA4Corners()
        // BGR order: B=160, G=180, R=190 → warm beige
        val bgColor = Scalar(160.0, 180.0, 190.0)
        val docColor = Scalar(240.0, 240.0, 240.0)
        val image = Mat(height, width, CvType.CV_8UC3, bgColor)
        fillQuad(image, corners, docColor)
        return Pair(image, corners)
    }

    /**
     * White document on tan tile surface with grout lines crossing the document boundary.
     * THE key regression test — simulates the exact in-field failure scenario.
     * Document RGB(230,230,230) on tan RGB(160,150,130) with 3H + 3V grout lines
     * (RGB ~120, 3px wide) spanning the full image including across the document boundary.
     * Scene analysis: mean ~170, stddev ~30-35 → isLowContrast, not isWhiteOnWhite.
     * Grout lines fragment document contours → tests line suppression in EdgeDetector.
     */
    fun docOnTanSurfaceWithGroutLines(
        width: Int = DEFAULT_WIDTH,
        height: Int = DEFAULT_HEIGHT
    ): Pair<Mat, List<Point>> {
        val corners = defaultA4Corners()
        // BGR order: B=130, G=150, R=160 → warm tan
        val bgColor = Scalar(130.0, 150.0, 160.0)
        val docColor = Scalar(230.0, 230.0, 230.0)
        val image = Mat(height, width, CvType.CV_8UC3, bgColor)
        fillQuad(image, corners, docColor)

        // Grout line color: dark gray RGB(120,120,120) → BGR(120,120,120)
        val groutColor = Scalar(120.0, 120.0, 120.0)
        val groutThickness = 3

        // 3 horizontal grout lines — evenly spaced, spanning full width
        val hSpacing = height / 4
        for (i in 1..3) {
            val y = i * hSpacing
            Imgproc.line(
                image,
                Point(0.0, y.toDouble()),
                Point(width.toDouble(), y.toDouble()),
                groutColor,
                groutThickness
            )
        }

        // 3 vertical grout lines — evenly spaced, spanning full height
        val vSpacing = width / 4
        for (i in 1..3) {
            val x = i * vSpacing
            Imgproc.line(
                image,
                Point(x.toDouble(), 0.0),
                Point(x.toDouble(), height.toDouble()),
                groutColor,
                groutThickness
            )
        }

        return Pair(image, corners)
    }

    /**
     * Light gray document on medium gray surface (~50 unit gradient, achromatic).
     * Document RGB(200,200,200) on surface RGB(150,150,150).
     * Scene analysis: mean ~176, stddev ~25 → isLowContrast (stddev < 40), not isWhiteOnWhite (mean < 180).
     * Tests DOG effectiveness on non-white, purely grayscale low-contrast scenes.
     */
    fun docOnGraySurfaceLowContrast(
        width: Int = DEFAULT_WIDTH,
        height: Int = DEFAULT_HEIGHT
    ): Pair<Mat, List<Point>> {
        val corners = defaultA4Corners()
        val bgColor = Scalar(150.0, 150.0, 150.0)
        val docColor = Scalar(200.0, 200.0, 200.0)
        val image = Mat(height, width, CvType.CV_8UC3, bgColor)
        fillQuad(image, corners, docColor)
        return Pair(image, corners)
    }

    /**
     * White document on tile floor pattern with grout grid.
     * Document RGB(235,235,235) on alternating tile colors (RGB ~180 and ~165)
     * with grout grid lines (RGB ~100, 2px wide) at ~80px intervals.
     * Scene analysis: mean ~175, stddev ~30 → isLowContrast, not isWhiteOnWhite.
     * Tests line suppression with a dense regular grid pattern.
     */
    fun docOnTileFloor(
        width: Int = DEFAULT_WIDTH,
        height: Int = DEFAULT_HEIGHT
    ): Pair<Mat, List<Point>> {
        val corners = defaultA4Corners()

        // Base tile pattern: alternating tile colors in a checkerboard
        val tileSize = 80
        val tileColor1 = Scalar(180.0, 180.0, 180.0)  // lighter tile
        val tileColor2 = Scalar(165.0, 165.0, 165.0)  // darker tile
        val image = Mat(height, width, CvType.CV_8UC3, tileColor1)

        // Fill darker tiles in checkerboard pattern
        for (tileRow in 0..(height / tileSize)) {
            for (tileCol in 0..(width / tileSize)) {
                if ((tileRow + tileCol) % 2 == 1) {
                    val x1 = tileCol * tileSize
                    val y1 = tileRow * tileSize
                    val x2 = minOf(x1 + tileSize, width)
                    val y2 = minOf(y1 + tileSize, height)
                    Imgproc.rectangle(
                        image,
                        Point(x1.toDouble(), y1.toDouble()),
                        Point(x2.toDouble(), y2.toDouble()),
                        tileColor2,
                        -1  // filled
                    )
                }
            }
        }

        // Grout grid lines at tile boundaries
        val groutColor = Scalar(100.0, 100.0, 100.0)
        val groutThickness = 2

        // Horizontal grout lines
        for (y in tileSize..height step tileSize) {
            Imgproc.line(
                image,
                Point(0.0, y.toDouble()),
                Point(width.toDouble(), y.toDouble()),
                groutColor,
                groutThickness
            )
        }

        // Vertical grout lines
        for (x in tileSize..width step tileSize) {
            Imgproc.line(
                image,
                Point(x.toDouble(), 0.0),
                Point(x.toDouble(), height.toDouble()),
                groutColor,
                groutThickness
            )
        }

        // Draw document on top of tile pattern
        fillQuad(image, corners, Scalar(235.0, 235.0, 235.0))
        return Pair(image, corners)
    }

    /**
     * NO document — just a surface with spanning lines. FALSE POSITIVE guard.
     * Surface RGB(170,170,170) with 5 spanning lines (2 horizontal, 3 vertical)
     * at various intensities (100-140) and widths (2-3px).
     * Scene analysis: mean ~165, stddev ~15.
     * No document should be detected — lines alone must not form a quad.
     */
    fun spanningLinesNoDocs(
        width: Int = DEFAULT_WIDTH,
        height: Int = DEFAULT_HEIGHT
    ): Mat {
        val bgColor = Scalar(170.0, 170.0, 170.0)
        val image = Mat(height, width, CvType.CV_8UC3, bgColor)

        // 2 horizontal spanning lines
        Imgproc.line(
            image,
            Point(0.0, 180.0),
            Point(width.toDouble(), 180.0),
            Scalar(100.0, 100.0, 100.0),
            3
        )
        Imgproc.line(
            image,
            Point(0.0, 420.0),
            Point(width.toDouble(), 420.0),
            Scalar(130.0, 130.0, 130.0),
            2
        )

        // 3 vertical spanning lines
        Imgproc.line(
            image,
            Point(200.0, 0.0),
            Point(200.0, height.toDouble()),
            Scalar(110.0, 110.0, 110.0),
            2
        )
        Imgproc.line(
            image,
            Point(450.0, 0.0),
            Point(450.0, height.toDouble()),
            Scalar(140.0, 140.0, 140.0),
            3
        )
        Imgproc.line(
            image,
            Point(650.0, 0.0),
            Point(650.0, height.toDouble()),
            Scalar(120.0, 120.0, 120.0),
            2
        )

        return image
    }

    /**
     * White document on medium surface with one diagonal seam/crease line.
     * Document RGB(235,235,235) on surface RGB(155,155,155) with a diagonal seam
     * (RGB ~110, 3px wide) running from top-left to bottom-right of the image.
     * Scene analysis: mean ~180, stddev ~30.
     * Tests that diagonal spanning lines are suppressed without affecting document detection.
     */
    fun docWithDiagonalSeam(
        width: Int = DEFAULT_WIDTH,
        height: Int = DEFAULT_HEIGHT
    ): Pair<Mat, List<Point>> {
        val corners = defaultA4Corners()
        val bgColor = Scalar(155.0, 155.0, 155.0)
        val docColor = Scalar(235.0, 235.0, 235.0)
        val image = Mat(height, width, CvType.CV_8UC3, bgColor)
        fillQuad(image, corners, docColor)

        // Diagonal seam from top-left corner to bottom-right corner of the image
        // (not aligned with document edges)
        Imgproc.line(
            image,
            Point(0.0, 0.0),
            Point(width.toDouble(), height.toDouble()),
            Scalar(110.0, 110.0, 110.0),
            3
        )

        return Pair(image, corners)
    }

    // ----------------------------------------------------------------
    // False positive guard generators (no document present)
    // ----------------------------------------------------------------

    /**
     * Horizontal brightness gradient (180→240) with no document.
     * Must not trigger a false positive detection.
     */
    fun brightnessGradientNoDocs(
        width: Int = DEFAULT_WIDTH,
        height: Int = DEFAULT_HEIGHT
    ): Mat {
        val gray = Mat(height, width, CvType.CV_8UC1)
        val rowData = ByteArray(width)
        for (col in 0 until width) {
            rowData[col] = (180 + 60 * col / width).toByte()
        }
        for (row in 0 until height) {
            gray.put(row, 0, rowData)
        }
        val image = Mat()
        Imgproc.cvtColor(gray, image, Imgproc.COLOR_GRAY2BGR)
        gray.release()
        return image
    }

    /**
     * Uniform white background with Gaussian noise (stddev=15), no document.
     * Must not trigger a false positive detection.
     */
    fun noisyWhiteNoDocs(
        width: Int = DEFAULT_WIDTH,
        height: Int = DEFAULT_HEIGHT
    ): Mat {
        val image = Mat(height, width, CvType.CV_8UC3, Scalar(220.0, 220.0, 220.0))
        val noisy = addNoise(image, stddev = 15.0)
        image.release()
        return noisy
    }

    // ----------------------------------------------------------------
    // Helpers
    // ----------------------------------------------------------------

    /**
     * Fills a quadrilateral region on the image with the given color.
     * Corners outside the image bounds are clipped by OpenCV.
     */
    private fun fillQuad(image: Mat, corners: List<Point>, color: Scalar) {
        val pts = MatOfPoint(*corners.toTypedArray())
        Imgproc.fillConvexPoly(image, pts, color)
        pts.release()
    }
}
