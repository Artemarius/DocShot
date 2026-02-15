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
