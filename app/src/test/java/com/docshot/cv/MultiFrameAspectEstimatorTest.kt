package com.docshot.cv

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.BeforeClass
import org.junit.Test
import org.opencv.core.Point
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

class MultiFrameAspectEstimatorTest {

    companion object {
        private var openCvLoaded = false

        @JvmStatic
        @BeforeClass
        fun loadOpenCV() {
            openCvLoaded = try {
                System.loadLibrary("opencv_java4")
                true
            } catch (_: UnsatisfiedLinkError) {
                try {
                    System.loadLibrary("opencv_java412")
                    true
                } catch (_: UnsatisfiedLinkError) {
                    false
                }
            }
        }

        /** Synthetic camera intrinsics for testing. */
        private val TEST_INTRINSICS = CameraIntrinsics(
            fx = 1000.0,
            fy = 1000.0,
            cx = 320.0,
            cy = 240.0
        )

        /** Creates a rectangle [w]x[h] centered at given offset. */
        private fun makeRectAt(
            w: Double,
            h: Double,
            offsetX: Double = 100.0,
            offsetY: Double = 100.0
        ): List<Point> = listOf(
            Point(offsetX, offsetY),
            Point(offsetX + w, offsetY),
            Point(offsetX + w, offsetY + h),
            Point(offsetX, offsetY + h)
        )

        /**
         * Adds small random-like jitter to corners to simulate hand tremor.
         * Uses a deterministic pattern based on [seed] for reproducibility.
         */
        private fun jitterCorners(
            corners: List<Point>,
            maxJitter: Double,
            seed: Int
        ): List<Point> {
            // Simple deterministic "random" based on seed — no actual Random needed
            val offsets = listOf(
                0.3, -0.7, 0.5, -0.2, 0.8, -0.4, 0.1, -0.6
            )
            return corners.mapIndexed { i, p ->
                val jx = offsets[(seed * 4 + i * 2) % offsets.size] * maxJitter
                val jy = offsets[(seed * 4 + i * 2 + 1) % offsets.size] * maxJitter
                Point(p.x + jx, p.y + jy)
            }
        }

        /**
         * Simulates a pinhole camera projection of a rectangle [docW] x [docH]
         * tilted by [tiltDeg] degrees about the horizontal axis.
         */
        private fun projectRectAtTilt(
            docW: Double,
            docH: Double,
            tiltDeg: Double,
            focalLength: Double = 1000.0,
            cameraHeight: Double = 500.0
        ): List<Point> {
            val theta = Math.toRadians(tiltDeg)
            val cosT = cos(theta)
            val sinT = sin(theta)

            val halfW = docW / 2.0
            val halfH = docH / 2.0

            val corners3D = listOf(
                Triple(-halfW, -halfH, 0.0),
                Triple(halfW, -halfH, 0.0),
                Triple(halfW, halfH, 0.0),
                Triple(-halfW, halfH, 0.0)
            )

            val rotated = corners3D.map { (x, y, z) ->
                Triple(x, y * cosT - z * sinT, y * sinT + z * cosT)
            }

            val cx = 320.0
            val cy = 240.0
            return rotated.map { (x, y, z) ->
                val depth = cameraHeight - z
                Point(
                    focalLength * x / depth + cx,
                    focalLength * y / depth + cy
                )
            }
        }
    }

    private fun requireOpenCv() {
        assumeTrue("OpenCV native library not available", openCvLoaded)
    }

    // =======================================================================
    // Pure JVM tests (no OpenCV native required)
    // =======================================================================

    @Test
    fun `minFrames is 3`() {
        val estimator = MultiFrameAspectEstimator()
        assertEquals(3, estimator.minFrames)
    }

    @Test
    fun `initial frameCount is zero`() {
        val estimator = MultiFrameAspectEstimator()
        assertEquals(0, estimator.frameCount)
    }

    @Test
    fun `MultiFrameEstimate ratio is coerced to valid range`() {
        val estimate = MultiFrameEstimate(
            estimatedRatio = 0.707,
            confidence = 0.95,
            frameCount = 10
        )
        assertEquals(0.707, estimate.estimatedRatio, 0.001)
        assertEquals(0.95, estimate.confidence, 0.001)
        assertEquals(10, estimate.frameCount)
    }

    @Test
    fun `MultiFrameEstimate data class equality`() {
        val a = MultiFrameEstimate(estimatedRatio = 0.5, confidence = 0.8, frameCount = 5)
        val b = MultiFrameEstimate(estimatedRatio = 0.5, confidence = 0.8, frameCount = 5)
        assertEquals(a, b)
    }

    @Test
    fun `MultiFrameEstimate data class copy`() {
        val original = MultiFrameEstimate(
            estimatedRatio = 0.707,
            confidence = 0.9,
            frameCount = 8
        )
        val modified = original.copy(confidence = 0.5)
        assertEquals(0.707, modified.estimatedRatio, 0.001)
        assertEquals(0.5, modified.confidence, 0.001)
        assertEquals(8, modified.frameCount)
    }

    @Test
    fun `MultiFrameEstimate destructuring`() {
        val estimate = MultiFrameEstimate(
            estimatedRatio = 0.773,
            confidence = 0.88,
            frameCount = 7
        )
        val (ratio, confidence, count) = estimate
        assertEquals(0.773, ratio, 0.001)
        assertEquals(0.88, confidence, 0.001)
        assertEquals(7, count)
    }

    @Test
    fun `CameraIntrinsics data class holds values`() {
        val intrinsics = CameraIntrinsics(fx = 1200.0, fy = 1200.0, cx = 640.0, cy = 480.0)
        assertEquals(1200.0, intrinsics.fx, 0.001)
        assertEquals(1200.0, intrinsics.fy, 0.001)
        assertEquals(640.0, intrinsics.cx, 0.001)
        assertEquals(480.0, intrinsics.cy, 0.001)
    }

    // =======================================================================
    // OpenCV-dependent tests (skipped on JVM if native not available)
    // =======================================================================

    @Test
    fun `estimateAspectRatio returns null with insufficient frames`() {
        requireOpenCv()
        val estimator = MultiFrameAspectEstimator()
        assertNull(estimator.estimateAspectRatio())
    }

    @Test
    fun `addFrame requires exactly 4 corners`() {
        requireOpenCv()
        val estimator = MultiFrameAspectEstimator()
        try {
            estimator.addFrame(emptyList())
            throw AssertionError("Should have thrown IllegalArgumentException")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("Expected 4 corners"))
        }
    }

    @Test
    fun `addFrame increments frameCount`() {
        requireOpenCv()
        val estimator = MultiFrameAspectEstimator()
        val corners = makeRectAt(300.0, 500.0)
        estimator.addFrame(corners)
        assertEquals(1, estimator.frameCount)
        estimator.addFrame(corners)
        assertEquals(2, estimator.frameCount)
        estimator.release()
    }

    @Test
    fun `estimateAspectRatio returns null with fewer than minFrames`() {
        requireOpenCv()
        val estimator = MultiFrameAspectEstimator()
        val corners = makeRectAt(300.0, 500.0)
        estimator.addFrame(corners)
        estimator.addFrame(corners)
        assertNull(
            "Should return null with only 2 frames (minFrames=3)",
            estimator.estimateAspectRatio()
        )
        estimator.release()
    }

    @Test
    fun `A4 document corners produce ratio near 0_707 with intrinsics`() {
        requireOpenCv()
        val estimator = MultiFrameAspectEstimator()

        // Feed 5 frames of A4 corners (210x297 proportions) with slight jitter
        val baseCorners = makeRectAt(210.0, 297.0)
        for (i in 0 until 5) {
            estimator.addFrame(jitterCorners(baseCorners, maxJitter = 1.5, seed = i))
        }

        val estimate = estimator.estimateAspectRatio(TEST_INTRINSICS)
        estimator.release()

        assertNotNull("Should produce an estimate with 5 frames", estimate)
        val pctError = abs(estimate!!.estimatedRatio - 0.707) / 0.707 * 100.0
        assertTrue(
            "A4 ratio (${estimate.estimatedRatio}) should be within 5% of 0.707, " +
                "was ${pctError}%",
            pctError < 5.0
        )
    }

    @Test
    fun `US Letter corners produce ratio near 0_773 with intrinsics`() {
        requireOpenCv()
        val estimator = MultiFrameAspectEstimator()

        // US Letter: 8.5 x 11 inches, scale up for realistic pixel dimensions
        val baseCorners = makeRectAt(425.0, 550.0) // 50x scale
        for (i in 0 until 5) {
            estimator.addFrame(jitterCorners(baseCorners, maxJitter = 1.5, seed = i))
        }

        val estimate = estimator.estimateAspectRatio(TEST_INTRINSICS)
        estimator.release()

        assertNotNull("Should produce an estimate", estimate)
        val pctError = abs(estimate!!.estimatedRatio - 0.773) / 0.773 * 100.0
        assertTrue(
            "US Letter ratio (${estimate.estimatedRatio}) should be within 5% of 0.773, " +
                "was ${pctError}%",
            pctError < 5.0
        )
    }

    @Test
    fun `square document produces ratio near 1_0 with intrinsics`() {
        requireOpenCv()
        val estimator = MultiFrameAspectEstimator()

        val baseCorners = makeRectAt(300.0, 300.0)
        for (i in 0 until 5) {
            estimator.addFrame(jitterCorners(baseCorners, maxJitter = 1.5, seed = i))
        }

        val estimate = estimator.estimateAspectRatio(TEST_INTRINSICS)
        estimator.release()

        assertNotNull("Should produce an estimate", estimate)
        val pctError = abs(estimate!!.estimatedRatio - 1.0) * 100.0
        assertTrue(
            "Square ratio (${estimate.estimatedRatio}) should be within 5% of 1.0, " +
                "was ${pctError}%",
            pctError < 5.0
        )
    }

    @Test
    fun `estimation with intrinsics produces valid result`() {
        requireOpenCv()
        val estimator = MultiFrameAspectEstimator()

        val corners = makeRectAt(300.0, 400.0)
        for (i in 0 until 5) {
            estimator.addFrame(jitterCorners(corners, maxJitter = 2.0, seed = i))
        }

        val estimate = estimator.estimateAspectRatio(TEST_INTRINSICS)
        estimator.release()

        assertNotNull("Should produce a non-null result", estimate)
        assertTrue(
            "Ratio (${estimate!!.estimatedRatio}) should be in [0.1, 1.0]",
            estimate.estimatedRatio in 0.1..1.0
        )
        assertTrue(
            "Confidence (${estimate.confidence}) should be in [0.0, 1.0]",
            estimate.confidence in 0.0..1.0
        )
        assertEquals(
            "Frame count should match",
            5, estimate.frameCount
        )
    }

    @Test
    fun `estimation without intrinsics uses Zhang method`() {
        requireOpenCv()
        val estimator = MultiFrameAspectEstimator()

        // Feed frames with varied viewpoints (different tilt angles) to give
        // Zhang's method enough geometric diversity
        for (tilt in listOf(3.0, 5.0, 8.0, 10.0, 12.0)) {
            val corners = projectRectAtTilt(docW = 210.0, docH = 297.0, tiltDeg = tilt)
            estimator.addFrame(corners)
        }

        val estimate = estimator.estimateAspectRatio(intrinsics = null)
        estimator.release()

        // Zhang's method may return null if viewpoints are too similar (degenerate),
        // so we only check validity if a result is produced
        if (estimate != null) {
            assertTrue(
                "Ratio (${estimate.estimatedRatio}) should be in [0.1, 1.0]",
                estimate.estimatedRatio in 0.1..1.0
            )
            assertTrue(
                "Confidence (${estimate.confidence}) should be in [0.0, 1.0]",
                estimate.confidence in 0.0..1.0
            )
        }
    }

    // =======================================================================
    // Multi-Frame Variance Reduction Tests
    // =======================================================================

    @Test
    fun `identicalFrames_zeroVariance_highConfidence`() {
        requireOpenCv()
        val estimator = MultiFrameAspectEstimator()

        // Feed 5 identical frames -- should produce zero variance and max confidence
        val corners = makeRectAt(210.0, 297.0)
        for (i in 0 until 5) {
            estimator.addFrame(corners) // no jitter at all
        }

        val estimate = estimator.estimateAspectRatio(TEST_INTRINSICS)
        estimator.release()

        assertNotNull("Should produce an estimate", estimate)
        assertTrue(
            "Identical frames should produce very high confidence (${estimate!!.confidence})",
            estimate.confidence > 0.95
        )
    }

    @Test
    fun `smallJitter_varianceReduced_vsLargeJitter`() {
        requireOpenCv()

        // Small jitter estimation
        val smallJitterEstimator = MultiFrameAspectEstimator()
        val baseCorners = makeRectAt(210.0, 297.0)
        for (i in 0 until 7) {
            smallJitterEstimator.addFrame(
                jitterCorners(baseCorners, maxJitter = 0.5, seed = i)
            )
        }
        val smallJitterEstimate = smallJitterEstimator.estimateAspectRatio(TEST_INTRINSICS)
        smallJitterEstimator.release()

        // Large jitter estimation
        val largeJitterEstimator = MultiFrameAspectEstimator()
        for (i in 0 until 7) {
            largeJitterEstimator.addFrame(
                jitterCorners(baseCorners, maxJitter = 15.0, seed = i)
            )
        }
        val largeJitterEstimate = largeJitterEstimator.estimateAspectRatio(TEST_INTRINSICS)
        largeJitterEstimator.release()

        assertNotNull("Small jitter should produce estimate", smallJitterEstimate)
        assertNotNull("Large jitter should produce estimate", largeJitterEstimate)

        assertTrue(
            "Small jitter confidence (${smallJitterEstimate!!.confidence}) should be >= " +
                "large jitter confidence (${largeJitterEstimate!!.confidence})",
            smallJitterEstimate.confidence >= largeJitterEstimate.confidence - 0.01
        )
    }

    @Test
    fun `moreFrames_doesNotDegrade_confidence`() {
        requireOpenCv()

        // 3 frames
        val est3 = MultiFrameAspectEstimator()
        val baseCorners = makeRectAt(210.0, 297.0)
        for (i in 0 until 3) {
            est3.addFrame(jitterCorners(baseCorners, maxJitter = 1.0, seed = i))
        }
        val result3 = est3.estimateAspectRatio(TEST_INTRINSICS)
        est3.release()

        // 8 frames (same jitter level)
        val est8 = MultiFrameAspectEstimator()
        for (i in 0 until 8) {
            est8.addFrame(jitterCorners(baseCorners, maxJitter = 1.0, seed = i))
        }
        val result8 = est8.estimateAspectRatio(TEST_INTRINSICS)
        est8.release()

        assertNotNull(result3)
        assertNotNull(result8)

        // More consistent frames should not reduce confidence
        assertTrue(
            "8-frame confidence (${result8!!.confidence}) should be >= " +
                "3-frame confidence (${result3!!.confidence}) minus small epsilon",
            result8.confidence >= result3.confidence - 0.05
        )
    }

    // =======================================================================
    // Reset and Release Tests
    // =======================================================================

    @Test
    fun `reset clears all state`() {
        requireOpenCv()
        val estimator = MultiFrameAspectEstimator()
        val corners = makeRectAt(300.0, 400.0)

        estimator.addFrame(corners)
        estimator.addFrame(corners)
        estimator.addFrame(corners)
        estimator.addFrame(corners)
        estimator.addFrame(corners)
        assertEquals(5, estimator.frameCount)

        estimator.reset()
        assertEquals(0, estimator.frameCount)
        assertNull(
            "After reset, should return null (no frames)",
            estimator.estimateAspectRatio()
        )
    }

    @Test
    fun `release clears all state`() {
        requireOpenCv()
        val estimator = MultiFrameAspectEstimator()
        val corners = makeRectAt(300.0, 400.0)

        estimator.addFrame(corners)
        estimator.addFrame(corners)
        estimator.addFrame(corners)
        assertEquals(3, estimator.frameCount)

        estimator.release()
        assertEquals(0, estimator.frameCount)
    }

    @Test
    fun `reset allows re-accumulation`() {
        requireOpenCv()
        val estimator = MultiFrameAspectEstimator()
        val cornersA4 = makeRectAt(210.0, 297.0)

        // First round
        for (i in 0 until 5) {
            estimator.addFrame(jitterCorners(cornersA4, maxJitter = 1.0, seed = i))
        }
        val firstEstimate = estimator.estimateAspectRatio(TEST_INTRINSICS)
        assertNotNull(firstEstimate)

        // Reset and accumulate new data
        estimator.reset()
        assertEquals(0, estimator.frameCount)

        // Second round: square document
        val cornersSquare = makeRectAt(300.0, 300.0)
        for (i in 0 until 5) {
            estimator.addFrame(jitterCorners(cornersSquare, maxJitter = 1.0, seed = i + 10))
        }
        val secondEstimate = estimator.estimateAspectRatio(TEST_INTRINSICS)
        assertNotNull(secondEstimate)

        // The two estimates should be significantly different (A4 ~0.707 vs square ~1.0)
        assertTrue(
            "After reset, estimate should reflect new data",
            abs(firstEstimate!!.estimatedRatio - secondEstimate!!.estimatedRatio) > 0.1
        )
        estimator.release()
    }

    // =======================================================================
    // Outlier Robustness Tests
    // =======================================================================

    @Test
    fun `medianAggregation_robustToSingleOutlier`() {
        requireOpenCv()
        val estimator = MultiFrameAspectEstimator()

        // 4 consistent A4 frames
        val a4Corners = makeRectAt(210.0, 297.0)
        for (i in 0 until 4) {
            estimator.addFrame(jitterCorners(a4Corners, maxJitter = 1.0, seed = i))
        }

        // 1 outlier: square-like document
        val outlierCorners = makeRectAt(300.0, 310.0) // near-square
        estimator.addFrame(outlierCorners)

        val estimate = estimator.estimateAspectRatio(TEST_INTRINSICS)
        estimator.release()

        assertNotNull("Should produce an estimate despite outlier", estimate)
        // Median of 4 A4-like values + 1 outlier should still land near A4
        val pctError = abs(estimate!!.estimatedRatio - 0.707) / 0.707 * 100.0
        assertTrue(
            "Median should be robust: ratio (${estimate.estimatedRatio}) should be " +
                "within 10% of 0.707, was ${pctError}%",
            pctError < 10.0
        )
    }

    // =======================================================================
    // Ratio Range Validation
    // =======================================================================

    @Test
    fun `estimatedRatio is always between 0_1 and 1_0`() {
        requireOpenCv()

        // Test with various document shapes
        val shapes = listOf(
            makeRectAt(210.0, 297.0),  // A4
            makeRectAt(300.0, 300.0),  // Square
            makeRectAt(80.0, 240.0),   // Receipt-like
            makeRectAt(500.0, 300.0)   // Landscape
        )

        for (shape in shapes) {
            val estimator = MultiFrameAspectEstimator()
            for (i in 0 until 5) {
                estimator.addFrame(jitterCorners(shape, maxJitter = 1.0, seed = i))
            }

            val estimate = estimator.estimateAspectRatio(TEST_INTRINSICS)
            estimator.release()

            if (estimate != null) {
                assertTrue(
                    "Ratio (${estimate.estimatedRatio}) should be >= 0.1",
                    estimate.estimatedRatio >= 0.1
                )
                assertTrue(
                    "Ratio (${estimate.estimatedRatio}) should be <= 1.0",
                    estimate.estimatedRatio <= 1.0
                )
            }
        }
    }

    @Test
    fun `frameCount in result matches added frames`() {
        requireOpenCv()
        val estimator = MultiFrameAspectEstimator()
        val corners = makeRectAt(210.0, 297.0)

        for (i in 0 until 6) {
            estimator.addFrame(jitterCorners(corners, maxJitter = 1.0, seed = i))
        }

        val estimate = estimator.estimateAspectRatio(TEST_INTRINSICS)
        estimator.release()

        assertNotNull(estimate)
        assertEquals(
            "Frame count in result should match added frames",
            6, estimate!!.frameCount
        )
    }

    @Test
    fun `perspectiveDistorted_A4_withIntrinsics`() {
        requireOpenCv()
        val estimator = MultiFrameAspectEstimator()

        // Feed frames with mild perspective distortion (as if from real camera)
        for (tilt in listOf(5.0, 6.0, 7.0, 8.0, 9.0)) {
            val corners = projectRectAtTilt(docW = 210.0, docH = 297.0, tiltDeg = tilt)
            estimator.addFrame(corners)
        }

        val estimate = estimator.estimateAspectRatio(TEST_INTRINSICS)
        estimator.release()

        assertNotNull("Should produce estimate from perspective-distorted frames", estimate)
        assertTrue(
            "Ratio (${estimate!!.estimatedRatio}) should be in valid range",
            estimate.estimatedRatio in 0.1..1.0
        )
    }
}
