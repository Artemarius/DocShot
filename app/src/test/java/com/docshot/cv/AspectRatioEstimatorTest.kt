package com.docshot.cv

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.opencv.core.Point
import kotlin.math.abs

class AspectRatioEstimatorTest {

    companion object {
        @JvmStatic
        @BeforeClass
        fun loadOpenCV() {
            try {
                System.loadLibrary("opencv_java4")
            } catch (_: UnsatisfiedLinkError) {
                try {
                    System.loadLibrary("opencv_java412")
                } catch (_: UnsatisfiedLinkError) {
                    throw org.junit.AssumptionViolatedException(
                        "OpenCV native library not available — skipping AspectRatioEstimatorTest"
                    )
                }
            }
        }

        /** Creates a perfect rectangle with given width/height centered at origin. */
        private fun makeRect(w: Double, h: Double): List<Point> {
            return listOf(
                Point(0.0, 0.0),      // TL
                Point(w, 0.0),        // TR
                Point(w, h),          // BR
                Point(0.0, h)         // BL
            )
        }
    }

    @Test
    fun `computeRawRatio returns ~0_707 for A4 rectangle`() {
        // A4: 210 x 297 mm, ratio = 210/297 = 0.7071
        val corners = makeRect(210.0, 297.0)
        val ratio = computeRawRatio(corners)
        assertEquals(1.0 / 1.414, ratio, 0.01)
    }

    @Test
    fun `computeRawRatio returns ~0_773 for US Letter rectangle`() {
        // US Letter: 8.5 x 11 inches, ratio = 8.5/11 = 0.7727
        val corners = makeRect(8.5, 11.0)
        val ratio = computeRawRatio(corners)
        assertEquals(1.0 / 1.294, ratio, 0.01)
    }

    @Test
    fun `computeRawRatio is orientation-independent`() {
        // Portrait
        val portrait = makeRect(210.0, 297.0)
        val portraitRatio = computeRawRatio(portrait)

        // Landscape (same document rotated)
        val landscape = makeRect(297.0, 210.0)
        val landscapeRatio = computeRawRatio(landscape)

        assertEquals(portraitRatio, landscapeRatio, 0.001)
    }

    @Test
    fun `estimateAspectRatio snaps to A4 for A4-like quad`() {
        val corners = makeRect(210.0, 297.0)
        val estimate = estimateAspectRatio(corners)

        assertNotNull("Should match a format", estimate.matchedFormat)
        assertEquals("A4", estimate.matchedFormat!!.name)
        assertTrue("Confidence should be high", estimate.confidence > 0.7)
    }

    @Test
    fun `estimateAspectRatio returns null format for extreme ratio`() {
        // 1:10 ratio — no standard document format
        val corners = makeRect(100.0, 1000.0)
        val estimate = estimateAspectRatio(corners)

        assertNull("Should not match any format", estimate.matchedFormat)
        assertEquals(0.1, estimate.estimatedRatio, 0.01)
    }

    @Test
    fun `estimateAspectRatio snaps to Receipt for 1_3 quad`() {
        // Receipt: 1:3 ratio
        val corners = makeRect(80.0, 240.0)
        val estimate = estimateAspectRatio(corners)

        assertNotNull("Should match a format", estimate.matchedFormat)
        assertEquals("Receipt", estimate.matchedFormat!!.name)
    }

    @Test
    fun `homographyError is lower for correct format than wrong format`() {
        // A4 rectangle with slight perspective distortion
        val corners = listOf(
            Point(50.0, 30.0),     // TL
            Point(260.0, 40.0),    // TR
            Point(255.0, 330.0),   // BR
            Point(55.0, 320.0)     // BL
        )

        // Synthetic camera intrinsics
        val intrinsics = CameraIntrinsics(fx = 1000.0, fy = 1000.0, cx = 320.0, cy = 240.0)

        val a4Error = homographyError(corners, 1.0 / 1.414, intrinsics)
        val squareError = homographyError(corners, 1.0, intrinsics)

        assertTrue(
            "A4 error ($a4Error) should be lower than Square error ($squareError)",
            a4Error < squareError
        )
    }

    @Test
    fun `estimateAspectRatio with intrinsics disambiguates A4 from Letter`() {
        // A4 quad: ratio ~0.707, Letter ~0.773 — these are close enough that
        // both are within SNAP_THRESHOLD of a moderately distorted A4
        val corners = makeRect(210.0, 297.0)
        val intrinsics = CameraIntrinsics(fx = 1000.0, fy = 1000.0, cx = 320.0, cy = 240.0)

        val estimate = estimateAspectRatio(corners, intrinsics)

        assertNotNull("Should match a format", estimate.matchedFormat)
        assertEquals("A4", estimate.matchedFormat!!.name)
        // With intrinsics available and a clean A4 quad, should still snap to A4
        assertTrue(
            "Ratio should be close to A4",
            abs(estimate.estimatedRatio - 1.0 / 1.414) < 0.01
        )
    }
}
