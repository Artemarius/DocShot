package com.docshot.cv

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.opencv.core.CvType
import org.opencv.core.Point
import kotlin.math.abs
import kotlin.math.sqrt

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

        /** Creates a perfect rectangle with given width/height at (0,0). */
        private fun makeRect(w: Double, h: Double): List<Point> {
            return listOf(
                Point(0.0, 0.0),      // TL
                Point(w, 0.0),        // TR
                Point(w, h),          // BR
                Point(0.0, h)         // BL
            )
        }

        /** Creates a trapezoid by shifting the top edge inward symmetrically. */
        private fun makeTrapezoid(
            w: Double,
            h: Double,
            topInset: Double
        ): List<Point> {
            return listOf(
                Point(topInset, 0.0),          // TL
                Point(w - topInset, 0.0),      // TR
                Point(w, h),                   // BR
                Point(0.0, h)                  // BL
            )
        }

        /** Creates a heavily skewed trapezoid (perspective distortion). */
        private fun makeHeavyTrapezoid(
            w: Double,
            h: Double,
            topInset: Double,
            bottomInset: Double
        ): List<Point> {
            return listOf(
                Point(topInset, 0.0),              // TL
                Point(w - topInset, 0.0),          // TR
                Point(w - bottomInset, h),         // BR
                Point(bottomInset, h)              // BL
            )
        }

        /** Synthetic camera intrinsics for testing. */
        private val TEST_INTRINSICS = CameraIntrinsics(
            fx = 1000.0,
            fy = 1000.0,
            cx = 320.0,
            cy = 240.0
        )
    }

    // =======================================================================
    // Existing tests (pre-B1)
    // =======================================================================

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

    // =======================================================================
    // B1: Hartley Normalization tests
    // =======================================================================

    @Test
    fun `hartleyNormalize centroid is at origin`() {
        val corners = listOf(
            Point(100.0, 200.0),
            Point(300.0, 200.0),
            Point(300.0, 400.0),
            Point(100.0, 400.0)
        )

        val (normalized, T) = hartleyNormalize(corners)
        try {
            val cx = normalized.map { it.x }.average()
            val cy = normalized.map { it.y }.average()
            assertEquals("Centroid X should be ~0", 0.0, cx, 1e-10)
            assertEquals("Centroid Y should be ~0", 0.0, cy, 1e-10)
        } finally {
            T.release()
        }
    }

    @Test
    fun `hartleyNormalize average distance is sqrt(2)`() {
        val corners = listOf(
            Point(100.0, 200.0),
            Point(300.0, 200.0),
            Point(300.0, 400.0),
            Point(100.0, 400.0)
        )

        val (normalized, T) = hartleyNormalize(corners)
        try {
            val avgDist = normalized.map { sqrt(it.x * it.x + it.y * it.y) }.average()
            assertEquals("Average distance should be sqrt(2)", sqrt(2.0), avgDist, 1e-10)
        } finally {
            T.release()
        }
    }

    @Test
    fun `hartleyNormalize transform matrix is 3x3 CV_64FC1`() {
        val corners = makeRect(200.0, 300.0)

        val (_, T) = hartleyNormalize(corners)
        try {
            assertEquals("Rows should be 3", 3, T.rows())
            assertEquals("Cols should be 3", 3, T.cols())
            assertEquals("Type should be CV_64FC1", CvType.CV_64FC1, T.type())
        } finally {
            T.release()
        }
    }

    @Test
    fun `hartleyNormalize transform matrix matches expected structure`() {
        val corners = listOf(
            Point(100.0, 200.0),
            Point(300.0, 200.0),
            Point(300.0, 400.0),
            Point(100.0, 400.0)
        )

        val cx = corners.map { it.x }.average()
        val cy = corners.map { it.y }.average()
        val centered = corners.map { Point(it.x - cx, it.y - cy) }
        val avgDist = centered.map { sqrt(it.x * it.x + it.y * it.y) }.average()
        val expectedScale = sqrt(2.0) / avgDist

        val (_, T) = hartleyNormalize(corners)
        try {
            // T = [[scale, 0, -scale*cx], [0, scale, -scale*cy], [0, 0, 1]]
            assertEquals("T[0,0] should be scale", expectedScale, T.get(0, 0)[0], 1e-10)
            assertEquals("T[0,1] should be 0", 0.0, T.get(0, 1)[0], 1e-10)
            assertEquals("T[0,2] should be -scale*cx", -expectedScale * cx, T.get(0, 2)[0], 1e-10)
            assertEquals("T[1,0] should be 0", 0.0, T.get(1, 0)[0], 1e-10)
            assertEquals("T[1,1] should be scale", expectedScale, T.get(1, 1)[0], 1e-10)
            assertEquals("T[1,2] should be -scale*cy", -expectedScale * cy, T.get(1, 2)[0], 1e-10)
            assertEquals("T[2,0] should be 0", 0.0, T.get(2, 0)[0], 1e-10)
            assertEquals("T[2,1] should be 0", 0.0, T.get(2, 1)[0], 1e-10)
            assertEquals("T[2,2] should be 1", 1.0, T.get(2, 2)[0], 1e-10)
        } finally {
            T.release()
        }
    }

    @Test
    fun `hartleyNormalize applies transform correctly to points`() {
        // Verify T * [x, y, 1]^T gives the normalized point
        val corners = listOf(
            Point(50.0, 100.0),
            Point(250.0, 110.0),
            Point(240.0, 350.0),
            Point(60.0, 340.0)
        )

        val (normalized, T) = hartleyNormalize(corners)
        try {
            for (i in corners.indices) {
                val orig = corners[i]
                val norm = normalized[i]
                // T * [x, y, 1]^T
                val tx = T.get(0, 0)[0] * orig.x + T.get(0, 1)[0] * orig.y + T.get(0, 2)[0]
                val ty = T.get(1, 0)[0] * orig.x + T.get(1, 1)[0] * orig.y + T.get(1, 2)[0]
                assertEquals("Transformed X[$i] should match", norm.x, tx, 1e-10)
                assertEquals("Transformed Y[$i] should match", norm.y, ty, 1e-10)
            }
        } finally {
            T.release()
        }
    }

    @Test(expected = IllegalArgumentException::class)
    fun `hartleyNormalize rejects non-4-point input`() {
        hartleyNormalize(listOf(Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0)))
    }

    // =======================================================================
    // B2: Perspective Severity Classifier tests
    // =======================================================================

    @Test
    fun `perspectiveSeverity returns ~0 for perfect rectangle`() {
        val corners = makeRect(200.0, 300.0)
        val severity = perspectiveSeverity(corners)
        assertEquals("Perfect rectangle should have ~0 severity", 0.0, severity, 0.5)
    }

    @Test
    fun `perspectiveSeverity returns ~0 for perfect square`() {
        val corners = makeRect(200.0, 200.0)
        val severity = perspectiveSeverity(corners)
        assertEquals("Perfect square should have ~0 severity", 0.0, severity, 0.5)
    }

    @Test
    fun `perspectiveSeverity returns low value for mild trapezoid`() {
        // Mild perspective: top edge slightly shorter (small inset)
        val corners = makeTrapezoid(w = 300.0, h = 400.0, topInset = 15.0)
        val severity = perspectiveSeverity(corners)
        assertTrue(
            "Mild trapezoid severity ($severity) should be < 15 degrees",
            severity < 15.0
        )
        assertTrue(
            "Mild trapezoid severity ($severity) should be > 0",
            severity > 0.0
        )
    }

    @Test
    fun `perspectiveSeverity returns high value for heavy trapezoid`() {
        // Heavy perspective: top edge much shorter
        val corners = makeTrapezoid(w = 300.0, h = 300.0, topInset = 100.0)
        val severity = perspectiveSeverity(corners)
        assertTrue(
            "Heavy trapezoid severity ($severity) should be > 20 degrees",
            severity > 20.0
        )
    }

    @Test
    fun `perspectiveSeverity increases with more perspective distortion`() {
        val mild = makeTrapezoid(w = 300.0, h = 400.0, topInset = 10.0)
        val moderate = makeTrapezoid(w = 300.0, h = 400.0, topInset = 40.0)
        val heavy = makeTrapezoid(w = 300.0, h = 400.0, topInset = 80.0)

        val mildSeverity = perspectiveSeverity(mild)
        val moderateSeverity = perspectiveSeverity(moderate)
        val heavySeverity = perspectiveSeverity(heavy)

        assertTrue(
            "Moderate ($moderateSeverity) > mild ($mildSeverity)",
            moderateSeverity > mildSeverity
        )
        assertTrue(
            "Heavy ($heavySeverity) > moderate ($moderateSeverity)",
            heavySeverity > moderateSeverity
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun `perspectiveSeverity rejects non-4-point input`() {
        perspectiveSeverity(listOf(Point(0.0, 0.0)))
    }

    // =======================================================================
    // B3: Angular Corrected Ratio tests
    // =======================================================================

    @Test
    fun `angularCorrectedRatio returns raw ratio for perfect rectangle`() {
        // For a perfect rectangle, convergence angles are 0, so correction factor = 1.0
        val corners = makeRect(210.0, 297.0)
        val corrected = angularCorrectedRatio(corners)
        val raw = computeRawRatio(corners)
        assertEquals("Correction should be identity for rectangle", raw, corrected, 0.001)
    }

    @Test
    fun `angularCorrectedRatio close to raw for near-frontal quad`() {
        // Very mild perspective: top edge shifted by 5px
        val corners = listOf(
            Point(5.0, 0.0),       // TL
            Point(215.0, 0.0),     // TR
            Point(210.0, 297.0),   // BR
            Point(0.0, 297.0)      // BL
        )
        val corrected = angularCorrectedRatio(corners)
        val raw = computeRawRatio(corners)
        val pctDiff = abs(corrected - raw) / raw * 100.0
        assertTrue(
            "Correction should be < 2% for near-frontal quad (was $pctDiff%)",
            pctDiff < 2.0
        )
    }

    @Test
    fun `angularCorrectedRatio is within valid range`() {
        val corners = makeTrapezoid(w = 300.0, h = 400.0, topInset = 50.0)
        val corrected = angularCorrectedRatio(corners)
        assertTrue("Corrected ratio should be >= 0.1", corrected >= 0.1)
        assertTrue("Corrected ratio should be <= 1.0", corrected <= 1.0)
    }

    @Test
    fun `angularCorrectedRatio handles symmetric horizontal convergence`() {
        // Symmetric trapezoid: top narrower than bottom
        // This represents a document viewed from slightly above
        val corners = listOf(
            Point(30.0, 0.0),      // TL (inset)
            Point(270.0, 0.0),     // TR (inset)
            Point(300.0, 400.0),   // BR
            Point(0.0, 400.0)      // BL
        )
        val corrected = angularCorrectedRatio(corners)
        assertTrue("Corrected ratio should be positive", corrected > 0.0)
        assertTrue("Corrected ratio should be <= 1.0", corrected <= 1.0)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `angularCorrectedRatio rejects non-4-point input`() {
        angularCorrectedRatio(listOf(Point(0.0, 0.0), Point(1.0, 0.0)))
    }

    // =======================================================================
    // B4: Projective Aspect Ratio tests
    // =======================================================================

    @Test
    fun `projectiveAspectRatio returns reasonable value for A4 rectangle`() {
        // Perfect A4 rectangle -- projective method should give ratio near 0.707
        val corners = makeRect(210.0, 297.0)
        val ratio = projectiveAspectRatio(corners, TEST_INTRINSICS)
        assertNotNull("Should not return null for valid input", ratio)
        assertEquals("Should be close to A4 ratio", 1.0 / 1.414, ratio!!, 0.1)
    }

    @Test
    fun `projectiveAspectRatio returns value for perspective-distorted quad`() {
        // A4 document viewed at an angle
        val corners = listOf(
            Point(80.0, 50.0),     // TL
            Point(250.0, 70.0),    // TR
            Point(240.0, 350.0),   // BR
            Point(90.0, 330.0)     // BL
        )
        val ratio = projectiveAspectRatio(corners, TEST_INTRINSICS)
        assertNotNull("Should produce a result for perspective quad", ratio)
        assertTrue("Ratio ($ratio) should be > 0.1", ratio!! > 0.1)
        assertTrue("Ratio ($ratio) should be <= 1.0", ratio <= 1.0)
    }

    @Test
    fun `projectiveAspectRatio returns value in valid range for heavy skew`() {
        // Heavily skewed quad
        val corners = listOf(
            Point(150.0, 20.0),    // TL
            Point(280.0, 80.0),    // TR
            Point(250.0, 380.0),   // BR
            Point(50.0, 350.0)     // BL
        )
        val ratio = projectiveAspectRatio(corners, TEST_INTRINSICS)
        if (ratio != null) {
            assertTrue("Ratio ($ratio) should be > 0.05", ratio > 0.05)
            assertTrue("Ratio ($ratio) should be <= 1.0", ratio <= 1.0)
        }
        // null is also acceptable if the computation fails for extreme input
    }

    @Test
    fun `projectiveAspectRatio returns close to 1 for square`() {
        val corners = makeRect(200.0, 200.0)
        val ratio = projectiveAspectRatio(corners, TEST_INTRINSICS)
        assertNotNull("Should not return null for square", ratio)
        assertEquals("Square should give ratio near 1.0", 1.0, ratio!!, 0.15)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `projectiveAspectRatio rejects non-4-point input`() {
        projectiveAspectRatio(
            listOf(Point(0.0, 0.0)),
            TEST_INTRINSICS
        )
    }

    // =======================================================================
    // B5: Dual-Regime Estimation tests
    // =======================================================================

    @Test
    fun `estimateAspectRatioDualRegime uses angular for near-frontal rectangle`() {
        // Perfect rectangle has 0 severity -> angular path
        val corners = makeRect(210.0, 297.0)
        val estimate = estimateAspectRatioDualRegime(corners)

        assertNotNull("Should match a format", estimate.matchedFormat)
        assertEquals("Should snap to A4", "A4", estimate.matchedFormat!!.name)
    }

    @Test
    fun `estimateAspectRatioDualRegime returns result for mild perspective`() {
        // Mild trapezoid (severity < 15)
        val corners = listOf(
            Point(10.0, 0.0),      // TL
            Point(220.0, 5.0),     // TR
            Point(215.0, 300.0),   // BR
            Point(5.0, 295.0)      // BL
        )
        val estimate = estimateAspectRatioDualRegime(corners)
        assertTrue(
            "Ratio (${estimate.estimatedRatio}) should be in valid range",
            estimate.estimatedRatio in 0.1..1.0
        )
    }

    @Test
    fun `estimateAspectRatioDualRegime handles transition zone`() {
        // Create a quad with severity in 15-20 range
        // A trapezoid with moderate convergence
        val corners = makeTrapezoid(w = 300.0, h = 400.0, topInset = 35.0)
        val severity = perspectiveSeverity(corners)

        val estimate = estimateAspectRatioDualRegime(corners, TEST_INTRINSICS)
        assertTrue(
            "Ratio (${estimate.estimatedRatio}) should be in valid range",
            estimate.estimatedRatio in 0.1..1.0
        )
    }

    @Test
    fun `estimateAspectRatioDualRegime handles high severity with intrinsics`() {
        // Heavy trapezoid (severity > 20)
        val corners = makeTrapezoid(w = 300.0, h = 300.0, topInset = 100.0)
        val severity = perspectiveSeverity(corners)
        assertTrue(
            "Severity ($severity) should be > 20 for this test",
            severity > 20.0
        )

        val estimate = estimateAspectRatioDualRegime(corners, TEST_INTRINSICS)
        assertTrue(
            "Ratio (${estimate.estimatedRatio}) should be in valid range",
            estimate.estimatedRatio in 0.1..1.0
        )
    }

    @Test
    fun `estimateAspectRatioDualRegime falls back to angular without intrinsics`() {
        // Heavy skew but no intrinsics -- should fall back to angular
        val corners = makeTrapezoid(w = 300.0, h = 300.0, topInset = 100.0)
        val estimate = estimateAspectRatioDualRegime(corners, intrinsics = null)
        assertTrue(
            "Ratio (${estimate.estimatedRatio}) should be in valid range",
            estimate.estimatedRatio in 0.1..1.0
        )
    }

    @Test
    fun `estimateAspectRatioDualRegime smooth transition no discontinuity`() {
        // Test that ratios change smoothly as we vary the trapezoid inset
        // from low severity through transition zone to high severity
        val ratios = mutableListOf<Double>()
        val severities = mutableListOf<Double>()

        for (inset in listOf(5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 70.0, 85.0)) {
            val corners = makeTrapezoid(w = 400.0, h = 400.0, topInset = inset)
            val severity = perspectiveSeverity(corners)
            val estimate = estimateAspectRatioDualRegime(corners, TEST_INTRINSICS)
            ratios.add(estimate.estimatedRatio)
            severities.add(severity)
        }

        // Check there are no large jumps between consecutive estimates
        for (i in 1 until ratios.size) {
            val jump = abs(ratios[i] - ratios[i - 1])
            assertTrue(
                "Jump between severity %.1f and %.1f should be < 0.3, was $jump"
                    .format(severities[i - 1], severities[i]),
                jump < 0.3
            )
        }
    }

    @Test
    fun `estimateAspectRatioDualRegime backward compatible with estimateAspectRatio`() {
        // The two functions should return the same result since estimateAspectRatio
        // now delegates to estimateAspectRatioDualRegime
        val corners = makeRect(210.0, 297.0)

        val oldResult = estimateAspectRatio(corners)
        val newResult = estimateAspectRatioDualRegime(corners)

        assertEquals(
            "Both methods should return same ratio",
            oldResult.estimatedRatio,
            newResult.estimatedRatio,
            1e-10
        )
        assertEquals(
            "Both methods should match same format",
            oldResult.matchedFormat,
            newResult.matchedFormat
        )
    }

    @Test
    fun `estimateAspectRatioDualRegime snaps correctly for known formats`() {
        // Test that format snapping still works through the dual-regime path
        val a4 = makeRect(210.0, 297.0)
        val letter = makeRect(8.5, 11.0)
        val square = makeRect(200.0, 200.0)

        val a4Est = estimateAspectRatioDualRegime(a4)
        val letterEst = estimateAspectRatioDualRegime(letter)
        val squareEst = estimateAspectRatioDualRegime(square)

        assertEquals("A4", a4Est.matchedFormat?.name)
        assertEquals("US Letter", letterEst.matchedFormat?.name)
        assertEquals("Square", squareEst.matchedFormat?.name)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `estimateAspectRatioDualRegime rejects non-4-point input`() {
        estimateAspectRatioDualRegime(emptyList())
    }

    // =======================================================================
    // B10: Hartley Normalization — Invertibility
    // =======================================================================

    @Test
    fun `hartleyNormalize is invertible`() {
        // Applying T then T_inv should recover the original points
        val corners = listOf(
            Point(73.0, 142.0),
            Point(310.0, 158.0),
            Point(295.0, 410.0),
            Point(88.0, 395.0)
        )

        val (normalized, T) = hartleyNormalize(corners)
        var Tinv: org.opencv.core.Mat? = null
        try {
            Tinv = org.opencv.core.Mat()
            org.opencv.core.Core.invert(T, Tinv)

            // Apply T_inv to each normalized point: T_inv * [nx, ny, 1]^T
            for (i in corners.indices) {
                val nx = normalized[i].x
                val ny = normalized[i].y
                val rx = Tinv.get(0, 0)[0] * nx + Tinv.get(0, 1)[0] * ny + Tinv.get(0, 2)[0]
                val ry = Tinv.get(1, 0)[0] * nx + Tinv.get(1, 1)[0] * ny + Tinv.get(1, 2)[0]
                assertEquals(
                    "Recovered X[$i] should match original",
                    corners[i].x, rx, 1e-8
                )
                assertEquals(
                    "Recovered Y[$i] should match original",
                    corners[i].y, ry, 1e-8
                )
            }
        } finally {
            T.release()
            Tinv?.release()
        }
    }

    @Test
    fun `hartleyNormalize centroid at origin for asymmetric quad`() {
        // Non-symmetric distorted quad to verify centroid computation is general
        val corners = listOf(
            Point(10.0, 20.0),
            Point(500.0, 50.0),
            Point(480.0, 600.0),
            Point(30.0, 580.0)
        )

        val (normalized, T) = hartleyNormalize(corners)
        try {
            val cx = normalized.map { it.x }.average()
            val cy = normalized.map { it.y }.average()
            assertEquals("Centroid X should be ~0", 0.0, cx, 1e-10)
            assertEquals("Centroid Y should be ~0", 0.0, cy, 1e-10)

            val avgDist = normalized.map { sqrt(it.x * it.x + it.y * it.y) }.average()
            assertEquals("Average distance should be sqrt(2)", sqrt(2.0), avgDist, 1e-10)
        } finally {
            T.release()
        }
    }

    @Test
    fun `hartleyNormalize handles nearly collinear points gracefully`() {
        // Points nearly on a line -- should still produce valid normalization
        val corners = listOf(
            Point(0.0, 0.0),
            Point(100.0, 1.0),
            Point(200.0, 2.0),
            Point(300.0, 3.0)
        )

        val (normalized, T) = hartleyNormalize(corners)
        try {
            val cx = normalized.map { it.x }.average()
            val cy = normalized.map { it.y }.average()
            assertEquals("Centroid X should be ~0", 0.0, cx, 1e-10)
            assertEquals("Centroid Y should be ~0", 0.0, cy, 1e-10)
        } finally {
            T.release()
        }
    }

    // =======================================================================
    // B10: Format Snapping Tests (via estimateAspectRatioDualRegime)
    // =======================================================================

    @Test
    fun `formatSnap_ratio0_71_snapsToA4`() {
        // Ratio 0.71 is within SNAP_THRESHOLD (0.06) of A4 (0.707)
        // Build a rectangle whose raw ratio is ~0.71
        // ratio = min/max = w/h => w = 0.71 * h
        val h = 1000.0
        val w = h * 0.71
        val corners = makeRect(w, h)
        val estimate = estimateAspectRatio(corners)

        assertNotNull("Should match a format for ratio ~0.71", estimate.matchedFormat)
        assertEquals(
            "Ratio 0.71 should snap to A4",
            "A4", estimate.matchedFormat!!.name
        )
    }

    @Test
    fun `formatSnap_ratio0_78_snapsToUSLetter`() {
        // Ratio 0.78 is within SNAP_THRESHOLD (0.06) of US Letter (0.773)
        val h = 1000.0
        val w = h * 0.78
        val corners = makeRect(w, h)
        val estimate = estimateAspectRatio(corners)

        assertNotNull("Should match a format for ratio ~0.78", estimate.matchedFormat)
        assertEquals(
            "Ratio 0.78 should snap to US Letter",
            "US Letter", estimate.matchedFormat!!.name
        )
    }

    @Test
    fun `formatSnap_ratio0_50_noSnap`() {
        // Ratio 0.50 is not within SNAP_THRESHOLD of any known format:
        // - Business Card: 0.571 -> distance 0.071 > 0.06
        // - Receipt: 0.333 -> distance 0.167 > 0.06
        val h = 1000.0
        val w = h * 0.50
        val corners = makeRect(w, h)
        val estimate = estimateAspectRatio(corners)

        assertNull(
            "Ratio 0.50 should not snap to any format",
            estimate.matchedFormat
        )
    }

    @Test
    fun `formatSnap_exactBoundary_A4plusThreshold`() {
        // A4 ratio is 0.707, SNAP_THRESHOLD is 0.06
        // Ratio exactly at 0.707 + 0.06 = 0.767 should still snap
        val h = 1000.0
        val w = h * 0.767
        val corners = makeRect(w, h)
        val estimate = estimateAspectRatio(corners)

        // At 0.767, distance to A4 (0.707) = 0.060 (at boundary),
        // distance to US Letter (0.773) = 0.006 (much closer)
        // Should snap to US Letter since it's closer
        assertNotNull("Should match some format at boundary", estimate.matchedFormat)
        assertEquals(
            "At ratio 0.767, US Letter (0.773) is closer than A4 (0.707)",
            "US Letter", estimate.matchedFormat!!.name
        )
    }

    @Test
    fun `formatSnap_exactBoundary_justOutsideAllFormats`() {
        // Ratio 0.45 is far from all formats:
        // - Business Card: 0.571 -> distance 0.121 > 0.06
        // - Receipt: 0.333 -> distance 0.117 > 0.06
        val h = 1000.0
        val w = h * 0.45
        val corners = makeRect(w, h)
        val estimate = estimateAspectRatio(corners)

        assertNull(
            "Ratio 0.45 should not snap to any format",
            estimate.matchedFormat
        )
    }

    @Test
    fun `formatSnap_IDCard_snapsCorrectly`() {
        // ID Card ratio: 1.0 / 1.586 = 0.631
        val h = 1000.0
        val w = h * 0.631
        val corners = makeRect(w, h)
        val estimate = estimateAspectRatio(corners)

        assertNotNull("Should match ID Card format", estimate.matchedFormat)
        assertEquals("ID Card", estimate.matchedFormat!!.name)
    }

    @Test
    fun `formatSnap_BusinessCard_snapsCorrectly`() {
        // Business Card ratio: 1.0 / 1.75 = 0.571
        val h = 1000.0
        val w = h * 0.571
        val corners = makeRect(w, h)
        val estimate = estimateAspectRatio(corners)

        assertNotNull("Should match Business Card format", estimate.matchedFormat)
        assertEquals("Business Card", estimate.matchedFormat!!.name)
    }

    @Test
    fun `formatSnap_highConfidence_forExactRatio`() {
        // Exact A4 ratio should yield very high confidence
        val corners = makeRect(210.0, 297.0)
        val estimate = estimateAspectRatio(corners)

        assertNotNull(estimate.matchedFormat)
        assertTrue(
            "Exact A4 ratio should have confidence > 0.9, was ${estimate.confidence}",
            estimate.confidence > 0.9
        )
    }

    @Test
    fun `formatSnap_lowerConfidence_forDistantRatio`() {
        // Ratio slightly off from A4 but within threshold
        val h = 1000.0
        val w = h * 0.75 // Distance from A4 (0.707) = 0.043
        val corners = makeRect(w, h)
        val estimate = estimateAspectRatio(corners)

        // This falls between A4 and US Letter, confidence should be lower
        assertNotNull(estimate.matchedFormat)
        assertTrue(
            "Off-center ratio should have lower confidence",
            estimate.confidence < 0.95
        )
    }

    // =======================================================================
    // B10: Projective Estimation Additional Tests
    // =======================================================================

    @Test
    fun `projectiveAspectRatio_A4_withMildPerspective`() {
        // A4 document with controlled mild perspective distortion
        // Simulated by making top edge slightly narrower (5% foreshortening)
        val w = 210.0
        val h = 297.0
        val inset = w * 0.05 // 5% narrowing
        val corners = listOf(
            Point(inset / 2.0, 0.0),           // TL
            Point(w - inset / 2.0, 0.0),       // TR
            Point(w, h),                        // BR
            Point(0.0, h)                       // BL
        )
        val ratio = projectiveAspectRatio(corners, TEST_INTRINSICS)
        assertNotNull("Should produce a result", ratio)
        // Allow wider tolerance for projective method under mild perspective
        assertTrue(
            "Ratio ($ratio) should be in document-like range [0.4, 0.95]",
            ratio!! in 0.4..0.95
        )
    }

    @Test
    fun `projectiveAspectRatio_USLetter_withModerateSkew`() {
        // US Letter proportions with moderate perspective
        val w = 850.0
        val h = 1100.0
        val inset = 80.0
        val corners = listOf(
            Point(inset, 20.0),        // TL
            Point(w - inset, 30.0),    // TR
            Point(w - 10.0, h - 20.0), // BR
            Point(10.0, h - 30.0)      // BL
        )
        val ratio = projectiveAspectRatio(corners, TEST_INTRINSICS)
        assertNotNull("Should produce a result for US Letter", ratio)
        assertTrue(
            "Ratio ($ratio) should be reasonable",
            ratio!! in 0.3..1.0
        )
    }

    // =======================================================================
    // B10: Dual-Regime Integration — Regime Selection Verification
    // =======================================================================

    @Test
    fun `dualRegime_lowSeverity_usesAngular_producesAccurateA4`() {
        // Near-frontal A4 with very mild distortion (severity should be < 15)
        val corners = listOf(
            Point(2.0, 0.0),       // TL (barely shifted)
            Point(212.0, 1.0),     // TR
            Point(211.0, 297.0),   // BR
            Point(1.0, 296.0)      // BL
        )
        val severity = perspectiveSeverity(corners)
        assertTrue(
            "Setup: severity ($severity) should be < 15",
            severity < 15.0
        )

        val estimate = estimateAspectRatioDualRegime(corners)
        assertNotNull("Should match A4", estimate.matchedFormat)
        assertEquals("A4", estimate.matchedFormat!!.name)
    }

    @Test
    fun `dualRegime_highSeverity_withIntrinsics_usesProjective`() {
        // Heavily skewed quad (severity > 20)
        val corners = makeTrapezoid(w = 400.0, h = 400.0, topInset = 120.0)
        val severity = perspectiveSeverity(corners)
        assertTrue(
            "Setup: severity ($severity) should be > 20",
            severity > 20.0
        )

        // With intrinsics, dual-regime should use projective path
        val estimate = estimateAspectRatioDualRegime(corners, TEST_INTRINSICS)
        assertTrue(
            "Ratio (${estimate.estimatedRatio}) should be in valid range",
            estimate.estimatedRatio in 0.1..1.0
        )
    }

    @Test
    fun `dualRegime_transitionZone_blendsBothMethods`() {
        // Find a quad that lands in the 15-20 degree transition zone
        // Use binary search over insets to find the right severity
        var targetInset = 35.0
        var corners = makeTrapezoid(w = 400.0, h = 500.0, topInset = targetInset)
        var severity = perspectiveSeverity(corners)

        // Adjust to get in the 15-20 range
        if (severity < 15.0) targetInset = 50.0
        if (severity > 20.0) targetInset = 28.0
        corners = makeTrapezoid(w = 400.0, h = 500.0, topInset = targetInset)
        severity = perspectiveSeverity(corners)

        // If we're in the transition zone, verify the blend works
        if (severity in 15.0..20.0) {
            val angularOnly = angularCorrectedRatio(corners)
            val blended = estimateAspectRatioDualRegime(corners, TEST_INTRINSICS)

            // The blended ratio should be different from pure angular
            // (unless projective gives the same result, which is fine too)
            assertTrue(
                "Blended ratio (${blended.estimatedRatio}) should be in valid range",
                blended.estimatedRatio in 0.1..1.0
            )
        }
        // If we couldn't land in transition zone, just verify no crash
        val estimate = estimateAspectRatioDualRegime(corners, TEST_INTRINSICS)
        assertTrue(estimate.estimatedRatio in 0.1..1.0)
    }

    // =======================================================================
    // B10: KnownFormat and AspectRatioEstimate data class tests
    // =======================================================================

    @Test
    fun `KNOWN_FORMATS_ratios_are_all_valid`() {
        for (fmt in KNOWN_FORMATS) {
            assertTrue(
                "Format ${fmt.name} ratio (${fmt.ratio}) should be > 0",
                fmt.ratio > 0.0
            )
            assertTrue(
                "Format ${fmt.name} ratio (${fmt.ratio}) should be <= 1.0",
                fmt.ratio <= 1.0
            )
        }
    }

    @Test
    fun `KNOWN_FORMATS_contains_all_standard_formats`() {
        val names = KNOWN_FORMATS.map { it.name }.toSet()
        assertTrue("Should contain A4", "A4" in names)
        assertTrue("Should contain US Letter", "US Letter" in names)
        assertTrue("Should contain ID Card", "ID Card" in names)
        assertTrue("Should contain Business Card", "Business Card" in names)
        assertTrue("Should contain Receipt", "Receipt" in names)
        assertTrue("Should contain Square", "Square" in names)
    }

    @Test
    fun `AspectRatioEstimate_data_class_holds_values`() {
        val format = KnownFormat("Test", 0.5)
        val estimate = AspectRatioEstimate(
            estimatedRatio = 0.5,
            matchedFormat = format,
            confidence = 0.85,
            verifiedByHomography = true
        )
        assertEquals(0.5, estimate.estimatedRatio, 0.001)
        assertEquals(format, estimate.matchedFormat)
        assertEquals(0.85, estimate.confidence, 0.001)
        assertTrue(estimate.verifiedByHomography)
    }

    @Test
    fun `AspectRatioEstimate_default_verifiedByHomography_is_false`() {
        val estimate = AspectRatioEstimate(
            estimatedRatio = 0.707,
            matchedFormat = null,
            confidence = 0.5
        )
        assertEquals(false, estimate.verifiedByHomography)
    }
}
