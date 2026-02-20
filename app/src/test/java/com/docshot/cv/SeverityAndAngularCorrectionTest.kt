package com.docshot.cv

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.opencv.core.Point
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tan

/**
 * Pure-math tests for [perspectiveSeverity], [angularCorrectedRatio], and [computeRawRatio].
 *
 * These functions use only [org.opencv.core.Point] (a plain Java class with x/y fields)
 * and standard math -- no OpenCV native library required. This class runs on JVM without
 * any OpenCV setup, unlike [AspectRatioEstimatorTest] which requires native libs.
 */
class SeverityAndAngularCorrectionTest {

    // =====================================================================
    // Helpers: Synthetic quad generation
    // =====================================================================

    /** Creates a perfect rectangle with given width/height at (0,0). TL, TR, BR, BL order. */
    private fun makeRect(w: Double, h: Double): List<Point> = listOf(
        Point(0.0, 0.0),
        Point(w, 0.0),
        Point(w, h),
        Point(0.0, h)
    )

    /** Creates a symmetric trapezoid: top edge narrowed by [topInset] on each side. */
    private fun makeTrapezoid(
        w: Double,
        h: Double,
        topInset: Double
    ): List<Point> = listOf(
        Point(topInset, 0.0),
        Point(w - topInset, 0.0),
        Point(w, h),
        Point(0.0, h)
    )

    /**
     * Simulates a pinhole camera projection of a rectangle [docW] x [docH]
     * tilted by [tiltDeg] degrees about the horizontal axis (looking down at the document).
     *
     * The camera is at (cx, cy, focalLength) looking straight down at the document
     * center. The tilt rotates the document plane about its horizontal center axis.
     *
     * Returns 4 projected corners in TL, TR, BR, BL order.
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

        // Document corners in 3D, centered at origin on XY plane:
        // TL(-w/2, -h/2, 0), TR(w/2, -h/2, 0), BR(w/2, h/2, 0), BL(-w/2, h/2, 0)
        // After tilt about X-axis by theta:
        //   y' = y * cos(theta), z' = y * sin(theta)
        val halfW = docW / 2.0
        val halfH = docH / 2.0

        val corners3D = listOf(
            Triple(-halfW, -halfH, 0.0), // TL
            Triple(halfW, -halfH, 0.0),  // TR
            Triple(halfW, halfH, 0.0),   // BR
            Triple(-halfW, halfH, 0.0)   // BL
        )

        // Apply tilt rotation about X-axis
        val rotated = corners3D.map { (x, y, z) ->
            Triple(x, y * cosT - z * sinT, y * sinT + z * cosT)
        }

        // Project: camera at (0, 0, cameraHeight), projecting onto image plane
        // u = fx * X / (Z_camera - Z_point) + cx
        // v = fy * Y / (Z_camera - Z_point) + cy
        val cx = 320.0
        val cy = 240.0
        return rotated.map { (x, y, z) ->
            val depth = cameraHeight - z
            if (depth <= 0) Point(cx, cy) // degenerate -- shouldn't happen for small angles
            else Point(
                focalLength * x / depth + cx,
                focalLength * y / depth + cy
            )
        }
    }

    // =====================================================================
    // 1. Perspective Severity Classifier Tests
    // =====================================================================

    @Test
    fun `severity_perfectRectangle_isZero`() {
        val corners = makeRect(200.0, 300.0)
        val severity = perspectiveSeverity(corners)
        assertEquals(
            "Perfect rectangle should have severity = 0",
            0.0, severity, 0.1
        )
    }

    @Test
    fun `severity_perfectSquare_isZero`() {
        val corners = makeRect(200.0, 200.0)
        val severity = perspectiveSeverity(corners)
        assertEquals(
            "Perfect square should have severity = 0",
            0.0, severity, 0.1
        )
    }

    @Test
    fun `severity_nearRectangularQuad_isLow`() {
        // Angles ~89-91 degrees: very slight distortion
        val corners = listOf(
            Point(2.0, 1.0),       // TL shifted slightly
            Point(302.0, 0.0),     // TR
            Point(301.0, 400.0),   // BR shifted slightly
            Point(0.0, 401.0)      // BL
        )
        val severity = perspectiveSeverity(corners)
        assertTrue(
            "Near-rectangular severity ($severity) should be < 15 (LOW regime)",
            severity < 15.0
        )
    }

    @Test
    fun `severity_mildTrapezoid_isLow`() {
        // Small top-edge narrowing -> mild foreshortening
        val corners = makeTrapezoid(w = 300.0, h = 400.0, topInset = 10.0)
        val severity = perspectiveSeverity(corners)
        assertTrue(
            "Mild trapezoid severity ($severity) should be < 15 (LOW regime)",
            severity < 15.0
        )
        assertTrue(
            "Mild trapezoid severity ($severity) should be > 0",
            severity > 0.0
        )
    }

    @Test
    fun `severity_moderateTrapezoid_isTransition`() {
        // Moderate convergence: top edge noticeably narrower
        // With inset=80 on 400x500, severity lands in a moderate range
        val corners = makeTrapezoid(w = 400.0, h = 500.0, topInset = 80.0)
        val severity = perspectiveSeverity(corners)
        assertTrue(
            "Moderate trapezoid severity ($severity) should be >= 5",
            severity >= 5.0
        )
        assertTrue(
            "Moderate trapezoid severity ($severity) should be < 25",
            severity < 25.0
        )
    }

    @Test
    fun `severity_heavyTrapezoid_isHigh`() {
        // Heavy convergence: top edge much narrower than bottom
        // inset=120 on 300x300 gives very visible trapezoid (top=60px wide, bottom=300px)
        val corners = makeTrapezoid(w = 300.0, h = 300.0, topInset = 120.0)
        val severity = perspectiveSeverity(corners)
        assertTrue(
            "Heavy trapezoid severity ($severity) should be > 20 (HIGH regime)",
            severity > 20.0
        )
    }

    @Test
    fun `severity_projectedRect_5deg_isLow`() {
        // A4 document tilted 5 degrees
        val corners = projectRectAtTilt(docW = 210.0, docH = 297.0, tiltDeg = 5.0)
        val severity = perspectiveSeverity(corners)
        assertTrue(
            "5-degree tilt severity ($severity) should be < 15 (LOW regime)",
            severity < 15.0
        )
    }

    @Test
    fun `severity_projectedRect_10deg_isLow`() {
        // A4 document tilted 10 degrees
        val corners = projectRectAtTilt(docW = 210.0, docH = 297.0, tiltDeg = 10.0)
        val severity = perspectiveSeverity(corners)
        assertTrue(
            "10-degree tilt severity ($severity) should be < 15 (LOW regime)",
            severity < 15.0
        )
    }

    @Test
    fun `severity_projectedRect_30deg_isSignificant`() {
        // A4 document tilted 30 degrees -- visible foreshortening
        // In our pinhole model the severity metric captures max corner angle deviation;
        // at 30deg tilt the projected quad has moderate severity (>5 degrees)
        val corners = projectRectAtTilt(docW = 210.0, docH = 297.0, tiltDeg = 30.0)
        val severity = perspectiveSeverity(corners)
        assertTrue(
            "30-degree tilt severity ($severity) should be > 5",
            severity > 5.0
        )
    }

    @Test
    fun `severity_projectedRect_60deg_isHigh`() {
        // A4 document tilted 60 degrees -- extreme foreshortening
        val corners = projectRectAtTilt(docW = 210.0, docH = 297.0, tiltDeg = 60.0)
        val severity = perspectiveSeverity(corners)
        assertTrue(
            "60-degree tilt severity ($severity) should be > 15",
            severity > 15.0
        )
    }

    @Test
    fun `severity_monotonicallyIncreases_withTiltAngle`() {
        val angles = listOf(0.0, 5.0, 10.0, 15.0, 20.0, 30.0)
        val severities = angles.map { angle ->
            val corners = projectRectAtTilt(docW = 210.0, docH = 297.0, tiltDeg = angle)
            perspectiveSeverity(corners)
        }

        for (i in 1 until severities.size) {
            assertTrue(
                "Severity at ${angles[i]}deg (${severities[i]}) should be >= " +
                    "severity at ${angles[i - 1]}deg (${severities[i - 1]})",
                severities[i] >= severities[i - 1] - 0.01 // tiny epsilon for FP
            )
        }
    }

    @Test
    fun `severity_symmetric_topVsBottomNarrowing`() {
        // Top narrow
        val topNarrow = makeTrapezoid(w = 300.0, h = 400.0, topInset = 30.0)
        // Bottom narrow (flip: bottom narrower, top wider)
        val bottomNarrow = listOf(
            Point(0.0, 0.0),
            Point(300.0, 0.0),
            Point(300.0 - 30.0, 400.0),
            Point(30.0, 400.0)
        )

        val severityTop = perspectiveSeverity(topNarrow)
        val severityBot = perspectiveSeverity(bottomNarrow)

        assertEquals(
            "Top-narrowing and bottom-narrowing should give same severity",
            severityTop, severityBot, 0.5
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun `severity_rejects_3points`() {
        perspectiveSeverity(listOf(Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0)))
    }

    @Test(expected = IllegalArgumentException::class)
    fun `severity_rejects_5points`() {
        perspectiveSeverity(
            listOf(
                Point(0.0, 0.0), Point(1.0, 0.0), Point(2.0, 0.0),
                Point(1.0, 1.0), Point(0.0, 1.0)
            )
        )
    }

    // =====================================================================
    // 2. Angular Correction Tests (Low Severity)
    // =====================================================================

    @Test
    fun `angularCorrection_perfectA4_returnsExactRatio`() {
        val corners = makeRect(210.0, 297.0)
        val corrected = angularCorrectedRatio(corners)
        val expected = 210.0 / 297.0 // 0.7071
        assertEquals(
            "Perfect A4 should return exact ratio",
            expected, corrected, 0.001
        )
    }

    @Test
    fun `angularCorrection_a4At5Degrees_withinOnePercent`() {
        // Project A4 at 5 degrees tilt
        val corners = projectRectAtTilt(docW = 210.0, docH = 297.0, tiltDeg = 5.0)
        val corrected = angularCorrectedRatio(corners)
        val expected = 210.0 / 297.0 // 0.7071

        val pctError = abs(corrected - expected) / expected * 100.0
        assertTrue(
            "A4 at 5 degrees: corrected ratio ($corrected) should be within 1% " +
                "of 0.707, was ${pctError}%",
            pctError < 1.0
        )
    }

    @Test
    fun `angularCorrection_a4At10Degrees_withinTwoPercent`() {
        // Project A4 at 10 degrees tilt
        val corners = projectRectAtTilt(docW = 210.0, docH = 297.0, tiltDeg = 10.0)
        val corrected = angularCorrectedRatio(corners)
        val expected = 210.0 / 297.0 // 0.7071

        val pctError = abs(corrected - expected) / expected * 100.0
        assertTrue(
            "A4 at 10 degrees: corrected ratio ($corrected) should be within 2% " +
                "of 0.707, was ${pctError}%",
            pctError < 2.0
        )
    }

    @Test
    fun `angularCorrection_usLetterAt10Degrees_withinTwoPercent`() {
        // Project US Letter at 10 degrees tilt
        // US Letter: 8.5 x 11 inches, ratio = 8.5/11 = 0.7727
        val corners = projectRectAtTilt(docW = 8.5, docH = 11.0, tiltDeg = 10.0)
        val corrected = angularCorrectedRatio(corners)
        val expected = 8.5 / 11.0 // 0.7727

        val pctError = abs(corrected - expected) / expected * 100.0
        assertTrue(
            "US Letter at 10 degrees: corrected ratio ($corrected) should be within 2% " +
                "of 0.773, was ${pctError}%",
            pctError < 2.0
        )
    }

    @Test
    fun `angularCorrection_squareAtMildAngle_approxOne`() {
        // Square document at mild angle should still give ratio ~1.0
        val corners = projectRectAtTilt(docW = 200.0, docH = 200.0, tiltDeg = 8.0)
        val corrected = angularCorrectedRatio(corners)

        val pctError = abs(corrected - 1.0) / 1.0 * 100.0
        assertTrue(
            "Square at 8 degrees: corrected ratio ($corrected) should be within 2% " +
                "of 1.0, was ${pctError}%",
            pctError < 2.0
        )
    }

    @Test
    fun `angularCorrection_squareAt0Degrees_exactlyOne`() {
        val corners = makeRect(200.0, 200.0)
        val corrected = angularCorrectedRatio(corners)
        assertEquals(
            "Square at 0 degrees should give ratio = 1.0",
            1.0, corrected, 0.001
        )
    }

    @Test
    fun `angularCorrection_squareAt5Degrees_nearOne`() {
        val corners = projectRectAtTilt(docW = 200.0, docH = 200.0, tiltDeg = 5.0)
        val corrected = angularCorrectedRatio(corners)

        val pctError = abs(corrected - 1.0) / 1.0 * 100.0
        assertTrue(
            "Square at 5 degrees: corrected ratio ($corrected) should be within 1% " +
                "of 1.0, was ${pctError}%",
            pctError < 1.0
        )
    }

    @Test
    fun `angularCorrection_correctionImproves_overRawRatio`() {
        // For a tilted document, the raw ratio is distorted by foreshortening.
        // Angular correction should bring it closer to the true ratio.
        val trueRatio = 210.0 / 297.0
        val corners = projectRectAtTilt(docW = 210.0, docH = 297.0, tiltDeg = 10.0)

        val rawRatio = computeRawRatio(corners)
        val correctedRatio = angularCorrectedRatio(corners)

        val rawError = abs(rawRatio - trueRatio)
        val correctedError = abs(correctedRatio - trueRatio)

        assertTrue(
            "Corrected error ($correctedError) should be <= raw error ($rawError)",
            correctedError <= rawError + 0.001 // small epsilon for FP
        )
    }

    @Test
    fun `angularCorrection_A4landscape_at10Degrees`() {
        // Landscape A4: 297 x 210
        val corners = projectRectAtTilt(docW = 297.0, docH = 210.0, tiltDeg = 10.0)
        val corrected = angularCorrectedRatio(corners)
        val expected = 210.0 / 297.0 // same ratio regardless of orientation

        val pctError = abs(corrected - expected) / expected * 100.0
        assertTrue(
            "Landscape A4 at 10 degrees: corrected ratio ($corrected) should be within 2% " +
                "of 0.707, was ${pctError}%",
            pctError < 2.0
        )
    }

    @Test
    fun `angularCorrection_resultAlwaysInValidRange`() {
        // Test with various tilt angles -- result should always be in [0.1, 1.0]
        for (tilt in listOf(0.0, 5.0, 10.0, 15.0, 20.0, 25.0)) {
            val corners = projectRectAtTilt(docW = 210.0, docH = 297.0, tiltDeg = tilt)
            val corrected = angularCorrectedRatio(corners)
            assertTrue(
                "At tilt=$tilt: corrected ratio ($corrected) should be >= 0.1",
                corrected >= 0.1
            )
            assertTrue(
                "At tilt=$tilt: corrected ratio ($corrected) should be <= 1.0",
                corrected <= 1.0
            )
        }
    }

    @Test
    fun `angularCorrection_identityForRectangle`() {
        // For a perfect rectangle, correction factor = cos(0)/cos(0) = 1
        val corners = makeRect(300.0, 500.0)
        val raw = computeRawRatio(corners)
        val corrected = angularCorrectedRatio(corners)
        assertEquals(
            "No correction needed for perfect rectangle",
            raw, corrected, 1e-10
        )
    }

    @Test
    fun `angularCorrection_receiptAt5Degrees`() {
        // Receipt: 1:3 ratio = 80x240
        val corners = projectRectAtTilt(docW = 80.0, docH = 240.0, tiltDeg = 5.0)
        val corrected = angularCorrectedRatio(corners)
        val expected = 80.0 / 240.0 // 0.333

        val pctError = abs(corrected - expected) / expected * 100.0
        assertTrue(
            "Receipt at 5 degrees: corrected ratio ($corrected) should be within 2% " +
                "of 0.333, was ${pctError}%",
            pctError < 2.0
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun `angularCorrection_rejectsEmptyList`() {
        angularCorrectedRatio(emptyList())
    }

    @Test(expected = IllegalArgumentException::class)
    fun `angularCorrection_rejectsSinglePoint`() {
        angularCorrectedRatio(listOf(Point(0.0, 0.0)))
    }

    // =====================================================================
    // 3. computeRawRatio Tests
    // =====================================================================

    @Test
    fun `rawRatio_A4_isCorrect`() {
        val corners = makeRect(210.0, 297.0)
        assertEquals(210.0 / 297.0, computeRawRatio(corners), 0.001)
    }

    @Test
    fun `rawRatio_USLetter_isCorrect`() {
        val corners = makeRect(8.5, 11.0)
        assertEquals(8.5 / 11.0, computeRawRatio(corners), 0.001)
    }

    @Test
    fun `rawRatio_square_isOne`() {
        val corners = makeRect(150.0, 150.0)
        assertEquals(1.0, computeRawRatio(corners), 0.001)
    }

    @Test
    fun `rawRatio_landscape_sameAsPortrait`() {
        val portrait = computeRawRatio(makeRect(100.0, 200.0))
        val landscape = computeRawRatio(makeRect(200.0, 100.0))
        assertEquals(portrait, landscape, 0.001)
    }

    @Test
    fun `rawRatio_trapezoid_averagesSides`() {
        // Trapezoid: top=200, bottom=300, left~right~400
        // side1 = (topLen + bottomLen) / 2 = (200 + 300) / 2 = 250
        // side2 = height ~= 400
        // ratio = min(250, 400) / max(250, 400) = 250/400 = 0.625
        val corners = listOf(
            Point(50.0, 0.0),   // TL
            Point(250.0, 0.0),  // TR
            Point(300.0, 400.0), // BR
            Point(0.0, 400.0)   // BL
        )
        val ratio = computeRawRatio(corners)
        // side1 (horizontal pair average) = (200 + 300) / 2 = 250
        // side2 (vertical pair) need to compute actual lengths
        // left edge = distance((50,0), (0,400)) = sqrt(50^2 + 400^2) = sqrt(162500) ~ 403.1
        // right edge = distance((250,0), (300,400)) = sqrt(50^2 + 400^2) ~ 403.1
        // side2 = 403.1
        // ratio = 250 / 403.1 = 0.620
        assertTrue(
            "Trapezoid raw ratio ($ratio) should be in [0.5, 0.75]",
            ratio in 0.5..0.75
        )
    }

    @Test
    fun `rawRatio_alwaysLessOrEqualToOne`() {
        val testCases = listOf(
            makeRect(100.0, 200.0),
            makeRect(500.0, 100.0),
            makeRect(1.0, 1000.0),
            makeTrapezoid(w = 300.0, h = 400.0, topInset = 50.0)
        )
        for (corners in testCases) {
            val ratio = computeRawRatio(corners)
            assertTrue("Raw ratio ($ratio) should be <= 1.0", ratio <= 1.0)
            assertTrue("Raw ratio ($ratio) should be > 0.0", ratio > 0.0)
        }
    }

    @Test(expected = IllegalArgumentException::class)
    fun `rawRatio_rejectsThreePoints`() {
        computeRawRatio(listOf(Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0)))
    }

    // =====================================================================
    // 4. Cross-function consistency
    // =====================================================================

    @Test
    fun `angularCorrection_converges_toRawRatio_atZeroTilt`() {
        // At 0 degrees, angular correction should exactly equal raw ratio
        val docs = listOf(
            makeRect(210.0, 297.0),  // A4
            makeRect(8.5, 11.0),     // US Letter
            makeRect(200.0, 200.0),  // Square
            makeRect(80.0, 240.0)    // Receipt
        )
        for (corners in docs) {
            val raw = computeRawRatio(corners)
            val corrected = angularCorrectedRatio(corners)
            assertEquals(
                "At zero tilt, corrected should equal raw",
                raw, corrected, 1e-10
            )
        }
    }

    @Test
    fun `severity_rectangle_allAnglesExactly90`() {
        // Verify the internal math: for a rectangle, all four angles are 90 degrees
        // so max deviation should be exactly 0
        val corners = makeRect(150.0, 300.0)
        val severity = perspectiveSeverity(corners)
        assertEquals(
            "Rectangle angles should all be exactly 90 degrees",
            0.0, severity, 1e-10
        )
    }

    @Test
    fun `angularCorrection_symmetric_leftTiltVsRightTilt`() {
        // Tilting left vs right about vertical axis should give same corrected ratio
        // We can simulate this by flipping the quad horizontally
        val corners = projectRectAtTilt(docW = 210.0, docH = 297.0, tiltDeg = 10.0)
        val flipped = corners.map { Point(-it.x + 640.0, it.y) }
        // Re-order to TL, TR, BR, BL after flip
        val reordered = listOf(flipped[1], flipped[0], flipped[3], flipped[2])

        val corrected = angularCorrectedRatio(corners)
        val correctedFlipped = angularCorrectedRatio(reordered)

        assertEquals(
            "Left and right tilt should give same corrected ratio",
            corrected, correctedFlipped, 0.01
        )
    }
}
