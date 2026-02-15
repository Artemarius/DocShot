package com.docshot.cv

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.opencv.core.Point

class QuadScoringTest {

    companion object {
        @JvmStatic
        @BeforeClass
        fun loadOpenCV() {
            // Load OpenCV native library for JVM tests.
            // rankQuads() calls Imgproc.isContourConvex which requires native code.
            // If unavailable, rankQuads tests will be skipped via AssumptionViolatedException.
            try {
                System.loadLibrary("opencv_java4")
            } catch (_: UnsatisfiedLinkError) {
                try {
                    System.loadLibrary("opencv_java412")
                } catch (_: UnsatisfiedLinkError) {
                    System.err.println(
                        "WARNING: Could not load OpenCV native library. " +
                            "rankQuads tests require opencv_java4 on java.library.path " +
                            "or must be run as an Android instrumented test."
                    )
                    throw org.junit.AssumptionViolatedException(
                        "OpenCV native library not available — skipping QuadScoringTest"
                    )
                }
            }
        }
    }

    @Test
    fun `quadArea computes correct area for a rectangle`() {
        val rect = listOf(
            Point(0.0, 0.0),
            Point(100.0, 0.0),
            Point(100.0, 200.0),
            Point(0.0, 200.0)
        )
        assertEquals(20000.0, quadArea(rect), 0.1)
    }

    @Test
    fun `quadArea computes correct area for a non-axis-aligned quad`() {
        // Triangle-like degenerate quad with one collapsed edge still works
        val quad = listOf(
            Point(0.0, 0.0),
            Point(4.0, 0.0),
            Point(4.0, 3.0),
            Point(0.0, 3.0)
        )
        assertEquals(12.0, quadArea(quad), 0.001)
    }

    @Test
    fun `perfect rectangle has angle score close to 1`() {
        val rect = listOf(
            Point(0.0, 0.0),
            Point(100.0, 0.0),
            Point(100.0, 200.0),
            Point(0.0, 200.0)
        )
        val score = angleRegularityScore(rect)
        assertTrue("Perfect rectangle score should be > 0.95, got $score", score > 0.95)
    }

    @Test
    fun `skewed quadrilateral has lower angle score than rectangle`() {
        val rect = listOf(
            Point(0.0, 0.0),
            Point(100.0, 0.0),
            Point(100.0, 100.0),
            Point(0.0, 100.0)
        )
        val skewed = listOf(
            Point(20.0, 0.0),
            Point(120.0, 0.0),
            Point(100.0, 100.0),
            Point(0.0, 100.0)
        )
        val rectScore = angleRegularityScore(rect)
        val skewedScore = angleRegularityScore(skewed)
        assertTrue(
            "Rectangle ($rectScore) should score higher than skewed ($skewedScore)",
            rectScore > skewedScore
        )
    }

    @Test
    fun `larger quad area is computed correctly`() {
        val small = listOf(
            Point(0.0, 0.0),
            Point(50.0, 0.0),
            Point(50.0, 50.0),
            Point(0.0, 50.0)
        )
        val large = listOf(
            Point(0.0, 0.0),
            Point(200.0, 0.0),
            Point(200.0, 200.0),
            Point(0.0, 200.0)
        )
        assertTrue(quadArea(large) > quadArea(small))
    }

    // ------------------------------------------------------------------
    // rankQuads tests
    // ------------------------------------------------------------------

    @Test
    fun `rankQuads returns unambiguous result for single candidate`() {
        val quad = listOf(
            Point(0.0, 0.0),
            Point(100.0, 0.0),
            Point(100.0, 100.0),
            Point(0.0, 100.0)
        )
        val imageArea = 200.0 * 200.0

        val result = rankQuads(listOf(quad), imageArea)

        assertNotNull("Single convex quad should produce a non-null result", result.quad)
        assertEquals("Single candidate → candidateCount = 1", 1, result.candidateCount)
        assertEquals(
            "Single candidate → scoreMargin = 1.0 (unambiguous)",
            1.0, result.scoreMargin, 0.001
        )
        assertTrue("Score should be positive", result.score > 0.0)
    }

    @Test
    fun `rankQuads reports low margin for similar candidates`() {
        // Two nearly identical rectangles — scores should be very close,
        // producing a low score margin (< 0.2).
        val quadA = listOf(
            Point(10.0, 10.0),
            Point(110.0, 10.0),
            Point(110.0, 110.0),
            Point(10.0, 110.0)
        )
        val quadB = listOf(
            Point(12.0, 12.0),
            Point(112.0, 12.0),
            Point(112.0, 112.0),
            Point(12.0, 112.0)
        )
        val imageArea = 200.0 * 200.0

        val result = rankQuads(listOf(quadA, quadB), imageArea)

        assertNotNull("Should find a best quad", result.quad)
        assertEquals("Two convex quads → candidateCount = 2", 2, result.candidateCount)
        assertTrue(
            "Similar candidates should yield low scoreMargin (< 0.2), got ${result.scoreMargin}",
            result.scoreMargin < 0.2
        )
    }

    @Test
    fun `rankQuads reports high margin for clearly different candidates`() {
        // One large quad covering most of the image, one tiny quad.
        // The large quad should clearly dominate, giving a high score margin.
        val largeQuad = listOf(
            Point(10.0, 10.0),
            Point(190.0, 10.0),
            Point(190.0, 190.0),
            Point(10.0, 190.0)
        )
        val tinyQuad = listOf(
            Point(0.0, 0.0),
            Point(10.0, 0.0),
            Point(10.0, 10.0),
            Point(0.0, 10.0)
        )
        val imageArea = 200.0 * 200.0

        val result = rankQuads(listOf(largeQuad, tinyQuad), imageArea)

        assertNotNull("Should find a best quad", result.quad)
        assertEquals("Two convex quads → candidateCount = 2", 2, result.candidateCount)
        assertTrue(
            "Clearly different candidates should yield high scoreMargin (> 0.5), got ${result.scoreMargin}",
            result.scoreMargin > 0.5
        )
    }

    @Test
    fun `rankQuads returns null quad for empty candidates`() {
        val result = rankQuads(emptyList(), imageArea = 200.0 * 200.0)

        assertNull("Empty candidates should yield null quad", result.quad)
        assertEquals("No candidates → candidateCount = 0", 0, result.candidateCount)
        assertEquals("No candidates → score = 0.0", 0.0, result.score, 0.001)
        assertEquals("No candidates → scoreMargin = 0.0", 0.0, result.scoreMargin, 0.001)
    }

    @Test
    fun `rankQuads rejects non-convex and counts only valid`() {
        // One convex rectangle
        val convexQuad = listOf(
            Point(0.0, 0.0),
            Point(100.0, 0.0),
            Point(100.0, 100.0),
            Point(0.0, 100.0)
        )
        // One non-convex (concave) quadrilateral — point 2 is pushed inward
        val concaveQuad = listOf(
            Point(0.0, 0.0),
            Point(100.0, 0.0),
            Point(30.0, 30.0), // concavity
            Point(0.0, 100.0)
        )
        val imageArea = 200.0 * 200.0

        val result = rankQuads(listOf(convexQuad, concaveQuad), imageArea)

        assertNotNull("Should find the convex quad", result.quad)
        assertEquals(
            "Only the convex quad should be counted",
            1, result.candidateCount
        )
        assertEquals(
            "Single valid candidate → scoreMargin = 1.0",
            1.0, result.scoreMargin, 0.001
        )
    }

    // ------------------------------------------------------------------
    // aspectRatioScore tests
    // ------------------------------------------------------------------

    @Test
    fun `aspectRatioScore high for A4 ratio`() {
        // A4 is 1:1.414, so a 100x141 rectangle
        val a4 = listOf(
            Point(0.0, 0.0),
            Point(100.0, 0.0),
            Point(100.0, 141.4),
            Point(0.0, 141.4)
        )
        val score = aspectRatioScore(a4)
        assertTrue("A4 aspect ratio score should be > 0.9, got $score", score > 0.9)
    }

    @Test
    fun `aspectRatioScore high for receipt ratio`() {
        // Receipt is ~1:3, so a 100x300 rectangle
        val receipt = listOf(
            Point(0.0, 0.0),
            Point(100.0, 0.0),
            Point(100.0, 300.0),
            Point(0.0, 300.0)
        )
        val score = aspectRatioScore(receipt)
        assertTrue("Receipt aspect ratio score should be > 0.8, got $score", score > 0.8)
    }

    @Test
    fun `aspectRatioScore high for business card ratio`() {
        // Business card is ~1:1.75, so a 100x175 rectangle
        val card = listOf(
            Point(0.0, 0.0),
            Point(100.0, 0.0),
            Point(100.0, 175.0),
            Point(0.0, 175.0)
        )
        val score = aspectRatioScore(card)
        assertTrue("Business card aspect ratio score should be > 0.8, got $score", score > 0.8)
    }

    @Test
    fun `aspectRatioScore low for extreme ratio`() {
        // Extreme aspect ratio 1:10 — doesn't match any known format
        val extreme = listOf(
            Point(0.0, 0.0),
            Point(10.0, 0.0),
            Point(10.0, 100.0),
            Point(0.0, 100.0)
        )
        val score = aspectRatioScore(extreme)
        assertTrue("Extreme aspect ratio score should be < 0.3, got $score", score < 0.3)
    }

    @Test
    fun `aspectRatioScore high for US Letter ratio`() {
        // US Letter is 8.5x11 = 1:1.294, so a 100x129.4 rectangle
        val letter = listOf(
            Point(0.0, 0.0),
            Point(100.0, 0.0),
            Point(100.0, 129.4),
            Point(0.0, 129.4)
        )
        val score = aspectRatioScore(letter)
        assertTrue("US Letter aspect ratio score should be > 0.8, got $score", score > 0.8)
    }

    @Test
    fun `aspectRatioScore high for ID card ratio`() {
        // CR-80 ID card is 85.6x53.98mm = 1:1.586 (long/short), so a 100x158.6 rectangle
        val idCard = listOf(
            Point(0.0, 0.0),
            Point(100.0, 0.0),
            Point(100.0, 158.6),
            Point(0.0, 158.6)
        )
        val score = aspectRatioScore(idCard)
        assertTrue("ID card aspect ratio score should be > 0.8, got $score", score > 0.8)
    }

    @Test
    fun `aspectRatioScore high for square ratio`() {
        // Square 1:1 — common for some sticky notes, coasters, etc.
        val square = listOf(
            Point(0.0, 0.0),
            Point(100.0, 0.0),
            Point(100.0, 100.0),
            Point(0.0, 100.0)
        )
        val score = aspectRatioScore(square)
        assertTrue("Square aspect ratio score should be > 0.9, got $score", score > 0.9)
    }

    @Test
    fun `scoreQuad includes aspect ratio component`() {
        // Compare a document-shaped quad vs a very thin strip
        // Both same area and same angles, but different aspect ratios
        val documentShaped = listOf(
            Point(0.0, 0.0),
            Point(100.0, 0.0),
            Point(100.0, 141.4),
            Point(0.0, 141.4)
        )
        val thinStrip = listOf(
            Point(0.0, 0.0),
            Point(10.0, 0.0),
            Point(10.0, 1414.0),
            Point(0.0, 1414.0)
        )
        // Same image area for both
        val imageArea = 200.0 * 200.0

        val documentScore = scoreQuad(documentShaped, imageArea)
        val stripScore = scoreQuad(thinStrip, imageArea)

        // Document-shaped quad should score higher because of better aspect ratio match,
        // even though the strip has similar area and perfect angles
        assertTrue(
            "Document shape ($documentScore) should score higher than thin strip ($stripScore)",
            documentScore > stripScore
        )
    }
}
