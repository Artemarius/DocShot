package com.docshot.cv

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc

/**
 * Tests for [QuadValidator.edgeDensityScore].
 *
 * Requires OpenCV native libs — run as androidTest or with opencv_java4 on classpath.
 * These tests create synthetic edge images with known geometry to verify that
 * edge-density scoring correctly distinguishes supported vs unsupported quads.
 */
class QuadValidatorTest {

    companion object {
        @JvmStatic
        @BeforeClass
        fun loadOpenCV() {
            // Load OpenCV native library for JVM tests.
            // This requires opencv_java4 (or opencv_java490, etc.) on java.library.path.
            // If unavailable, these tests must be run as Android instrumented tests instead.
            try {
                System.loadLibrary("opencv_java4")
            } catch (_: UnsatisfiedLinkError) {
                try {
                    System.loadLibrary("opencv_java412")
                } catch (_: UnsatisfiedLinkError) {
                    System.err.println(
                        "WARNING: Could not load OpenCV native library. " +
                            "QuadValidatorTest requires opencv_java4 on java.library.path " +
                            "or must be run as an Android instrumented test."
                    )
                    throw org.junit.AssumptionViolatedException(
                        "OpenCV native library not available — skipping QuadValidatorTest"
                    )
                }
            }
        }
    }

    /**
     * A quad whose edges are fully drawn on the edge image should score > 0.9.
     */
    @Test
    fun `perfect edge density scores above 0_9`() {
        val edges = Mat.zeros(200, 200, CvType.CV_8UC1)
        val quad = listOf(
            Point(30.0, 30.0),   // TL
            Point(170.0, 30.0),  // TR
            Point(170.0, 170.0), // BR
            Point(30.0, 170.0)   // BL
        )
        // Draw all 4 edges of the quad onto the edge image
        drawQuadEdges(edges, quad)

        val score = QuadValidator.edgeDensityScore(edges, quad)
        edges.release()

        assertTrue(
            "Perfect edge density should be > 0.9, got $score",
            score > 0.9
        )
    }

    /**
     * An empty (all-black) edge image with a quad in the middle should score ~0.0.
     */
    @Test
    fun `zero edge density on empty image scores 0`() {
        val edges = Mat.zeros(200, 200, CvType.CV_8UC1)
        val quad = listOf(
            Point(40.0, 40.0),
            Point(160.0, 40.0),
            Point(160.0, 160.0),
            Point(40.0, 160.0)
        )

        val score = QuadValidator.edgeDensityScore(edges, quad)
        edges.release()

        assertEquals("Empty image should produce score 0.0", 0.0, score, 0.001)
    }

    /**
     * When only 2 of 4 quad edges have supporting edge pixels, the score
     * should be approximately 0.5 (half the perimeter is supported).
     */
    @Test
    fun `partial edge density with 2 of 4 edges scores around 0_5`() {
        val edges = Mat.zeros(200, 200, CvType.CV_8UC1)
        val quad = listOf(
            Point(30.0, 30.0),   // TL
            Point(170.0, 30.0),  // TR
            Point(170.0, 170.0), // BR
            Point(30.0, 170.0)   // BL
        )
        // Draw only edges TL->TR (top) and BR->BL (bottom)
        Imgproc.line(edges, quad[0], quad[1], Scalar(255.0), 1)
        Imgproc.line(edges, quad[2], quad[3], Scalar(255.0), 1)

        val score = QuadValidator.edgeDensityScore(edges, quad)
        edges.release()

        assertTrue(
            "Partial (2/4 edges) density should be in [0.35, 0.65], got $score",
            score in 0.35..0.65
        )
    }

    /**
     * A quad with corners near or at image boundaries should not crash
     * and should return a reasonable score.
     */
    @Test
    fun `quad at image boundary does not crash`() {
        val edges = Mat.zeros(200, 200, CvType.CV_8UC1)
        val quad = listOf(
            Point(0.0, 0.0),
            Point(199.0, 0.0),
            Point(199.0, 199.0),
            Point(0.0, 199.0)
        )
        // Draw all edges along the boundary
        drawQuadEdges(edges, quad)

        val score = QuadValidator.edgeDensityScore(edges, quad)
        edges.release()

        assertTrue(
            "Boundary quad with edges drawn should score > 0.8, got $score",
            score > 0.8
        )
    }

    /**
     * With edges offset by 2 pixels from the quad, a larger search radius
     * should find more support than a zero search radius.
     */
    @Test
    fun `larger search radius finds offset edges better`() {
        val edges = Mat.zeros(200, 200, CvType.CV_8UC1)
        // Quad centered in the image
        val quad = listOf(
            Point(50.0, 50.0),
            Point(150.0, 50.0),
            Point(150.0, 150.0),
            Point(50.0, 150.0)
        )
        // Draw edges offset inward by 2 pixels from the quad
        val offsetQuad = listOf(
            Point(52.0, 52.0),
            Point(148.0, 52.0),
            Point(148.0, 148.0),
            Point(52.0, 148.0)
        )
        drawQuadEdges(edges, offsetQuad)

        val scoreNarrow = QuadValidator.edgeDensityScore(
            edges,
            quad,
            searchRadius = 0
        )
        val scoreWide = QuadValidator.edgeDensityScore(
            edges,
            quad,
            searchRadius = 3
        )
        edges.release()

        assertTrue(
            "searchRadius=3 ($scoreWide) should score higher than " +
                "searchRadius=0 ($scoreNarrow) for 2px offset edges",
            scoreWide > scoreNarrow
        )
    }

    // ------------------------------------------------------------------
    // Helper
    // ------------------------------------------------------------------

    /** Draws all 4 edges of a quad onto a binary edge image. */
    private fun drawQuadEdges(edges: Mat, quad: List<Point>) {
        for (i in 0 until 4) {
            Imgproc.line(edges, quad[i], quad[(i + 1) % 4], Scalar(255.0), 1)
        }
    }
}
