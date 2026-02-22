package com.docshot.cv

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

private const val TAG = "DocShot:LineSuppressionTest"

/**
 * Tests for [suppressSpanningLines] in [EdgeDetector.kt] and its integration
 * with the full detection pipeline via [detectDocumentWithStatus].
 *
 * Direct suppression tests verify line detection, border-margin filtering,
 * and morph-close junction healing. Integration tests verify that synthetic
 * images with grout/seam lines are detected after suppression. A false positive
 * guard ensures spanning lines alone do not produce a spurious detection.
 *
 * These are instrumented tests because the CV pipeline uses `android.util.Log`
 * and requires the OpenCV native library loaded via [OpenCVLoader.initLocal].
 */
@RunWith(AndroidJUnit4::class)
class LineSuppressionTest {

    companion object {
        @JvmStatic
        @BeforeClass
        fun initOpenCV() {
            val success = OpenCVLoader.initLocal()
            check(success) { "Failed to initialize OpenCV" }
            Log.d(TAG, "OpenCV initialized successfully")
        }
    }

    // ----------------------------------------------------------------
    // Direct suppression tests
    // ----------------------------------------------------------------

    /**
     * A single horizontal white line spanning the full width near the top
     * (y=20, both endpoints at the left and right borders) should be
     * detected and zeroed by [suppressSpanningLines].
     */
    @Test
    fun horizontalSpanningLine_suppressed() {
        val width = 640
        val height = 480
        val edges = Mat.zeros(height, width, CvType.CV_8UC1)
        try {
            // Draw a horizontal spanning line near the top edge (y=20).
            // thickness=1 matches real Canny edge output (single-pixel edges).
            Imgproc.line(
                edges,
                Point(0.0, 20.0),
                Point((width - 1).toDouble(), 20.0),
                Scalar(255.0),
                1
            )

            // Verify the line exists before suppression
            val whitePixelsBefore = Core.countNonZero(edges)
            assertTrue("Line should have white pixels before suppression", whitePixelsBefore > 0)

            val start = System.nanoTime()
            val suppressed = suppressSpanningLines(edges, Size(width.toDouble(), height.toDouble()))
            val ms = (System.nanoTime() - start) / 1_000_000.0
            Log.d(TAG, "horizontalSpanningLine_suppressed: %.2f ms, suppressed=$suppressed".format(ms))

            assertEquals("Should suppress exactly 1 line", 1, suppressed)

            // Check that the original line area (y=20 row) is zeroed.
            // The morph-close may create a few white pixels elsewhere, but the
            // line itself (row 20, +/- masking thickness) should be gone.
            val lineRow = edges.row(20)
            val lineRowWhite = Core.countNonZero(lineRow)
            // Allow small residual from morph close (kernel overlap), but the
            // vast majority of the original line should be zeroed
            assertTrue(
                "Line row pixels should be mostly zeroed after suppression (found $lineRowWhite white pixels)",
                lineRowWhite < width / 10
            )
            lineRow.release()
        } finally {
            edges.release()
        }
    }

    /**
     * A single vertical white line spanning the full height near the left
     * edge (x=10, both endpoints at the top and bottom borders) should be
     * detected and zeroed.
     */
    @Test
    fun verticalSpanningLine_suppressed() {
        val width = 640
        val height = 480
        val edges = Mat.zeros(height, width, CvType.CV_8UC1)
        try {
            // Draw a vertical spanning line near the left edge (x=10).
            // thickness=1 matches real Canny edge output (single-pixel edges).
            Imgproc.line(
                edges,
                Point(10.0, 0.0),
                Point(10.0, (height - 1).toDouble()),
                Scalar(255.0),
                1
            )

            val start = System.nanoTime()
            val suppressed = suppressSpanningLines(edges, Size(width.toDouble(), height.toDouble()))
            val ms = (System.nanoTime() - start) / 1_000_000.0
            Log.d(TAG, "verticalSpanningLine_suppressed: %.2f ms, suppressed=$suppressed".format(ms))

            assertEquals("Should suppress exactly 1 line", 1, suppressed)

            // Check that the original line area (col 10) is zeroed
            val lineCol = edges.col(10)
            val lineColWhite = Core.countNonZero(lineCol)
            assertTrue(
                "Line column pixels should be mostly zeroed after suppression (found $lineColWhite white pixels)",
                lineColWhite < height / 10
            )
            lineCol.release()
        } finally {
            edges.release()
        }
    }

    /**
     * A short line (40% of image width) should NOT be suppressed --
     * it falls below the 70% spanning threshold.
     */
    @Test
    fun shortLine_notSuppressed() {
        val width = 640
        val height = 480
        val edges = Mat.zeros(height, width, CvType.CV_8UC1)
        try {
            // Draw a line that is only 40% of image width, starting from the left border
            val lineLength = (width * 0.40).toInt()
            Imgproc.line(
                edges,
                Point(0.0, 100.0),
                Point(lineLength.toDouble(), 100.0),
                Scalar(255.0),
                1
            )

            val whitePixelsBefore = Core.countNonZero(edges)

            val start = System.nanoTime()
            val suppressed = suppressSpanningLines(edges, Size(width.toDouble(), height.toDouble()))
            val ms = (System.nanoTime() - start) / 1_000_000.0
            Log.d(TAG, "shortLine_notSuppressed: %.2f ms, suppressed=$suppressed".format(ms))

            assertEquals("Short line should not be suppressed", 0, suppressed)

            // Verify the line is still there (white pixels unchanged)
            val whitePixelsAfter = Core.countNonZero(edges)
            assertEquals(
                "White pixel count should be unchanged when nothing is suppressed",
                whitePixelsBefore,
                whitePixelsAfter
            )
        } finally {
            edges.release()
        }
    }

    /**
     * A long line (>70% of max dim) with at least one endpoint far from
     * any border (>15px interior) should NOT be suppressed -- the
     * border-margin check protects document edges whose endpoints are
     * typically 50-100px inside the frame.
     */
    @Test
    fun interiorLine_notSuppressed() {
        val width = 640
        val height = 480
        val edges = Mat.zeros(height, width, CvType.CV_8UC1)
        try {
            // Draw a long horizontal line (80% of width) with the left endpoint
            // well inside the frame (x=50, which is >15px from the left border)
            // and the right endpoint also inside (x=width*0.8+50)
            val x1 = 50.0  // >15px from left border
            val x2 = x1 + width * 0.80  // long enough to exceed 70% threshold
            Imgproc.line(
                edges,
                Point(x1, 200.0),
                Point(x2, 200.0),
                Scalar(255.0),
                1
            )

            val whitePixelsBefore = Core.countNonZero(edges)

            val start = System.nanoTime()
            val suppressed = suppressSpanningLines(edges, Size(width.toDouble(), height.toDouble()))
            val ms = (System.nanoTime() - start) / 1_000_000.0
            Log.d(TAG, "interiorLine_notSuppressed: %.2f ms, suppressed=$suppressed".format(ms))

            assertEquals("Interior line should not be suppressed", 0, suppressed)

            val whitePixelsAfter = Core.countNonZero(edges)
            assertEquals(
                "White pixel count should be unchanged when nothing is suppressed",
                whitePixelsBefore,
                whitePixelsAfter
            )
        } finally {
            edges.release()
        }
    }

    // ----------------------------------------------------------------
    // Junction healing test
    // ----------------------------------------------------------------

    /**
     * When a spanning line crosses a perpendicular short edge segment
     * (simulating a document boundary at a grout junction), the spanning
     * line is removed and the morph-close heals the junction gap so that
     * the short edge segment is mostly preserved.
     */
    @Test
    fun junctionGap_healedByMorphClose() {
        val width = 640
        val height = 480
        val edges = Mat.zeros(height, width, CvType.CV_8UC1)
        try {
            // Draw a horizontal spanning line at y=240 (mid-height, near no border vertically
            // but endpoints at x=0 and x=639, which are at the borders)
            Imgproc.line(
                edges,
                Point(0.0, 240.0),
                Point((width - 1).toDouble(), 240.0),
                Scalar(255.0),
                1
            )

            // Draw a perpendicular vertical edge segment crossing the spanning line.
            // This simulates a document boundary edge at x=300, from y=180 to y=300
            // (120px long -- short, not spanning). thickness=1 matches Canny output.
            val segmentX = 300
            val segmentYStart = 180
            val segmentYEnd = 300
            Imgproc.line(
                edges,
                Point(segmentX.toDouble(), segmentYStart.toDouble()),
                Point(segmentX.toDouble(), segmentYEnd.toDouble()),
                Scalar(255.0),
                1
            )

            val start = System.nanoTime()
            val suppressed = suppressSpanningLines(edges, Size(width.toDouble(), height.toDouble()))
            val ms = (System.nanoTime() - start) / 1_000_000.0
            Log.d(TAG, "junctionGap_healedByMorphClose: %.2f ms, suppressed=$suppressed".format(ms))

            assertEquals("Should suppress exactly 1 spanning line", 1, suppressed)

            // Check that the short edge segment is mostly preserved.
            // Sample pixels along the segment away from the junction zone.
            // The junction area (near y=240) may have a small gap, but the
            // morph-close (5x5 kernel) should heal it. Check pixels well
            // above and below the junction point.
            var preservedPixels = 0
            val sampleYs = listOf(
                segmentYStart,          // top of segment
                segmentYStart + 10,
                segmentYStart + 20,
                segmentYEnd - 20,
                segmentYEnd - 10,
                segmentYEnd             // bottom of segment
            )
            for (y in sampleYs) {
                // Check a small neighborhood around x=300 for white pixels
                // (morph close may shift positions slightly)
                for (dx in -2..2) {
                    val px = edges.get(y, segmentX + dx)
                    if (px != null && px[0] > 0) {
                        preservedPixels++
                        break  // found a white pixel at this y, move to next y
                    }
                }
            }

            assertTrue(
                "At least 4 of ${sampleYs.size} sample points on the short edge segment " +
                    "should be preserved after morph close (found $preservedPixels)",
                preservedPixels >= 4
            )
        } finally {
            edges.release()
        }
    }

    // ----------------------------------------------------------------
    // Integration detection tests
    // ----------------------------------------------------------------

    /**
     * Document on a tan surface with grout lines -- the key regression scenario.
     * Grout lines fragment the document contour; line suppression should clean
     * them up so the document is detected.
     */
    @Test
    fun docOnTanSurfaceWithGroutLines_detectedAfterSuppression() {
        val (image, _) = SyntheticImageFactory.docOnTanSurfaceWithGroutLines()
        try {
            invalidateSceneCache()
            val start = System.nanoTime()
            val status = detectDocumentWithStatus(image)
            val ms = (System.nanoTime() - start) / 1_000_000.0
            Log.d(TAG, "docOnTanSurfaceWithGroutLines: %.1f ms, detected=${status.result != null}, " +
                "confidence=${status.result?.confidence ?: "N/A"}".format(ms))

            assertNotNull(
                "Document on tan surface with grout lines should be detected after line suppression",
                status.result
            )
            assertTrue(
                "Confidence should be >= 0.35, got ${status.result!!.confidence}",
                status.result.confidence >= 0.35
            )
        } finally {
            image.release()
        }
    }

    /**
     * Document on a tile floor with a dense grout grid.
     * Line suppression should remove the spanning grout lines so the
     * document contour is cleanly extractable.
     */
    @Test
    fun docOnTileFloor_detectedAfterSuppression() {
        val (image, _) = SyntheticImageFactory.docOnTileFloor()
        try {
            invalidateSceneCache()
            val start = System.nanoTime()
            val status = detectDocumentWithStatus(image)
            val ms = (System.nanoTime() - start) / 1_000_000.0
            Log.d(TAG, "docOnTileFloor: %.1f ms, detected=${status.result != null}, " +
                "confidence=${status.result?.confidence ?: "N/A"}".format(ms))

            assertNotNull(
                "Document on tile floor should be detected after line suppression",
                status.result
            )
            assertTrue(
                "Confidence should be >= 0.35, got ${status.result!!.confidence}",
                status.result.confidence >= 0.35
            )
        } finally {
            image.release()
        }
    }

    /**
     * Document on a surface with a diagonal seam/crease line.
     * The diagonal line spans corner-to-corner of the image so both
     * endpoints are near the border; suppression should remove it.
     */
    @Test
    fun docWithDiagonalSeam_detectedAfterSuppression() {
        val (image, _) = SyntheticImageFactory.docWithDiagonalSeam()
        try {
            invalidateSceneCache()
            val start = System.nanoTime()
            val status = detectDocumentWithStatus(image)
            val ms = (System.nanoTime() - start) / 1_000_000.0
            Log.d(TAG, "docWithDiagonalSeam: %.1f ms, detected=${status.result != null}, " +
                "confidence=${status.result?.confidence ?: "N/A"}".format(ms))

            assertNotNull(
                "Document with diagonal seam should be detected after line suppression",
                status.result
            )
            assertTrue(
                "Confidence should be >= 0.35, got ${status.result!!.confidence}",
                status.result.confidence >= 0.35
            )
        } finally {
            image.release()
        }
    }

    // ----------------------------------------------------------------
    // False positive guard
    // ----------------------------------------------------------------

    /**
     * Image with spanning lines but NO document present.
     * Line suppression should remove the lines, and the pipeline should
     * not produce a false positive detection from whatever residual
     * artifacts remain.
     */
    @Test
    fun spanningLinesNoDocs_noFalsePositive() {
        val image = SyntheticImageFactory.spanningLinesNoDocs()
        try {
            invalidateSceneCache()
            val start = System.nanoTime()
            val status = detectDocumentWithStatus(image)
            val ms = (System.nanoTime() - start) / 1_000_000.0
            Log.d(TAG, "spanningLinesNoDocs: %.1f ms, detected=${status.result != null}, " +
                "confidence=${status.result?.confidence ?: "N/A"}".format(ms))

            val noFalsePositive = status.result == null || status.result.confidence < MIN_CONFIDENCE_THRESHOLD
            assertTrue(
                "Spanning lines with no document should not produce a false positive " +
                    "(result=${status.result != null}, confidence=${status.result?.confidence ?: 0.0})",
                noFalsePositive
            )
        } finally {
            image.release()
        }
    }
}
