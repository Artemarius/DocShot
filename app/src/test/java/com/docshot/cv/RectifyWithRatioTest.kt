package com.docshot.cv

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import kotlin.math.abs
import kotlin.math.min

class RectifyWithRatioTest {

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
                        "OpenCV native library not available — skipping RectifyWithRatioTest"
                    )
                }
            }
        }
    }

    /** Creates a test image (solid color) and corners matching a rect inside it. */
    private fun makeTestCase(
        imageWidth: Int = 800,
        imageHeight: Int = 600,
        rectLeft: Double = 100.0,
        rectTop: Double = 50.0,
        rectRight: Double = 700.0,
        rectBottom: Double = 550.0
    ): Pair<Mat, List<Point>> {
        val mat = Mat(imageHeight, imageWidth, CvType.CV_8UC3)
        val corners = listOf(
            Point(rectLeft, rectTop),
            Point(rectRight, rectTop),
            Point(rectRight, rectBottom),
            Point(rectLeft, rectBottom)
        )
        return mat to corners
    }

    @Test
    fun `output dimensions match target ratio within 1px`() {
        val (mat, corners) = makeTestCase()
        val targetRatio = 1.0 / 1.414 // A4

        val output = rectifyWithAspectRatio(mat, corners, targetRatio)
        try {
            val w = output.cols()
            val h = output.rows()
            val actualRatio = min(w, h).toDouble() / maxOf(w, h).toDouble()
            assertTrue(
                "Output ratio $actualRatio should match target $targetRatio within tolerance",
                abs(actualRatio - targetRatio) < 0.01
            )
        } finally {
            output.release()
            mat.release()
        }
    }

    @Test
    fun `longer dimension preserved matches standard rectify`() {
        val (mat, corners) = makeTestCase()
        val targetRatio = 1.0 / 1.414

        val standard = rectify(mat, corners)
        val withRatio = rectifyWithAspectRatio(mat, corners, targetRatio)

        try {
            val stdLong = maxOf(standard.cols(), standard.rows())
            val ratioLong = maxOf(withRatio.cols(), withRatio.rows())
            assertEquals(
                "Longer dimension should be preserved",
                stdLong, ratioLong
            )
        } finally {
            standard.release()
            withRatio.release()
            mat.release()
        }
    }

    @Test
    fun `handles landscape orientation correctly`() {
        // Landscape: wider than tall
        val (mat, corners) = makeTestCase(
            imageWidth = 1000, imageHeight = 600,
            rectLeft = 50.0, rectTop = 50.0,
            rectRight = 950.0, rectBottom = 550.0
        )
        val targetRatio = 1.0 / 1.414

        val output = rectifyWithAspectRatio(mat, corners, targetRatio)
        try {
            // In landscape, width is the long dimension
            assertTrue("Output should be landscape", output.cols() >= output.rows())

            val w = output.cols()
            val h = output.rows()
            val actualRatio = min(w, h).toDouble() / maxOf(w, h).toDouble()
            assertTrue(
                "Output ratio $actualRatio should match target $targetRatio within tolerance",
                abs(actualRatio - targetRatio) < 0.01
            )
        } finally {
            output.release()
            mat.release()
        }
    }
}
