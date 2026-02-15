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
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.min

private const val TAG = "DocShot:AspectRatioTest"

/**
 * Instrumented tests for aspect ratio estimation and re-warp.
 * Requires OpenCV native libraries (runs on device/emulator).
 */
@RunWith(AndroidJUnit4::class)
class AspectRatioInstrumentedTest {

    companion object {
        @JvmStatic
        @BeforeClass
        fun initOpenCV() {
            val success = OpenCVLoader.initLocal()
            if (!success) {
                throw IllegalStateException("Failed to initialize OpenCV")
            }
            Log.d(TAG, "OpenCV initialized successfully")
        }
    }

    /**
     * Full pipeline: create synthetic A4 document -> detect -> rectify -> estimate -> verify A4 snap.
     */
    @Test
    fun syntheticA4_detectAndEstimate_snapsToA4() {
        // Create a synthetic image with a white A4-ratio rectangle on dark background
        val imgW = 800
        val imgH = 600
        val image = Mat(imgH, imgW, CvType.CV_8UC3, Scalar(40.0, 40.0, 40.0))

        // A4 ratio: 210x297. Scale to fit in image.
        // Place a white rectangle with A4 proportions
        val docW = 350.0
        val docH = docW * 1.414 // A4 ratio
        val left = (imgW - docW) / 2.0
        val top = (imgH - docH) / 2.0

        Imgproc.rectangle(
            image,
            Point(left, top),
            Point(left + docW, top + docH),
            Scalar(240.0, 240.0, 240.0),
            -1 // filled
        )

        try {
            val detection = detectDocument(image)
            assertNotNull("Should detect the A4 document", detection)

            val corners = detection!!.corners
            val estimate = estimateAspectRatio(corners)

            assertNotNull("Should snap to a known format", estimate.matchedFormat)
            assertEquals("Should snap to A4", "A4", estimate.matchedFormat!!.name)
            assertTrue(
                "Confidence should be >= 0.5, got ${estimate.confidence}",
                estimate.confidence >= 0.5
            )
            Log.d(TAG, "A4 estimate: ratio=%.4f, format=%s, conf=%.3f".format(
                estimate.estimatedRatio, estimate.matchedFormat?.name, estimate.confidence))
        } finally {
            image.release()
        }
    }

    /**
     * Re-warp: rectify normally, then re-warp with A4 ratio -> verify output dimensions.
     */
    @Test
    fun reWarp_producesCorrectDimensions() {
        val imgW = 800
        val imgH = 600
        val image = Mat(imgH, imgW, CvType.CV_8UC3, Scalar(128.0, 128.0, 128.0))

        // Define a slightly trapezoidal quad (simulates perspective)
        val corners = listOf(
            Point(150.0, 80.0),
            Point(650.0, 100.0),
            Point(630.0, 520.0),
            Point(170.0, 500.0)
        )
        val targetRatio = 1.0 / 1.414 // A4

        val standard = rectify(image, corners)
        val reWarped = rectifyWithAspectRatio(image, corners, targetRatio)

        try {
            // Standard rectify preserves raw edges
            assertTrue("Standard output should be valid", standard.cols() > 0 && standard.rows() > 0)

            // Re-warped should match target ratio
            val w = reWarped.cols()
            val h = reWarped.rows()
            val actualRatio = min(w, h).toDouble() / maxOf(w, h).toDouble()
            assertTrue(
                "Re-warped ratio $actualRatio should match target $targetRatio (tolerance 0.01)",
                abs(actualRatio - targetRatio) < 0.01
            )

            // Longer dimension should be preserved
            val stdLong = maxOf(standard.cols(), standard.rows())
            val rewarpLong = maxOf(reWarped.cols(), reWarped.rows())
            assertEquals("Longer dimension should match", stdLong, rewarpLong)

            Log.d(TAG, "Re-warp: standard=%dx%d, reWarped=%dx%d (target ratio=%.3f, actual=%.3f)".format(
                standard.cols(), standard.rows(), w, h, targetRatio, actualRatio))
        } finally {
            standard.release()
            reWarped.release()
            image.release()
        }
    }
}
