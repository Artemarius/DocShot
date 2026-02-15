package com.docshot.cv

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.Point
import org.opencv.core.Scalar

private const val TAG = "DocShot:RegressionTest"

/**
 * End-to-end regression tests for the document detection pipeline.
 * Uses [SyntheticImageFactory] to generate test images with known ground-truth
 * corners, then verifies that [detectDocument] and [detectDocumentWithStatus]
 * produce correct results.
 *
 * These are instrumented tests because the CV pipeline uses `android.util.Log`.
 */
@RunWith(AndroidJUnit4::class)
class DetectionRegressionTest {

    companion object {
        /** Maximum acceptable corner deviation in pixels (at 800x600 resolution). */
        private const val MAX_CORNER_ERROR_PX = 15.0

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

    @Test
    fun highContrastWhiteOnDark_detectsWithHighConfidence() {
        val image = SyntheticImageFactory.whiteDocOnSolidBg()
        try {
            val result = detectDocument(image)
            assertNotNull("Should detect high-contrast document", result)
            assertTrue(
                "Confidence should be >= 0.6, got ${result!!.confidence}",
                result.confidence >= 0.6
            )
            assertCornersClose(result.corners, SyntheticImageFactory.defaultA4Corners())
        } finally {
            image.release()
        }
    }

    @Test
    fun lowContrast_detectsAfterCLAHE() {
        val image = SyntheticImageFactory.lowContrastDoc()
        try {
            val result = detectDocument(image)
            assertNotNull("Should detect low-contrast document (CLAHE strategy)", result)
            Log.d(TAG, "Low contrast confidence: ${result!!.confidence}")
        } finally {
            image.release()
        }
    }

    @Test
    fun lowLight_detectsAfterCLAHE() {
        val image = SyntheticImageFactory.lowLightDoc(brightness = 0.25)
        try {
            val result = detectDocument(image)
            assertNotNull("Should detect low-light document (CLAHE strategy)", result)
            Log.d(TAG, "Low light confidence: ${result!!.confidence}")
        } finally {
            image.release()
        }
    }

    @Test
    fun shadowAcrossDoc_detects() {
        val image = SyntheticImageFactory.shadowedDoc(shadowIntensity = 0.6)
        try {
            val result = detectDocument(image)
            assertNotNull("Should detect document with shadow", result)
            Log.d(TAG, "Shadow confidence: ${result!!.confidence}")
        } finally {
            image.release()
        }
    }

    @Test
    fun whiteDocOnBlueBg_detects() {
        val image = SyntheticImageFactory.coloredBgDoc(
            bgColor = Scalar(200.0, 100.0, 50.0) // blue desk in BGR
        )
        try {
            val result = detectDocument(image)
            assertNotNull("Should detect document on colored background", result)
            Log.d(TAG, "Colored bg confidence: ${result!!.confidence}")
        } finally {
            image.release()
        }
    }

    @Test
    fun patternedBg_detects() {
        val image = SyntheticImageFactory.patternedBgDoc(patternType = "lines")
        try {
            val result = detectDocument(image)
            assertNotNull("Should detect document on patterned background", result)
            Log.d(TAG, "Patterned bg confidence: ${result!!.confidence}")
        } finally {
            image.release()
        }
    }

    @Test
    fun smallReceipt_detects() {
        val image = SyntheticImageFactory.smallDoc(
            corners = SyntheticImageFactory.defaultReceiptCorners()
        )
        try {
            val result = detectDocument(image)
            assertNotNull("Should detect small receipt-sized document", result)
            Log.d(TAG, "Small receipt confidence: ${result!!.confidence}")
        } finally {
            image.release()
        }
    }

    @Test
    fun smallBusinessCard_detects() {
        val image = SyntheticImageFactory.smallDoc(
            corners = SyntheticImageFactory.defaultBusinessCardCorners()
        )
        try {
            val result = detectDocument(image)
            assertNotNull("Should detect small business card-sized document", result)
            Log.d(TAG, "Small business card confidence: ${result!!.confidence}")
        } finally {
            image.release()
        }
    }

    @Test
    fun partialDoc_flagsPartialDocument() {
        val image = SyntheticImageFactory.partialDoc(visibleCorners = 3)
        try {
            val status = detectDocumentWithStatus(image)
            // A quad touching 2+ frame edges should always flag isPartialDocument,
            // whether or not the quad is accepted as a valid detection.
            assertTrue(
                "Partial doc should set isPartialDocument=true",
                status.isPartialDocument
            )
            Log.d(TAG, "Partial doc: result=${status.result != null}, partial=${status.isPartialDocument}")
        } finally {
            image.release()
        }
    }

    @Test
    fun noDocument_returnsNullNoFalsePositive() {
        // Plain background with no document — should not falsely detect anything
        val image = org.opencv.core.Mat(
            600, 800, org.opencv.core.CvType.CV_8UC3,
            Scalar(120.0, 120.0, 120.0)
        )
        try {
            val result = detectDocument(image)
            assertNull("Plain background should not trigger false positive", result)
        } finally {
            image.release()
        }
    }

    @Test
    fun highContrastWithNoise_stillDetects() {
        val clean = SyntheticImageFactory.whiteDocOnSolidBg()
        val noisy = SyntheticImageFactory.addNoise(clean, stddev = 20.0)
        clean.release()
        try {
            val result = detectDocument(noisy)
            assertNotNull("Should detect document despite sensor noise", result)
            Log.d(TAG, "Noisy image confidence: ${result!!.confidence}")
        } finally {
            noisy.release()
        }
    }

    // ----------------------------------------------------------------
    // Group D: Document types, challenging conditions, expanded coverage
    // ----------------------------------------------------------------

    @Test
    fun usLetterFormat_detects() {
        val image = SyntheticImageFactory.whiteDocOnSolidBg(
            corners = SyntheticImageFactory.defaultLetterCorners()
        )
        try {
            val result = detectDocument(image)
            assertNotNull("Should detect US Letter-sized document", result)
            Log.d(TAG, "US Letter confidence: ${result!!.confidence}")
        } finally {
            image.release()
        }
    }

    @Test
    fun idCardFormat_detects() {
        val image = SyntheticImageFactory.smallDoc(
            corners = SyntheticImageFactory.defaultIdCardCorners()
        )
        try {
            val result = detectDocument(image)
            assertNotNull("Should detect ID card-sized document", result)
            Log.d(TAG, "ID card confidence: ${result!!.confidence}")
        } finally {
            image.release()
        }
    }

    @Test
    fun extremeLowLight_detectsOrGracefullyFails() {
        // Extremely dim scene — pipeline must not crash even if detection fails
        val image = SyntheticImageFactory.lowLightDoc(brightness = 0.10)
        try {
            val result = detectDocument(image)
            // No assertion on detection — just verifying no crash
            Log.d(TAG, "Extreme low light: detected=${result != null}, " +
                "confidence=${result?.confidence ?: "N/A"}")
        } finally {
            image.release()
        }
    }

    @Test
    fun overexposed_detects() {
        val image = SyntheticImageFactory.overexposedDoc(exposure = 1.8)
        try {
            val result = detectDocument(image)
            assertNotNull("Should detect document in overexposed scene", result)
            Log.d(TAG, "Overexposed confidence: ${result!!.confidence}")
        } finally {
            image.release()
        }
    }

    @Test
    fun verticalShadow_detects() {
        val image = SyntheticImageFactory.verticalShadowDoc(shadowIntensity = 0.6)
        try {
            val result = detectDocument(image)
            assertNotNull("Should detect document with vertical shadow", result)
            Log.d(TAG, "Vertical shadow confidence: ${result!!.confidence}")
        } finally {
            image.release()
        }
    }

    @Test
    fun redBackground_detects() {
        val image = SyntheticImageFactory.coloredBgDoc(
            bgColor = Scalar(50.0, 50.0, 200.0)  // red in BGR
        )
        try {
            val result = detectDocument(image)
            assertNotNull("Should detect document on red background", result)
            Log.d(TAG, "Red bg confidence: ${result!!.confidence}")
        } finally {
            image.release()
        }
    }

    @Test
    fun greenBackground_detects() {
        val image = SyntheticImageFactory.coloredBgDoc(
            bgColor = Scalar(50.0, 180.0, 50.0)  // green in BGR
        )
        try {
            val result = detectDocument(image)
            assertNotNull("Should detect document on green background", result)
            Log.d(TAG, "Green bg confidence: ${result!!.confidence}")
        } finally {
            image.release()
        }
    }

    @Test
    fun gridPatternBg_detects() {
        val image = SyntheticImageFactory.patternedBgDoc(patternType = "grid")
        try {
            val result = detectDocument(image)
            assertNotNull("Should detect document on grid-patterned background", result)
            Log.d(TAG, "Grid pattern bg confidence: ${result!!.confidence}")
        } finally {
            image.release()
        }
    }

    @Test
    fun lowLightWithNoise_detects() {
        val dim = SyntheticImageFactory.lowLightDoc(brightness = 0.30)
        val noisy = SyntheticImageFactory.addNoise(dim, stddev = 15.0)
        dim.release()
        try {
            val result = detectDocument(noisy)
            assertNotNull("Should detect document in low light with noise", result)
            Log.d(TAG, "Low light + noise confidence: ${result!!.confidence}")
        } finally {
            noisy.release()
        }
    }

    @Test
    fun smallReceiptOnColoredBg_detects() {
        val image = SyntheticImageFactory.smallDoc(
            corners = SyntheticImageFactory.defaultReceiptCorners(),
            bgColor = Scalar(100.0, 130.0, 170.0)  // warm wood-tone in BGR
        )
        try {
            val result = detectDocument(image)
            assertNotNull("Should detect small receipt on colored background", result)
            Log.d(TAG, "Small receipt on colored bg confidence: ${result!!.confidence}")
        } finally {
            image.release()
        }
    }

    // ----------------------------------------------------------------
    // Helpers
    // ----------------------------------------------------------------

    private fun assertCornersClose(
        detected: List<Point>,
        expected: List<Point>,
        maxError: Double = MAX_CORNER_ERROR_PX
    ) {
        assertTrue("Expected 4 corners, got ${detected.size}", detected.size == 4)
        // Order both sets consistently for comparison
        val orderedDetected = orderCorners(detected)
        val orderedExpected = orderCorners(expected)
        for (i in orderedDetected.indices) {
            val d = orderedDetected[i]
            val e = orderedExpected[i]
            val dist = kotlin.math.sqrt(
                (d.x - e.x) * (d.x - e.x) + (d.y - e.y) * (d.y - e.y)
            )
            assertTrue(
                "Corner $i deviation ${dist}px exceeds max ${maxError}px " +
                    "(detected=${d.x},${d.y} expected=${e.x},${e.y})",
                dist <= maxError
            )
        }
    }
}
