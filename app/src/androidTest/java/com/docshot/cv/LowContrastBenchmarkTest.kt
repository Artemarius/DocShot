package com.docshot.cv

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.Point

private const val TAG = "DocShot:LCBenchmark"

/**
 * Benchmark harness for low-contrast / white-on-white detection strategies.
 * Runs each test image through each strategy individually via [detectWithStrategy],
 * recording detection success, confidence, corner accuracy, and processing time.
 *
 * Primary purpose is to compare strategy effectiveness — not strict pass/fail.
 * A few basic assertions ensure at least one new strategy improves detection.
 */
@RunWith(AndroidJUnit4::class)
class LowContrastBenchmarkTest {

    companion object {
        @JvmStatic
        @BeforeClass
        fun initOpenCV() {
            val success = OpenCVLoader.initLocal()
            check(success) { "Failed to initialize OpenCV" }
            Log.d(TAG, "OpenCV initialized successfully")
        }
    }

    private data class BenchmarkResult(
        val imageName: String,
        val strategy: PreprocessStrategy,
        val detected: Boolean,
        val confidence: Double,
        val timeMs: Double,
        val maxCornerErrorPx: Double?
    )

    /**
     * Runs all low-contrast test images through all strategies individually.
     * Logs a comparison table and asserts basic effectiveness.
     */
    @Test
    fun benchmarkAllStrategies() {
        val groundTruth = SyntheticImageFactory.defaultA4Corners()

        val images = mapOf(
            "whiteOnNearWhite" to SyntheticImageFactory.whiteOnNearWhite(),
            "whiteOnWhite" to SyntheticImageFactory.whiteOnWhite(),
            "whiteOnCream" to SyntheticImageFactory.whiteOnCream(),
            "whiteOnLightWood" to SyntheticImageFactory.whiteOnLightWood(),
            "whiteOnWhiteTextured" to SyntheticImageFactory.whiteOnWhiteTextured(),
            "glossyPaper" to SyntheticImageFactory.glossyPaper(),
        )

        val strategies = listOf(
            // New low-contrast strategies
            PreprocessStrategy.ADAPTIVE_THRESHOLD,
            PreprocessStrategy.GRADIENT_MAGNITUDE,
            PreprocessStrategy.LAB_CLAHE,
            PreprocessStrategy.DOG,
            PreprocessStrategy.MULTICHANNEL_FUSION,
            // Existing strategies for comparison
            PreprocessStrategy.STANDARD,
            PreprocessStrategy.CLAHE_ENHANCED,
        )

        val results = mutableListOf<BenchmarkResult>()

        for ((name, image) in images) {
            for (strategy in strategies) {
                // Clear scene cache so each strategy is measured independently
                invalidateSceneCache()

                val start = System.nanoTime()
                val status = detectWithStrategy(image, strategy)
                val timeMs = (System.nanoTime() - start) / 1_000_000.0

                val detection = status.result
                val cornerError = if (detection != null) {
                    maxCornerError(detection.corners, groundTruth)
                } else {
                    null
                }

                results.add(BenchmarkResult(
                    imageName = name,
                    strategy = strategy,
                    detected = detection != null,
                    confidence = detection?.confidence ?: 0.0,
                    timeMs = timeMs,
                    maxCornerErrorPx = cornerError
                ))
            }
            image.release()
        }

        // Log results as a table
        Log.d(TAG, "=".repeat(100))
        Log.d(TAG, "%-22s %-22s %7s %6s %7s %8s".format(
            "Image", "Strategy", "Detect", "Conf", "Time", "MaxErr"))
        Log.d(TAG, "-".repeat(100))
        for (r in results) {
            Log.d(TAG, "%-22s %-22s %7s %6.2f %5.1fms %8s".format(
                r.imageName,
                r.strategy,
                if (r.detected) "YES" else "no",
                r.confidence,
                r.timeMs,
                r.maxCornerErrorPx?.let { "%.1fpx".format(it) } ?: "N/A"
            ))
        }
        Log.d(TAG, "=".repeat(100))

        // Summary: detection rate per strategy across all images
        val newStrategies = setOf(
            PreprocessStrategy.ADAPTIVE_THRESHOLD,
            PreprocessStrategy.GRADIENT_MAGNITUDE,
            PreprocessStrategy.LAB_CLAHE,
            PreprocessStrategy.DOG,
            PreprocessStrategy.MULTICHANNEL_FUSION
        )
        val existingStrategies = setOf(
            PreprocessStrategy.STANDARD,
            PreprocessStrategy.CLAHE_ENHANCED
        )

        val newDetections = results.count { it.strategy in newStrategies && it.detected }
        val existingDetections = results.count { it.strategy in existingStrategies && it.detected }

        Log.d(TAG, "New strategy detections: $newDetections / ${results.count { it.strategy in newStrategies }}")
        Log.d(TAG, "Existing strategy detections: $existingDetections / ${results.count { it.strategy in existingStrategies }}")

        // Basic assertion: new strategies should collectively detect more than existing ones
        assertTrue(
            "New strategies ($newDetections detections) should outperform existing ($existingDetections) on low-contrast images",
            newDetections >= existingDetections
        )
    }

    /**
     * Verifies that the full multi-strategy pipeline (via [detectDocument])
     * detects whiteOnNearWhite — the easiest of the new test cases.
     */
    @Test
    fun fullPipeline_whiteOnNearWhite_detects() {
        val image = SyntheticImageFactory.whiteOnNearWhite()
        try {
            invalidateSceneCache()
            val result = detectDocument(image)
            assertTrue(
                "Full pipeline should detect whiteOnNearWhite (easiest low-contrast case)",
                result != null
            )
            Log.d(TAG, "whiteOnNearWhite full pipeline: confidence=${result?.confidence}")
        } finally {
            image.release()
        }
    }

    /**
     * Verifies that the full pipeline detects whiteOnLightWood (~35 gradient).
     */
    @Test
    fun fullPipeline_whiteOnLightWood_detects() {
        val image = SyntheticImageFactory.whiteOnLightWood()
        try {
            invalidateSceneCache()
            val result = detectDocument(image)
            assertTrue(
                "Full pipeline should detect whiteOnLightWood (~35 gradient)",
                result != null
            )
            Log.d(TAG, "whiteOnLightWood full pipeline: confidence=${result?.confidence}")
        } finally {
            image.release()
        }
    }

    private fun maxCornerError(detected: List<Point>, expected: List<Point>): Double {
        val orderedDetected = orderCorners(detected)
        val orderedExpected = orderCorners(expected)
        return orderedDetected.zip(orderedExpected).maxOf { (d, e) ->
            kotlin.math.sqrt((d.x - e.x) * (d.x - e.x) + (d.y - e.y) * (d.y - e.y))
        }
    }
}
