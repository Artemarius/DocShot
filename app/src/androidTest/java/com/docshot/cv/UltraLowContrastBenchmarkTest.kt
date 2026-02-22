package com.docshot.cv

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.Point

private const val TAG = "DocShot:ULCBenchmark"

/**
 * Benchmark harness for ultra-low-contrast detection (3-unit and 5-unit gradients).
 * Runs each synthetic test image through strategies individually via [detectWithStrategy],
 * recording detection success, confidence, corner accuracy, and processing time.
 *
 * These are benchmark-only tests — they log results for analysis but do not assert
 * detection success, since the v1.2.5 DIRECTIONAL_GRADIENT and LSD+Radon strategies
 * that target these gradients may not yet be fully implemented.
 */
@RunWith(AndroidJUnit4::class)
class UltraLowContrastBenchmarkTest {

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
     * Runs all ultra-low-contrast test images through all strategies individually.
     * Logs a comparison table for analysis. Does NOT assert detection success —
     * these gradients are below what most existing strategies can handle.
     */
    @Test
    fun benchmarkAllStrategies() {
        // Each generator returns Pair<Mat, List<Point>> (image + ground truth corners)
        val images = mapOf(
            "ultraLC_3unit" to SyntheticImageFactory.ultraLowContrast3Unit(),
            "ultraLC_5unit" to SyntheticImageFactory.ultraLowContrast5Unit(),
            "ultraLC_5unit_noisy" to SyntheticImageFactory.ultraLowContrast5UnitNoisy(),
            "ultraLC_tilted8" to SyntheticImageFactory.ultraLowContrastTilted8deg(),
            "ultraLC_3unit_warm" to SyntheticImageFactory.ultraLowContrast3UnitWarm()
        )

        val strategies = listOf(
            // v1.2.5 target strategies
            PreprocessStrategy.DIRECTIONAL_GRADIENT,
            // v1.2.4 low-contrast strategies for comparison
            PreprocessStrategy.DOG,
            PreprocessStrategy.GRADIENT_MAGNITUDE,
            PreprocessStrategy.LAB_CLAHE,
            PreprocessStrategy.MULTICHANNEL_FUSION,
            PreprocessStrategy.ADAPTIVE_THRESHOLD,
            // Original strategies for baseline
            PreprocessStrategy.STANDARD,
            PreprocessStrategy.CLAHE_ENHANCED,
        )

        val results = mutableListOf<BenchmarkResult>()

        for ((name, imageAndCorners) in images) {
            val (image, groundTruth) = imageAndCorners
            for (strategy in strategies) {
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
        Log.d(TAG, "=".repeat(105))
        Log.d(TAG, "%-22s %-22s %7s %6s %7s %8s".format(
            "Image", "Strategy", "Detect", "Conf", "Time", "MaxErr"))
        Log.d(TAG, "-".repeat(105))
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
        Log.d(TAG, "=".repeat(105))

        // Summary: detection rate per strategy
        Log.d(TAG, "\nDetection summary per strategy:")
        for (strategy in strategies) {
            val stratResults = results.filter { it.strategy == strategy }
            val detected = stratResults.count { it.detected }
            val avgTime = stratResults.map { it.timeMs }.average()
            Log.d(TAG, "  %-22s %d/%d detected, avg %.1fms".format(
                strategy, detected, stratResults.size, avgTime))
        }

        // Summary: detection rate per image
        Log.d(TAG, "\nDetection summary per image:")
        for ((name, _) in images) {
            val imgResults = results.filter { it.imageName == name }
            val detected = imgResults.count { it.detected }
            val bestConf = imgResults.maxOfOrNull { it.confidence } ?: 0.0
            Log.d(TAG, "  %-22s %d/%d strategies detected, best confidence=%.2f".format(
                name, detected, imgResults.size, bestConf))
        }
    }

    /**
     * Runs each ultra-low-contrast image through the full multi-strategy pipeline
     * via [detectDocumentWithStatus]. Logs whether the pipeline finds the document
     * when all strategies are tried with time budget and short-circuit logic.
     */
    @Test
    fun benchmarkFullPipeline() {
        val images = mapOf(
            "ultraLC_3unit" to SyntheticImageFactory.ultraLowContrast3Unit(),
            "ultraLC_5unit" to SyntheticImageFactory.ultraLowContrast5Unit(),
            "ultraLC_5unit_noisy" to SyntheticImageFactory.ultraLowContrast5UnitNoisy(),
            "ultraLC_tilted8" to SyntheticImageFactory.ultraLowContrastTilted8deg(),
            "ultraLC_3unit_warm" to SyntheticImageFactory.ultraLowContrast3UnitWarm()
        )

        Log.d(TAG, "=".repeat(80))
        Log.d(TAG, "Full pipeline results (multi-strategy with time budget):")
        Log.d(TAG, "-".repeat(80))

        for ((name, imageAndCorners) in images) {
            val (image, groundTruth) = imageAndCorners
            invalidateSceneCache()

            val start = System.nanoTime()
            val status = detectDocumentWithStatus(image)
            val timeMs = (System.nanoTime() - start) / 1_000_000.0

            val detection = status.result
            val cornerError = if (detection != null) {
                maxCornerError(detection.corners, groundTruth)
            } else {
                null
            }

            Log.d(TAG, "%-22s detect=%-3s conf=%.2f time=%.1fms maxErr=%s partial=%s".format(
                name,
                if (detection != null) "YES" else "no",
                detection?.confidence ?: 0.0,
                timeMs,
                cornerError?.let { "%.1fpx".format(it) } ?: "N/A",
                status.isPartialDocument
            ))

            image.release()
        }
        Log.d(TAG, "=".repeat(80))
    }

    private fun maxCornerError(detected: List<Point>, expected: List<Point>): Double {
        val orderedDetected = orderCorners(detected)
        val orderedExpected = orderCorners(expected)
        return orderedDetected.zip(orderedExpected).maxOf { (d, e) ->
            kotlin.math.sqrt((d.x - e.x) * (d.x - e.x) + (d.y - e.y) * (d.y - e.y))
        }
    }
}
