package com.docshot.cv

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertNull
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

private const val TAG = "DocShot:LsdBenchmark"

/**
 * Benchmark and false-positive test harness for the LSD+Radon detection cascade
 * ([detectDocumentLsd]). Exercises all three tiers (LSD fast path, corner-constrained
 * Radon, joint Radon rectangle fit) across the full set of synthetic test images.
 *
 * Most tests are benchmark-only (log results, no assertions) to track detection
 * capabilities as the cascade evolves. The false-positive tests DO assert — the
 * LSD path must not hallucinate documents where none exist.
 */
@RunWith(AndroidJUnit4::class)
class LsdRadonBenchmarkTest {

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
        val detected: Boolean,
        val confidence: Double,
        val timeMs: Double,
        val maxCornerErrorPx: Double?,
        val tier: String
    )

    private data class SegmentResult(
        val imageName: String,
        val segmentCount: Int,
        val longestSegmentPx: Float,
        val timeMs: Double
    )

    // -----------------------------------------------------------------
    // Test image loading helpers
    // -----------------------------------------------------------------

    /**
     * All 5 ultra-low-contrast images (v1.2.5). Each returns Pair<Mat, List<Point>>.
     */
    private fun ultraLowContrastImages(): Map<String, Pair<Mat, List<Point>>> = mapOf(
        "ultraLC_3unit" to SyntheticImageFactory.ultraLowContrast3Unit(),
        "ultraLC_5unit" to SyntheticImageFactory.ultraLowContrast5Unit(),
        "ultraLC_5unit_noisy" to SyntheticImageFactory.ultraLowContrast5UnitNoisy(),
        "ultraLC_tilted8" to SyntheticImageFactory.ultraLowContrastTilted8deg(),
        "ultraLC_3unit_warm" to SyntheticImageFactory.ultraLowContrast3UnitWarm()
    )

    /**
     * All 6 low-contrast images (v1.2.4). These return Mat only — ground truth
     * is [SyntheticImageFactory.defaultA4Corners].
     */
    private fun lowContrastImages(): Map<String, Pair<Mat, List<Point>>> {
        val gt = SyntheticImageFactory.defaultA4Corners()
        return mapOf(
            "whiteOnNearWhite" to Pair(SyntheticImageFactory.whiteOnNearWhite(), gt),
            "whiteOnWhite" to Pair(SyntheticImageFactory.whiteOnWhite(), gt),
            "whiteOnCream" to Pair(SyntheticImageFactory.whiteOnCream(), gt),
            "whiteOnLightWood" to Pair(SyntheticImageFactory.whiteOnLightWood(), gt),
            "whiteOnWhiteTextured" to Pair(SyntheticImageFactory.whiteOnWhiteTextured(), gt),
            "glossyPaper" to Pair(SyntheticImageFactory.glossyPaper(), gt)
        )
    }

    /** All 11 test images (5 ultra-low-contrast + 6 low-contrast). */
    private fun allTestImages(): Map<String, Pair<Mat, List<Point>>> {
        val images = LinkedHashMap<String, Pair<Mat, List<Point>>>()
        images.putAll(ultraLowContrastImages())
        images.putAll(lowContrastImages())
        return images
    }

    // -----------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------

    /** Convert a BGR Mat to grayscale for LSD input. Caller must release the result. */
    private fun toGray(bgr: Mat): Mat {
        val gray = Mat()
        Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY)
        return gray
    }

    /** Maximum corner distance between detected and expected corners, after ordering. */
    private fun maxCornerError(detected: List<Point>, expected: List<Point>): Double {
        val orderedDetected = orderCorners(detected)
        val orderedExpected = orderCorners(expected)
        return orderedDetected.zip(orderedExpected).maxOf { (d, e) ->
            kotlin.math.sqrt((d.x - e.x) * (d.x - e.x) + (d.y - e.y) * (d.y - e.y))
        }
    }

    /**
     * Infers the detection tier from confidence range.
     * Tier 1: 0.50-0.85, Tier 2: 0.45-0.75, Tier 3: 0.40-0.65.
     * When ranges overlap, the highest tier is assumed (Tier 1 has priority).
     */
    private fun inferTier(confidence: Double): String {
        return when {
            confidence >= 0.50 -> "Tier1"
            confidence >= 0.45 -> "Tier2"
            confidence >= 0.40 -> "Tier3"
            confidence > 0.0 -> "Below"
            else -> "None"
        }
    }

    // -----------------------------------------------------------------
    // 1. Per-tier benchmark
    // -----------------------------------------------------------------

    /**
     * Runs all 11 test images through the full [detectDocumentLsd] cascade.
     * Logs detection success, confidence, timing, inferred tier, and corner error.
     *
     * Benchmark only — does NOT assert detection success.
     */
    @Test
    fun benchmarkLsdTiers() {
        val images = allTestImages()
        val results = mutableListOf<BenchmarkResult>()

        for ((name, imageAndCorners) in images) {
            val (image, groundTruth) = imageAndCorners
            var gray: Mat? = null
            try {
                gray = toGray(image)
                val start = System.nanoTime()
                val detection = detectDocumentLsd(
                    gray = gray,
                    imageWidth = image.cols(),
                    imageHeight = image.rows()
                )
                val timeMs = (System.nanoTime() - start) / 1_000_000.0

                val cornerError = if (detection != null) {
                    maxCornerError(detection.corners, groundTruth)
                } else {
                    null
                }

                results.add(BenchmarkResult(
                    imageName = name,
                    detected = detection != null,
                    confidence = detection?.confidence ?: 0.0,
                    timeMs = timeMs,
                    maxCornerErrorPx = cornerError,
                    tier = if (detection != null) inferTier(detection.confidence) else "None"
                ))
            } finally {
                gray?.release()
                image.release()
            }
        }

        // Log results table
        Log.d(TAG, "=".repeat(105))
        Log.d(TAG, "LSD+Radon Per-Tier Benchmark (11 images)")
        Log.d(TAG, "%-22s %7s %6s %7s %8s %6s".format(
            "Image", "Detect", "Conf", "Time", "MaxErr", "Tier"))
        Log.d(TAG, "-".repeat(105))
        for (r in results) {
            Log.d(TAG, "%-22s %7s %6.2f %5.1fms %8s %6s".format(
                r.imageName,
                if (r.detected) "YES" else "no",
                r.confidence,
                r.timeMs,
                r.maxCornerErrorPx?.let { "%.1fpx".format(it) } ?: "N/A",
                r.tier
            ))
        }
        Log.d(TAG, "-".repeat(105))

        // Summary
        val detected = results.count { it.detected }
        val avgTime = results.map { it.timeMs }.average()
        val tierCounts = results.filter { it.detected }.groupingBy { it.tier }.eachCount()
        Log.d(TAG, "Total: %d/%d detected, avg %.1fms".format(detected, results.size, avgTime))
        Log.d(TAG, "Tier breakdown: %s".format(tierCounts))
        Log.d(TAG, "=".repeat(105))
    }

    // -----------------------------------------------------------------
    // 2. LSD-only segment detection benchmark
    // -----------------------------------------------------------------

    /**
     * Runs all 11 images through [detectSegments] only (LSD B1).
     * Logs segment count, longest segment length, and timing.
     * Isolates LSD performance from clustering/rectangle formation.
     *
     * Benchmark only — does NOT assert.
     */
    @Test
    fun benchmarkLsdSegmentDetection() {
        val images = allTestImages()
        val results = mutableListOf<SegmentResult>()

        for ((name, imageAndCorners) in images) {
            val (image, _) = imageAndCorners
            var gray: Mat? = null
            try {
                gray = toGray(image)
                val start = System.nanoTime()
                val segments = detectSegments(gray)
                val timeMs = (System.nanoTime() - start) / 1_000_000.0

                val longestPx = segments.maxOfOrNull { it.length } ?: 0f

                results.add(SegmentResult(
                    imageName = name,
                    segmentCount = segments.size,
                    longestSegmentPx = longestPx,
                    timeMs = timeMs
                ))
            } finally {
                gray?.release()
                image.release()
            }
        }

        // Log results table
        Log.d(TAG, "=".repeat(80))
        Log.d(TAG, "LSD Segment Detection Benchmark (11 images)")
        Log.d(TAG, "%-22s %8s %12s %7s".format("Image", "Segments", "Longest(px)", "Time"))
        Log.d(TAG, "-".repeat(80))
        for (r in results) {
            Log.d(TAG, "%-22s %8d %12.1f %5.1fms".format(
                r.imageName,
                r.segmentCount,
                r.longestSegmentPx,
                r.timeMs
            ))
        }
        Log.d(TAG, "-".repeat(80))

        val avgSegments = results.map { it.segmentCount }.average()
        val avgTime = results.map { it.timeMs }.average()
        Log.d(TAG, "Average: %.1f segments, %.1fms".format(avgSegments, avgTime))
        Log.d(TAG, "=".repeat(80))
    }

    // -----------------------------------------------------------------
    // 3. False positive rejection (ASSERTS)
    // -----------------------------------------------------------------

    /**
     * Verifies that the LSD+Radon cascade does NOT detect documents in
     * no-document images. These are assertion-based tests — false positive
     * rejection is critical for the LSD path.
     */
    @Test
    fun testFalsePositiveRejection() {
        val noDocImages = mapOf(
            "brightnessGradient" to SyntheticImageFactory.brightnessGradientNoDocs(),
            "noisyWhite" to SyntheticImageFactory.noisyWhiteNoDocs()
        )

        for ((name, image) in noDocImages) {
            var gray: Mat? = null
            try {
                gray = toGray(image)
                val start = System.nanoTime()
                val detection = detectDocumentLsd(
                    gray = gray,
                    imageWidth = image.cols(),
                    imageHeight = image.rows()
                )
                val timeMs = (System.nanoTime() - start) / 1_000_000.0

                Log.d(TAG, "FP test %-22s detect=%-3s conf=%.2f time=%.1fms".format(
                    name,
                    if (detection != null) "YES" else "no",
                    detection?.confidence ?: 0.0,
                    timeMs
                ))

                // Assert: no detection, or confidence below suppression threshold
                assertTrue(
                    "LSD cascade must not hallucinate documents in '$name' " +
                        "(got confidence=${detection?.confidence ?: 0.0})",
                    detection == null || detection.confidence < MIN_CONFIDENCE_THRESHOLD
                )
            } finally {
                gray?.release()
                image.release()
            }
        }
    }

    /**
     * Additional false positive test: uniform solid white image (no gradients at all).
     * LSD should find zero or near-zero segments, and the cascade must return null.
     */
    @Test
    fun testFalsePositive_uniformWhite() {
        val image = Mat(600, 800, CvType.CV_8UC3, Scalar(230.0, 230.0, 230.0))
        var gray: Mat? = null
        try {
            gray = toGray(image)
            val detection = detectDocumentLsd(
                gray = gray,
                imageWidth = image.cols(),
                imageHeight = image.rows()
            )

            Log.d(TAG, "FP test uniformWhite: detect=${detection != null}, " +
                "conf=${detection?.confidence ?: 0.0}")

            assertNull(
                "LSD cascade must return null for uniform white image",
                detection
            )
        } finally {
            gray?.release()
            image.release()
        }
    }

    // -----------------------------------------------------------------
    // 4. Tilted document benchmark
    // -----------------------------------------------------------------

    /**
     * Compares LSD cascade detection on tilted vs axis-aligned documents.
     * Logs detection rate and corner accuracy for each group.
     *
     * Benchmark only — does NOT assert.
     */
    @Test
    fun benchmarkTiltedDocuments() {
        // Tilted image (8 degrees)
        val tilted = mapOf(
            "ultraLC_tilted8" to SyntheticImageFactory.ultraLowContrastTilted8deg()
        )

        // Axis-aligned images for comparison (subset that represents typical scenarios)
        val axisAligned = mapOf(
            "ultraLC_5unit" to SyntheticImageFactory.ultraLowContrast5Unit(),
            "whiteOnNearWhite" to Pair(
                SyntheticImageFactory.whiteOnNearWhite(),
                SyntheticImageFactory.defaultA4Corners()
            ),
            "whiteOnWhite" to Pair(
                SyntheticImageFactory.whiteOnWhite(),
                SyntheticImageFactory.defaultA4Corners()
            ),
            "whiteOnCream" to Pair(
                SyntheticImageFactory.whiteOnCream(),
                SyntheticImageFactory.defaultA4Corners()
            ),
            "whiteOnLightWood" to Pair(
                SyntheticImageFactory.whiteOnLightWood(),
                SyntheticImageFactory.defaultA4Corners()
            )
        )

        val tiltedResults = mutableListOf<BenchmarkResult>()
        val alignedResults = mutableListOf<BenchmarkResult>()

        fun runBatch(
            images: Map<String, Pair<Mat, List<Point>>>,
            resultList: MutableList<BenchmarkResult>
        ) {
            for ((name, imageAndCorners) in images) {
                val (image, groundTruth) = imageAndCorners
                var gray: Mat? = null
                try {
                    gray = toGray(image)
                    val start = System.nanoTime()
                    val detection = detectDocumentLsd(
                        gray = gray,
                        imageWidth = image.cols(),
                        imageHeight = image.rows()
                    )
                    val timeMs = (System.nanoTime() - start) / 1_000_000.0

                    val cornerError = if (detection != null) {
                        maxCornerError(detection.corners, groundTruth)
                    } else {
                        null
                    }

                    resultList.add(BenchmarkResult(
                        imageName = name,
                        detected = detection != null,
                        confidence = detection?.confidence ?: 0.0,
                        timeMs = timeMs,
                        maxCornerErrorPx = cornerError,
                        tier = if (detection != null) inferTier(detection.confidence) else "None"
                    ))
                } finally {
                    gray?.release()
                    image.release()
                }
            }
        }

        runBatch(tilted, tiltedResults)
        runBatch(axisAligned, alignedResults)

        // Log tilted results
        Log.d(TAG, "=".repeat(105))
        Log.d(TAG, "Tilted vs Axis-Aligned Document Benchmark")
        Log.d(TAG, "-".repeat(105))

        Log.d(TAG, "TILTED (8 deg):")
        for (r in tiltedResults) {
            Log.d(TAG, "  %-22s detect=%-3s conf=%.2f time=%.1fms maxErr=%s tier=%s".format(
                r.imageName,
                if (r.detected) "YES" else "no",
                r.confidence,
                r.timeMs,
                r.maxCornerErrorPx?.let { "%.1fpx".format(it) } ?: "N/A",
                r.tier
            ))
        }

        Log.d(TAG, "AXIS-ALIGNED:")
        for (r in alignedResults) {
            Log.d(TAG, "  %-22s detect=%-3s conf=%.2f time=%.1fms maxErr=%s tier=%s".format(
                r.imageName,
                if (r.detected) "YES" else "no",
                r.confidence,
                r.timeMs,
                r.maxCornerErrorPx?.let { "%.1fpx".format(it) } ?: "N/A",
                r.tier
            ))
        }

        val tiltDetected = tiltedResults.count { it.detected }
        val alignDetected = alignedResults.count { it.detected }
        Log.d(TAG, "Tilted: %d/%d detected".format(tiltDetected, tiltedResults.size))
        Log.d(TAG, "Aligned: %d/%d detected".format(alignDetected, alignedResults.size))
        Log.d(TAG, "=".repeat(105))
    }

    // -----------------------------------------------------------------
    // 5. Contour pipeline vs LSD cascade comparison
    // -----------------------------------------------------------------

    /**
     * Runs all 11 images through BOTH the contour pipeline ([detectDocumentWithStatus])
     * and the LSD cascade ([detectDocumentLsd]), logging side-by-side results.
     *
     * This shows where the LSD cascade adds value — ultra-low-contrast images
     * that the contour pipeline cannot detect.
     *
     * Benchmark only — does NOT assert.
     */
    @Test
    fun benchmarkContourVsLsd() {
        val images = allTestImages()

        data class ComparisonResult(
            val imageName: String,
            val contourDetected: Boolean,
            val contourConfidence: Double,
            val contourTimeMs: Double,
            val contourMaxErr: Double?,
            val lsdDetected: Boolean,
            val lsdConfidence: Double,
            val lsdTimeMs: Double,
            val lsdMaxErr: Double?,
            val lsdTier: String
        )

        val results = mutableListOf<ComparisonResult>()

        for ((name, imageAndCorners) in images) {
            val (image, groundTruth) = imageAndCorners
            var gray: Mat? = null
            try {
                // --- Contour pipeline ---
                invalidateSceneCache()
                val contourStart = System.nanoTime()
                val contourStatus = detectDocumentWithStatus(image)
                val contourTimeMs = (System.nanoTime() - contourStart) / 1_000_000.0

                val contourDetection = contourStatus.result
                val contourMaxErr = if (contourDetection != null) {
                    maxCornerError(contourDetection.corners, groundTruth)
                } else {
                    null
                }

                // --- LSD cascade ---
                gray = toGray(image)
                val lsdStart = System.nanoTime()
                val lsdDetection = detectDocumentLsd(
                    gray = gray,
                    imageWidth = image.cols(),
                    imageHeight = image.rows()
                )
                val lsdTimeMs = (System.nanoTime() - lsdStart) / 1_000_000.0

                val lsdMaxErr = if (lsdDetection != null) {
                    maxCornerError(lsdDetection.corners, groundTruth)
                } else {
                    null
                }

                results.add(ComparisonResult(
                    imageName = name,
                    contourDetected = contourDetection != null,
                    contourConfidence = contourDetection?.confidence ?: 0.0,
                    contourTimeMs = contourTimeMs,
                    contourMaxErr = contourMaxErr,
                    lsdDetected = lsdDetection != null,
                    lsdConfidence = lsdDetection?.confidence ?: 0.0,
                    lsdTimeMs = lsdTimeMs,
                    lsdMaxErr = lsdMaxErr,
                    lsdTier = if (lsdDetection != null) inferTier(lsdDetection.confidence) else "None"
                ))
            } finally {
                gray?.release()
                image.release()
            }
        }

        // Log comparison table
        Log.d(TAG, "=".repeat(120))
        Log.d(TAG, "Contour Pipeline vs LSD Cascade Comparison (11 images)")
        Log.d(TAG, "%-22s | %-7s %6s %7s %8s | %-7s %6s %7s %8s %6s".format(
            "Image",
            "C:Det", "C:Conf", "C:Time", "C:Err",
            "L:Det", "L:Conf", "L:Time", "L:Err", "L:Tier"))
        Log.d(TAG, "-".repeat(120))
        for (r in results) {
            Log.d(TAG, "%-22s | %-7s %6.2f %5.1fms %8s | %-7s %6.2f %5.1fms %8s %6s".format(
                r.imageName,
                if (r.contourDetected) "YES" else "no",
                r.contourConfidence,
                r.contourTimeMs,
                r.contourMaxErr?.let { "%.1fpx".format(it) } ?: "N/A",
                if (r.lsdDetected) "YES" else "no",
                r.lsdConfidence,
                r.lsdTimeMs,
                r.lsdMaxErr?.let { "%.1fpx".format(it) } ?: "N/A",
                r.lsdTier
            ))
        }
        Log.d(TAG, "-".repeat(120))

        // Summary
        val contourDetected = results.count { it.contourDetected }
        val lsdDetected = results.count { it.lsdDetected }
        val lsdOnly = results.count { it.lsdDetected && !it.contourDetected }
        val contourOnly = results.count { it.contourDetected && !it.lsdDetected }
        val both = results.count { it.contourDetected && it.lsdDetected }
        val neither = results.count { !it.contourDetected && !it.lsdDetected }

        Log.d(TAG, "Contour: %d/%d detected".format(contourDetected, results.size))
        Log.d(TAG, "LSD:     %d/%d detected".format(lsdDetected, results.size))
        Log.d(TAG, "LSD-only: %d, Contour-only: %d, Both: %d, Neither: %d".format(
            lsdOnly, contourOnly, both, neither))

        val avgContourTime = results.map { it.contourTimeMs }.average()
        val avgLsdTime = results.map { it.lsdTimeMs }.average()
        Log.d(TAG, "Avg time — Contour: %.1fms, LSD: %.1fms".format(avgContourTime, avgLsdTime))
        Log.d(TAG, "=".repeat(120))
    }
}
