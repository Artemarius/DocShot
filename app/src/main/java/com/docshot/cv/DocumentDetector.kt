package com.docshot.cv

import android.util.Log
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

private const val TAG = "DocShot:Detector"

/** Minimum confidence to accept a detection. Below this, detectDocument() returns null. */
const val MIN_CONFIDENCE_THRESHOLD = 0.35

/** Multi-strategy time budget in milliseconds. Stops trying additional strategies after this. */
private const val STRATEGY_TIME_BUDGET_MS = 25.0

/** Confidence threshold above which we short-circuit (no need to try more strategies). */
private const val STRATEGY_SHORT_CIRCUIT_CONFIDENCE = 0.65

/**
 * Result of document detection (corners only, no rectification).
 * @param corners Detected corner positions [TL, TR, BR, BL] in source image coordinates.
 * @param detectionMs Detection processing time in milliseconds.
 */
data class DocumentCorners(
    val corners: List<Point>,
    val detectionMs: Double,
    val confidence: Double = 0.0
)

/**
 * Result of document detection and rectification.
 * @param rectified The perspective-corrected document image (caller must release).
 * @param corners Detected corner positions [TL, TR, BR, BL] in source image coordinates.
 * @param pipelineMs Total processing time in milliseconds.
 */
data class DetectionResult(
    val rectified: Mat,
    val corners: List<Point>,
    val pipelineMs: Double,
    val confidence: Double = 0.0
)

/**
 * Status from detection including partial-document flag.
 * @param result Detection result, or null if no valid document found.
 * @param isPartialDocument True if a large contour touching 2+ frame edges was
 *   found but no valid quad could be extracted — suggests the document extends
 *   beyond the camera frame.
 */
data class DetectionStatus(
    val result: DocumentCorners?,
    val isPartialDocument: Boolean,
    /** Total detection time in ms, always populated regardless of whether a document was found. */
    val detectionMs: Double = 0.0
)

// Module-level MatPool shared across strategy attempts within a single detection call.
// Not thread-safe — detection runs single-threaded on Dispatchers.Default.
private val matPool = MatPool(maxSize = 8)

/**
 * Runs detection only: finds the best document quadrilateral without rectifying.
 * Uses multi-strategy preprocessing — tries alternative strategies if the primary
 * one yields low confidence, within a 25ms time budget.
 *
 * Pipeline stages (per strategy):
 * 1. Preprocess (strategy-dependent)
 * 2. Canny edge detection (auto-threshold)
 * 3. Contour finding + polygon approximation
 * 4. Quadrilateral scoring and ranking
 * 5. Edge density validation + confidence scoring
 *
 * All intermediate Mats are released. Detections with confidence below
 * [MIN_CONFIDENCE_THRESHOLD] are suppressed (returns null).
 *
 * @param input BGR, RGBA, or grayscale image (not modified).
 */
fun detectDocument(input: Mat): DocumentCorners? {
    return detectDocumentWithStatus(input).result
}

/**
 * Like [detectDocument] but also reports whether a partial document was detected.
 * The `isPartialDocument` flag is OR'd across all strategies tried.
 *
 * @param input BGR, RGBA, or grayscale image (not modified).
 */
fun detectDocumentWithStatus(input: Mat): DetectionStatus {
    val start = System.nanoTime()

    val sceneAnalysis = analyzeScene(input)
    val strategies = sceneAnalysis.strategies
    val sharedGray = sceneAnalysis.grayMat

    var bestResult: DocumentCorners? = null
    var anyPartialDocument = false

    try {
        for (strategy in strategies) {
            val elapsed = (System.nanoTime() - start) / 1_000_000.0
            if (elapsed > STRATEGY_TIME_BUDGET_MS && bestResult != null) {
                Log.d(TAG, "Strategy time budget exceeded (%.1f ms), stopping".format(elapsed))
                break
            }

            // Share the gray Mat from analyzeScene with the first strategy that uses it
            val strategyResult = detectWithStrategy(input, strategy, sharedGray)
            anyPartialDocument = anyPartialDocument || strategyResult.isPartialDocument

            val candidate = strategyResult.result
            if (candidate != null) {
                if (bestResult == null || candidate.confidence > bestResult.confidence) {
                    bestResult = candidate
                }
                if (bestResult.confidence >= STRATEGY_SHORT_CIRCUIT_CONFIDENCE) {
                    Log.d(TAG, "Short-circuit: confidence %.2f >= %.2f with $strategy".format(
                        bestResult.confidence, STRATEGY_SHORT_CIRCUIT_CONFIDENCE))
                    break
                }
            }
        }
    } finally {
        // Release the shared gray Mat from analyzeScene
        sharedGray?.release()
    }

    // Apply minimum confidence threshold
    if (bestResult != null && bestResult.confidence < MIN_CONFIDENCE_THRESHOLD) {
        val ms = (System.nanoTime() - start) / 1_000_000.0
        Log.d(TAG, "Detection suppressed: confidence %.2f < %.2f threshold (%.1f ms)".format(
            bestResult.confidence, MIN_CONFIDENCE_THRESHOLD, ms))
        return DetectionStatus(result = null, isPartialDocument = anyPartialDocument, detectionMs = ms)
    }

    val ms = (System.nanoTime() - start) / 1_000_000.0
    if (bestResult != null) {
        // Update detection time to include the full multi-strategy cost
        val finalResult = bestResult.copy(detectionMs = ms)
        Log.d(TAG, "detectDocument: %.1f ms, confidence: %.2f".format(ms, finalResult.confidence))
        return DetectionStatus(result = finalResult, isPartialDocument = anyPartialDocument, detectionMs = ms)
    }

    Log.d(TAG, "detectDocument: %.1f ms — no document found across %d strategies".format(
        ms, strategies.size))
    return DetectionStatus(result = null, isPartialDocument = anyPartialDocument, detectionMs = ms)
}

/**
 * Runs detection with a single preprocessing strategy.
 * Extracted from the original `detectDocument()` body — identical logic,
 * parameterized by [strategy].
 *
 * Wrapped in try/finally to guarantee intermediate Mat cleanup even if
 * scoring or contour analysis throws.
 */
private fun detectWithStrategy(
    input: Mat,
    strategy: PreprocessStrategy,
    sharedGray: Mat? = null
): DetectionStatus {
    val start = System.nanoTime()
    val imageSize = Size(input.cols().toDouble(), input.rows().toDouble())
    val imageArea = imageSize.width * imageSize.height

    var preprocessed: Mat? = null
    var edges: Mat? = null

    try {
        preprocessed = preprocessWithStrategy(input, strategy, sharedGray = sharedGray)
        edges = when (strategy) {
            PreprocessStrategy.HEAVY_MORPH -> detectEdgesHeavyMorph(preprocessed)
            PreprocessStrategy.CLAHE_ENHANCED -> {
                // CLAHE-enhanced images need lower Canny thresholds: the auto-threshold
                // formula (0.67*median) overestimates for low-contrast scenes where edge
                // gradients are subtle even after histogram equalization.
                detectEdges(preprocessed, thresholdLow = 30.0, thresholdHigh = 60.0)
            }
            else -> detectEdges(preprocessed)
        }
        preprocessed.release()
        preprocessed = null

        val contourAnalysis = analyzeContours(edges, imageSize)
        val quads = contourAnalysis.quads

        if (quads.isEmpty()) {
            edges.release()
            edges = null
            val ms = (System.nanoTime() - start) / 1_000_000.0
            Log.d(TAG, "detectWithStrategy($strategy): %.1f ms — no quads".format(ms))
            return DetectionStatus(
                result = null,
                isPartialDocument = contourAnalysis.hasPartialDocument
            )
        }

        val rankResult = rankQuads(quads, imageArea)
        if (rankResult.quad == null) {
            edges.release()
            edges = null
            val ms = (System.nanoTime() - start) / 1_000_000.0
            Log.d(TAG, "detectWithStrategy($strategy): %.1f ms — no valid quad after scoring".format(ms))
            return DetectionStatus(
                result = null,
                isPartialDocument = contourAnalysis.hasPartialDocument
            )
        }
        val corners = rankResult.quad

        // Measure how well the detected quad aligns with actual Canny edge pixels.
        val edgeDensity = QuadValidator.edgeDensityScore(edges, corners)
        edges.release()
        edges = null

        // Confidence = weighted combination of two complementary signals:
        //   1. Quad score (60%) — geometric quality: area + angle regularity + aspect ratio.
        //      Area is accounted for here (40% of quadScore = 24% effective weight).
        //   2. Edge density (40%) — fraction of the quad perimeter supported by edge pixels.
        //      Strongest indicator that the quad sits on real document edges.
        // Previous formula double-counted area (44% effective weight), which made
        // small documents (ID cards, business cards) unable to reach auto-capture threshold.
        val quadScore = scoreQuad(corners, imageArea)

        // Score margin penalty for ambiguous multi-candidate scenes
        val marginFactor = if (rankResult.candidateCount >= 2) {
            0.5 + 0.5 * rankResult.scoreMargin
        } else {
            1.0
        }
        val confidence = (0.6 * quadScore + 0.4 * edgeDensity) * marginFactor

        val ms = (System.nanoTime() - start) / 1_000_000.0
        Log.d(TAG, "detectWithStrategy($strategy): %.1f ms, confidence=%.2f (quad=%.2f, edge=%.2f, margin=%.2f)".format(
            ms, confidence, quadScore, edgeDensity, marginFactor))

        return DetectionStatus(
            result = DocumentCorners(
                corners = corners,
                detectionMs = ms,
                confidence = confidence
            ),
            isPartialDocument = contourAnalysis.hasPartialDocument
        )
    } finally {
        preprocessed?.release()
        edges?.release()
    }
}

/**
 * Runs the full detection → rectification pipeline on a single image.
 * Returns null if no document-like quadrilateral was found.
 *
 * All intermediate Mats are released. Caller must release the Mat inside [DetectionResult].
 *
 * @param input BGR or RGBA image (not modified).
 * @param interpolation Warp interpolation mode. Use INTER_CUBIC for captures,
 *   INTER_LINEAR for preview.
 */
fun detectAndRectify(
    input: Mat,
    interpolation: Int = Imgproc.INTER_CUBIC
): DetectionResult? {
    val start = System.nanoTime()

    val detection = detectDocument(input) ?: return null
    val rectified = rectify(input, detection.corners, interpolation)

    val ms = (System.nanoTime() - start) / 1_000_000.0
    Log.d(TAG, "detectAndRectify: %.1f ms total".format(ms))

    return DetectionResult(
        rectified = rectified,
        corners = detection.corners,
        pipelineMs = ms,
        confidence = detection.confidence
    )
}
