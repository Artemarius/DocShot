package com.docshot.camera

import android.util.Log
import org.opencv.core.Point
import kotlin.math.sqrt

private const val TAG = "DocShot:QuadSmoother"

/** If a new raw detection deviates from the smoothed average by more than this
 *  fraction of the quad diagonal, the buffer is cleared and stability resets.
 *  Prevents the smoother from gradually blending a completely different contour
 *  into the tracked quad and inheriting its stability progress. */
private const val JUMP_DETECTION_FRACTION = 0.10

/**
 * Temporal smoothing for detected document corners.
 * Maintains a rolling window of the last [windowSize] detections
 * and returns the averaged corner positions to reduce jitter.
 *
 * After [missThreshold] consecutive frames with no detection,
 * clears the buffer and returns null (overlay fades out).
 *
 * Stability tracking: after [stableThreshold] consecutive frames with valid
 * detection AND low corner variance (max corner movement < 2% of image diagonal),
 * [isStable] becomes true and [stabilityProgress] reaches 1.0.
 */
class QuadSmoother(
    private val windowSize: Int = 5,
    private val missThreshold: Int = 10,
    private val stableThreshold: Int = 20,
    private val maxCornerDriftFraction: Double = 0.025
) {
    private val buffer = ArrayDeque<List<Point>>()
    private val confidenceBuffer = ArrayDeque<Double>()
    private var consecutiveMisses = 0
    private var consecutiveStable = 0
    private var previousSmoothed: List<Point>? = null

    /** True when the document has been stably detected for [stableThreshold] consecutive frames. */
    var isStable: Boolean = false
        private set

    /** Progress toward stability: 0.0 to 1.0 (consecutiveStable / stableThreshold). */
    var stabilityProgress: Float = 0f
        private set

    /** Rolling average of recent detection confidence values. */
    val averageConfidence: Double
        get() = if (confidenceBuffer.isEmpty()) 0.0
                else confidenceBuffer.sum() / confidenceBuffer.size

    /**
     * Feed a new detection result. Pass null when no document was detected.
     * Returns smoothed corners or null if detection has been lost.
     *
     * @param corners detected corner positions, or null if no detection this frame.
     * @param confidence detection confidence for this frame (ignored on miss).
     */
    fun update(corners: List<Point>?, confidence: Double = 0.0): List<Point>? {
        if (corners == null) {
            consecutiveMisses++
            resetStability()
            if (consecutiveMisses >= missThreshold) {
                buffer.clear()
                confidenceBuffer.clear()
                previousSmoothed = null
                return null
            }
            // On miss, don't add to confidence buffer — keep existing values
            return if (buffer.isNotEmpty()) average() else null
        }

        consecutiveMisses = 0

        // Detect sudden quad jumps BEFORE smoothing absorbs them.
        // Compare new raw corners against the current smoothed average.
        // If the jump exceeds the threshold, clear the buffer so the
        // smoother snaps to the new position and stability resets.
        if (buffer.isNotEmpty()) {
            val currentAvg = average()
            val diagonal = estimateDiagonal(currentAvg)
            if (diagonal > 1.0) {
                var sumDist = 0.0
                for (i in 0 until 4) {
                    val dx = corners[i].x - currentAvg[i].x
                    val dy = corners[i].y - currentAvg[i].y
                    sumDist += sqrt(dx * dx + dy * dy)
                }
                val jumpFraction = (sumDist / 4.0) / diagonal
                if (jumpFraction > JUMP_DETECTION_FRACTION) {
                    Log.d(TAG, "Quad jump detected (%.1f%% of diagonal) — clearing buffer".format(
                        jumpFraction * 100))
                    buffer.clear()
                    confidenceBuffer.clear()
                    resetStability()
                }
            }
        }

        if (buffer.size >= windowSize) buffer.removeFirst()
        buffer.addLast(corners)

        // Track confidence in a parallel rolling buffer (same window as corners)
        if (confidenceBuffer.size >= windowSize) confidenceBuffer.removeFirst()
        confidenceBuffer.addLast(confidence)

        val smoothed = average()
        updateStability(smoothed)
        previousSmoothed = smoothed
        return smoothed
    }

    fun clear() {
        buffer.clear()
        confidenceBuffer.clear()
        consecutiveMisses = 0
        previousSmoothed = null
        resetStability()
    }

    /**
     * Computes the max corner movement between the current smoothed quad and the
     * previous one as a fraction of the image diagonal estimated from the quad.
     * If movement is below [maxCornerDriftFraction], increments the stable counter.
     */
    private fun updateStability(current: List<Point>) {
        val prev = previousSmoothed
        if (prev == null) {
            // First valid smoothed frame — start counting but can't measure drift yet
            consecutiveStable = 1
            stabilityProgress = consecutiveStable.toFloat() / stableThreshold
            isStable = false
            return
        }

        // Estimate image diagonal from the quad's bounding box for normalization.
        // Use the larger of current/prev bounding boxes to avoid division by tiny values.
        val diagonal = estimateDiagonal(current)
        if (diagonal < 1.0) {
            resetStability()
            return
        }

        // Average corner movement (Euclidean distance) across all 4 corners.
        // Using average instead of max prevents one wobbly corner from
        // resetting the entire stability streak.
        var sumDrift = 0.0
        for (i in 0 until 4) {
            val dx = current[i].x - prev[i].x
            val dy = current[i].y - prev[i].y
            sumDrift += sqrt(dx * dx + dy * dy)
        }

        val driftFraction = (sumDrift / 4.0) / diagonal

        if (driftFraction < maxCornerDriftFraction) {
            consecutiveStable++
        } else if (driftFraction < maxCornerDriftFraction * 4) {
            // Minor wobble (2.5–10% of diagonal): likely rounded-corner jitter
            // or slight hand movement. Halve progress instead of wiping it.
            consecutiveStable = maxOf(1, consecutiveStable / 2)
        } else {
            // Major jump (>10% of diagonal): a different contour was detected.
            // Hard reset — don't let the previous quad's stability carry over.
            consecutiveStable = 1
        }

        stabilityProgress = (consecutiveStable.toFloat() / stableThreshold).coerceAtMost(1f)
        isStable = consecutiveStable >= stableThreshold

        if (isStable) {
            Log.d(TAG, "Stable! consecutive=$consecutiveStable, drift=%.4f".format(driftFraction))
        }
    }

    private fun resetStability() {
        consecutiveStable = 0
        stabilityProgress = 0f
        isStable = false
    }

    /**
     * Estimates the diagonal of the bounding box encompassing the 4 corners.
     * Used to normalize drift distances against image scale.
     */
    private fun estimateDiagonal(points: List<Point>): Double {
        var minX = Double.MAX_VALUE
        var minY = Double.MAX_VALUE
        var maxX = Double.MIN_VALUE
        var maxY = Double.MIN_VALUE
        for (pt in points) {
            if (pt.x < minX) minX = pt.x
            if (pt.y < minY) minY = pt.y
            if (pt.x > maxX) maxX = pt.x
            if (pt.y > maxY) maxY = pt.y
        }
        val dx = maxX - minX
        val dy = maxY - minY
        return sqrt(dx * dx + dy * dy)
    }

    private fun average(): List<Point> {
        val n = buffer.size
        return List(4) { i ->
            var sumX = 0.0
            var sumY = 0.0
            for (quad in buffer) {
                sumX += quad[i].x
                sumY += quad[i].y
            }
            Point(sumX / n, sumY / n)
        }
    }
}
