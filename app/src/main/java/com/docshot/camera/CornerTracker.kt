package com.docshot.camera

import android.util.Log
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.core.TermCriteria
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video
import kotlin.math.sqrt

private const val TAG = "DocShot:CornerTracker"

/** KLT search window size (pixels). 15x15 balances precision with robustness. */
private const val KLT_WINDOW_SIZE = 15

/** Pyramid levels for Lucas-Kanade. 2 levels handles moderate inter-frame motion. */
private const val KLT_PYRAMID_LEVELS = 2

/** Max KLT iterations per point. */
private const val KLT_MAX_ITERATIONS = 20

/** Sub-pixel convergence epsilon for KLT. */
private const val KLT_EPSILON = 0.03

/** Max per-point error from KLT before declaring that point lost. */
private const val KLT_MAX_ERROR = 12.0f

/** If correction drift (avg corner distance) exceeds this many pixels,
 *  tracking is reset and we fall back to detection. */
private const val CORRECTION_DRIFT_PX = 8.0

/** Minimum tracked quad area in pixels to accept (rejects degenerate quads). */
private const val MIN_QUAD_AREA_PX = 100.0

/** How often to run full detection as a correction step during tracking.
 *  Every Nth frame gets a full detect; others are KLT-only. */
private const val CORRECTION_INTERVAL = 3

/**
 * Tracking state machine.
 * - [DETECT_ONLY]: no tracking active; every frame runs full detection.
 * - [TRACKING]: KLT tracks 4 corners; full detection runs every [CORRECTION_INTERVAL] frames.
 */
enum class TrackingState {
    DETECT_ONLY,
    TRACKING
}

/**
 * Result from [CornerTracker.processFrame].
 *
 * @property corners tracked or detected corners [TL, TR, BR, BL], null if no quad available.
 * @property isTracked true if corners came from KLT tracking (sub-pixel consistent),
 *   false if from full detection.
 * @property state current tracking state after this frame.
 */
data class TrackingResult(
    val corners: List<Point>?,
    val isTracked: Boolean,
    val state: TrackingState
)

/**
 * KLT optical flow corner tracker for document quadrilaterals.
 *
 * Tracks 4 document corners between frames using pyramidal Lucas-Kanade,
 * providing sub-pixel consistent corners (~0.03px) at ~0.5ms per frame.
 * Full detection runs as a correction step every [CORRECTION_INTERVAL] frames
 * during tracking.
 *
 * State machine:
 * ```
 * DETECT_ONLY  ──(high-confidence detection)──>  TRACKING
 * TRACKING     ──(KLT fails / drift too high)──>  DETECT_ONLY
 * ```
 *
 * Thread safety: not thread-safe. Call from a single thread (ImageAnalysis executor).
 */
class CornerTracker(
    /** Minimum detection confidence to enter TRACKING state. */
    private val minTrackingConfidence: Double = 0.65
) {
    /** Current state of the tracker. */
    var state: TrackingState = TrackingState.DETECT_ONLY
        private set

    /** Previous grayscale frame, retained for KLT. */
    private var prevGray: Mat? = null

    /** Last known corner positions (analysis-resolution pixels). */
    private var trackedCorners: List<Point>? = null

    /** Frame counter within current tracking session (for correction interval). */
    private var trackingFrameCount = 0

    /**
     * Process a frame with optional detection result.
     *
     * In DETECT_ONLY: uses [detectedCorners] to initialize tracking.
     * In TRACKING: runs KLT on [currentGray], uses [detectedCorners] for correction
     * on every [CORRECTION_INTERVAL]-th frame.
     *
     * @param currentGray grayscale frame (8-bit, analysis resolution). Caller retains
     *   ownership; this class clones what it needs.
     * @param detectedCorners corners from full detection this frame, or null if detection
     *   was skipped / failed. [TL, TR, BR, BL] in analysis-resolution pixels.
     * @param detectionConfidence confidence of the detection (0.0 if no detection).
     * @return [TrackingResult] with current corners and tracking metadata.
     */
    fun processFrame(
        currentGray: Mat,
        detectedCorners: List<Point>?,
        detectionConfidence: Double
    ): TrackingResult {
        return when (state) {
            TrackingState.DETECT_ONLY -> handleDetectOnly(currentGray, detectedCorners, detectionConfidence)
            TrackingState.TRACKING -> handleTracking(currentGray, detectedCorners)
        }
    }

    /** Resets tracker to DETECT_ONLY, releasing all state. Call on enterIdle / quad loss. */
    fun reset() {
        state = TrackingState.DETECT_ONLY
        prevGray?.release()
        prevGray = null
        trackedCorners = null
        trackingFrameCount = 0
        Log.d(TAG, "Tracker reset to DETECT_ONLY")
    }

    /** Releases all native resources. Call from ViewModel.onCleared(). */
    fun release() {
        prevGray?.release()
        prevGray = null
        trackedCorners = null
        trackingFrameCount = 0
        state = TrackingState.DETECT_ONLY
        Log.d(TAG, "Tracker released")
    }

    /** True if tracking should run a correction detection this frame. */
    fun needsCorrectionDetection(): Boolean =
        state == TrackingState.TRACKING && (trackingFrameCount % CORRECTION_INTERVAL == 0)

    // ── State handlers ──────────────────────────────────────────────────

    private fun handleDetectOnly(
        currentGray: Mat,
        detectedCorners: List<Point>?,
        detectionConfidence: Double
    ): TrackingResult {
        if (detectedCorners != null && detectionConfidence >= minTrackingConfidence) {
            // Transition to TRACKING
            storePreviousFrame(currentGray)
            trackedCorners = detectedCorners.map { Point(it.x, it.y) }
            trackingFrameCount = 1
            state = TrackingState.TRACKING
            Log.d(TAG, "DETECT_ONLY -> TRACKING (conf=%.2f)".format(detectionConfidence))
            return TrackingResult(
                corners = trackedCorners,
                isTracked = false, // first frame is from detection, not KLT
                state = state
            )
        }

        // Stay in DETECT_ONLY, pass through detection result
        storePreviousFrame(currentGray)
        return TrackingResult(
            corners = detectedCorners,
            isTracked = false,
            state = state
        )
    }

    private fun handleTracking(
        currentGray: Mat,
        detectedCorners: List<Point>?
    ): TrackingResult {
        val prev = prevGray
        val prevCorners = trackedCorners

        if (prev == null || prevCorners == null) {
            // Shouldn't happen, but recover gracefully
            resetToDetectOnly("missing previous state")
            storePreviousFrame(currentGray)
            return TrackingResult(corners = detectedCorners, isTracked = false, state = state)
        }

        // Validate frame dimensions match (camera config change guard)
        if (prev.cols() != currentGray.cols() || prev.rows() != currentGray.rows()) {
            resetToDetectOnly("frame size mismatch (${prev.cols()}x${prev.rows()} -> ${currentGray.cols()}x${currentGray.rows()})")
            storePreviousFrame(currentGray)
            return TrackingResult(corners = detectedCorners, isTracked = false, state = state)
        }

        trackingFrameCount++

        // Run KLT
        val kltResult = runKlt(prev, currentGray, prevCorners)

        if (kltResult == null) {
            // KLT failed — all points lost or non-convex result
            resetToDetectOnly("KLT failed")
            storePreviousFrame(currentGray)
            return TrackingResult(corners = detectedCorners, isTracked = false, state = state)
        }

        // Correction step: compare KLT result with detection
        if (detectedCorners != null && trackingFrameCount % CORRECTION_INTERVAL == 0) {
            val drift = averageCornerDistance(kltResult, detectedCorners)
            if (drift > CORRECTION_DRIFT_PX) {
                // KLT has diverged too far from detection — reset
                resetToDetectOnly("correction drift %.1fpx > ${CORRECTION_DRIFT_PX}px".format(drift))
                storePreviousFrame(currentGray)
                return TrackingResult(corners = detectedCorners, isTracked = false, state = state)
            }
            // Drift is acceptable — trust KLT (sub-pixel consistent)
            Log.d(TAG, "Correction OK: drift=%.1fpx".format(drift))
        }

        // Accept KLT result
        trackedCorners = kltResult
        storePreviousFrame(currentGray)
        return TrackingResult(corners = kltResult, isTracked = true, state = state)
    }

    // ── KLT core ────────────────────────────────────────────────────────

    /**
     * Runs pyramidal Lucas-Kanade optical flow on 4 corner points.
     * Returns the tracked corners, or null if tracking failed (any point lost,
     * high error, or result is non-convex / too small).
     */
    private fun runKlt(prevFrame: Mat, currFrame: Mat, corners: List<Point>): List<Point>? {
        val prevPts = MatOfPoint2f(*corners.toTypedArray())
        val nextPts = MatOfPoint2f()
        val status = MatOfByte()
        val err = MatOfFloat()

        try {
            Video.calcOpticalFlowPyrLK(
                prevFrame,
                currFrame,
                prevPts,
                nextPts,
                status,
                err,
                Size(KLT_WINDOW_SIZE.toDouble(), KLT_WINDOW_SIZE.toDouble()),
                KLT_PYRAMID_LEVELS,
                TermCriteria(
                    TermCriteria.COUNT + TermCriteria.EPS,
                    KLT_MAX_ITERATIONS,
                    KLT_EPSILON
                )
            )

            val statusArr = status.toArray()
            val errArr = err.toArray()
            val nextArr = nextPts.toArray()

            // Validate: all 4 points must be found with acceptable error
            for (i in 0 until 4) {
                if (i >= statusArr.size || statusArr[i].toInt() != 1) {
                    Log.d(TAG, "KLT: point $i lost (status=${statusArr.getOrNull(i)})")
                    return null
                }
                if (i < errArr.size && errArr[i] > KLT_MAX_ERROR) {
                    Log.d(TAG, "KLT: point $i error too high (${errArr[i]})")
                    return null
                }
            }

            if (nextArr.size < 4) return null

            val tracked = nextArr.take(4)

            // Bounds check: all points must be within frame
            val cols = currFrame.cols()
            val rows = currFrame.rows()
            for (pt in tracked) {
                if (pt.x < 0 || pt.x >= cols || pt.y < 0 || pt.y >= rows) {
                    Log.d(TAG, "KLT: point out of bounds (%.1f, %.1f)".format(pt.x, pt.y))
                    return null
                }
            }

            // Convexity guard: tracked quad must remain convex
            if (!isConvex(tracked)) {
                Log.d(TAG, "KLT: tracked quad is non-convex")
                return null
            }

            // Area guard: reject degenerate quads
            if (quadArea(tracked) < MIN_QUAD_AREA_PX) {
                Log.d(TAG, "KLT: tracked quad area too small")
                return null
            }

            return tracked
        } finally {
            prevPts.release()
            nextPts.release()
            status.release()
            err.release()
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    private fun storePreviousFrame(gray: Mat) {
        prevGray?.release()
        prevGray = gray.clone()
    }

    private fun resetToDetectOnly(reason: String) {
        Log.d(TAG, "TRACKING -> DETECT_ONLY ($reason)")
        state = TrackingState.DETECT_ONLY
        trackedCorners = null
        trackingFrameCount = 0
        // Keep prevGray — caller will update it
    }

    companion object {
        /**
         * Average Euclidean distance between two sets of 4 corners.
         * Used for correction drift measurement.
         */
        fun averageCornerDistance(a: List<Point>, b: List<Point>): Double {
            require(a.size == 4 && b.size == 4) { "Expected 4 corners each" }
            var sum = 0.0
            for (i in 0 until 4) {
                val dx = a[i].x - b[i].x
                val dy = a[i].y - b[i].y
                sum += sqrt(dx * dx + dy * dy)
            }
            return sum / 4.0
        }

        /**
         * Tests convexity of a quadrilateral via cross-product sign consistency.
         * Corners must be in order (CW or CCW). Returns false for degenerate/self-intersecting quads.
         */
        fun isConvex(quad: List<Point>): Boolean {
            require(quad.size == 4) { "Expected 4 corners" }
            var positiveCount = 0
            var negativeCount = 0
            for (i in 0 until 4) {
                val a = quad[i]
                val b = quad[(i + 1) % 4]
                val c = quad[(i + 2) % 4]
                val cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x)
                if (cross > 0) positiveCount++
                else if (cross < 0) negativeCount++
            }
            // All cross products must have the same sign (all positive or all negative)
            return positiveCount == 0 || negativeCount == 0
        }

        /**
         * Computes the area of a quadrilateral using the shoelace formula.
         * Works for any simple (non-self-intersecting) polygon.
         */
        fun quadArea(quad: List<Point>): Double {
            require(quad.size == 4) { "Expected 4 corners" }
            var area = 0.0
            for (i in 0 until 4) {
                val j = (i + 1) % 4
                area += quad[i].x * quad[j].y
                area -= quad[j].x * quad[i].y
            }
            return kotlin.math.abs(area) / 2.0
        }
    }
}
