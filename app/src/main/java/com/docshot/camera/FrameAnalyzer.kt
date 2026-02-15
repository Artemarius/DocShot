package com.docshot.camera

import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.docshot.cv.detectDocumentWithStatus
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

private const val TAG = "DocShot:Analyzer"
private const val MAX_ANALYSIS_WIDTH = 640

/** Consecutive misses before skipping every other frame. */
private const val SKIP_TIER1_THRESHOLD = 5

/** Consecutive misses before skipping 2 of every 3 frames. */
private const val SKIP_TIER2_THRESHOLD = 15

/**
 * CameraX ImageAnalysis.Analyzer that runs document detection on each camera frame.
 *
 * Extracts the Y (luminance) plane from YUV_420_888 frames, downscales if needed,
 * runs the detection pipeline, applies temporal smoothing, and reports results
 * via the [onResult] callback.
 *
 * Implements adaptive frame skipping: when detection repeatedly fails on complex
 * scenes, processing rate is reduced to save CPU. When a detection is found,
 * immediately resumes full-rate processing.
 *
 * Results are reported as normalized [0,1] coordinates with rotation already
 * applied for display orientation.
 */
class FrameAnalyzer(
    private val onResult: (FrameDetectionResult) -> Unit
) : ImageAnalysis.Analyzer {

    private val smoother = QuadSmoother()

    /** Number of consecutive frames with no valid detection (confidence < threshold). */
    private var consecutiveMisses = 0

    /** Frame counter for skip logic (resets when detection found). */
    private var frameCounter = 0

    /** Last reported result, used when skipping frames. */
    private var lastResult: FrameDetectionResult? = null

    /** Clears the smoother and miss counter so stability builds from scratch. */
    fun resetSmoothing() {
        smoother.clear()
        consecutiveMisses = 0
        frameCounter = 0
        lastResult = null
    }

    override fun analyze(imageProxy: ImageProxy) {
        frameCounter++

        // Adaptive frame skipping: reduce processing on complex scenes where
        // detection repeatedly fails, to save CPU for other work.
        if (shouldSkipFrame()) {
            imageProxy.close()
            // Report last known result so UI stays responsive
            lastResult?.let { onResult(it) }
            return
        }

        val start = System.nanoTime()
        try {
            val grayMat = extractYPlane(imageProxy)
            val analysisWidth = grayMat.cols()
            val analysisHeight = grayMat.rows()

            // Downscale for speed if the camera picked a high-res analysis stream
            val downscaled = downscaleIfNeeded(grayMat)
            val scaleFactor = analysisWidth.toDouble() / downscaled.cols()

            val status = detectDocumentWithStatus(downscaled)
            val detection = status.result
            if (downscaled !== grayMat) downscaled.release()
            grayMat.release()

            // Update consecutive miss counter for adaptive frame skipping
            if (detection != null) {
                consecutiveMisses = 0
                frameCounter = 0
            } else {
                consecutiveMisses++
            }

            // Scale corners back to original analysis dimensions
            val rawCorners = detection?.corners?.map { pt ->
                Point(pt.x * scaleFactor, pt.y * scaleFactor)
            }

            // Temporal smoothing (pass confidence for rolling average tracking)
            val smoothed = smoother.update(
                corners = rawCorners,
                confidence = detection?.confidence ?: 0.0
            )

            // Rotate corners for display orientation and normalize to [0,1]
            val rotation = imageProxy.imageInfo.rotationDegrees
            val normalized = smoothed?.let {
                rotateAndNormalize(it, analysisWidth, analysisHeight, rotation)
            }

            // Image dimensions in display orientation (for overlay coordinate mapping)
            val displayWidth: Int
            val displayHeight: Int
            if (rotation == 90 || rotation == 270) {
                displayWidth = analysisHeight
                displayHeight = analysisWidth
            } else {
                displayWidth = analysisWidth
                displayHeight = analysisHeight
            }

            val totalMs = (System.nanoTime() - start) / 1_000_000.0
            Log.d(TAG, "analyze: %.1f ms (detect=%.1f ms, rotation=%d°, misses=%d)".format(
                totalMs, detection?.detectionMs ?: 0.0, rotation, consecutiveMisses))

            val result = FrameDetectionResult(
                normalizedCorners = normalized,
                sourceWidth = displayWidth,
                sourceHeight = displayHeight,
                detectionMs = detection?.detectionMs ?: 0.0,
                totalMs = totalMs,
                isStable = smoother.isStable,
                stabilityProgress = smoother.stabilityProgress,
                confidence = smoother.averageConfidence,
                isPartialDocument = status.isPartialDocument
            )
            lastResult = result
            onResult(result)
        } finally {
            imageProxy.close()
        }
    }

    /**
     * Determines whether the current frame should be skipped based on
     * consecutive detection misses.
     *
     * - High confidence / recent detection: process every frame
     * - 5+ consecutive misses: skip every other frame (process 1 of 2)
     * - 15+ consecutive misses: skip 2 of every 3 frames (process 1 of 3)
     */
    private fun shouldSkipFrame(): Boolean {
        if (consecutiveMisses < SKIP_TIER1_THRESHOLD) return false

        if (consecutiveMisses >= SKIP_TIER2_THRESHOLD) {
            // Process every 3rd frame
            return (frameCounter % 3) != 0
        }

        // Process every 2nd frame
        return (frameCounter % 2) != 0
    }

    /**
     * Extracts the Y (luminance) plane from a YUV_420_888 ImageProxy as a grayscale Mat.
     * Handles row stride padding correctly.
     */
    private fun extractYPlane(imageProxy: ImageProxy): Mat {
        val yPlane = imageProxy.planes[0]
        val width = imageProxy.width
        val height = imageProxy.height
        val rowStride = yPlane.rowStride
        val buffer = yPlane.buffer
        buffer.rewind()

        val mat = Mat(height, width, CvType.CV_8UC1)
        if (rowStride == width) {
            val data = ByteArray(width * height)
            buffer.get(data)
            mat.put(0, 0, data)
        } else {
            val rowData = ByteArray(width)
            for (row in 0 until height) {
                buffer.position(row * rowStride)
                buffer.get(rowData, 0, width)
                mat.put(row, 0, rowData)
            }
        }
        return mat
    }

    /**
     * Downscales a Mat if wider than [MAX_ANALYSIS_WIDTH].
     * Returns the original Mat if no downscale is needed (caller must track this
     * to avoid double-releasing).
     */
    private fun downscaleIfNeeded(gray: Mat): Mat {
        if (gray.cols() <= MAX_ANALYSIS_WIDTH) return gray

        val scale = MAX_ANALYSIS_WIDTH.toDouble() / gray.cols()
        val newHeight = (gray.rows() * scale).toInt()
        val resized = Mat()
        Imgproc.resize(
            gray, resized,
            Size(MAX_ANALYSIS_WIDTH.toDouble(), newHeight.toDouble()),
            0.0, 0.0, Imgproc.INTER_LINEAR
        )
        return resized
    }

    /**
     * Rotates corner points by [rotationDegrees] CW and normalizes to [0,1].
     * Maps from sensor coordinates to display coordinates.
     */
    private fun rotateAndNormalize(
        corners: List<Point>,
        imageWidth: Int,
        imageHeight: Int,
        rotationDegrees: Int
    ): List<FloatArray> {
        val w = imageWidth.toDouble()
        val h = imageHeight.toDouble()

        return corners.map { pt ->
            val (nx, ny) = when (rotationDegrees) {
                90 -> (1.0 - pt.y / h) to (pt.x / w)
                180 -> (1.0 - pt.x / w) to (1.0 - pt.y / h)
                270 -> (pt.y / h) to (1.0 - pt.x / w)
                else -> (pt.x / w) to (pt.y / h)
            }
            floatArrayOf(nx.toFloat(), ny.toFloat())
        }
    }
}

/**
 * Result from a single frame's analysis, ready for UI consumption.
 * Corners are in normalized [0,1] display coordinates (rotation already applied).
 */
data class FrameDetectionResult(
    val normalizedCorners: List<FloatArray>?,
    val sourceWidth: Int,
    val sourceHeight: Int,
    val detectionMs: Double,
    val totalMs: Double,
    val isStable: Boolean = false,
    val stabilityProgress: Float = 0f,
    val confidence: Double = 0.0,
    val isPartialDocument: Boolean = false
)
