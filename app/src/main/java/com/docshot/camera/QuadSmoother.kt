package com.docshot.camera

import org.opencv.core.Point

/**
 * Temporal smoothing for detected document corners.
 * Maintains a rolling window of the last [windowSize] detections
 * and returns the averaged corner positions to reduce jitter.
 *
 * After [missThreshold] consecutive frames with no detection,
 * clears the buffer and returns null (overlay fades out).
 */
class QuadSmoother(
    private val windowSize: Int = 5,
    private val missThreshold: Int = 10
) {
    private val buffer = ArrayDeque<List<Point>>()
    private var consecutiveMisses = 0

    /**
     * Feed a new detection result. Pass null when no document was detected.
     * Returns smoothed corners or null if detection has been lost.
     */
    fun update(corners: List<Point>?): List<Point>? {
        if (corners == null) {
            consecutiveMisses++
            if (consecutiveMisses >= missThreshold) {
                buffer.clear()
                return null
            }
            return if (buffer.isNotEmpty()) average() else null
        }

        consecutiveMisses = 0
        if (buffer.size >= windowSize) buffer.removeFirst()
        buffer.addLast(corners)
        return average()
    }

    fun clear() {
        buffer.clear()
        consecutiveMisses = 0
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
