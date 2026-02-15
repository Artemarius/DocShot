package com.docshot.cv

import org.opencv.core.CvType
import org.opencv.core.Mat

/**
 * Lightweight object pool for OpenCV Mats to reduce per-frame allocation churn
 * in the detection hot path.
 *
 * Not thread-safe — designed for single-threaded detection on Dispatchers.Default.
 * Only intermediate scratch Mats should use the pool; result Mats returned to callers
 * are still freshly allocated (caller-owned).
 *
 * @param maxSize Maximum number of Mats to keep in the pool. Excess Mats are released.
 */
class MatPool(private val maxSize: Int = 8) {

    private val pool = ArrayDeque<Mat>(maxSize)
    private var acquireCount = 0
    private var reuseCount = 0

    /**
     * Returns a Mat from the pool, resized/reformatted if needed, or creates a new one.
     * The returned Mat's content is undefined — caller must write to it before reading.
     */
    fun acquire(rows: Int, cols: Int, type: Int): Mat {
        acquireCount++
        val mat = pool.removeLastOrNull()
        if (mat != null) {
            // Reuse pooled Mat — create() is a no-op if size/type already match
            mat.create(rows, cols, type)
            reuseCount++
            return mat
        }
        return Mat(rows, cols, type)
    }

    /**
     * Returns a Mat to the pool for future reuse. Does NOT call [Mat.release].
     * If the pool is full, the Mat is released immediately.
     */
    fun release(mat: Mat) {
        if (pool.size < maxSize) {
            pool.addLast(mat)
        } else {
            mat.release()
        }
    }

    /**
     * Releases all pooled Mats and clears the pool. Call on lifecycle destroy.
     */
    fun clear() {
        for (mat in pool) {
            mat.release()
        }
        pool.clear()
        acquireCount = 0
        reuseCount = 0
    }

    /** Number of Mats currently in the pool. */
    val size: Int get() = pool.size

    /** Total acquire() calls since last clear(). */
    val totalAcquires: Int get() = acquireCount

    /** Number of acquire() calls satisfied from pool (reuses). */
    val totalReuses: Int get() = reuseCount
}
