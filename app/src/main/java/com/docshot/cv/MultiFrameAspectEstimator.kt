package com.docshot.cv

import android.util.Log
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.min
import kotlin.math.sqrt

private const val TAG = "DocShot:MultiFrameAR"

/**
 * Result from multi-frame aspect ratio estimation.
 *
 * @property estimatedRatio aspect ratio (min/max, always <= 1.0)
 * @property confidence 0.0-1.0 based on inter-frame consistency (lower variance = higher confidence)
 * @property frameCount number of frames used in the estimation
 */
data class MultiFrameEstimate(
    val estimatedRatio: Double,
    val confidence: Double,
    val frameCount: Int
)

/**
 * Accumulates document corner observations during the auto-capture stabilization
 * window and estimates the true document aspect ratio using multi-frame constraints.
 *
 * Each observed quadrilateral provides a homography H to a canonical unit square.
 * For a planar surface, H = K * [r1 r2 t] where K is the camera matrix.
 * The rotation columns must satisfy:
 *   - r1^T * r2 = 0 (orthogonality)
 *   - ||r1|| = ||r2|| (equal norm)
 *
 * With N frames, this gives 2N constraints on omega = (K*K^T)^{-1}, the image
 * of the absolute conic. The least-squares solution for omega yields K, from
 * which the true aspect ratio can be recovered.
 *
 * Two estimation paths:
 * 1. **With intrinsics**: Decompose each H via K_inv * H = [r1 r2 t], compute
 *    the norm ratio ||r1||/||r2|| per frame, take the median.
 * 2. **Without intrinsics** (Zhang's method): Solve for omega from homography
 *    constraints alone, recover K, then use path 1.
 *
 * Thread safety: not thread-safe. Call from a single thread.
 */
class MultiFrameAspectEstimator {

    /** Accumulated homographies (document corners -> unit square). */
    private val homographies = mutableListOf<Mat>()

    /** Cached estimate — invalidated on addFrame() or reset(). */
    private var cachedEstimate: MultiFrameEstimate? = null
    private var cacheValid = false

    /** Number of accumulated frames. */
    val frameCount: Int get() = homographies.size

    /** Minimum frames needed for a meaningful estimate. */
    val minFrames: Int = 3

    /**
     * Add a frame's document corners to the accumulator.
     * Computes the homography from the detected corners to a canonical unit square
     * and stores it.
     *
     * @param corners 4 document corners [TL, TR, BR, BL] in image coordinates
     */
    fun addFrame(corners: List<Point>) {
        require(corners.size == 4) { "Expected 4 corners, got ${corners.size}" }

        // Homography from detected quad to unit square [0,1]x[0,1]
        val srcPts = MatOfPoint2f(*corners.toTypedArray())
        val dstPts = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(1.0, 0.0),
            Point(1.0, 1.0),
            Point(0.0, 1.0)
        )

        try {
            val H = Imgproc.getPerspectiveTransform(srcPts, dstPts)
            homographies.add(H)
            cacheValid = false
            Log.d(TAG, "Added frame ${homographies.size}")
        } finally {
            srcPts.release()
            dstPts.release()
        }
    }

    /**
     * Estimates the document aspect ratio from accumulated frames.
     *
     * Uses the Zhang calibration approach: each homography H_i provides two
     * constraints on omega (the image of the absolute conic):
     *   h1^T * omega * h2 = 0  (orthogonality)
     *   h1^T * omega * h1 - h2^T * omega * h2 = 0  (equal norm)
     *
     * With omega parameterized as a symmetric 3x3 matrix [a b c; b d e; c e f],
     * each constraint becomes a linear equation in [a, b, c, d, e, f].
     *
     * @param intrinsics optional camera intrinsics. If provided, uses the simpler
     *   decomposition approach. If null, solves from scratch via Zhang (needs >= 3 frames).
     * @return estimate with ratio and confidence, or null if insufficient data
     */
    fun estimateAspectRatio(intrinsics: CameraIntrinsics? = null): MultiFrameEstimate? {
        if (homographies.size < minFrames) {
            Log.d(TAG, "Insufficient frames: ${homographies.size} < $minFrames")
            return null
        }

        // Return cached result if data hasn't changed since last solve
        if (cacheValid) return cachedEstimate

        // If intrinsics are available, use the simpler decomposition approach
        val result = if (intrinsics != null) {
            estimateWithIntrinsics(intrinsics)
        } else {
            // Otherwise, solve for omega via Zhang's method
            estimateViaZhang()
        }

        cachedEstimate = result
        cacheValid = true
        return result
    }

    /**
     * Simpler approach when camera intrinsics are known:
     * Decompose each H via K_inv * H = [r1 r2 t], compute ||r1||/||r2|| per frame,
     * take the median ratio.
     */
    private fun estimateWithIntrinsics(intrinsics: CameraIntrinsics): MultiFrameEstimate? {
        var K: Mat? = null
        var Kinv: Mat? = null

        try {
            // Build camera matrix K
            K = Mat.zeros(3, 3, CvType.CV_64FC1)
            K.put(0, 0, intrinsics.fx)
            K.put(1, 1, intrinsics.fy)
            K.put(0, 2, intrinsics.cx)
            K.put(1, 2, intrinsics.cy)
            K.put(2, 2, 1.0)

            // K_inv
            Kinv = Mat()
            Core.invert(K, Kinv)

            val ratios = mutableListOf<Double>()

            for (H in homographies) {
                var M: Mat? = null
                try {
                    // M = K_inv * H
                    M = Mat()
                    Core.gemm(Kinv, H, 1.0, Mat(), 0.0, M)

                    // Extract rotation columns r1 (col 0) and r2 (col 1)
                    val r1 = doubleArrayOf(
                        M.get(0, 0)[0],
                        M.get(1, 0)[0],
                        M.get(2, 0)[0]
                    )
                    val r2 = doubleArrayOf(
                        M.get(0, 1)[0],
                        M.get(1, 1)[0],
                        M.get(2, 1)[0]
                    )

                    val normR1 = sqrt(r1[0] * r1[0] + r1[1] * r1[1] + r1[2] * r1[2])
                    val normR2 = sqrt(r2[0] * r2[0] + r2[1] * r2[1] + r2[2] * r2[2])

                    if (normR1 > 0 && normR2 > 0) {
                        val ratio = min(normR1, normR2) / maxOf(normR1, normR2)
                        ratios.add(ratio)
                    }
                } finally {
                    M?.release()
                }
            }

            if (ratios.isEmpty()) return null

            // Use median for robustness against outliers
            val sorted = ratios.sorted()
            val median = sorted[sorted.size / 2]

            // Confidence from consistency (inverse of variance)
            val variance = ratios.map { (it - median) * (it - median) }.average()
            val confidence = (1.0 / (1.0 + variance * 1000.0)).coerceIn(0.0, 1.0)

            Log.d(
                TAG,
                "Intrinsics estimate: ratio=%.4f, confidence=%.2f, frames=%d, variance=%.6f".format(
                    median, confidence, ratios.size, variance
                )
            )

            return MultiFrameEstimate(
                estimatedRatio = median.coerceIn(0.1, 1.0),
                confidence = confidence,
                frameCount = ratios.size
            )
        } finally {
            K?.release()
            Kinv?.release()
        }
    }

    /**
     * Zhang's method: solve for omega from homography constraints alone.
     *
     * Each H gives 2 linear constraints on the 6 elements of symmetric omega.
     * Stack into a Vb = 0 system and solve via SVD.
     *
     * omega is parameterized as b = [B11, B12, B22, B13, B23, B33].
     * From b we recover the intrinsic parameters fx, fy, cx, cy, then
     * delegate to [estimateWithIntrinsics] for per-frame ratio extraction.
     */
    private fun estimateViaZhang(): MultiFrameEstimate? {
        val numConstraints = homographies.size * 2

        var V: Mat? = null

        try {
            V = Mat(numConstraints, 6, CvType.CV_64FC1)

            for ((idx, H) in homographies.withIndex()) {
                // Read homography elements into a local array for fast access
                val h = Array(3) { row ->
                    DoubleArray(3) { col -> H.get(row, col)[0] }
                }

                // Build the v_ij vector for the Zhang constraint equations.
                // v_ij = [h_i1*h_j1,
                //         h_i1*h_j2 + h_i2*h_j1,
                //         h_i2*h_j2,
                //         h_i3*h_j1 + h_i1*h_j3,
                //         h_i3*h_j2 + h_i2*h_j3,
                //         h_i3*h_j3]
                // where h_ij = H[j-1][i-1] (i = column index in homography, j = row index)
                fun vij(i: Int, j: Int): DoubleArray = doubleArrayOf(
                    h[0][i] * h[0][j],
                    h[0][i] * h[1][j] + h[1][i] * h[0][j],
                    h[1][i] * h[1][j],
                    h[2][i] * h[0][j] + h[0][i] * h[2][j],
                    h[2][i] * h[1][j] + h[1][i] * h[2][j],
                    h[2][i] * h[2][j]
                )

                // Constraint 1: v_12 = 0 (orthogonality of rotation columns)
                val v12 = vij(0, 1)
                for (c in 0 until 6) V.put(idx * 2, c, v12[c])

                // Constraint 2: v_11 - v_22 = 0 (equal norm of rotation columns)
                val v11 = vij(0, 0)
                val v22 = vij(1, 1)
                for (c in 0 until 6) V.put(idx * 2 + 1, c, v11[c] - v22[c])
            }

            // Solve Vb = 0 via SVD — b is the right singular vector with smallest singular value
            var w: Mat? = null
            var u: Mat? = null
            var vt: Mat? = null

            try {
                w = Mat()
                u = Mat()
                vt = Mat()
                Core.SVDecomp(V, w, u, vt)

                // Last row of vt is the solution (smallest singular value)
                val lastRow = vt.rows() - 1
                val b = DoubleArray(6) { vt.get(lastRow, it)[0] }

                // Reconstruct omega (B) from b = [B11, B12, B22, B13, B23, B33]
                val B11 = b[0]
                val B12 = b[1]
                val B22 = b[2]
                val B13 = b[3]
                val B23 = b[4]
                val B33 = b[5]

                // Extract intrinsics from omega using Zhang's closed-form equations:
                // v0 = (B12*B13 - B11*B23) / (B11*B22 - B12^2)
                val denom = B11 * B22 - B12 * B12
                if (abs(denom) < 1e-12) {
                    Log.d(TAG, "Zhang: degenerate omega (denom ~ 0)")
                    return null
                }

                val v0 = (B12 * B13 - B11 * B23) / denom
                val lambda = B33 - (B13 * B13 + v0 * (B12 * B13 - B11 * B23)) / B11

                if (B11 == 0.0 || lambda / B11 <= 0 || lambda / denom <= 0) {
                    Log.d(
                        TAG,
                        "Zhang: negative under sqrt (lambda=%.6f, B11=%.6f, denom=%.6f)".format(
                            lambda, B11, denom
                        )
                    )
                    return null
                }

                val fx = sqrt(lambda / B11)
                val fy = sqrt(lambda * B11 / denom)
                val cx = -B13 / B11 * fx  // approximate principal point
                val cy = v0

                Log.d(
                    TAG,
                    "Zhang: recovered fx=%.1f, fy=%.1f, cx=%.1f, cy=%.1f".format(fx, fy, cx, cy)
                )

                // Decompose each H with recovered K and get per-frame ratios
                val recoveredIntrinsics = CameraIntrinsics(
                    fx = fx,
                    fy = fy,
                    cx = cx,
                    cy = cy
                )

                return estimateWithIntrinsics(recoveredIntrinsics)
            } finally {
                w?.release()
                u?.release()
                vt?.release()
            }
        } finally {
            V?.release()
        }
    }

    /** Clears all accumulated data. Call when tracking is lost or scene changes. */
    fun reset() {
        val count = homographies.size
        homographies.forEach { it.release() }
        homographies.clear()
        cachedEstimate = null
        cacheValid = false
        Log.d(TAG, "Reset (cleared $count homographies)")
    }

    /** Releases all native resources. Call from ViewModel.onCleared(). */
    fun release() {
        homographies.forEach { it.release() }
        homographies.clear()
        cachedEstimate = null
        cacheValid = false
    }
}
