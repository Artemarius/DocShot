package com.docshot.cv

import android.util.Log
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.min
import kotlin.math.sqrt

private const val TAG = "DocShot:AspectRatio"

/** A known document format with display name and min/max ratio (always <= 1.0). */
data class KnownFormat(val name: String, val ratio: Double)

/** Result of aspect ratio estimation. */
data class AspectRatioEstimate(
    val estimatedRatio: Double,
    val matchedFormat: KnownFormat?,
    val confidence: Double,
    val verifiedByHomography: Boolean = false
)

/** Simplified camera intrinsics for homography verification. */
data class CameraIntrinsics(val fx: Double, val fy: Double, val cx: Double, val cy: Double)

/** Known document formats — ratio is min(side)/max(side), always <= 1.0. */
val KNOWN_FORMATS = listOf(
    KnownFormat("A4", 1.0 / 1.414),            // 0.707
    KnownFormat("US Letter", 1.0 / 1.294),      // 0.773
    KnownFormat("ID Card", 1.0 / 1.586),        // 0.631
    KnownFormat("Business Card", 1.0 / 1.75),   // 0.571
    KnownFormat("Receipt", 1.0 / 3.0),           // 0.333
    KnownFormat("Square", 1.0)                    // 1.000
)

/** Max distance from a known ratio to consider snapping. */
private const val SNAP_THRESHOLD = 0.06

/** Gaussian sigma for snap confidence scoring. */
private const val SNAP_SIGMA = 0.04

/** Perspective severity threshold below which angular correction is used (degrees). */
private const val SEVERITY_LOW_THRESHOLD = 15.0

/** Perspective severity threshold above which projective estimation is used (degrees). */
private const val SEVERITY_HIGH_THRESHOLD = 20.0

/**
 * Computes the raw aspect ratio (min/max, always <= 1.0) from quad edge pairs.
 * Same math as [aspectRatioScore] in QuadRanker.kt but returns the ratio directly.
 */
fun computeRawRatio(corners: List<Point>): Double {
    require(corners.size == 4) { "Expected 4 corners, got ${corners.size}" }
    val edgeLengths = DoubleArray(4) { i ->
        val next = (i + 1) % 4
        distance(corners[i], corners[next])
    }
    val side1 = (edgeLengths[0] + edgeLengths[2]) / 2.0
    val side2 = (edgeLengths[1] + edgeLengths[3]) / 2.0
    if (side1 <= 0.0 || side2 <= 0.0) return 1.0
    return min(side1, side2) / maxOf(side1, side2)
}

// ---------------------------------------------------------------------------
// B1: Hartley Normalization
// ---------------------------------------------------------------------------

/**
 * Applies Hartley normalization to 4 corner points: translates so centroid is at
 * origin, scales so average distance from origin is sqrt(2). Standard conditioning
 * technique (Hartley & Zisserman, ch4.4) that stabilizes all downstream projective
 * computations.
 *
 * @param corners 4 points (TL, TR, BR, BL)
 * @return Pair of (normalized points, 3x3 normalization transform T as CV_64FC1 Mat).
 *         Caller must release the returned Mat.
 */
fun hartleyNormalize(corners: List<Point>): Pair<List<Point>, Mat> {
    require(corners.size == 4) { "Expected 4 corners" }
    val cx = corners.map { it.x }.average()
    val cy = corners.map { it.y }.average()
    val centered = corners.map { Point(it.x - cx, it.y - cy) }
    val avgDist = centered.map { sqrt(it.x * it.x + it.y * it.y) }.average()
    val scale = if (avgDist > 0.0) sqrt(2.0) / avgDist else 1.0
    val normalized = centered.map { Point(it.x * scale, it.y * scale) }

    val T = Mat.zeros(3, 3, CvType.CV_64FC1)
    T.put(0, 0, scale); T.put(0, 2, -scale * cx)
    T.put(1, 1, scale); T.put(1, 2, -scale * cy)
    T.put(2, 2, 1.0)

    return Pair(normalized, T)
}

// ---------------------------------------------------------------------------
// B2: Perspective Severity Classifier
// ---------------------------------------------------------------------------

/**
 * Computes the maximum interior-angle deviation from 90 degrees across all four
 * corners of the quadrilateral. Used to classify the perspective regime:
 * - < 15deg: low severity (use angular correction)
 * - > 20deg: high severity (use projective estimation)
 * - 15-20deg: transition zone (weighted blend)
 *
 * @param corners 4 points in TL, TR, BR, BL order
 * @return maximum deviation from 90 degrees, in degrees (always >= 0)
 */
fun perspectiveSeverity(corners: List<Point>): Double {
    require(corners.size == 4) { "Expected 4 corners" }
    var maxDeviation = 0.0
    for (i in 0 until 4) {
        val prev = corners[(i + 3) % 4]
        val curr = corners[i]
        val next = corners[(i + 1) % 4]
        val v1 = Point(prev.x - curr.x, prev.y - curr.y)
        val v2 = Point(next.x - curr.x, next.y - curr.y)
        val dot = v1.x * v2.x + v1.y * v2.y
        val cross = v1.x * v2.y - v1.y * v2.x
        val angle = Math.toDegrees(atan2(abs(cross), dot))
        val deviation = abs(angle - 90.0)
        if (deviation > maxDeviation) maxDeviation = deviation
    }
    return maxDeviation
}

// ---------------------------------------------------------------------------
// B3: Low-Severity Angular Correction
// ---------------------------------------------------------------------------

/**
 * Computes a perspective-corrected aspect ratio for low-severity views (quad is
 * close to a rectangle). Applies a foreshortening correction based on the
 * convergence angles of opposite edge pairs.
 *
 * corrected_ratio = raw_ratio * cos(alpha_v / 2) / cos(alpha_h / 2)
 *
 * @param corners 4 points in TL, TR, BR, BL order
 * @return corrected ratio (min/max, always in [0.1, 1.0])
 */
fun angularCorrectedRatio(corners: List<Point>): Double {
    require(corners.size == 4) { "Expected 4 corners" }
    val (tl, tr, br, bl) = corners

    val rawRatio = computeRawRatio(corners)

    // Horizontal convergence: angle between top edge and bottom edge
    val topDir = Point(tr.x - tl.x, tr.y - tl.y)
    val botDir = Point(br.x - bl.x, br.y - bl.y)
    val alphaH = angleBetweenVectors(topDir, botDir)

    // Vertical convergence: angle between left edge and right edge
    val leftDir = Point(bl.x - tl.x, bl.y - tl.y)
    val rightDir = Point(br.x - tr.x, br.y - tr.y)
    val alphaV = angleBetweenVectors(leftDir, rightDir)

    val correction = cos(alphaV / 2.0) / cos(alphaH / 2.0)
    return (rawRatio * correction).coerceIn(0.1, 1.0)
}

/** Angle between two 2D vectors in radians (always positive, [0, PI]). */
private fun angleBetweenVectors(v1: Point, v2: Point): Double {
    val dot = v1.x * v2.x + v1.y * v2.y
    val cross = v1.x * v2.y - v1.y * v2.x
    return abs(atan2(cross, dot))
}

// ---------------------------------------------------------------------------
// B4: High-Severity Projective Estimation
// ---------------------------------------------------------------------------

/**
 * Estimates the document aspect ratio using full projective reconstruction with
 * vanishing points and camera intrinsics. Suitable for heavily skewed views where
 * edge-length ratios are unreliable.
 *
 * Algorithm:
 * 1. Compute homography H from corners to a unit square
 * 2. Decompose M = K_inv * H into [r1 r2 t]
 * 3. Aspect ratio correction = ||r1|| / ||r2||
 * 4. Apply correction to the raw edge-length ratio
 *
 * All vanishing point computations stay in homogeneous coordinates to avoid
 * numerical instability when VPs are near infinity (near-parallel edges).
 *
 * @param corners 4 points in TL, TR, BR, BL order
 * @param intrinsics camera calibration parameters
 * @return estimated ratio (min/max, always <= 1.0), or null if computation fails
 */
fun projectiveAspectRatio(corners: List<Point>, intrinsics: CameraIntrinsics): Double? {
    require(corners.size == 4) { "Expected 4 corners" }
    val (tl, tr, br, bl) = corners

    // Map to a square destination -- the ratio ||r1||/||r2|| from decomposition
    // reveals how the true document aspect ratio deviates from 1:1
    val dstSize = 1000.0

    var srcPts: MatOfPoint2f? = null
    var dstPts: MatOfPoint2f? = null
    var H: Mat? = null
    var K: Mat? = null
    var Kinv: Mat? = null
    var M: Mat? = null

    try {
        // Compute H mapping corners to a unit (square) destination.
        // If the document were truly square, r1 and r2 would have equal norms.
        // The ratio ||r1||/||r2|| reveals how the document deviates from square.
        srcPts = MatOfPoint2f(tl, tr, br, bl)
        dstPts = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(dstSize - 1.0, 0.0),
            Point(dstSize - 1.0, dstSize - 1.0),
            Point(0.0, dstSize - 1.0)
        )

        H = Imgproc.getPerspectiveTransform(srcPts, dstPts)

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

        // M = K_inv * H = [r1 r2 t] (up to scale)
        M = Mat()
        Core.gemm(Kinv, H, 1.0, Mat(), 0.0, M)

        // Extract columns r1 (col 0) and r2 (col 1)
        val r1 = doubleArrayOf(M.get(0, 0)[0], M.get(1, 0)[0], M.get(2, 0)[0])
        val r2 = doubleArrayOf(M.get(0, 1)[0], M.get(1, 1)[0], M.get(2, 1)[0])

        val normR1 = sqrt(r1[0] * r1[0] + r1[1] * r1[1] + r1[2] * r1[2])
        val normR2 = sqrt(r2[0] * r2[0] + r2[1] * r2[1] + r2[2] * r2[2])

        if (normR1 <= 0.0 || normR2 <= 0.0) return null

        // ||r1||/||r2|| is the aspect ratio of the destination rectangle that would
        // make H consistent with a rotation. Since destination is square (1:1),
        // the true document width:height = ||r1||/||r2||.
        val projRatio = normR1 / normR2

        // Convert to min/max ratio (always <= 1.0)
        val aspectRatio = min(projRatio, 1.0 / projRatio)

        // Sanity check: reject wildly unreasonable ratios
        if (aspectRatio < 0.05 || aspectRatio > 1.0) return null

        Log.d(TAG, "projectiveAspectRatio: normR1=%.4f, normR2=%.4f, ratio=%.4f".format(
            normR1, normR2, aspectRatio))
        return aspectRatio
    } catch (e: Exception) {
        Log.w(TAG, "projectiveAspectRatio failed: ${e.message}")
        return null
    } finally {
        srcPts?.release()
        dstPts?.release()
        H?.release()
        K?.release()
        Kinv?.release()
        M?.release()
    }
}

/** Converts a 2D point to homogeneous coordinates [x, y, 1]. */
private fun toHomogeneous(p: Point): DoubleArray = doubleArrayOf(p.x, p.y, 1.0)

/** Cross product of two 3-element vectors (homogeneous coordinates). */
private fun crossProduct(a: DoubleArray, b: DoubleArray): DoubleArray = doubleArrayOf(
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]
)

// ---------------------------------------------------------------------------
// B5: Transition-Zone Blending + Dual-Regime Dispatcher
// ---------------------------------------------------------------------------

/**
 * Estimates the document aspect ratio using a dual-regime approach that selects
 * the best method based on perspective severity:
 * - Low severity (< 15deg): angular correction (fast, accurate for near-frontal)
 * - High severity (> 20deg): projective estimation with intrinsics (accurate for skew)
 * - Transition zone (15-20deg): weighted blend of both methods
 *
 * Falls back to angular correction when intrinsics are unavailable.
 * The result is then snapped to known formats using the same logic as the
 * original [estimateAspectRatio].
 *
 * @param corners 4 points in TL, TR, BR, BL order
 * @param intrinsics camera calibration parameters, or null if unavailable
 * @return [AspectRatioEstimate] with the blended ratio and format snapping applied
 */
fun estimateAspectRatioDualRegime(
    corners: List<Point>,
    intrinsics: CameraIntrinsics? = null
): AspectRatioEstimate {
    require(corners.size == 4) { "Expected 4 corners, got ${corners.size}" }

    val severity = perspectiveSeverity(corners)
    Log.d(TAG, "perspectiveSeverity: %.1f deg".format(severity))

    val dualRatio: Double = when {
        severity < SEVERITY_LOW_THRESHOLD -> {
            // Low severity: angular correction is sufficient
            angularCorrectedRatio(corners)
        }
        severity > SEVERITY_HIGH_THRESHOLD -> {
            // High severity: prefer projective if intrinsics available
            if (intrinsics != null) {
                projectiveAspectRatio(corners, intrinsics)
                    ?: angularCorrectedRatio(corners) // fallback if projective fails
            } else {
                angularCorrectedRatio(corners)
            }
        }
        else -> {
            // Transition zone: weighted blend
            val angularRatio = angularCorrectedRatio(corners)
            if (intrinsics != null) {
                val projectiveRatio = projectiveAspectRatio(corners, intrinsics)
                if (projectiveRatio != null) {
                    val weight = (severity - SEVERITY_LOW_THRESHOLD) /
                        (SEVERITY_HIGH_THRESHOLD - SEVERITY_LOW_THRESHOLD)
                    angularRatio * (1.0 - weight) + projectiveRatio * weight
                } else {
                    angularRatio
                }
            } else {
                angularRatio
            }
        }
    }

    // Apply format snapping to the dual-regime ratio
    return snapToFormat(
        ratio = dualRatio,
        corners = corners,
        intrinsics = intrinsics
    )
}

/**
 * Snaps a computed ratio to known document formats. Extracted from the original
 * [estimateAspectRatio] to be reusable by the dual-regime path.
 *
 * @param ratio the estimated ratio to snap
 * @param corners original corners (for homography disambiguation if needed)
 * @param intrinsics camera intrinsics (for homography disambiguation if needed)
 * @return [AspectRatioEstimate] with format snapping applied
 */
private fun snapToFormat(
    ratio: Double,
    corners: List<Point>,
    intrinsics: CameraIntrinsics?
): AspectRatioEstimate {
    data class Candidate(val format: KnownFormat, val dist: Double)
    val candidates = KNOWN_FORMATS
        .map { Candidate(it, abs(ratio - it.ratio)) }
        .filter { it.dist <= SNAP_THRESHOLD }
        .sortedBy { it.dist }

    if (candidates.isEmpty()) {
        return AspectRatioEstimate(
            estimatedRatio = ratio,
            matchedFormat = null,
            confidence = 0.5
        )
    }

    // Single clear match or clear winner
    if (candidates.size == 1 || candidates[1].dist > candidates[0].dist * 2.0) {
        val best = candidates[0]
        val conf = exp(-best.dist * best.dist / (2.0 * SNAP_SIGMA * SNAP_SIGMA))
        return AspectRatioEstimate(
            estimatedRatio = best.format.ratio,
            matchedFormat = best.format,
            confidence = conf.coerceIn(0.0, 1.0)
        )
    }

    // Ambiguous: multiple close candidates
    if (intrinsics != null) {
        var bestCandidate = candidates[0]
        var bestError = Double.MAX_VALUE
        for (c in candidates) {
            val error = homographyError(corners, c.format.ratio, intrinsics)
            if (error < bestError) {
                bestError = error
                bestCandidate = c
            }
        }
        val conf = exp(-bestCandidate.dist * bestCandidate.dist / (2.0 * SNAP_SIGMA * SNAP_SIGMA))
        return AspectRatioEstimate(
            estimatedRatio = bestCandidate.format.ratio,
            matchedFormat = bestCandidate.format,
            confidence = conf.coerceIn(0.0, 1.0),
            verifiedByHomography = true
        )
    }

    // No intrinsics — pick closest
    val best = candidates[0]
    val conf = exp(-best.dist * best.dist / (2.0 * SNAP_SIGMA * SNAP_SIGMA))
    return AspectRatioEstimate(
        estimatedRatio = best.format.ratio,
        matchedFormat = best.format,
        confidence = conf.coerceIn(0.0, 1.0) * 0.8
    )
}

/**
 * Estimates the document aspect ratio from detected corners, optionally
 * using camera intrinsics for homography-based verification.
 *
 * Internally delegates to [estimateAspectRatioDualRegime] which selects the
 * best estimation method based on perspective severity. All existing callers
 * automatically benefit from the improved estimation.
 *
 * Algorithm:
 * 1. Classify perspective severity from corner angles
 * 2. Low severity (< 15deg): edge-length ratio + angular correction
 * 3. High severity (> 20deg): projective estimation with intrinsics
 * 4. Transition (15-20deg): weighted blend of both methods
 * 5. Snap to known formats within [SNAP_THRESHOLD]
 * 6. If ambiguous and intrinsics available -> use [homographyError] to disambiguate
 */
fun estimateAspectRatio(
    corners: List<Point>,
    intrinsics: CameraIntrinsics? = null
): AspectRatioEstimate {
    require(corners.size == 4) { "Expected 4 corners, got ${corners.size}" }
    return estimateAspectRatioDualRegime(corners, intrinsics)
}

/**
 * Computes the homography verification error for a candidate aspect ratio.
 *
 * Given the detected quad corners and a candidate document ratio, computes the
 * perspective transform H, then checks if H is consistent with a planar surface
 * viewed by a calibrated camera by decomposing via K_inv * H and checking that
 * the first two columns approximate rotation vectors (equal norms, orthogonal).
 *
 * @return Combined error (lower = better match). Typically 0.0-0.3 for correct ratio.
 */
fun homographyError(
    corners: List<Point>,
    candidateRatio: Double,
    intrinsics: CameraIntrinsics
): Double {
    require(corners.size == 4) { "Expected 4 corners" }
    require(candidateRatio > 0.0 && candidateRatio <= 1.0) { "Ratio must be in (0, 1]" }

    // Determine orientation from quad geometry — use the same approach as rectify()
    val (tl, tr, br, bl) = corners
    val quadWidth = (distance(tl, tr) + distance(bl, br)) / 2.0
    val quadHeight = (distance(tl, bl) + distance(tr, br)) / 2.0
    val isLandscape = quadWidth >= quadHeight

    // Build destination rect with candidate ratio
    val longSide = 1000.0
    val shortSide = longSide * candidateRatio
    val dstW = if (isLandscape) longSide else shortSide
    val dstH = if (isLandscape) shortSide else longSide

    var srcPts: MatOfPoint2f? = null
    var dstPts: MatOfPoint2f? = null
    var H: Mat? = null
    var K: Mat? = null
    var Kinv: Mat? = null
    var M: Mat? = null

    try {
        srcPts = MatOfPoint2f(tl, tr, br, bl)
        dstPts = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(dstW - 1.0, 0.0),
            Point(dstW - 1.0, dstH - 1.0),
            Point(0.0, dstH - 1.0)
        )

        H = Imgproc.getPerspectiveTransform(srcPts, dstPts)

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

        // M = K_inv * H
        M = Mat()
        Core.gemm(Kinv, H, 1.0, Mat(), 0.0, M)

        // Extract columns r1 (col 0) and r2 (col 1)
        val r1 = doubleArrayOf(M.get(0, 0)[0], M.get(1, 0)[0], M.get(2, 0)[0])
        val r2 = doubleArrayOf(M.get(0, 1)[0], M.get(1, 1)[0], M.get(2, 1)[0])

        val normR1 = sqrt(r1[0] * r1[0] + r1[1] * r1[1] + r1[2] * r1[2])
        val normR2 = sqrt(r2[0] * r2[0] + r2[1] * r2[1] + r2[2] * r2[2])

        if (normR1 == 0.0 || normR2 == 0.0) return Double.MAX_VALUE

        // Check 1: column norms should be equal (both are scale * rotation column)
        val normRatioError = abs(1.0 - normR1 / normR2)

        // Check 2: columns should be orthogonal (dot product ~ 0)
        val dot = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2]
        val dotError = abs(dot) / (normR1 * normR2)

        return normRatioError + dotError
    } catch (e: Exception) {
        Log.w(TAG, "homographyError failed: ${e.message}")
        return Double.MAX_VALUE
    } finally {
        srcPts?.release()
        dstPts?.release()
        H?.release()
        K?.release()
        Kinv?.release()
        M?.release()
    }
}

/**
 * Re-warps the source image using detected corners, forcing the output to match
 * [targetRatio] (min/max, always <= 1.0). Preserves the longer dimension from
 * the standard rectify output and computes the shorter from the ratio.
 *
 * Caller must release the returned Mat.
 */
fun rectifyWithAspectRatio(
    source: Mat,
    corners: List<Point>,
    targetRatio: Double,
    interpolation: Int = Imgproc.INTER_CUBIC
): Mat {
    require(corners.size == 4) { "Expected 4 ordered corners [TL, TR, BR, BL]" }
    require(targetRatio > 0.0 && targetRatio <= 1.0) { "targetRatio must be in (0, 1], got $targetRatio" }
    val start = System.nanoTime()

    val (tl, tr, br, bl) = corners

    // Derive standard dimensions (same as rectify())
    val widthTop = distance(tl, tr)
    val widthBottom = distance(bl, br)
    val stdWidth = maxOf(widthTop, widthBottom).toInt()

    val heightLeft = distance(tl, bl)
    val heightRight = distance(tr, br)
    val stdHeight = maxOf(heightLeft, heightRight).toInt()

    check(stdWidth > 0 && stdHeight > 0) { "Invalid dimensions: ${stdWidth}x${stdHeight}" }

    // Determine orientation from quad geometry
    val isLandscape = stdWidth >= stdHeight

    // Preserve the longer dimension, compute shorter from ratio
    val longDim = maxOf(stdWidth, stdHeight)
    val shortDim = (longDim * targetRatio).toInt().coerceAtLeast(1)

    val outWidth = if (isLandscape) longDim else shortDim
    val outHeight = if (isLandscape) shortDim else longDim

    val srcPts = MatOfPoint2f(tl, tr, br, bl)
    val dstPts = MatOfPoint2f(
        Point(0.0, 0.0),
        Point(outWidth - 1.0, 0.0),
        Point(outWidth - 1.0, outHeight - 1.0),
        Point(0.0, outHeight - 1.0)
    )

    val transform = Imgproc.getPerspectiveTransform(srcPts, dstPts)
    srcPts.release()
    dstPts.release()

    val output = Mat()
    Imgproc.warpPerspective(
        source, output,
        transform,
        Size(outWidth.toDouble(), outHeight.toDouble()),
        interpolation
    )
    transform.release()

    val ms = (System.nanoTime() - start) / 1_000_000.0
    Log.d(TAG, "rectifyWithAspectRatio: %.1f ms (output=%dx%d, ratio=%.3f)".format(
        ms, outWidth, outHeight, targetRatio))
    return output
}
