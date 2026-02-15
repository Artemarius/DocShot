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

/**
 * Estimates the document aspect ratio from detected corners, optionally
 * using camera intrinsics for homography-based verification.
 *
 * Algorithm:
 * 1. Compute raw ratio from quad edges
 * 2. Find known formats within [SNAP_THRESHOLD]
 * 3. If single clear match -> snap with Gaussian confidence
 * 4. If ambiguous and intrinsics available -> use [homographyError] to disambiguate
 * 5. If no candidates -> return raw ratio with null format
 */
fun estimateAspectRatio(
    corners: List<Point>,
    intrinsics: CameraIntrinsics? = null
): AspectRatioEstimate {
    require(corners.size == 4) { "Expected 4 corners, got ${corners.size}" }

    val rawRatio = computeRawRatio(corners)

    // Find all known formats within snap threshold, sorted by distance
    data class Candidate(val format: KnownFormat, val dist: Double)
    val candidates = KNOWN_FORMATS
        .map { Candidate(it, abs(rawRatio - it.ratio)) }
        .filter { it.dist <= SNAP_THRESHOLD }
        .sortedBy { it.dist }

    if (candidates.isEmpty()) {
        return AspectRatioEstimate(
            estimatedRatio = rawRatio,
            matchedFormat = null,
            confidence = 0.5
        )
    }

    // Single clear match or clear winner (second best is much worse)
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
        // Use homography verification to disambiguate
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

    // No intrinsics — pick closest by distance
    val best = candidates[0]
    val conf = exp(-best.dist * best.dist / (2.0 * SNAP_SIGMA * SNAP_SIGMA))
    return AspectRatioEstimate(
        estimatedRatio = best.format.ratio,
        matchedFormat = best.format,
        confidence = conf.coerceIn(0.0, 1.0) * 0.8 // lower confidence without homography verification
    )
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
