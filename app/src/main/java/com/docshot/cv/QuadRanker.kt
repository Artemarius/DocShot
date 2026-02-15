package com.docshot.cv

import android.util.Log
import org.opencv.core.Point
import org.opencv.imgproc.Imgproc
import org.opencv.core.MatOfPoint2f
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.sqrt

private const val TAG = "DocShot:QuadRank"

// Scoring weights for quadrilateral ranking
private const val WEIGHT_AREA = 0.5
private const val WEIGHT_ANGLE = 0.5

/**
 * Result of quadrilateral ranking with scoring metadata.
 * @param quad Ordered corners [TL, TR, BR, BL], or null if no valid quad.
 * @param score Best candidate's score [0.0, 1.0].
 * @param candidateCount Number of valid (convex) candidates considered.
 * @param scoreMargin Difference between best and second-best scores.
 *   1.0 when only one candidate exists (unambiguous).
 *   Lower values indicate ambiguity between multiple similar candidates.
 */
data class QuadRankResult(
    val quad: List<Point>?,
    val score: Double,
    val candidateCount: Int,
    val scoreMargin: Double
)

/**
 * Ranks all candidate quadrilaterals and returns the best one with scoring metadata.
 * Scoring uses a weighted combination of:
 * - Area (largest preferred, normalized to image area)
 * - Angle regularity (corners close to 90° preferred)
 * Non-convex quads are rejected outright.
 *
 * The [scoreMargin] in the result indicates how clearly the best candidate
 * dominates. A low margin means multiple quads scored similarly — ambiguous.
 * A margin of 1.0 means only one valid candidate existed (unambiguous).
 *
 * @param candidates List of 4-point polygons to evaluate.
 * @param imageArea Total image area for normalization.
 */
fun rankQuads(candidates: List<List<Point>>, imageArea: Double): QuadRankResult {
    val start = System.nanoTime()

    var bestScore = -1.0
    var secondBestScore = -1.0
    var bestQuad: List<Point>? = null
    var convexCount = 0

    for (quad in candidates) {
        // Reject non-convex quadrilaterals
        val mat = MatOfPoint2f(*quad.toTypedArray())
        val convex = Imgproc.isContourConvex(
            org.opencv.core.MatOfPoint(*quad.toTypedArray())
        )
        mat.release()
        if (!convex) continue

        convexCount++

        val area = quadArea(quad)
        val areaScore = (area / imageArea).coerceIn(0.0, 1.0)

        // Angle regularity: how close each interior angle is to 90°
        val angleScore = angleRegularityScore(quad)

        val score = WEIGHT_AREA * areaScore + WEIGHT_ANGLE * angleScore
        if (score > bestScore) {
            secondBestScore = bestScore
            bestScore = score
            bestQuad = quad
        } else if (score > secondBestScore) {
            secondBestScore = score
        }
    }

    // Score margin: how clearly the best candidate stands out.
    // 1.0 = sole candidate (unambiguous), 0.0 = tied with runner-up.
    val scoreMargin = when {
        convexCount == 0 -> 0.0
        convexCount == 1 -> 1.0
        bestScore <= 0.0 -> 0.0
        else -> ((bestScore - secondBestScore) / bestScore).coerceIn(0.0, 1.0)
    }

    val ms = (System.nanoTime() - start) / 1_000_000.0
    Log.d(TAG, "rankQuads: %.1f ms (candidates=%d, convex=%d, bestScore=%.3f, margin=%.3f)".format(
        ms, candidates.size, convexCount, bestScore, scoreMargin))

    return QuadRankResult(
        quad = bestQuad?.let { orderCorners(it) },
        score = bestScore.coerceAtLeast(0.0),
        candidateCount = convexCount,
        scoreMargin = scoreMargin
    )
}

/**
 * Picks the best quadrilateral from a list of candidates.
 * Scoring uses a weighted combination of:
 * - Area (largest preferred, normalized to image area)
 * - Angle regularity (corners close to 90° preferred)
 * Non-convex quads are rejected outright.
 * Returns ordered corners [TL, TR, BR, BL] or null if no valid quad found.
 */
fun bestQuad(candidates: List<List<Point>>, imageArea: Double): List<Point>? {
    return rankQuads(candidates, imageArea).quad
}

/**
 * Scores a single quadrilateral using the same weighted combination as [bestQuad]:
 * area (normalized to image area) and angle regularity.
 * Returns a value in [0.0, 1.0] where higher is better.
 *
 * @param quad 4-point polygon (any winding order).
 * @param imageArea total image area used for normalization.
 */
fun scoreQuad(quad: List<Point>, imageArea: Double): Double {
    require(quad.size == 4) { "Expected 4 points, got ${quad.size}" }
    require(imageArea > 0.0) { "imageArea must be positive" }

    val areaScore = (quadArea(quad) / imageArea).coerceIn(0.0, 1.0)
    val angleScore = angleRegularityScore(quad)
    return WEIGHT_AREA * areaScore + WEIGHT_ANGLE * angleScore
}

/**
 * Computes the area of a quadrilateral using the shoelace formula.
 */
internal fun quadArea(points: List<Point>): Double {
    val n = points.size
    var area = 0.0
    for (i in 0 until n) {
        val j = (i + 1) % n
        area += points[i].x * points[j].y
        area -= points[j].x * points[i].y
    }
    return abs(area) / 2.0
}

/**
 * Scores how close the four interior angles are to 90°.
 * Returns 1.0 for a perfect rectangle, approaches 0.0 for degenerate shapes.
 */
internal fun angleRegularityScore(quad: List<Point>): Double {
    require(quad.size == 4) { "Expected 4 points, got ${quad.size}" }
    var totalDeviation = 0.0
    for (i in 0 until 4) {
        val prev = quad[(i + 3) % 4]
        val curr = quad[i]
        val next = quad[(i + 1) % 4]
        val angle = interiorAngleDeg(prev, curr, next)
        totalDeviation += abs(angle - 90.0)
    }
    // Max total deviation is 360° (degenerate), normalize so 0° deviation = 1.0
    return (1.0 - totalDeviation / 360.0).coerceIn(0.0, 1.0)
}

/**
 * Computes the interior angle at vertex `b` in degrees, given points a-b-c.
 */
private fun interiorAngleDeg(a: Point, b: Point, c: Point): Double {
    val ba = doubleArrayOf(a.x - b.x, a.y - b.y)
    val bc = doubleArrayOf(c.x - b.x, c.y - b.y)
    val dot = ba[0] * bc[0] + ba[1] * bc[1]
    val magBa = sqrt(ba[0] * ba[0] + ba[1] * ba[1])
    val magBc = sqrt(bc[0] * bc[0] + bc[1] * bc[1])
    if (magBa == 0.0 || magBc == 0.0) return 0.0
    val cosAngle = (dot / (magBa * magBc)).coerceIn(-1.0, 1.0)
    return Math.toDegrees(kotlin.math.acos(cosAngle))
}

/**
 * Orders 4 points into consistent [TL, TR, BR, BL] order.
 * Algorithm: TL has smallest (x+y), BR has largest (x+y),
 * TR has smallest (y-x), BL has largest (y-x).
 */
fun orderCorners(points: List<Point>): List<Point> {
    require(points.size == 4) { "Expected 4 points, got ${points.size}" }

    val sums = DoubleArray(4) { points[it].x + points[it].y }
    val diffs = DoubleArray(4) { points[it].y - points[it].x }

    val tl = points[sums.indexOfMin()]
    val br = points[sums.indexOfMax()]
    val tr = points[diffs.indexOfMin()]
    val bl = points[diffs.indexOfMax()]

    return listOf(tl, tr, br, bl)
}

private fun DoubleArray.indexOfMin(): Int {
    var minIdx = 0
    for (i in 1 until size) {
        if (this[i] < this[minIdx]) minIdx = i
    }
    return minIdx
}

private fun DoubleArray.indexOfMax(): Int {
    var maxIdx = 0
    for (i in 1 until size) {
        if (this[i] > this[maxIdx]) maxIdx = i
    }
    return maxIdx
}
