package com.docshot.cv

import android.util.Log
import org.opencv.core.Point
import org.opencv.imgproc.Imgproc
import org.opencv.core.MatOfPoint2f
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.exp
import kotlin.math.min
import kotlin.math.sqrt

private const val TAG = "DocShot:QuadRank"

// Scoring weights for quadrilateral ranking
private const val WEIGHT_AREA = 0.4
private const val WEIGHT_ANGLE = 0.4
private const val WEIGHT_ASPECT = 0.2

/**
 * Known document aspect ratios (width/height, always <= 1.0 so we use min/max).
 * Used for aspect-ratio scoring to prefer shapes that match real document formats.
 */
private val KNOWN_ASPECT_RATIOS = doubleArrayOf(
    1.0 / 1.414,  // A4 / A-series (ISO 216)
    1.0 / 1.294,  // US Letter (8.5 x 11)
    1.0 / 3.0,    // Receipt (mid-range, covers 1:2.5 to 1:3.5)
    1.0 / 1.75,   // Business card (3.5 x 2 inches)
    1.0 / 1.586,  // ID card / credit card (CR-80: 85.6 x 53.98 mm)
    1.0,          // Square
)

/**
 * Gaussian falloff sigma for aspect ratio scoring. Controls how quickly
 * the score drops as the aspect ratio deviates from a known format.
 * A difference of 0.15 from any known ratio drops the score significantly.
 */
private const val ASPECT_SIGMA = 0.10

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
 * - Area (40%, largest preferred, normalized to image area)
 * - Angle regularity (40%, corners close to 90° preferred)
 * - Aspect ratio (20%, matches known document formats preferred)
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

        // Aspect ratio: how well the quad matches known document formats
        val aspectScore = aspectRatioScore(quad)

        val score = WEIGHT_AREA * areaScore + WEIGHT_ANGLE * angleScore + WEIGHT_ASPECT * aspectScore
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
 * - Aspect ratio (matches known document formats preferred)
 * Non-convex quads are rejected outright.
 * Returns ordered corners [TL, TR, BR, BL] or null if no valid quad found.
 */
fun bestQuad(candidates: List<List<Point>>, imageArea: Double): List<Point>? {
    return rankQuads(candidates, imageArea).quad
}

/**
 * Scores a single quadrilateral using the same weighted combination as [bestQuad]:
 * area (normalized to image area), angle regularity, and aspect ratio.
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
    val aspectScore = aspectRatioScore(quad)
    return WEIGHT_AREA * areaScore + WEIGHT_ANGLE * angleScore + WEIGHT_ASPECT * aspectScore
}

/**
 * Scores how well the quadrilateral's aspect ratio matches known document formats.
 * Uses a Gaussian falloff from the closest known ratio.
 *
 * Returns 1.0 for a perfect match with any known format, drops rapidly
 * for ratios that don't match any standard document size.
 *
 * @param quad 4-point polygon (any winding order).
 * @return Score in [0.0, 1.0].
 */
internal fun aspectRatioScore(quad: List<Point>): Double {
    require(quad.size == 4) { "Expected 4 points, got ${quad.size}" }

    // Compute edge lengths (4 edges of the quad)
    val edgeLengths = DoubleArray(4) { i ->
        val next = (i + 1) % 4
        distance(quad[i], quad[next])
    }

    // Average opposite edge pairs to get width and height
    val side1 = (edgeLengths[0] + edgeLengths[2]) / 2.0
    val side2 = (edgeLengths[1] + edgeLengths[3]) / 2.0

    if (side1 <= 0.0 || side2 <= 0.0) return 0.0

    // Normalize: aspect ratio as min/max to be orientation-independent (always <= 1.0)
    val ratio = min(side1, side2) / maxOf(side1, side2)

    // Find minimum distance to any known ratio
    var minDist = Double.MAX_VALUE
    for (known in KNOWN_ASPECT_RATIOS) {
        val dist = abs(ratio - known)
        if (dist < minDist) minDist = dist
    }

    // Gaussian falloff: score = exp(-dist² / (2 * sigma²))
    return exp(-minDist * minDist / (2.0 * ASPECT_SIGMA * ASPECT_SIGMA))
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
 * Euclidean distance between two points.
 */
internal fun distance(a: Point, b: Point): Double {
    val dx = a.x - b.x
    val dy = a.y - b.y
    return sqrt(dx * dx + dy * dy)
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
 * Orders 4 points into consistent [TL, TR, BR, BL] clockwise order.
 *
 * Uses centroid-based angular sorting, which is robust at all document
 * rotation angles including ~45° where the sum/difference method degenerates
 * (two corners can share the same x+y or y-x, causing duplicate assignments).
 *
 * Algorithm:
 * 1. Compute centroid of the 4 points.
 * 2. Sort by atan2 angle from centroid (ascending = clockwise in image coords).
 * 3. Rotate the cycle so TL (smallest x+y) comes first.
 */
fun orderCorners(points: List<Point>): List<Point> {
    require(points.size == 4) { "Expected 4 points, got ${points.size}" }

    // Centroid
    val cx = points.sumOf { it.x } / 4.0
    val cy = points.sumOf { it.y } / 4.0

    // Sort by angle from centroid — ascending atan2 gives CW order
    // in image coordinates (y-axis points downward).
    val sorted = points.sortedBy { atan2(it.y - cy, it.x - cx) }

    // Rotate so TL (smallest x+y sum) comes first
    val tlIdx = sorted.indices.minBy { sorted[it].x + sorted[it].y }

    return List(4) { sorted[(tlIdx + it) % 4] }
}
