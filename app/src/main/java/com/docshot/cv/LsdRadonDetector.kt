package com.docshot.cv

import android.graphics.PointF
import android.util.Log
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

private const val TAG = "DocShot:LsdRadon"

// ---------------------------------------------------------------------------
// B1: LSD Segment Detection
// ---------------------------------------------------------------------------

/**
 * Minimum segment length as a fraction of the image's longest dimension.
 * Segments shorter than 5% of max(width, height) are rejected — too short
 * to be a document edge and they add noise to clustering.
 */
private const val MIN_LENGTH_FRACTION = 0.05f

/**
 * A line segment detected by OpenCV's Line Segment Detector (LSD).
 *
 * Coordinates are in image space (origin top-left, x right, y down).
 * [width] is the NFA-validated line width reported by LSD — wider lines
 * have stronger gradient support.
 */
data class LineSegment(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val width: Float
) {
    /** Euclidean length of the segment in pixels. */
    val length: Float
        get() {
            val dx = x2 - x1
            val dy = y2 - y1
            return sqrt(dx * dx + dy * dy)
        }

    /**
     * Angle of the segment in degrees, normalized to [0, 180).
     * 0 = horizontal (pointing right), 90 = vertical (pointing down).
     * The range is [0, 180) because a line segment has no direction — only orientation.
     */
    val angle: Float
        get() {
            val dx = x2 - x1
            val dy = y2 - y1
            // atan2 returns [-PI, PI]. We map to [0, 180) for undirected orientation.
            var deg = Math.toDegrees(atan2(dy.toDouble(), dx.toDouble())).toFloat()
            if (deg < 0f) deg += 180f
            if (deg >= 180f) deg -= 180f
            return deg
        }

    /** Midpoint of the segment. */
    val midpoint: PointF
        get() = PointF((x1 + x2) / 2f, (y1 + y2) / 2f)
}

/**
 * Detects line segments in a grayscale image using OpenCV's LSD algorithm.
 *
 * LSD (Line Segment Detector) operates on the gradient direction field and uses
 * the NFA (Number of False Alarms) framework to validate detections — it does NOT
 * require Canny binarization, making it capable of detecting much fainter edges.
 *
 * With `quant=1.0` (half of the default 2.0), the quantization of gradient angles
 * is finer, halving the effective detection threshold from ~5.2 to ~2.6 gradient
 * units. This is critical for ultra-low-contrast white-on-white scenes.
 *
 * @param gray Single-channel 8-bit grayscale image. Not modified.
 * @return List of detected segments passing the minimum length filter, sorted by
 *   length descending (longest first). Empty if no segments found.
 */
fun detectSegments(gray: Mat): List<LineSegment> {
    require(gray.channels() == 1) { "Expected single-channel input, got ${gray.channels()}" }
    require(!gray.empty()) { "Input Mat is empty" }

    val start = System.nanoTime()
    val imageWidth = gray.cols()
    val imageHeight = gray.rows()
    val minLength = max(imageWidth, imageHeight) * MIN_LENGTH_FRACTION

    // LSD parameters (positional, matching createLineSegmentDetector signature):
    //   refineType  = LSD_REFINE_STD  — standard refinement (adjust endpoints)
    //   scale       = 0.8             — image scale for the Gaussian pyramid
    //   sigma_scale = 0.6             — sigma = sigma_scale / scale for Gaussian
    //   quant       = 1.0             — CRITICAL: halves gradient angle quantization
    //                                   (default 2.0 → ~5.2 unit threshold;
    //                                    1.0 → ~2.6 unit threshold)
    //   ang_th      = 22.5            — gradient angle tolerance in degrees
    //   log_eps     = 0.0             — log10(NFA) detection threshold
    //   density_th  = 0.7             — minimum density of aligned points in a region
    //   n_bins      = 1024            — number of bins for gradient angle histogram
    val lsd = Imgproc.createLineSegmentDetector(
        Imgproc.LSD_REFINE_STD,
        0.8,
        0.6,
        1.0,   // quant=1.0: ~2.6 unit gradient detection threshold
        22.5,
        0.0,
        0.7,
        1024
    )

    val lines = Mat()
    try {
        lsd.detect(gray, lines)

        if (lines.empty()) {
            val ms = (System.nanoTime() - start) / 1_000_000.0
            Log.d(TAG, "detectSegments: %.1f ms — no segments detected".format(ms))
            return emptyList()
        }

        // LSD output: N x 1 Mat of type CV_32FC4 or N x 4 of CV_32F,
        // but with width column appended it's actually 5 columns.
        // Each row: [x1, y1, x2, y2] (width is separate in the detect overload).
        // OpenCV's Kotlin/Java binding returns lines as rows of 4 floats.
        val segmentCount = lines.rows()
        val segments = ArrayList<LineSegment>(segmentCount)

        for (i in 0 until segmentCount) {
            // lines is N x 1 x CV_32FC4, so get(row, 0) returns [x1, y1, x2, y2]
            val data = lines.get(i, 0) ?: continue
            if (data.size < 4) continue

            val seg = LineSegment(
                x1 = data[0].toFloat(),
                y1 = data[1].toFloat(),
                x2 = data[2].toFloat(),
                y2 = data[3].toFloat(),
                // LSD detect() without the width output Mat doesn't provide width;
                // use 1.0 as default (all segments passed NFA validation)
                width = if (data.size >= 5) data[4].toFloat() else 1.0f
            )

            if (seg.length >= minLength) {
                segments.add(seg)
            }
        }

        // Sort by length descending — longest segments are most likely document edges
        segments.sortByDescending { it.length }

        val ms = (System.nanoTime() - start) / 1_000_000.0
        Log.d(TAG, "detectSegments: %.1f ms (%d raw, %d after length filter >= %.0fpx)".format(
            ms, segmentCount, segments.size, minLength))

        return segments
    } finally {
        lines.release()
        // LSD detector holds internal state — release to free native memory
    }
}

// ---------------------------------------------------------------------------
// B2: Segment Clustering
// ---------------------------------------------------------------------------

/**
 * Edge orientation classification based on segment angle.
 * Horizontal: 0-45 or 135-180 degrees (near-horizontal lines).
 * Vertical: 45-135 degrees (near-vertical lines).
 */
enum class EdgeOrientation {
    HORIZONTAL,
    VERTICAL
}

/**
 * A cluster of collinear line segments representing a candidate document edge.
 *
 * Multiple LSD segments that share roughly the same angle and perpendicular offset
 * (rho) from the image center are merged into one cluster. The cluster's aggregate
 * length and position determine its viability as a document edge.
 *
 * @param segments The constituent line segments in this cluster.
 * @param angle Average angle of the cluster in degrees [0, 180).
 * @param rho Average signed perpendicular distance from the image center to the
 *   cluster line, in pixels. Positive = below/right of center.
 * @param totalLength Sum of all segment lengths in this cluster, in pixels.
 * @param isHorizontal True if this cluster represents a near-horizontal edge.
 */
data class EdgeCluster(
    val segments: List<LineSegment>,
    val angle: Float,
    val rho: Float,
    val totalLength: Float,
    val isHorizontal: Boolean
) {
    /**
     * Start point of the cluster — the extreme endpoint projected onto the cluster
     * line direction, closest to the top-left of the image.
     * Computed from all segment endpoints projected along the cluster direction.
     */
    val startPoint: PointF
        get() {
            val rad = Math.toRadians(angle.toDouble())
            val dx = cos(rad).toFloat()
            val dy = sin(rad).toFloat()
            // Project all endpoints onto the cluster direction, find minimum
            var minProj = Float.MAX_VALUE
            var bestPoint = segments.first().let { PointF(it.x1, it.y1) }
            for (seg in segments) {
                for (pt in listOf(PointF(seg.x1, seg.y1), PointF(seg.x2, seg.y2))) {
                    val proj = pt.x * dx + pt.y * dy
                    if (proj < minProj) {
                        minProj = proj
                        bestPoint = pt
                    }
                }
            }
            return bestPoint
        }

    /**
     * End point of the cluster — the extreme endpoint projected onto the cluster
     * line direction, farthest from the top-left of the image.
     */
    val endPoint: PointF
        get() {
            val rad = Math.toRadians(angle.toDouble())
            val dx = cos(rad).toFloat()
            val dy = sin(rad).toFloat()
            // Project all endpoints onto the cluster direction, find maximum
            var maxProj = -Float.MAX_VALUE
            var bestPoint = segments.first().let { PointF(it.x1, it.y1) }
            for (seg in segments) {
                for (pt in listOf(PointF(seg.x1, seg.y1), PointF(seg.x2, seg.y2))) {
                    val proj = pt.x * dx + pt.y * dy
                    if (proj > maxProj) {
                        maxProj = proj
                        bestPoint = pt
                    }
                }
            }
            return bestPoint
        }
}

/** Angle tolerance in degrees for merging segments into the same cluster. */
private const val CLUSTER_ANGLE_TOLERANCE_DEG = 8.0f

/** Perpendicular distance tolerance in pixels for merging segments into the same cluster. */
private const val CLUSTER_RHO_TOLERANCE_PX = 15.0f

/**
 * Minimum total cluster length as a fraction of the image diagonal.
 * A cluster must represent at least 20% of the diagonal to be a plausible document edge.
 */
private const val MIN_CLUSTER_LENGTH_FRACTION = 0.20f

/** Maximum number of clusters per orientation to return (limits combinatorial explosion in B3). */
private const val MAX_CLUSTERS_PER_ORIENTATION = 6

/**
 * Groups detected line segments into edge clusters representing candidate document edges.
 *
 * The algorithm:
 * 1. Classifies each segment as horizontal or vertical by angle.
 * 2. Within each orientation group, sorts segments by angle.
 * 3. Greedily merges segments within [CLUSTER_ANGLE_TOLERANCE_DEG] degrees
 *    and [CLUSTER_RHO_TOLERANCE_PX] perpendicular distance of each other.
 * 4. Filters clusters by minimum total length (20% of image diagonal).
 * 5. Returns the top clusters sorted by total length (max 6 per orientation).
 *
 * @param segments Line segments from [detectSegments]. Order does not matter.
 * @param imageWidth Width of the source image in pixels.
 * @param imageHeight Height of the source image in pixels.
 * @return List of edge clusters sorted by total length descending, at most
 *   [MAX_CLUSTERS_PER_ORIENTATION] horizontal + [MAX_CLUSTERS_PER_ORIENTATION] vertical.
 *   Empty if no clusters meet the minimum length threshold.
 */
fun clusterSegments(
    segments: List<LineSegment>,
    imageWidth: Int,
    imageHeight: Int
): List<EdgeCluster> {
    require(imageWidth > 0) { "imageWidth must be positive, got $imageWidth" }
    require(imageHeight > 0) { "imageHeight must be positive, got $imageHeight" }

    if (segments.isEmpty()) return emptyList()

    val start = System.nanoTime()
    val imageCenter = PointF(imageWidth / 2f, imageHeight / 2f)
    val diagonal = sqrt((imageWidth.toFloat() * imageWidth + imageHeight.toFloat() * imageHeight).toDouble()).toFloat()
    val minClusterLength = diagonal * MIN_CLUSTER_LENGTH_FRACTION

    // Step 1: Classify segments by orientation
    val horizontalSegments = mutableListOf<LineSegment>()
    val verticalSegments = mutableListOf<LineSegment>()

    for (seg in segments) {
        when (classifyOrientation(seg.angle)) {
            EdgeOrientation.HORIZONTAL -> horizontalSegments.add(seg)
            EdgeOrientation.VERTICAL -> verticalSegments.add(seg)
        }
    }

    // Step 2-3: Cluster each orientation group
    val hClusters = clusterOrientationGroup(
        segments = horizontalSegments,
        imageCenter = imageCenter,
        isHorizontal = true
    )
    val vClusters = clusterOrientationGroup(
        segments = verticalSegments,
        imageCenter = imageCenter,
        isHorizontal = false
    )

    // Step 4: Filter by minimum total length
    val filteredH = hClusters.filter { it.totalLength >= minClusterLength }
    val filteredV = vClusters.filter { it.totalLength >= minClusterLength }

    // Step 5: Sort by total length descending, cap at MAX_CLUSTERS_PER_ORIENTATION each
    val topH = filteredH
        .sortedByDescending { it.totalLength }
        .take(MAX_CLUSTERS_PER_ORIENTATION)
    val topV = filteredV
        .sortedByDescending { it.totalLength }
        .take(MAX_CLUSTERS_PER_ORIENTATION)

    val result = (topH + topV).sortedByDescending { it.totalLength }

    val ms = (System.nanoTime() - start) / 1_000_000.0
    Log.d(TAG, "clusterSegments: %.1f ms (segs=%d, H=%d/%d, V=%d/%d, minLen=%.0f)".format(
        ms, segments.size,
        topH.size, horizontalSegments.size,
        topV.size, verticalSegments.size,
        minClusterLength
    ))

    return result
}

/**
 * Classifies a segment's orientation based on its angle.
 *
 * Horizontal: angle in [0, 45) or [135, 180) — lines that are more horizontal than vertical.
 * Vertical: angle in [45, 135) — lines that are more vertical than horizontal.
 *
 * The 45-degree boundary is where horizontal and vertical projections are equal.
 */
private fun classifyOrientation(angleDeg: Float): EdgeOrientation {
    return if (angleDeg < 45f || angleDeg >= 135f) {
        EdgeOrientation.HORIZONTAL
    } else {
        EdgeOrientation.VERTICAL
    }
}

/**
 * Clusters segments of the same orientation using greedy merge.
 *
 * Sorts segments by angle, then iterates: each segment either joins an existing
 * cluster (if within angle and rho tolerance of the cluster's running average)
 * or starts a new cluster. This is O(N*K) where K is the number of clusters,
 * which is fast because K is small (typically < 20 for a document scene).
 */
private fun clusterOrientationGroup(
    segments: List<LineSegment>,
    imageCenter: PointF,
    isHorizontal: Boolean
): List<EdgeCluster> {
    if (segments.isEmpty()) return emptyList()

    // Sort by angle for greedy clustering — nearby angles are adjacent
    val sorted = segments.sortedBy { it.angle }

    // Each cluster is tracked as a mutable accumulator
    data class ClusterAccumulator(
        val members: MutableList<LineSegment>,
        var angleSum: Float,     // weighted by segment length
        var rhoSum: Float,       // weighted by segment length
        var totalLength: Float
    ) {
        val avgAngle: Float get() = if (totalLength > 0f) angleSum / totalLength else 0f
        val avgRho: Float get() = if (totalLength > 0f) rhoSum / totalLength else 0f
    }

    val clusters = mutableListOf<ClusterAccumulator>()

    for (seg in sorted) {
        val segRho = perpendicularDistance(seg, seg.angle, imageCenter)
        val segLen = seg.length

        // Try to merge into an existing cluster
        var merged = false
        for (cluster in clusters) {
            val angleDiff = angleDifference(seg.angle, cluster.avgAngle)
            val rhoDiff = abs(segRho - cluster.avgRho)

            if (angleDiff <= CLUSTER_ANGLE_TOLERANCE_DEG && rhoDiff <= CLUSTER_RHO_TOLERANCE_PX) {
                cluster.members.add(seg)
                cluster.angleSum += seg.angle * segLen
                cluster.rhoSum += segRho * segLen
                cluster.totalLength += segLen
                merged = true
                break
            }
        }

        if (!merged) {
            clusters.add(ClusterAccumulator(
                members = mutableListOf(seg),
                angleSum = seg.angle * segLen,
                rhoSum = segRho * segLen,
                totalLength = segLen
            ))
        }
    }

    return clusters.map { acc ->
        EdgeCluster(
            segments = acc.members.toList(),
            angle = acc.avgAngle,
            rho = acc.avgRho,
            totalLength = acc.totalLength,
            isHorizontal = isHorizontal
        )
    }
}

/**
 * Computes the angular difference between two angles in [0, 180), accounting
 * for the wraparound at 0/180 degrees (e.g., 2 deg and 178 deg are 4 deg apart).
 *
 * @return Difference in [0, 90] degrees.
 */
private fun angleDifference(a: Float, b: Float): Float {
    val diff = abs(a - b)
    return if (diff > 90f) 180f - diff else diff
}

/**
 * Computes the signed perpendicular distance from the image center to the line
 * defined by a segment, projected along the direction perpendicular to [referenceAngle].
 *
 * This is the "rho" in Hough-like parameterization: the perpendicular offset from
 * the image center to the closest point on the line. Segments on the same
 * document edge will have similar rho values regardless of where along the edge
 * they appear.
 *
 * @param segment The line segment to measure.
 * @param referenceAngle The angle (in degrees, [0, 180)) defining the line direction.
 *   The perpendicular distance is measured along the normal to this direction.
 * @param imageCenter The reference point (typically image center) for the distance.
 * @return Signed perpendicular distance in pixels. Positive values indicate the
 *   line is below/right of the center (depending on orientation).
 */
fun perpendicularDistance(
    segment: LineSegment,
    referenceAngle: Float,
    imageCenter: PointF
): Float {
    // Normal direction to the reference angle (perpendicular)
    val normalRad = Math.toRadians((referenceAngle + 90.0).toDouble())
    val nx = cos(normalRad).toFloat()
    val ny = sin(normalRad).toFloat()

    // Signed distance from image center to the segment's midpoint,
    // projected onto the normal direction
    val mid = segment.midpoint
    val dx = mid.x - imageCenter.x
    val dy = mid.y - imageCenter.y

    return dx * nx + dy * ny
}

// ---------------------------------------------------------------------------
// B3: Tier 1 — LSD Rectangle Formation
// ---------------------------------------------------------------------------

/**
 * Overflow margin for intersection points beyond the image boundary.
 * Documents partially outside the frame can have corners slightly beyond
 * the image edge — allow 5% overflow to handle this case.
 */
private const val BOUNDS_OVERFLOW_FRACTION = 0.05f

/** Minimum area of a valid quad as a fraction of image area. */
private const val MIN_QUAD_AREA_FRACTION = 0.10

/** Minimum interior angle (degrees) for a valid document quad. */
private const val MIN_INTERIOR_ANGLE_DEG = 60.0

/** Maximum interior angle (degrees) for a valid document quad. */
private const val MAX_INTERIOR_ANGLE_DEG = 120.0

/** Threshold for considering a homogeneous coordinate's w as "parallel" (no intersection). */
private const val PARALLEL_W_THRESHOLD = 1e-8

/**
 * Minimum confidence assigned to LSD Tier 1 detections. LSD-based detections
 * are slightly less confident than the main contour pipeline because they lack
 * edge density verification against Canny edges.
 */
private const val LSD_MIN_CONFIDENCE = 0.50

/** Maximum confidence cap for LSD Tier 1 detections. */
private const val LSD_MAX_CONFIDENCE = 0.85

/**
 * Converts an [EdgeCluster]'s line representation (angle + rho from image center)
 * to homogeneous line coefficients `(a, b, c)` where `ax + by + c = 0`.
 *
 * The cluster defines a line via:
 * - `angle`: direction of the line in degrees [0, 180)
 * - `rho`: signed perpendicular distance from the image center to the line
 *
 * The line normal is perpendicular to the line direction at angle `theta`:
 *   normal = (cos(theta + 90), sin(theta + 90)) = (-sin(theta), cos(theta))
 *
 * A point P on the line satisfies: `(P - center) . normal = rho`, which expands to:
 *   `a*x + b*y + c = 0` where `a = nx`, `b = ny`, `c = -(nx*cx + ny*cy + rho)`
 *
 * @param cluster The edge cluster to convert.
 * @param imageCenterX X coordinate of the image center.
 * @param imageCenterY Y coordinate of the image center.
 * @return Triple of (a, b, c) in the equation `ax + by + c = 0`.
 */
private fun clusterToLine(
    cluster: EdgeCluster,
    imageCenterX: Float,
    imageCenterY: Float
): Triple<Double, Double, Double> {
    val normalRad = Math.toRadians((cluster.angle + 90.0).toDouble())
    val a = cos(normalRad)
    val b = sin(normalRad)
    val c = -(a * imageCenterX + b * imageCenterY) - cluster.rho.toDouble()
    return Triple(a, b, c)
}

/**
 * Computes the intersection of two lines in homogeneous coordinates.
 *
 * Given line1 = (a1, b1, c1) and line2 = (a2, b2, c2), the intersection
 * in homogeneous coordinates is the cross product `line1 x line2`.
 * The result is converted to Euclidean coordinates by dividing by w.
 *
 * @return The intersection point, or null if the lines are parallel (w ≈ 0).
 */
private fun intersectLines(
    line1: Triple<Double, Double, Double>,
    line2: Triple<Double, Double, Double>
): Point? {
    val (a1, b1, c1) = line1
    val (a2, b2, c2) = line2

    // Cross product: point = line1 × line2
    val x = b1 * c2 - b2 * c1
    val y = c1 * a2 - c2 * a1
    val w = a1 * b2 - a2 * b1

    if (abs(w) < PARALLEL_W_THRESHOLD) return null

    return Point(x / w, y / w)
}

/**
 * Computes the interior angle at vertex `b` in degrees, given points a-b-c.
 * Uses the dot product formula: angle = acos(ba . bc / (|ba| * |bc|)).
 */
private fun lsdInteriorAngleDeg(a: Point, b: Point, c: Point): Double {
    val bax = a.x - b.x
    val bay = a.y - b.y
    val bcx = c.x - b.x
    val bcy = c.y - b.y

    val dot = bax * bcx + bay * bcy
    val magBa = sqrt(bax * bax + bay * bay)
    val magBc = sqrt(bcx * bcx + bcy * bcy)

    if (magBa == 0.0 || magBc == 0.0) return 0.0

    val cosAngle = (dot / (magBa * magBc)).coerceIn(-1.0, 1.0)
    return Math.toDegrees(acos(cosAngle))
}

/**
 * Checks if a quadrilateral is convex by verifying that the cross products
 * of consecutive edge vectors all have the same sign.
 *
 * @param corners 4 points in order (any consistent winding).
 * @return True if the quad is convex.
 */
private fun isConvex(corners: List<Point>): Boolean {
    require(corners.size == 4) { "Expected 4 corners, got ${corners.size}" }

    var positiveCount = 0
    var negativeCount = 0

    for (i in 0 until 4) {
        val curr = corners[i]
        val next = corners[(i + 1) % 4]
        val nextNext = corners[(i + 2) % 4]

        // Cross product of vectors (curr→next) × (next→nextNext)
        val cross = (next.x - curr.x) * (nextNext.y - next.y) -
                (next.y - curr.y) * (nextNext.x - next.x)

        if (cross > 0.0) positiveCount++
        else if (cross < 0.0) negativeCount++
    }

    // Convex iff all cross products have the same sign
    return positiveCount == 0 || negativeCount == 0
}

/**
 * Computes the area of a quadrilateral using the shoelace formula.
 */
private fun lsdQuadArea(corners: List<Point>): Double {
    val n = corners.size
    var area = 0.0
    for (i in 0 until n) {
        val j = (i + 1) % n
        area += corners[i].x * corners[j].y
        area -= corners[j].x * corners[i].y
    }
    return abs(area) / 2.0
}

/**
 * Tier 1 LSD rectangle formation: from clustered edges, select the best
 * 2 horizontal + 2 vertical clusters, compute their 4 intersection points,
 * validate the resulting quadrilateral, and score it.
 *
 * This is the fast path (~0.1ms) — purely geometric intersection of the
 * strongest LSD edge clusters. Works well when LSD detects all 4 document
 * edges with sufficient total segment length.
 *
 * @param clusters Edge clusters from [clusterSegments], containing both
 *   horizontal and vertical clusters sorted by total length.
 * @param imageWidth Width of the source image in pixels.
 * @param imageHeight Height of the source image in pixels.
 * @return [DocumentCorners] with ordered corners and confidence, or null
 *   if no valid quad can be formed from the clusters.
 */
fun detectRectangleTier1(
    clusters: List<EdgeCluster>,
    imageWidth: Int,
    imageHeight: Int
): DocumentCorners? {
    require(imageWidth > 0) { "imageWidth must be positive, got $imageWidth" }
    require(imageHeight > 0) { "imageHeight must be positive, got $imageHeight" }

    val start = System.nanoTime()

    // Separate clusters by orientation
    val hClusters = clusters.filter { it.isHorizontal }
    val vClusters = clusters.filter { !it.isHorizontal }

    if (hClusters.size < 2 || vClusters.size < 2) {
        val ms = (System.nanoTime() - start) / 1_000_000.0
        Log.d(TAG, "detectRectangleTier1: %.1f ms — insufficient clusters (H=%d, V=%d, need 2+2)".format(
            ms, hClusters.size, vClusters.size))
        return null
    }

    val imageCenterX = imageWidth / 2f
    val imageCenterY = imageHeight / 2f
    val imageArea = imageWidth.toDouble() * imageHeight
    val imagePerimeter = 2.0 * (imageWidth + imageHeight)
    val boundsMarginX = imageWidth * BOUNDS_OVERFLOW_FRACTION
    val boundsMarginY = imageHeight * BOUNDS_OVERFLOW_FRACTION

    // Pre-convert all clusters to homogeneous line representations
    val hLines = hClusters.map { clusterToLine(it, imageCenterX, imageCenterY) }
    val vLines = vClusters.map { clusterToLine(it, imageCenterX, imageCenterY) }

    var bestCorners: List<Point>? = null
    var bestScore = -1.0

    // Try all combinations of 2 horizontal × 2 vertical clusters.
    // At most 6C2 × 6C2 = 15 × 15 = 225 combinations, but typically 2-4
    // clusters per orientation → 1-6 × 1-6 = 1-36 combinations.
    for (hi in hClusters.indices) {
        for (hj in hi + 1 until hClusters.size) {
            for (vi in vClusters.indices) {
                for (vj in vi + 1 until vClusters.size) {
                    val candidate = evaluateQuadCandidate(
                        hLine1 = hLines[hi],
                        hLine2 = hLines[hj],
                        vLine1 = vLines[vi],
                        vLine2 = vLines[vj],
                        hCluster1 = hClusters[hi],
                        hCluster2 = hClusters[hj],
                        vCluster1 = vClusters[vi],
                        vCluster2 = vClusters[vj],
                        imageWidth = imageWidth,
                        imageHeight = imageHeight,
                        imageArea = imageArea,
                        imagePerimeter = imagePerimeter,
                        boundsMarginX = boundsMarginX,
                        boundsMarginY = boundsMarginY
                    )

                    if (candidate != null && candidate.second > bestScore) {
                        bestCorners = candidate.first
                        bestScore = candidate.second
                    }
                }
            }
        }
    }

    val ms = (System.nanoTime() - start) / 1_000_000.0

    if (bestCorners == null) {
        Log.d(TAG, "detectRectangleTier1: %.1f ms — no valid quad from %d H × %d V combinations".format(
            ms, hClusters.size, vClusters.size))
        return null
    }

    // Map score [0, 1] to confidence [LSD_MIN_CONFIDENCE, LSD_MAX_CONFIDENCE]
    val confidence = LSD_MIN_CONFIDENCE + bestScore * (LSD_MAX_CONFIDENCE - LSD_MIN_CONFIDENCE)

    Log.d(TAG, "detectRectangleTier1: %.1f ms, score=%.3f, confidence=%.2f".format(
        ms, bestScore, confidence))

    return DocumentCorners(
        corners = bestCorners,
        detectionMs = ms,
        confidence = confidence
    )
}

/**
 * Evaluates a single quad candidate formed by 2 horizontal + 2 vertical cluster lines.
 * Computes the 4 intersection points, validates geometry, and scores the quad.
 *
 * @return Pair of (ordered corners, score) if valid, or null if any validation fails.
 */
private fun evaluateQuadCandidate(
    hLine1: Triple<Double, Double, Double>,
    hLine2: Triple<Double, Double, Double>,
    vLine1: Triple<Double, Double, Double>,
    vLine2: Triple<Double, Double, Double>,
    hCluster1: EdgeCluster,
    hCluster2: EdgeCluster,
    vCluster1: EdgeCluster,
    vCluster2: EdgeCluster,
    imageWidth: Int,
    imageHeight: Int,
    imageArea: Double,
    imagePerimeter: Double,
    boundsMarginX: Float,
    boundsMarginY: Float
): Pair<List<Point>, Double>? {
    // Compute 4 intersections: each H line meets each V line
    val p1 = intersectLines(hLine1, vLine1) ?: return null
    val p2 = intersectLines(hLine1, vLine2) ?: return null
    val p3 = intersectLines(hLine2, vLine1) ?: return null
    val p4 = intersectLines(hLine2, vLine2) ?: return null

    val rawCorners = listOf(p1, p2, p3, p4)

    // Validate: all intersection points within image bounds (with overflow margin)
    for (pt in rawCorners) {
        if (pt.x < -boundsMarginX || pt.x > imageWidth + boundsMarginX) return null
        if (pt.y < -boundsMarginY || pt.y > imageHeight + boundsMarginY) return null
    }

    // Order corners TL, TR, BR, BL using centroid-based angular sorting
    // (same algorithm as QuadRanker.orderCorners)
    val ordered = orderCorners(rawCorners)

    // Validate convexity — a valid document quad must be convex
    if (!isConvex(ordered)) return null

    // Validate area — must be at least MIN_QUAD_AREA_FRACTION of image area
    val area = lsdQuadArea(ordered)
    if (area < imageArea * MIN_QUAD_AREA_FRACTION) return null

    // Validate interior angles — all must be in [60°, 120°] for a reasonable document
    for (i in 0 until 4) {
        val prev = ordered[(i + 3) % 4]
        val curr = ordered[i]
        val next = ordered[(i + 1) % 4]
        val angle = lsdInteriorAngleDeg(prev, curr, next)
        if (angle < MIN_INTERIOR_ANGLE_DEG || angle > MAX_INTERIOR_ANGLE_DEG) return null
    }

    // Score the valid quad
    val score = scoreQuadCandidate(
        ordered = ordered,
        hCluster1 = hCluster1,
        hCluster2 = hCluster2,
        vCluster1 = vCluster1,
        vCluster2 = vCluster2,
        imageArea = imageArea,
        imagePerimeter = imagePerimeter
    )

    return Pair(ordered, score)
}

/**
 * Scores a valid quad candidate using three complementary signals:
 *
 * - **LSD evidence (40%):** Total segment length across the 4 clusters,
 *   normalized by the image perimeter. Higher values mean stronger gradient
 *   support along the quad edges. Capped at 1.0 (full-perimeter coverage).
 *
 * - **Area ratio (30%):** Quad area relative to the image area. Larger
 *   documents that fill the frame score higher.
 *
 * - **Angle regularity (30%):** How close the 4 interior angles are to 90°.
 *   Perfect rectangles score 1.0; heavily skewed quads score lower. Uses the
 *   same deviation-based formula as QuadRanker.angleRegularityScore.
 *
 * @return Score in [0.0, 1.0].
 */
private fun scoreQuadCandidate(
    ordered: List<Point>,
    hCluster1: EdgeCluster,
    hCluster2: EdgeCluster,
    vCluster1: EdgeCluster,
    vCluster2: EdgeCluster,
    imageArea: Double,
    imagePerimeter: Double
): Double {
    // LSD evidence: sum of total segment lengths across all 4 clusters,
    // normalized by image perimeter. Full-perimeter coverage scores 1.0.
    val totalLsdLength = (hCluster1.totalLength + hCluster2.totalLength +
            vCluster1.totalLength + vCluster2.totalLength).toDouble()
    val lsdEvidence = (totalLsdLength / imagePerimeter).coerceIn(0.0, 1.0)

    // Area ratio: quad area / image area
    val area = lsdQuadArea(ordered)
    val areaRatio = (area / imageArea).coerceIn(0.0, 1.0)

    // Angle regularity: how close interior angles are to 90°
    // Same formula as QuadRanker — total deviation from 90° / 360°
    var totalAngleDeviation = 0.0
    for (i in 0 until 4) {
        val prev = ordered[(i + 3) % 4]
        val curr = ordered[i]
        val next = ordered[(i + 1) % 4]
        val angle = lsdInteriorAngleDeg(prev, curr, next)
        totalAngleDeviation += abs(angle - 90.0)
    }
    val angleRegularity = (1.0 - totalAngleDeviation / 360.0).coerceIn(0.0, 1.0)

    return lsdEvidence * 0.4 + areaRatio * 0.3 + angleRegularity * 0.3
}

// ---------------------------------------------------------------------------
// Top-level LSD+Radon Detection
// ---------------------------------------------------------------------------

/**
 * Detects a document in a grayscale image using the LSD + Radon cascade.
 *
 * This is a separate detection path from the Canny/contour pipeline. It
 * operates on the raw gradient field (via LSD) rather than binarized edges,
 * making it capable of detecting much fainter document boundaries (down to
 * ~2.6 gradient units vs ~10+ for Canny-based detection).
 *
 * Three-tier cascade with progressive fallback:
 * - **Tier 1:** LSD fast path — cluster LSD segments into 2H+2V edges, intersect (~0.1ms)
 * - **Tier 2:** Corner-constrained Radon — rescue partial LSD detections (2-3 edges) (~2ms)
 * - **Tier 3:** Joint Radon rectangle fit — full Radon scan when LSD gives nothing (~4ms)
 *
 * @param gray Single-channel 8-bit grayscale image. Not modified.
 * @param imageWidth Width of the source image in pixels.
 * @param imageHeight Height of the source image in pixels.
 * @return [DocumentCorners] with ordered corners and confidence, or null
 *   if no valid document quad was found.
 */
fun detectDocumentLsd(
    gray: Mat,
    imageWidth: Int,
    imageHeight: Int
): DocumentCorners? {
    require(gray.channels() == 1) { "Expected single-channel input, got ${gray.channels()}" }
    require(!gray.empty()) { "Input Mat is empty" }
    require(imageWidth > 0) { "imageWidth must be positive, got $imageWidth" }
    require(imageHeight > 0) { "imageHeight must be positive, got $imageHeight" }

    val start = System.nanoTime()

    // Tier 1: LSD fast path — detect segments, cluster, form rectangle
    val segments = detectSegments(gray)
    var tier1Result: DocumentCorners? = null

    // Hoist clusters to function scope so Tier 2 can access them if Tier 1 fails
    val clusters: List<EdgeCluster> = if (segments.isNotEmpty()) {
        clusterSegments(
            segments = segments,
            imageWidth = imageWidth,
            imageHeight = imageHeight
        )
    } else {
        emptyList()
    }

    if (clusters.isNotEmpty()) {
        tier1Result = detectRectangleTier1(
            clusters = clusters,
            imageWidth = imageWidth,
            imageHeight = imageHeight
        )
    }

    if (tier1Result != null) {
        val ms = (System.nanoTime() - start) / 1_000_000.0
        val result = tier1Result.copy(detectionMs = ms)
        Log.d(TAG, "detectDocumentLsd: %.1f ms, Tier 1 success (confidence=%.2f)".format(
            ms, result.confidence))
        return result
    }

    // Tier 2: Corner-constrained Radon — rescue partial LSD detections (2-3 edges).
    // When Tier 1 fails but LSD found >= 2 clusters, use known edge constraints
    // to search for missing edges via restricted Radon accumulation (~2ms).
    if (clusters.size >= 2) {
        val tier2Result = detectRectangleTier2(
            gray = gray,
            clusters = clusters,
            imageWidth = imageWidth,
            imageHeight = imageHeight
        )
        if (tier2Result != null) {
            val ms = (System.nanoTime() - start) / 1_000_000.0
            val result = tier2Result.copy(detectionMs = ms)
            Log.d(TAG, "detectDocumentLsd: %.1f ms, Tier 2 success (confidence=%.2f)".format(
                ms, result.confidence))
            return result
        }
    }

    // Tier 3: Joint Radon rectangle fit — last resort when LSD finds 0-1 usable edges.
    // Performs a full restricted Radon scan over 9 candidate orientations to find
    // a rectangle directly from gradient accumulation.
    val tier3Start = System.nanoTime()
    val tier3Result = detectRectangleTier3(
        gray = gray,
        imageWidth = imageWidth,
        imageHeight = imageHeight
    )
    val tier3Ms = (System.nanoTime() - tier3Start) / 1_000_000.0

    if (tier3Result != null) {
        val ms = (System.nanoTime() - start) / 1_000_000.0
        val result = tier3Result.copy(detectionMs = ms)
        Log.d(TAG, "detectDocumentLsd: %.1f ms (Tier 3: %.1f ms), Tier 3 success (confidence=%.2f)".format(
            ms, tier3Ms, result.confidence))
        return result
    }

    val ms = (System.nanoTime() - start) / 1_000_000.0
    Log.d(TAG, "detectDocumentLsd: %.1f ms — no document found (all tiers exhausted)".format(ms))
    return null
}

// ---------------------------------------------------------------------------
// B4: Gradient Density Verification
// ---------------------------------------------------------------------------

/**
 * Reference gradient value for normalizing per-side scores.
 *
 * A moderate document edge on a typical scene produces ~20-50 units of
 * perpendicular gradient. For ultra-low-contrast white-on-white scenes,
 * even 3-5 units of average perpendicular gradient is meaningful (SNR
 * improves with accumulation along the edge: sigma_noise/sqrt(N) shrinks
 * as N increases). Setting the reference at 20.0 means:
 *   - A clear edge (~40 units) scores 1.0 (clamped)
 *   - A moderate edge (~20 units) scores 1.0
 *   - A faint white-on-white edge (~5 units) scores 0.25
 *   - A very faint edge (~3 units) scores 0.15
 * This allows the LSD path to accept faint edges that Canny would reject
 * while still penalizing noise-only "edges" (random gradient ~1-2 units).
 */
private const val REFERENCE_GRADIENT = 20.0f

/**
 * Minimum per-side score threshold. A side must have at least this score to
 * count as having gradient evidence. A document should have edges on most
 * sides — if fewer than 3 of 4 sides pass this minimum, the quad is rejected.
 */
private const val MIN_SIDE_SCORE = 0.1f

/** Minimum number of sides that must pass [MIN_SIDE_SCORE] for the quad to be accepted. */
private const val MIN_PASSING_SIDES = 3

/**
 * Computes the average perpendicular gradient component along a single quad side.
 *
 * Samples [numSamples] evenly spaced points along the line from [p1] to [p2],
 * reads the Sobel gradient (Gx, Gy) at each point, projects it onto the
 * perpendicular (normal) direction of the side, and returns the average
 * absolute perpendicular component.
 *
 * This measures how much gradient evidence exists at the document boundary in the
 * direction that a real edge would produce — a document edge creates gradient
 * perpendicular to its direction, while texture/noise is randomly oriented.
 *
 * @param gx Horizontal Sobel gradient (CV_16S). Not modified or released.
 * @param gy Vertical Sobel gradient (CV_16S). Not modified or released.
 * @param p1 Start point of the side (image coordinates).
 * @param p2 End point of the side (image coordinates).
 * @param numSamples Number of evenly spaced sample points along the side.
 * @return Average absolute perpendicular gradient component across all valid
 *   samples. Returns 0.0 if no valid samples (all points outside image bounds).
 */
fun gradientDensityForSide(
    gx: Mat,
    gy: Mat,
    p1: Point,
    p2: Point,
    numSamples: Int = 50
): Float {
    require(numSamples > 0) { "numSamples must be positive, got $numSamples" }

    val rows = gx.rows()
    val cols = gx.cols()
    if (rows == 0 || cols == 0) return 0.0f

    // Side direction vector
    val sdx = p2.x - p1.x
    val sdy = p2.y - p1.y
    val sideLen = sqrt(sdx * sdx + sdy * sdy)
    if (sideLen < 1e-6) return 0.0f

    // Normal direction: perpendicular to the side, normalized
    // Side direction (dx, dy) → normal (-dy, dx)
    val normalX = (-sdy / sideLen).toFloat()
    val normalY = (sdx / sideLen).toFloat()

    val maxCol = cols - 1
    val maxRow = rows - 1
    var perpSum = 0.0f
    var validCount = 0

    for (s in 0 until numSamples) {
        // t ranges from 0.0 to 1.0 across the side (evenly spaced including endpoints
        // when numSamples > 1, or midpoint when numSamples == 1)
        val t = if (numSamples > 1) s.toDouble() / (numSamples - 1) else 0.5
        val px = p1.x + t * sdx
        val py = p1.y + t * sdy

        // Nearest-neighbor: round to integer pixel coordinates
        val col = px.roundToInt()
        val row = py.roundToInt()

        // Bounds check — skip samples outside image
        if (col < 0 || col > maxCol || row < 0 || row > maxRow) continue

        // Read Sobel values (CV_16S: Mat.get returns DoubleArray with 1 element)
        val gxVal = gx.get(row, col) ?: continue
        val gyVal = gy.get(row, col) ?: continue

        // Perpendicular gradient component: |dot(gradient, normal)|
        val perpComponent = abs(gxVal[0].toFloat() * normalX + gyVal[0].toFloat() * normalY)
        perpSum += perpComponent
        validCount++
    }

    return if (validCount > 0) perpSum / validCount else 0.0f
}

/**
 * Verifies gradient density along the 4 sides of a detected quadrilateral.
 *
 * This is the LSD-path equivalent of [QuadValidator.edgeDensityScore]. Since the
 * LSD detection path has no Canny edge image to validate against, we instead
 * sample the Sobel gradient perpendicular to each quad side. A real document
 * boundary produces consistent perpendicular gradient; noise or false detections
 * do not.
 *
 * The function computes Sobel gradients internally and releases them before
 * returning. For callers that already have Sobel Mats (e.g., from the Radon
 * accumulation in B5/B6), use the overload that accepts pre-computed gx/gy.
 *
 * @param gray Single-channel 8-bit grayscale image. Not modified.
 * @param corners Four quad corners in order [TL, TR, BR, BL].
 * @param numSamplesPerSide Number of evenly spaced sample points per side.
 * @return Score in [0.0, 1.0] indicating gradient evidence at the quad boundary.
 *   Returns 0.0 if fewer than [MIN_PASSING_SIDES] (3) sides have score > [MIN_SIDE_SCORE] (0.1).
 */
fun verifyGradientDensity(
    gray: Mat,
    corners: List<Point>,
    numSamplesPerSide: Int = 50
): Float {
    require(corners.size == 4) { "Expected 4 corners, got ${corners.size}" }
    require(gray.channels() == 1) { "Expected single-channel input, got ${gray.channels()}" }
    require(!gray.empty()) { "Input Mat is empty" }

    val start = System.nanoTime()

    // Compute Sobel gradients (CV_16S to preserve sign for directional projection)
    val gx = Mat()
    val gy = Mat()
    try {
        Imgproc.Sobel(gray, gx, CvType.CV_16S, 1, 0, 3)
        Imgproc.Sobel(gray, gy, CvType.CV_16S, 0, 1, 3)

        val score = verifyGradientDensity(
            gx = gx,
            gy = gy,
            corners = corners,
            numSamplesPerSide = numSamplesPerSide
        )

        val ms = (System.nanoTime() - start) / 1_000_000.0
        Log.d(TAG, "verifyGradientDensity: %.2f ms, score=%.3f (incl. Sobel)".format(ms, score))
        return score
    } finally {
        gx.release()
        gy.release()
    }
}

/**
 * Overload that accepts pre-computed Sobel gradient Mats, avoiding redundant
 * Sobel computation when the caller already has them (e.g., Radon accumulation
 * in Tier 2/3).
 *
 * @param gx Horizontal Sobel gradient (CV_16S). Not modified or released.
 * @param gy Vertical Sobel gradient (CV_16S). Not modified or released.
 * @param corners Four quad corners in order [TL, TR, BR, BL].
 * @param numSamplesPerSide Number of evenly spaced sample points per side.
 * @return Score in [0.0, 1.0]. Returns 0.0 if fewer than 3 of 4 sides have score > 0.1.
 */
fun verifyGradientDensity(
    gx: Mat,
    gy: Mat,
    corners: List<Point>,
    numSamplesPerSide: Int = 50
): Float {
    require(corners.size == 4) { "Expected 4 corners, got ${corners.size}" }

    // Side order: TL→TR, TR→BR, BR→BL, BL→TL (same as QuadValidator.edgeDensityScore)
    val sideScores = FloatArray(4)
    for (i in 0 until 4) {
        val p1 = corners[i]
        val p2 = corners[(i + 1) % 4]

        val avgPerpGradient = gradientDensityForSide(
            gx = gx,
            gy = gy,
            p1 = p1,
            p2 = p2,
            numSamples = numSamplesPerSide
        )

        // Normalize against reference gradient (see REFERENCE_GRADIENT doc)
        sideScores[i] = min(1.0f, avgPerpGradient / REFERENCE_GRADIENT)
    }

    // Require at least 3 of 4 sides to have meaningful gradient evidence.
    // A real document has edges on all 4 sides; allowing 1 weak side tolerates
    // occlusion, shadow, or one side blending into the background.
    val passingSides = sideScores.count { it > MIN_SIDE_SCORE }
    if (passingSides < MIN_PASSING_SIDES) {
        return 0.0f
    }

    // Overall score: average of all 4 side scores
    return sideScores.sum() / 4.0f
}

// ---------------------------------------------------------------------------
// B5: Tier 2 — Corner-Constrained Radon Search
// ---------------------------------------------------------------------------

/**
 * Minimum confidence assigned to LSD Tier 2 detections.
 * Tier 2 is slightly less confident than Tier 1 because at least one edge
 * is inferred via Radon accumulation rather than directly observed by LSD.
 */
private const val LSD_TIER2_MIN_CONFIDENCE = 0.45

/** Maximum confidence cap for LSD Tier 2 detections. */
private const val LSD_TIER2_MAX_CONFIDENCE = 0.75

/**
 * Coarse rho step in pixels for the initial Radon scan.
 * 8px spacing across a typical 300-500px search range gives ~40-60 coarse
 * samples — enough to locate peaks without excessive computation.
 */
private const val TIER2_COARSE_RHO_STEP = 8.0f

/**
 * Fine rho step in pixels for refinement around coarse peaks.
 * 1px resolution within +/-12px windows gives precise edge localization.
 */
private const val TIER2_FINE_RHO_STEP = 1.0f

/**
 * Half-width of the refinement window around each coarse peak in pixels.
 * Fine search covers [peak - 12, peak + 12] = 25 rho values per peak.
 */
private const val TIER2_FINE_WINDOW_HALF = 12.0f

/** Number of top coarse peaks to refine in Tier 2. Limits fine-pass work. */
private const val TIER2_COARSE_TOP_PEAKS = 3

/** Number of evenly spaced sample points along each Radon candidate line. */
private const val TIER2_RADON_SAMPLES = 100

/**
 * Minimum distance of a missing edge from its parallel known edge,
 * as a fraction of the image dimension along the search direction.
 * Prevents degenerate zero-width document detections.
 */
private const val TIER2_MIN_EDGE_DISTANCE_FRACTION = 0.15f

/**
 * Maximum rho search range extension beyond the known edge span,
 * as a fraction of the image dimension. Allows the missing edge to
 * be up to 50% of the image dimension away from the known edges.
 */
private const val TIER2_SEARCH_RANGE_EXTENSION_FRACTION = 0.50f

/** Maximum number of Radon peaks to try per search direction. */
private const val TIER2_MAX_CANDIDATE_PEAKS = 5

/**
 * A Radon response peak: a candidate line parameterized by (angle, rho) with
 * an associated gradient accumulation score.
 *
 * @param angleDeg Line angle in degrees [0, 180).
 * @param rho Signed perpendicular distance from the image center to the line.
 * @param response Average absolute perpendicular gradient along the line.
 */
private data class RadonPeak(
    val angleDeg: Float,
    val rho: Float,
    val response: Float
)

/**
 * Performs a restricted Radon line search by accumulating perpendicular gradient
 * along candidate lines at a fixed angle, sweeping rho across a specified range.
 *
 * **Core concept:** For each candidate rho, sample ~100 points along the line
 * and sum the perpendicular gradient component. Horizontal edge search accumulates
 * |Gy|; vertical edge search accumulates |Gx|. This naturally suppresses text
 * and texture (which have random gradient directions) while boosting coherent
 * boundary gradients.
 *
 * **Coarse-to-fine strategy:**
 * 1. Coarse pass: sweep rho in [rhoMin, rhoMax] with [TIER2_COARSE_RHO_STEP] (8px).
 *    Locates approximate peak positions with ~50 rho evaluations.
 * 2. Fine pass: refine top [TIER2_COARSE_TOP_PEAKS] (3) coarse peaks with
 *    [TIER2_FINE_RHO_STEP] (1px) in +/-[TIER2_FINE_WINDOW_HALF] (12px) windows.
 *    ~25 rho evaluations per peak x 3 peaks = ~75 evaluations.
 * Total: ~125 rho evaluations x 100 samples/line = ~12,500 pixel reads.
 *
 * @param gx Pre-computed horizontal Sobel gradient (CV_16S). Not modified.
 * @param gy Pre-computed vertical Sobel gradient (CV_16S). Not modified.
 * @param searchAngleDeg Angle of the candidate lines in degrees [0, 180).
 * @param rhoMin Minimum rho value (signed perpendicular distance from image center).
 * @param rhoMax Maximum rho value (must be >= rhoMin).
 * @param imageWidth Width of the source image in pixels.
 * @param imageHeight Height of the source image in pixels.
 * @param isHorizontalEdge True if searching for horizontal edges (accumulate |Gy|);
 *   false for vertical edges (accumulate |Gx|).
 * @return List of [RadonPeak] results sorted by response descending (strongest first).
 *   Empty if no valid peaks found.
 */
private fun radonLineSearch(
    gx: Mat,
    gy: Mat,
    searchAngleDeg: Float,
    rhoMin: Float,
    rhoMax: Float,
    imageWidth: Int,
    imageHeight: Int,
    isHorizontalEdge: Boolean
): List<RadonPeak> {
    require(rhoMax >= rhoMin) { "rhoMax ($rhoMax) must be >= rhoMin ($rhoMin)" }

    val cols = gx.cols()
    val rows = gx.rows()
    val maxCol = cols - 1
    val maxRow = rows - 1
    val cx = imageWidth / 2.0f
    val cy = imageHeight / 2.0f

    // Pre-compute line direction vector from the search angle
    val lineRad = Math.toRadians(searchAngleDeg.toDouble())
    val lineDx = cos(lineRad).toFloat()
    val lineDy = sin(lineRad).toFloat()
    // Normal direction (perpendicular to line direction, used for rho offset)
    val normalDx = -lineDy
    val normalDy = lineDx

    // Half-diagonal: ensures the line spans the entire visible image
    val halfDiag = sqrt(
        imageWidth.toFloat() * imageWidth + imageHeight.toFloat() * imageHeight
    ) / 2f

    /**
     * Evaluates the Radon response for a single (angle, rho) line.
     * Returns the average absolute perpendicular gradient component
     * across all valid sample points.
     */
    fun evaluateRho(rho: Float): Float {
        // Base point: image center offset by rho along the normal direction.
        // This is the closest point on the line to the image center.
        val baseX = cx + rho * normalDx
        val baseY = cy + rho * normalDy

        // Sample TIER2_RADON_SAMPLES points evenly across the line's visible extent
        val tStep = (2.0f * halfDiag) / (TIER2_RADON_SAMPLES - 1)

        var accumSum = 0.0f
        var validCount = 0

        for (s in 0 until TIER2_RADON_SAMPLES) {
            val t = -halfDiag + s * tStep
            val px = baseX + t * lineDx
            val py = baseY + t * lineDy

            val col = px.roundToInt()
            val row = py.roundToInt()

            // Bounds check — skip samples outside image
            if (col < 0 || col > maxCol || row < 0 || row > maxRow) continue

            // Read the perpendicular gradient component:
            //   For horizontal edges: boundary gradient is vertical -> accumulate |Gy|
            //   For vertical edges: boundary gradient is horizontal -> accumulate |Gx|
            if (isHorizontalEdge) {
                val gyVal = gy.get(row, col) ?: continue
                accumSum += abs(gyVal[0].toFloat())
            } else {
                val gxVal = gx.get(row, col) ?: continue
                accumSum += abs(gxVal[0].toFloat())
            }
            validCount++
        }

        return if (validCount > 0) accumSum / validCount else 0.0f
    }

    // --- Coarse pass: sweep rho with TIER2_COARSE_RHO_STEP ---
    val coarseResults = mutableListOf<RadonPeak>()
    var rho = rhoMin
    while (rho <= rhoMax) {
        val response = evaluateRho(rho)
        if (response > 0.0f) {
            coarseResults.add(RadonPeak(
                angleDeg = searchAngleDeg,
                rho = rho,
                response = response
            ))
        }
        rho += TIER2_COARSE_RHO_STEP
    }

    if (coarseResults.isEmpty()) return emptyList()

    // Select top coarse peaks for refinement
    coarseResults.sortByDescending { it.response }
    val topCoarse = coarseResults.take(TIER2_COARSE_TOP_PEAKS)

    // --- Fine pass: refine each coarse peak with TIER2_FINE_RHO_STEP ---
    val fineResults = mutableListOf<RadonPeak>()
    for (coarsePeak in topCoarse) {
        val fineMin = max(rhoMin, coarsePeak.rho - TIER2_FINE_WINDOW_HALF)
        val fineMax = min(rhoMax, coarsePeak.rho + TIER2_FINE_WINDOW_HALF)

        var fineRho = fineMin
        while (fineRho <= fineMax) {
            val response = evaluateRho(fineRho)
            if (response > 0.0f) {
                fineResults.add(RadonPeak(
                    angleDeg = searchAngleDeg,
                    rho = fineRho,
                    response = response
                ))
            }
            fineRho += TIER2_FINE_RHO_STEP
        }
    }

    // Fine results subsume coarse — return sorted by response descending
    fineResults.sortByDescending { it.response }
    return fineResults
}

/**
 * Converts a [RadonPeak] to a homogeneous line representation (a, b, c)
 * where ax + by + c = 0, using the same parameterization as [clusterToLine].
 *
 * @param peak The Radon peak (angle, rho relative to image center).
 * @param imageCenterX X coordinate of the image center.
 * @param imageCenterY Y coordinate of the image center.
 * @return Triple of (a, b, c) in the equation ax + by + c = 0.
 */
private fun peakToLine(
    peak: RadonPeak,
    imageCenterX: Float,
    imageCenterY: Float
): Triple<Double, Double, Double> {
    val normalRad = Math.toRadians((peak.angleDeg + 90.0).toDouble())
    val a = cos(normalRad)
    val b = sin(normalRad)
    val c = -(a * imageCenterX + b * imageCenterY) - peak.rho.toDouble()
    return Triple(a, b, c)
}

/**
 * Attempts to form and validate a quad from 4 homogeneous lines (2H + 2V),
 * using the same geometric checks as [evaluateQuadCandidate] but without
 * requiring [EdgeCluster] objects for scoring. Uses a simplified scoring
 * based on area ratio, angle regularity, and optional gradient density.
 *
 * @param hLine1 First horizontal line (a, b, c).
 * @param hLine2 Second horizontal line (a, b, c).
 * @param vLine1 First vertical line (a, b, c).
 * @param vLine2 Second vertical line (a, b, c).
 * @param imageWidth Width of the source image in pixels.
 * @param imageHeight Height of the source image in pixels.
 * @param gx Pre-computed horizontal Sobel gradient (CV_16S), or null.
 * @param gy Pre-computed vertical Sobel gradient (CV_16S), or null.
 * @return Pair of (ordered corners, score) if valid, or null if validation fails.
 */
private fun tryFormQuad(
    hLine1: Triple<Double, Double, Double>,
    hLine2: Triple<Double, Double, Double>,
    vLine1: Triple<Double, Double, Double>,
    vLine2: Triple<Double, Double, Double>,
    imageWidth: Int,
    imageHeight: Int,
    gx: Mat? = null,
    gy: Mat? = null
): Pair<List<Point>, Double>? {
    val boundsMarginX = imageWidth * BOUNDS_OVERFLOW_FRACTION
    val boundsMarginY = imageHeight * BOUNDS_OVERFLOW_FRACTION
    val imageArea = imageWidth.toDouble() * imageHeight

    // Compute 4 intersections: each H line meets each V line
    val p1 = intersectLines(hLine1, vLine1) ?: return null
    val p2 = intersectLines(hLine1, vLine2) ?: return null
    val p3 = intersectLines(hLine2, vLine1) ?: return null
    val p4 = intersectLines(hLine2, vLine2) ?: return null

    val rawCorners = listOf(p1, p2, p3, p4)

    // Validate bounds (with 5% overflow margin)
    for (pt in rawCorners) {
        if (pt.x < -boundsMarginX || pt.x > imageWidth + boundsMarginX) return null
        if (pt.y < -boundsMarginY || pt.y > imageHeight + boundsMarginY) return null
    }

    // Order corners TL, TR, BR, BL (same algorithm as QuadRanker.orderCorners)
    val ordered = orderCorners(rawCorners)

    // Validate convexity
    if (!isConvex(ordered)) return null

    // Validate area — must be at least MIN_QUAD_AREA_FRACTION of image area
    val area = lsdQuadArea(ordered)
    if (area < imageArea * MIN_QUAD_AREA_FRACTION) return null

    // Validate interior angles — all must be in [60, 120] degrees
    for (i in 0 until 4) {
        val prev = ordered[(i + 3) % 4]
        val curr = ordered[i]
        val next = ordered[(i + 1) % 4]
        val angle = lsdInteriorAngleDeg(prev, curr, next)
        if (angle < MIN_INTERIOR_ANGLE_DEG || angle > MAX_INTERIOR_ANGLE_DEG) return null
    }

    // Score: area ratio (50%) + angle regularity (50%)
    // No LSD evidence weight since at least one edge is Radon-inferred.
    val areaRatio = (area / imageArea).coerceIn(0.0, 1.0)

    var totalAngleDeviation = 0.0
    for (i in 0 until 4) {
        val prev = ordered[(i + 3) % 4]
        val curr = ordered[i]
        val next = ordered[(i + 1) % 4]
        val angle = lsdInteriorAngleDeg(prev, curr, next)
        totalAngleDeviation += abs(angle - 90.0)
    }
    val angleRegularity = (1.0 - totalAngleDeviation / 360.0).coerceIn(0.0, 1.0)

    var score = areaRatio * 0.5 + angleRegularity * 0.5

    // Gradient density verification if pre-computed Sobel Mats are available.
    // Acts as a quality gate: quads with poor gradient evidence are rejected.
    if (gx != null && gy != null) {
        val gradientScore = verifyGradientDensity(
            gx = gx,
            gy = gy,
            corners = ordered,
            numSamplesPerSide = 50
        )
        if (gradientScore <= 0.0f) return null
        // Blend: 40% geometry + 60% gradient. Gradient evidence is the strongest
        // indicator that the inferred edges sit on real document boundaries.
        score = score * 0.4 + gradientScore * 0.6
    }

    return Pair(ordered, score)
}

/**
 * Tier 2: Corner-constrained Radon search.
 *
 * When Tier 1 fails (no valid 2H+2V rectangle from LSD clusters) but LSD found
 * 2-3 strong edges, we use corner constraints from the known edges to search for
 * the missing edge(s) via restricted Radon accumulation.
 *
 * **Case classification:**
 * - **3 edges (2H+1V or 1H+2V):** The missing edge is fully constrained — it must
 *   intersect 2 known perpendicular edges. 1D search along rho with the angle
 *   inferred from the parallel known edge (if available) or perpendicular+90.
 * - **2 perpendicular edges (1H+1V):** One known corner = intersection of the 2
 *   known edges. Search for the opposite H edge and opposite V edge independently,
 *   constrained by minimum document size (15% of image dimension).
 * - **2 parallel edges (2H+0V or 0H+2V):** Two independent 1D rho searches for
 *   perpendicular edges, one near each end of the known parallel edges.
 * - **Fewer than 2 edges:** Not enough constraints, return null.
 *
 * Pre-computes Sobel Gx/Gy once and reuses for all Radon searches and gradient
 * density verification.
 *
 * @param gray Single-channel 8-bit grayscale image. Not modified.
 * @param clusters Edge clusters from [clusterSegments], already filtered by 20%
 *   diagonal minimum length.
 * @param imageWidth Width of the source image in pixels.
 * @param imageHeight Height of the source image in pixels.
 * @return [DocumentCorners] with ordered corners and confidence in
 *   [LSD_TIER2_MIN_CONFIDENCE, LSD_TIER2_MAX_CONFIDENCE], or null if no valid
 *   quad can be formed.
 */
fun detectRectangleTier2(
    gray: Mat,
    clusters: List<EdgeCluster>,
    imageWidth: Int,
    imageHeight: Int
): DocumentCorners? {
    require(gray.channels() == 1) { "Expected single-channel input, got ${gray.channels()}" }
    require(!gray.empty()) { "Input Mat is empty" }
    require(imageWidth > 0) { "imageWidth must be positive, got $imageWidth" }
    require(imageHeight > 0) { "imageHeight must be positive, got $imageHeight" }

    val start = System.nanoTime()

    // Step 1: Determine what we have — count usable H and V clusters
    val hClusters = clusters.filter { it.isHorizontal }
    val vClusters = clusters.filter { !it.isHorizontal }
    val hCount = hClusters.size
    val vCount = vClusters.size
    val totalEdges = hCount + vCount

    if (totalEdges < 2) {
        val ms = (System.nanoTime() - start) / 1_000_000.0
        Log.d(TAG, "detectRectangleTier2: %.1f ms — only %d edges, need >= 2".format(ms, totalEdges))
        return null
    }

    val imageCenterX = imageWidth / 2f
    val imageCenterY = imageHeight / 2f

    // Pre-compute Sobel gradients once — reused by all radonLineSearch calls
    // and by gradient density verification in tryFormQuad
    val gx = Mat()
    val gy = Mat()
    try {
        Imgproc.Sobel(gray, gx, CvType.CV_16S, 1, 0, 3)
        Imgproc.Sobel(gray, gy, CvType.CV_16S, 0, 1, 3)

        // Convert known clusters to homogeneous line representations
        val hLines = hClusters.map { clusterToLine(it, imageCenterX, imageCenterY) }
        val vLines = vClusters.map { clusterToLine(it, imageCenterX, imageCenterY) }

        val result: DocumentCorners? = when {
            // 3-edge case: 2H+1V — search for the missing second vertical edge
            hCount >= 2 && vCount == 1 ->
                tier2SearchMissingEdge(
                    knownParallelClusters = hClusters, knownParallelLines = hLines,
                    knownPerpCluster = vClusters[0], knownPerpLine = vLines[0],
                    searchForHorizontal = false,
                    gx = gx, gy = gy,
                    imageWidth = imageWidth, imageHeight = imageHeight,
                    imageCenterX = imageCenterX, imageCenterY = imageCenterY
                )

            // 3-edge case: 1H+2V — search for the missing second horizontal edge
            hCount == 1 && vCount >= 2 ->
                tier2SearchMissingEdge(
                    knownParallelClusters = vClusters, knownParallelLines = vLines,
                    knownPerpCluster = hClusters[0], knownPerpLine = hLines[0],
                    searchForHorizontal = true,
                    gx = gx, gy = gy,
                    imageWidth = imageWidth, imageHeight = imageHeight,
                    imageCenterX = imageCenterX, imageCenterY = imageCenterY
                )

            // 2-perpendicular case: 1H+1V — one known corner, search for 2 missing edges
            hCount >= 1 && vCount >= 1 ->
                tier2SearchTwoMissing(
                    hCluster = hClusters[0], hLine = hLines[0],
                    vCluster = vClusters[0], vLine = vLines[0],
                    gx = gx, gy = gy,
                    imageWidth = imageWidth, imageHeight = imageHeight,
                    imageCenterX = imageCenterX, imageCenterY = imageCenterY
                )

            // 2-parallel case: 2H+0V — search for 2 vertical edges
            hCount >= 2 && vCount == 0 ->
                tier2SearchTwoPerpendicularEdges(
                    knownClusters = hClusters.take(2),
                    knownLines = hLines.take(2),
                    searchForHorizontal = false,
                    gx = gx, gy = gy,
                    imageWidth = imageWidth, imageHeight = imageHeight,
                    imageCenterX = imageCenterX, imageCenterY = imageCenterY
                )

            // 2-parallel case: 0H+2V — search for 2 horizontal edges
            vCount >= 2 && hCount == 0 ->
                tier2SearchTwoPerpendicularEdges(
                    knownClusters = vClusters.take(2),
                    knownLines = vLines.take(2),
                    searchForHorizontal = true,
                    gx = gx, gy = gy,
                    imageWidth = imageWidth, imageHeight = imageHeight,
                    imageCenterX = imageCenterX, imageCenterY = imageCenterY
                )

            else -> null
        }

        val ms = (System.nanoTime() - start) / 1_000_000.0
        if (result != null) {
            val finalResult = result.copy(detectionMs = ms)
            Log.d(TAG, "detectRectangleTier2: %.1f ms, H=%d V=%d, confidence=%.2f".format(
                ms, hCount, vCount, finalResult.confidence))
            return finalResult
        }

        Log.d(TAG, "detectRectangleTier2: %.1f ms, H=%d V=%d — no valid quad".format(
            ms, hCount, vCount))
        return null
    } finally {
        gx.release()
        gy.release()
    }
}

// ---------------------------------------------------------------------------
// Tier 2 Case Handlers
// ---------------------------------------------------------------------------

/**
 * 3-edge case: search for a single missing edge.
 *
 * We have 2 parallel edges (e.g., 2H) and 1 perpendicular edge (e.g., 1V).
 * The missing edge is the same orientation as the lone perpendicular edge and
 * must intersect both known parallel edges. Its angle is inferred from the
 * known perpendicular cluster. The rho search range is bounded by the known
 * perpendicular edge's rho, extended by [TIER2_SEARCH_RANGE_EXTENSION_FRACTION].
 *
 * @param knownParallelClusters The 2 known parallel clusters (both H or both V).
 * @param knownParallelLines The 2 known parallel lines in homogeneous form.
 * @param knownPerpCluster The 1 known perpendicular cluster.
 * @param knownPerpLine The known perpendicular line in homogeneous form.
 * @param searchForHorizontal True if the missing edge is horizontal;
 *   false if the missing edge is vertical.
 * @param gx Pre-computed horizontal Sobel gradient (CV_16S).
 * @param gy Pre-computed vertical Sobel gradient (CV_16S).
 * @param imageWidth Width of the source image in pixels.
 * @param imageHeight Height of the source image in pixels.
 * @param imageCenterX X coordinate of the image center.
 * @param imageCenterY Y coordinate of the image center.
 * @return [DocumentCorners] or null if no valid quad found.
 */
private fun tier2SearchMissingEdge(
    knownParallelClusters: List<EdgeCluster>,
    knownParallelLines: List<Triple<Double, Double, Double>>,
    knownPerpCluster: EdgeCluster,
    knownPerpLine: Triple<Double, Double, Double>,
    searchForHorizontal: Boolean,
    gx: Mat,
    gy: Mat,
    imageWidth: Int,
    imageHeight: Int,
    imageCenterX: Float,
    imageCenterY: Float
): DocumentCorners? {
    // Search angle: same as the known perpendicular edge (the missing edge
    // should be roughly parallel to it — both are perpendicular to the known pair)
    val searchAngle = knownPerpCluster.angle
    val knownPerpRho = knownPerpCluster.rho

    // The image dimension along the search direction determines the rho range.
    // For searching V edges, use imageWidth; for H edges, use imageHeight.
    val searchDimension = if (searchForHorizontal) imageHeight.toFloat() else imageWidth.toFloat()
    val extension = searchDimension * TIER2_SEARCH_RANGE_EXTENSION_FRACTION
    val minDistance = searchDimension * TIER2_MIN_EDGE_DISTANCE_FRACTION

    // Search on both sides of the known perpendicular edge: the missing edge
    // could be on either side (we don't know which side of the document it's on).
    // rhoRange1: positive side (rho > knownPerpRho)
    // rhoRange2: negative side (rho < knownPerpRho)
    val rhoRange1 = Pair(knownPerpRho + minDistance, knownPerpRho + extension)
    val rhoRange2 = Pair(knownPerpRho - extension, knownPerpRho - minDistance)

    var bestCorners: List<Point>? = null
    var bestScore = -1.0

    for (rhoRange in listOf(rhoRange1, rhoRange2)) {
        // Skip degenerate ranges
        if (rhoRange.first > rhoRange.second) continue

        val peaks = radonLineSearch(
            gx = gx, gy = gy,
            searchAngleDeg = searchAngle,
            rhoMin = rhoRange.first,
            rhoMax = rhoRange.second,
            imageWidth = imageWidth,
            imageHeight = imageHeight,
            isHorizontalEdge = searchForHorizontal
        )

        // Try each peak as the missing edge
        for (peak in peaks.take(TIER2_MAX_CANDIDATE_PEAKS)) {
            val candidateLine = peakToLine(peak, imageCenterX, imageCenterY)

            // Try with each pair of known parallel lines
            for (pi in knownParallelLines.indices) {
                for (pj in pi + 1 until knownParallelLines.size) {
                    // Assign H/V roles based on what's known vs found
                    val candidate = if (searchForHorizontal) {
                        // Missing edge is horizontal; known parallel are vertical
                        tryFormQuad(
                            hLine1 = knownPerpLine,
                            hLine2 = candidateLine,
                            vLine1 = knownParallelLines[pi],
                            vLine2 = knownParallelLines[pj],
                            imageWidth = imageWidth,
                            imageHeight = imageHeight,
                            gx = gx, gy = gy
                        )
                    } else {
                        // Missing edge is vertical; known parallel are horizontal
                        tryFormQuad(
                            hLine1 = knownParallelLines[pi],
                            hLine2 = knownParallelLines[pj],
                            vLine1 = knownPerpLine,
                            vLine2 = candidateLine,
                            imageWidth = imageWidth,
                            imageHeight = imageHeight,
                            gx = gx, gy = gy
                        )
                    }

                    if (candidate != null && candidate.second > bestScore) {
                        bestCorners = candidate.first
                        bestScore = candidate.second
                    }
                }
            }
        }
    }

    return buildTier2Result(bestCorners, bestScore)
}

/**
 * 2-perpendicular case: 1H + 1V — one known corner, search for 2 missing edges.
 *
 * The intersection of the known H and V edges gives one document corner.
 * We search for the opposite horizontal edge (rho scan at the H angle) and
 * the opposite vertical edge (rho scan at the V angle), then try all
 * combinations of found H x found V peaks.
 *
 * @param hCluster The known horizontal cluster.
 * @param hLine The known horizontal line in homogeneous form.
 * @param vCluster The known vertical cluster.
 * @param vLine The known vertical line in homogeneous form.
 * @param gx Pre-computed horizontal Sobel gradient (CV_16S).
 * @param gy Pre-computed vertical Sobel gradient (CV_16S).
 * @param imageWidth Width of the source image in pixels.
 * @param imageHeight Height of the source image in pixels.
 * @param imageCenterX X coordinate of the image center.
 * @param imageCenterY Y coordinate of the image center.
 * @return [DocumentCorners] or null if no valid quad found.
 */
private fun tier2SearchTwoMissing(
    hCluster: EdgeCluster,
    hLine: Triple<Double, Double, Double>,
    vCluster: EdgeCluster,
    vLine: Triple<Double, Double, Double>,
    gx: Mat,
    gy: Mat,
    imageWidth: Int,
    imageHeight: Int,
    imageCenterX: Float,
    imageCenterY: Float
): DocumentCorners? {
    val hRho = hCluster.rho
    val vRho = vCluster.rho

    // Search for the opposite horizontal edge (parallel to known H, on opposite side)
    val hExtension = imageHeight * TIER2_SEARCH_RANGE_EXTENSION_FRACTION
    val hMinDist = imageHeight * TIER2_MIN_EDGE_DISTANCE_FRACTION

    val hPeaks = mutableListOf<RadonPeak>()
    for (rhoRange in listOf(
        Pair(hRho + hMinDist, hRho + hExtension),
        Pair(hRho - hExtension, hRho - hMinDist)
    )) {
        if (rhoRange.first > rhoRange.second) continue
        hPeaks.addAll(radonLineSearch(
            gx = gx, gy = gy,
            searchAngleDeg = hCluster.angle,
            rhoMin = rhoRange.first,
            rhoMax = rhoRange.second,
            imageWidth = imageWidth,
            imageHeight = imageHeight,
            isHorizontalEdge = true  // searching for horizontal edge -> accumulate |Gy|
        ))
    }

    // Search for the opposite vertical edge (parallel to known V, on opposite side)
    val vExtension = imageWidth * TIER2_SEARCH_RANGE_EXTENSION_FRACTION
    val vMinDist = imageWidth * TIER2_MIN_EDGE_DISTANCE_FRACTION

    val vPeaks = mutableListOf<RadonPeak>()
    for (rhoRange in listOf(
        Pair(vRho + vMinDist, vRho + vExtension),
        Pair(vRho - vExtension, vRho - vMinDist)
    )) {
        if (rhoRange.first > rhoRange.second) continue
        vPeaks.addAll(radonLineSearch(
            gx = gx, gy = gy,
            searchAngleDeg = vCluster.angle,
            rhoMin = rhoRange.first,
            rhoMax = rhoRange.second,
            imageWidth = imageWidth,
            imageHeight = imageHeight,
            isHorizontalEdge = false  // searching for vertical edge -> accumulate |Gx|
        ))
    }

    // Sort by response and take top candidates
    hPeaks.sortByDescending { it.response }
    vPeaks.sortByDescending { it.response }

    var bestCorners: List<Point>? = null
    var bestScore = -1.0

    // Try all combinations of found H x found V peaks
    for (hp in hPeaks.take(TIER2_MAX_CANDIDATE_PEAKS)) {
        val candidateHLine = peakToLine(hp, imageCenterX, imageCenterY)
        for (vp in vPeaks.take(TIER2_MAX_CANDIDATE_PEAKS)) {
            val candidateVLine = peakToLine(vp, imageCenterX, imageCenterY)

            val candidate = tryFormQuad(
                hLine1 = hLine,
                hLine2 = candidateHLine,
                vLine1 = vLine,
                vLine2 = candidateVLine,
                imageWidth = imageWidth,
                imageHeight = imageHeight,
                gx = gx, gy = gy
            )
            if (candidate != null && candidate.second > bestScore) {
                bestCorners = candidate.first
                bestScore = candidate.second
            }
        }
    }

    return buildTier2Result(bestCorners, bestScore)
}

/**
 * 2-parallel case: 2H+0V or 0H+2V — search for 2 perpendicular edges.
 *
 * Both missing edges are perpendicular to the known pair. Run two independent
 * Radon searches: one near each end of the known parallel edges' span.
 *
 * The search regions are computed from the endpoints of the known parallel
 * clusters: the perpendicular edges should be near the extremes of the
 * known edges' extent along the search direction.
 *
 * @param knownClusters The 2 known parallel clusters (both H or both V).
 * @param knownLines The 2 known parallel lines in homogeneous form.
 * @param searchForHorizontal True if the missing edges are horizontal (known are V);
 *   false if missing edges are vertical (known are H).
 * @param gx Pre-computed horizontal Sobel gradient (CV_16S).
 * @param gy Pre-computed vertical Sobel gradient (CV_16S).
 * @param imageWidth Width of the source image in pixels.
 * @param imageHeight Height of the source image in pixels.
 * @param imageCenterX X coordinate of the image center.
 * @param imageCenterY Y coordinate of the image center.
 * @return [DocumentCorners] or null if no valid quad found.
 */
private fun tier2SearchTwoPerpendicularEdges(
    knownClusters: List<EdgeCluster>,
    knownLines: List<Triple<Double, Double, Double>>,
    searchForHorizontal: Boolean,
    gx: Mat,
    gy: Mat,
    imageWidth: Int,
    imageHeight: Int,
    imageCenterX: Float,
    imageCenterY: Float
): DocumentCorners? {
    require(knownClusters.size == 2) { "Expected 2 known clusters, got ${knownClusters.size}" }

    // The search angle is perpendicular to the known edges.
    // Known H edges (angle ~0 or ~180) -> search for V edges (angle ~90)
    // Known V edges (angle ~90) -> search for H edges (angle ~0)
    val knownAngle = (knownClusters[0].angle + knownClusters[1].angle) / 2f
    val searchAngle = run {
        var a = if (searchForHorizontal) {
            knownAngle - 90f
        } else {
            knownAngle + 90f
        }
        // Normalize to [0, 180)
        if (a < 0f) a += 180f
        if (a >= 180f) a -= 180f
        a
    }

    // Rho search range: based on the endpoints of the known parallel edges.
    // The perpendicular edges should be near the ends of the known edges.
    // Project all endpoints onto the search direction's normal to find the
    // rho extent of the document.
    val normalRad = Math.toRadians((searchAngle + 90.0).toDouble())
    val nx = cos(normalRad).toFloat()
    val ny = sin(normalRad).toFloat()

    val endpointRhos = FloatArray(4)
    var idx = 0
    for (cluster in knownClusters) {
        val sp = cluster.startPoint
        val ep = cluster.endPoint
        endpointRhos[idx++] = (sp.x - imageCenterX) * nx + (sp.y - imageCenterY) * ny
        endpointRhos[idx++] = (ep.x - imageCenterX) * nx + (ep.y - imageCenterY) * ny
    }

    val minEndRho = endpointRhos.min()
    val maxEndRho = endpointRhos.max()

    // Search extension: 30% of the full extension (narrower than 3-edge case
    // because we have less constraint — keep search regions focused near the
    // known endpoints to avoid false positives)
    val dimension = if (searchForHorizontal) imageHeight.toFloat() else imageWidth.toFloat()
    val searchExtension = dimension * TIER2_SEARCH_RANGE_EXTENSION_FRACTION * 0.3f

    // Search region 1: near the min-rho end of the known edges
    val search1Min = minEndRho - searchExtension
    val search1Max = minEndRho + searchExtension

    // Search region 2: near the max-rho end of the known edges
    val search2Min = maxEndRho - searchExtension
    val search2Max = maxEndRho + searchExtension

    val peaks1 = radonLineSearch(
        gx = gx, gy = gy,
        searchAngleDeg = searchAngle,
        rhoMin = search1Min,
        rhoMax = search1Max,
        imageWidth = imageWidth,
        imageHeight = imageHeight,
        isHorizontalEdge = searchForHorizontal
    )

    val peaks2 = radonLineSearch(
        gx = gx, gy = gy,
        searchAngleDeg = searchAngle,
        rhoMin = search2Min,
        rhoMax = search2Max,
        imageWidth = imageWidth,
        imageHeight = imageHeight,
        isHorizontalEdge = searchForHorizontal
    )

    var bestCorners: List<Point>? = null
    var bestScore = -1.0

    // Try combinations: one peak from each search region
    for (p1 in peaks1.take(TIER2_MAX_CANDIDATE_PEAKS)) {
        for (p2 in peaks2.take(TIER2_MAX_CANDIDATE_PEAKS)) {
            val line1 = peakToLine(p1, imageCenterX, imageCenterY)
            val line2 = peakToLine(p2, imageCenterX, imageCenterY)

            // Assign H/V roles based on what's known vs found
            val candidate = if (searchForHorizontal) {
                // Found edges are horizontal, known edges are vertical
                tryFormQuad(
                    hLine1 = line1, hLine2 = line2,
                    vLine1 = knownLines[0], vLine2 = knownLines[1],
                    imageWidth = imageWidth, imageHeight = imageHeight,
                    gx = gx, gy = gy
                )
            } else {
                // Found edges are vertical, known edges are horizontal
                tryFormQuad(
                    hLine1 = knownLines[0], hLine2 = knownLines[1],
                    vLine1 = line1, vLine2 = line2,
                    imageWidth = imageWidth, imageHeight = imageHeight,
                    gx = gx, gy = gy
                )
            }

            if (candidate != null && candidate.second > bestScore) {
                bestCorners = candidate.first
                bestScore = candidate.second
            }
        }
    }

    return buildTier2Result(bestCorners, bestScore)
}

/**
 * Converts a best-quad result into a [DocumentCorners] for Tier 2, or null
 * if no valid quad was found. Maps the raw score [0, 1] to the Tier 2
 * confidence range [LSD_TIER2_MIN_CONFIDENCE, LSD_TIER2_MAX_CONFIDENCE].
 */
private fun buildTier2Result(
    bestCorners: List<Point>?,
    bestScore: Double
): DocumentCorners? {
    if (bestCorners == null || bestScore < 0.0) return null

    val confidence = LSD_TIER2_MIN_CONFIDENCE +
            bestScore * (LSD_TIER2_MAX_CONFIDENCE - LSD_TIER2_MIN_CONFIDENCE)

    return DocumentCorners(
        corners = bestCorners,
        detectionMs = 0.0,  // will be overwritten by detectRectangleTier2
        confidence = confidence
    )
}

// ---------------------------------------------------------------------------
// B6: Tier 3 — Joint Radon Rectangle Fit
// ---------------------------------------------------------------------------

/**
 * Minimum confidence assigned to Tier 3 (Radon-only) detections.
 * Lower than Tier 1 (0.50) because Radon accumulation without LSD segment
 * evidence is weaker — the geometric prior does more of the heavy lifting.
 */
private const val TIER3_MIN_CONFIDENCE = 0.40

/** Maximum confidence cap for Tier 3 detections. */
private const val TIER3_MAX_CONFIDENCE = 0.65

/**
 * Minimum gradient density score from [verifyGradientDensity] for a Tier 3
 * candidate to be accepted. Very low threshold because Tier 3 fires for
 * extremely faint edges (~3 unit gradients), but must still reject pure noise.
 */
private const val TIER3_MIN_GRADIENT_DENSITY = 0.05f

/**
 * Rho range for Radon scan: document edges cannot be at the very edge of the
 * frame. Scan from 15% to 85% of the image dimension.
 * This is a reasonable geometric prior — a user's document will occupy the
 * central portion of the viewfinder, not hug the frame border.
 */
private const val RADON_RHO_MIN_FRACTION = 0.15f
private const val RADON_RHO_MAX_FRACTION = 0.85f

/**
 * Coarse scan step in pixels for Radon accumulation.
 * 8px is enough resolution for the coarse pass — fine refinement will
 * narrow down to 1px precision.
 */
private const val RADON_COARSE_STEP = 8

/**
 * Refinement window: ±12px around each coarse peak, scanned at 1px resolution.
 */
private const val RADON_REFINE_HALF_WINDOW = 12

/** Maximum peaks to extract per scan direction (H or V). */
private const val RADON_MAX_PEAKS = 4

/**
 * Minimum separation between peaks as a fraction of the image dimension
 * in that direction. Prevents two peaks from representing the same edge.
 * 10% of the dimension ensures peaks correspond to distinct document edges.
 */
private const val RADON_PEAK_MIN_SEPARATION_FRACTION = 0.10f

/**
 * Number of sample points along each Radon line. 100 points provides
 * enough accumulation for SNR improvement while staying within the
 * performance budget (~225k total pixel reads across all angles).
 */
private const val RADON_NUM_SAMPLES = 100

/**
 * Candidate orientations for the Radon scan, in degrees from axis-aligned.
 * ±8 degrees covers the typical range where a user holds a camera over a
 * document — beyond ±8 deg the user is deliberately tilting and LSD/contour
 * detection should have caught it in earlier tiers.
 */
private val RADON_THETA_OFFSETS = floatArrayOf(-8f, -6f, -4f, -2f, 0f, 2f, 4f, 6f, 8f)

/**
 * Accumulates gradient magnitude along a line parameterized by (angle, rho)
 * in Hough-like coordinates, relative to the image center.
 *
 * For a line at angle [angleDeg] (degrees, 0 = horizontal) and perpendicular
 * offset [rhoPixels] from the image center, samples [numSamples] evenly
 * spaced points and accumulates the absolute value from [gradient] (which
 * should be the perpendicular Sobel component: |Gy| for near-horizontal
 * lines, |Gx| for near-vertical lines).
 *
 * Using only the perpendicular gradient component provides natural text and
 * texture suppression — document text creates gradients parallel to the edge,
 * while the document boundary creates gradient perpendicular to it.
 *
 * @param gradient Sobel gradient Mat (CV_16S). Either Gx or Gy depending on
 *   the line orientation. Not modified or released.
 * @param angleDeg Line direction in degrees from horizontal.
 * @param rhoPixels Perpendicular distance from image center to the line, in pixels.
 * @param imageWidth Width of the image in pixels.
 * @param imageHeight Height of the image in pixels.
 * @param numSamples Number of evenly spaced sample points along the line.
 * @return Average absolute gradient magnitude along the line. 0.0 if no valid
 *   samples (all points fell outside image bounds).
 */
private fun radonAccumulate(
    gradient: Mat,
    angleDeg: Float,
    rhoPixels: Float,
    imageWidth: Int,
    imageHeight: Int,
    numSamples: Int = RADON_NUM_SAMPLES
): Float {
    val rows = gradient.rows()
    val cols = gradient.cols()
    if (rows == 0 || cols == 0) return 0.0f

    val cx = imageWidth / 2.0f
    val cy = imageHeight / 2.0f

    // Line direction: angle in radians
    val lineRad = Math.toRadians(angleDeg.toDouble())
    val dirX = cos(lineRad).toFloat()
    val dirY = sin(lineRad).toFloat()

    // Normal direction (perpendicular to line direction)
    val normX = -dirY
    val normY = dirX

    // A point on the line: center + rho * normal
    val lineBaseX = cx + rhoPixels * normX
    val lineBaseY = cy + rhoPixels * normY

    // Sample extent: half the image diagonal ensures the line spans the entire image
    val halfExtent = sqrt(imageWidth.toFloat() * imageWidth + imageHeight.toFloat() * imageHeight) / 2.0f

    val maxCol = cols - 1
    val maxRow = rows - 1
    var accum = 0.0f
    var validCount = 0

    for (s in 0 until numSamples) {
        // t ranges from -halfExtent to +halfExtent
        val t = if (numSamples > 1) {
            -halfExtent + (2.0f * halfExtent * s) / (numSamples - 1)
        } else {
            0.0f
        }

        val px = lineBaseX + t * dirX
        val py = lineBaseY + t * dirY

        val col = px.roundToInt()
        val row = py.roundToInt()

        // Bounds check — skip samples outside image
        if (col < 0 || col > maxCol || row < 0 || row > maxRow) continue

        // Read CV_16S value and take absolute value
        val v = gradient.get(row, col) ?: continue
        accum += abs(v[0].toFloat())
        validCount++
    }

    return if (validCount > 0) accum / validCount else 0.0f
}

/**
 * Finds local maxima (peaks) in a 1D response array from a Radon scan.
 *
 * Peaks are local maxima that are strictly greater than both neighbors.
 * A minimum separation constraint ensures peaks correspond to distinct
 * document edges (not noise around a single edge). Results are returned
 * sorted by response descending.
 *
 * @param responses Radon response values, one per rho position.
 * @param rhoValues Corresponding rho positions in pixels, parallel array to [responses].
 * @param minSeparation Minimum distance (in pixels) between accepted peaks.
 * @param maxPeaks Maximum number of peaks to return.
 * @return List of (rho, response) pairs sorted by response descending,
 *   at most [maxPeaks] entries.
 */
private fun findRadonPeaks(
    responses: FloatArray,
    rhoValues: FloatArray,
    minSeparation: Float,
    maxPeaks: Int = RADON_MAX_PEAKS
): List<Pair<Float, Float>> {
    require(responses.size == rhoValues.size) {
        "responses and rhoValues must have the same size: ${responses.size} vs ${rhoValues.size}"
    }
    if (responses.size < 3) return emptyList()

    // Find all local maxima (strictly greater than both neighbors)
    val candidates = mutableListOf<Pair<Float, Float>>() // (rho, response)
    for (i in 1 until responses.size - 1) {
        if (responses[i] > responses[i - 1] && responses[i] > responses[i + 1]) {
            candidates.add(Pair(rhoValues[i], responses[i]))
        }
    }

    // Sort by response descending
    candidates.sortByDescending { it.second }

    // Greedily select peaks with minimum separation
    val selected = mutableListOf<Pair<Float, Float>>()
    for (candidate in candidates) {
        if (selected.size >= maxPeaks) break

        val tooClose = selected.any { abs(it.first - candidate.first) < minSeparation }
        if (!tooClose) {
            selected.add(candidate)
        }
    }

    return selected
}

/**
 * Tier 3 — Joint Radon rectangle fit: full restricted Radon scan to find a
 * document rectangle directly from gradient accumulation, without LSD.
 *
 * This is the last-resort detection for scenes where LSD finds no usable
 * edges at all (gradient < ~2.6 units). It exploits the geometric prior
 * that documents are typically near-axis-aligned rectangles filling most
 * of the viewfinder.
 *
 * **Algorithm:**
 * 1. Pre-compute Sobel Gx/Gy once (reused across all angle iterations)
 * 2. Scan 9 candidate orientations: [-8, -6, -4, -2, 0, +2, +4, +6, +8] deg
 *    from axis-aligned (covers the ±8 deg range of typical camera angles)
 * 3. For each theta, find best H and V peaks independently via coarse+fine
 *    Radon accumulation of the perpendicular gradient component
 * 4. Combine H and V peaks into rectangle candidates (up to C(4,2)^2 = 36 per theta)
 * 5. Validate convexity, area, angles; score with Radon strength + geometry;
 *    apply gradient density verification as a rejection gate
 * 6. Apply soft geometric priors (centering, aspect ratio)
 * 7. Return best quad with confidence mapped to [0.40, 0.65]
 *
 * **Performance:** ~4ms total on S21 (Snapdragon 888) at 640x480.
 * 9 angles x ~25 coarse rho x 100 samples = ~45k coarse reads (H+V),
 * plus ~180k fine reads = ~225k total pixel reads.
 *
 * @param gray Single-channel 8-bit grayscale image. Not modified.
 * @param imageWidth Width of the source image in pixels.
 * @param imageHeight Height of the source image in pixels.
 * @return [DocumentCorners] with ordered corners and confidence in [0.40, 0.65],
 *   or null if no valid rectangle was found.
 */
fun detectRectangleTier3(
    gray: Mat,
    imageWidth: Int,
    imageHeight: Int
): DocumentCorners? {
    require(gray.channels() == 1) { "Expected single-channel input, got ${gray.channels()}" }
    require(!gray.empty()) { "Input Mat is empty" }
    require(imageWidth > 0) { "imageWidth must be positive, got $imageWidth" }
    require(imageHeight > 0) { "imageHeight must be positive, got $imageHeight" }

    val start = System.nanoTime()

    val imageArea = imageWidth.toDouble() * imageHeight
    val cx = imageWidth / 2.0
    val cy = imageHeight / 2.0

    // Step 1: Pre-compute Sobel gradients (CV_16S to preserve sign).
    // Gx captures vertical edges, Gy captures horizontal edges.
    // These are reused for all 9 angle iterations.
    val gx = Mat()
    val gy = Mat()
    try {
        Imgproc.Sobel(gray, gx, CvType.CV_16S, 1, 0, 3)
        Imgproc.Sobel(gray, gy, CvType.CV_16S, 0, 1, 3)

        // Rho ranges: 15%-85% of each dimension — document can't hug the frame border
        val hRhoMin = imageHeight * RADON_RHO_MIN_FRACTION
        val hRhoMax = imageHeight * RADON_RHO_MAX_FRACTION
        val vRhoMin = imageWidth * RADON_RHO_MIN_FRACTION
        val vRhoMax = imageWidth * RADON_RHO_MAX_FRACTION

        // Minimum separation between peaks: 10% of the dimension
        val hMinSep = imageHeight * RADON_PEAK_MIN_SEPARATION_FRACTION
        val vMinSep = imageWidth * RADON_PEAK_MIN_SEPARATION_FRACTION

        // Bounds for intersection validation (5% overflow, same as Tier 1)
        val boundsMarginX = imageWidth * BOUNDS_OVERFLOW_FRACTION
        val boundsMarginY = imageHeight * BOUNDS_OVERFLOW_FRACTION

        var bestCorners: List<Point>? = null
        var bestScore = -1.0

        // Step 2: Scan over candidate orientations
        for (theta in RADON_THETA_OFFSETS) {

            // Step 3a: Find horizontal edge peaks at angle theta.
            // Horizontal edges produce perpendicular gradient in Y direction → accumulate |Gy|.
            // Rho is the perpendicular offset from image center in the direction of the
            // horizontal line normal (roughly vertical for near-horizontal lines).
            val hCoarseRhoCount = ((hRhoMax - hRhoMin) / RADON_COARSE_STEP).roundToInt() + 1
            val hCoarseRho = FloatArray(hCoarseRhoCount)
            val hCoarseResp = FloatArray(hCoarseRhoCount)

            for (i in 0 until hCoarseRhoCount) {
                // rho is offset from center: convert [hRhoMin..hRhoMax] (absolute position)
                // to signed offset from center. hRhoMin/Max are fractions of imageHeight
                // measured from the top, so offset from center = position - cy.
                val absPos = hRhoMin + i * RADON_COARSE_STEP
                if (absPos > hRhoMax) break
                val rho = absPos - imageHeight / 2.0f
                hCoarseRho[i] = rho
                hCoarseResp[i] = radonAccumulate(
                    gradient = gy,
                    angleDeg = theta,
                    rhoPixels = rho,
                    imageWidth = imageWidth,
                    imageHeight = imageHeight
                )
            }

            // Find coarse peaks and refine
            val hCoarsePeaks = findRadonPeaks(
                responses = hCoarseResp,
                rhoValues = hCoarseRho,
                minSeparation = hMinSep
            )

            // Refine each coarse peak at 1px resolution in ±RADON_REFINE_HALF_WINDOW
            val hRefinedPeaks = mutableListOf<Pair<Float, Float>>() // (rho, response)
            for ((coarseRho, _) in hCoarsePeaks) {
                val fineStart = coarseRho - RADON_REFINE_HALF_WINDOW
                val fineEnd = coarseRho + RADON_REFINE_HALF_WINDOW
                val fineCount = (fineEnd - fineStart).roundToInt() + 1
                val fineRho = FloatArray(fineCount)
                val fineResp = FloatArray(fineCount)

                for (j in 0 until fineCount) {
                    val rho = fineStart + j
                    fineRho[j] = rho
                    fineResp[j] = radonAccumulate(
                        gradient = gy,
                        angleDeg = theta,
                        rhoPixels = rho,
                        imageWidth = imageWidth,
                        imageHeight = imageHeight
                    )
                }
                val finePeaks = findRadonPeaks(
                    responses = fineResp,
                    rhoValues = fineRho,
                    minSeparation = hMinSep,
                    maxPeaks = 1
                )
                if (finePeaks.isNotEmpty()) {
                    hRefinedPeaks.add(finePeaks[0])
                } else {
                    // Fallback: use coarse peak position with refined response
                    val midIdx = fineCount / 2
                    hRefinedPeaks.add(Pair(fineRho[midIdx], fineResp[midIdx]))
                }
            }

            // Step 3b: Find vertical edge peaks at angle (theta + 90).
            // Vertical edges produce perpendicular gradient in X direction → accumulate |Gx|.
            val vAngle = theta + 90f
            val vCoarseRhoCount = ((vRhoMax - vRhoMin) / RADON_COARSE_STEP).roundToInt() + 1
            val vCoarseRho = FloatArray(vCoarseRhoCount)
            val vCoarseResp = FloatArray(vCoarseRhoCount)

            for (i in 0 until vCoarseRhoCount) {
                val absPos = vRhoMin + i * RADON_COARSE_STEP
                if (absPos > vRhoMax) break
                val rho = absPos - imageWidth / 2.0f
                vCoarseRho[i] = rho
                vCoarseResp[i] = radonAccumulate(
                    gradient = gx,
                    angleDeg = vAngle,
                    rhoPixels = rho,
                    imageWidth = imageWidth,
                    imageHeight = imageHeight
                )
            }

            val vCoarsePeaks = findRadonPeaks(
                responses = vCoarseResp,
                rhoValues = vCoarseRho,
                minSeparation = vMinSep
            )

            // Refine vertical peaks
            val vRefinedPeaks = mutableListOf<Pair<Float, Float>>()
            for ((coarseRho, _) in vCoarsePeaks) {
                val fineStart = coarseRho - RADON_REFINE_HALF_WINDOW
                val fineEnd = coarseRho + RADON_REFINE_HALF_WINDOW
                val fineCount = (fineEnd - fineStart).roundToInt() + 1
                val fineRho = FloatArray(fineCount)
                val fineResp = FloatArray(fineCount)

                for (j in 0 until fineCount) {
                    val rho = fineStart + j
                    fineRho[j] = rho
                    fineResp[j] = radonAccumulate(
                        gradient = gx,
                        angleDeg = vAngle,
                        rhoPixels = rho,
                        imageWidth = imageWidth,
                        imageHeight = imageHeight
                    )
                }
                val finePeaks = findRadonPeaks(
                    responses = fineResp,
                    rhoValues = fineRho,
                    minSeparation = vMinSep,
                    maxPeaks = 1
                )
                if (finePeaks.isNotEmpty()) {
                    vRefinedPeaks.add(finePeaks[0])
                } else {
                    val midIdx = fineCount / 2
                    vRefinedPeaks.add(Pair(fineRho[midIdx], fineResp[midIdx]))
                }
            }

            // Step 4: Combine H and V peaks into rectangle candidates.
            // Need at least 2 H peaks and 2 V peaks to form a rectangle.
            if (hRefinedPeaks.size < 2 || vRefinedPeaks.size < 2) continue

            // Try all combinations of 2H × 2V (at most C(4,2) × C(4,2) = 36 per theta)
            for (hi in hRefinedPeaks.indices) {
                for (hj in hi + 1 until hRefinedPeaks.size) {
                    for (vi in vRefinedPeaks.indices) {
                        for (vj in vi + 1 until vRefinedPeaks.size) {
                            val (hRho1, hResp1) = hRefinedPeaks[hi]
                            val (hRho2, hResp2) = hRefinedPeaks[hj]
                            val (vRho1, vResp1) = vRefinedPeaks[vi]
                            val (vRho2, vResp2) = vRefinedPeaks[vj]

                            // Convert Radon (theta, rho) lines to homogeneous (a, b, c)
                            val hLine1 = radonLineToHomogeneous(
                                angleDeg = theta,
                                rhoPixels = hRho1,
                                imageCenterX = imageWidth / 2.0f,
                                imageCenterY = imageHeight / 2.0f
                            )
                            val hLine2 = radonLineToHomogeneous(
                                angleDeg = theta,
                                rhoPixels = hRho2,
                                imageCenterX = imageWidth / 2.0f,
                                imageCenterY = imageHeight / 2.0f
                            )
                            val vLine1 = radonLineToHomogeneous(
                                angleDeg = vAngle,
                                rhoPixels = vRho1,
                                imageCenterX = imageWidth / 2.0f,
                                imageCenterY = imageHeight / 2.0f
                            )
                            val vLine2 = radonLineToHomogeneous(
                                angleDeg = vAngle,
                                rhoPixels = vRho2,
                                imageCenterX = imageWidth / 2.0f,
                                imageCenterY = imageHeight / 2.0f
                            )

                            // Compute 4 intersections
                            val p1 = intersectLines(hLine1, vLine1) ?: continue
                            val p2 = intersectLines(hLine1, vLine2) ?: continue
                            val p3 = intersectLines(hLine2, vLine1) ?: continue
                            val p4 = intersectLines(hLine2, vLine2) ?: continue

                            val rawCorners = listOf(p1, p2, p3, p4)

                            // Bounds check with overflow margin
                            val outOfBounds = rawCorners.any { pt ->
                                pt.x < -boundsMarginX || pt.x > imageWidth + boundsMarginX ||
                                        pt.y < -boundsMarginY || pt.y > imageHeight + boundsMarginY
                            }
                            if (outOfBounds) continue

                            // Order corners: TL, TR, BR, BL
                            val ordered = orderCorners(rawCorners)

                            // Validate convexity
                            if (!isConvex(ordered)) continue

                            // Validate area >= 10% of image area
                            val area = lsdQuadArea(ordered)
                            if (area < imageArea * MIN_QUAD_AREA_FRACTION) continue

                            // Validate interior angles (60-120 degrees)
                            var anglesValid = true
                            for (i in 0 until 4) {
                                val prev = ordered[(i + 3) % 4]
                                val curr = ordered[i]
                                val next = ordered[(i + 1) % 4]
                                val angle = lsdInteriorAngleDeg(prev, curr, next)
                                if (angle < MIN_INTERIOR_ANGLE_DEG || angle > MAX_INTERIOR_ANGLE_DEG) {
                                    anglesValid = false
                                    break
                                }
                            }
                            if (!anglesValid) continue

                            // Gradient density verification (B4, pre-computed Sobel overload)
                            val gradDensity = verifyGradientDensity(
                                gx = gx,
                                gy = gy,
                                corners = ordered,
                                numSamplesPerSide = 50
                            )
                            if (gradDensity < TIER3_MIN_GRADIENT_DENSITY) continue

                            // Scoring: Radon response strength + geometric quality

                            // Radon score: average of all 4 peak responses, normalized
                            // by REFERENCE_GRADIENT (same reference as gradient density)
                            val avgRadonResponse = (hResp1 + hResp2 + vResp1 + vResp2) / 4.0f
                            val radonScore = (avgRadonResponse / REFERENCE_GRADIENT)
                                .coerceIn(0.0f, 1.0f).toDouble()

                            // Geometric score: area ratio + angle regularity (same formula as Tier 1)
                            val areaRatio = (area / imageArea).coerceIn(0.0, 1.0)
                            var totalAngleDeviation = 0.0
                            for (i in 0 until 4) {
                                val prev = ordered[(i + 3) % 4]
                                val curr = ordered[i]
                                val next = ordered[(i + 1) % 4]
                                val ang = lsdInteriorAngleDeg(prev, curr, next)
                                totalAngleDeviation += abs(ang - 90.0)
                            }
                            val angleRegularity = (1.0 - totalAngleDeviation / 360.0)
                                .coerceIn(0.0, 1.0)
                            val geometricScore = areaRatio * 0.5 + angleRegularity * 0.5

                            var combinedScore = radonScore * 0.5 + geometricScore * 0.5

                            // Step 5: Soft geometric priors for scoring boost

                            // Centering: document center close to image center → up to 10% bonus
                            val quadCx = ordered.sumOf { it.x } / 4.0
                            val quadCy = ordered.sumOf { it.y } / 4.0
                            val centerDistNorm = sqrt(
                                (quadCx - cx) * (quadCx - cx) +
                                        (quadCy - cy) * (quadCy - cy)
                            ) / sqrt(cx * cx + cy * cy)
                            // centerDistNorm is 0 when perfectly centered, ~1 when at a corner
                            val centeringBonus = 0.10 * (1.0 - centerDistNorm).coerceIn(0.0, 1.0)

                            // Aspect ratio: ratios between 0.5 and 1.0 (common document shapes) → up to 5% bonus
                            val sideLengths = DoubleArray(4)
                            for (i in 0 until 4) {
                                val pa = ordered[i]
                                val pb = ordered[(i + 1) % 4]
                                sideLengths[i] = sqrt(
                                    (pb.x - pa.x) * (pb.x - pa.x) +
                                            (pb.y - pa.y) * (pb.y - pa.y)
                                )
                            }
                            // Average opposing side pairs for aspect ratio
                            val avgWidth = (sideLengths[0] + sideLengths[2]) / 2.0
                            val avgHeight = (sideLengths[1] + sideLengths[3]) / 2.0
                            val aspectRatio = if (avgWidth > avgHeight) {
                                avgHeight / avgWidth
                            } else {
                                avgWidth / avgHeight
                            }
                            // aspectRatio is in (0, 1]. Common documents: 0.5 (wide) to 1.0 (square)
                            val arBonus = if (aspectRatio in 0.5..1.0) {
                                0.05 * (1.0 - abs(aspectRatio - 0.75) / 0.25).coerceIn(0.0, 1.0)
                            } else {
                                0.0
                            }

                            combinedScore += centeringBonus + arBonus
                            combinedScore = combinedScore.coerceIn(0.0, 1.0)

                            if (combinedScore > bestScore) {
                                bestCorners = ordered
                                bestScore = combinedScore
                            }
                        }
                    }
                }
            }
        }

        val ms = (System.nanoTime() - start) / 1_000_000.0

        if (bestCorners == null) {
            Log.d(TAG, "detectRectangleTier3: %.1f ms — no valid rectangle found".format(ms))
            return null
        }

        // Step 6: Map score to confidence in [TIER3_MIN_CONFIDENCE, TIER3_MAX_CONFIDENCE]
        val confidence = TIER3_MIN_CONFIDENCE +
                bestScore * (TIER3_MAX_CONFIDENCE - TIER3_MIN_CONFIDENCE)

        Log.d(TAG, "detectRectangleTier3: %.1f ms, score=%.3f, confidence=%.2f".format(
            ms, bestScore, confidence))

        return DocumentCorners(
            corners = bestCorners,
            detectionMs = ms,
            confidence = confidence
        )
    } finally {
        gx.release()
        gy.release()
    }
}

/**
 * Converts a Radon-parameterized line (angle + rho from image center) to
 * homogeneous line coefficients (a, b, c) where ax + by + c = 0.
 *
 * This is the Radon equivalent of [clusterToLine] for edge clusters.
 * The line direction is at [angleDeg] from horizontal, with perpendicular
 * offset [rhoPixels] from the image center.
 *
 * @param angleDeg Line direction in degrees from horizontal.
 * @param rhoPixels Perpendicular offset from image center, in pixels.
 * @param imageCenterX X coordinate of the image center.
 * @param imageCenterY Y coordinate of the image center.
 * @return Triple of (a, b, c) in the equation ax + by + c = 0.
 */
private fun radonLineToHomogeneous(
    angleDeg: Float,
    rhoPixels: Float,
    imageCenterX: Float,
    imageCenterY: Float
): Triple<Double, Double, Double> {
    // Normal direction: perpendicular to the line direction
    val normalRad = Math.toRadians((angleDeg + 90.0).toDouble())
    val a = cos(normalRad)
    val b = sin(normalRad)
    // Line equation: a*(x - cx) + b*(y - cy) = rho
    // → ax + by - (a*cx + b*cy + rho) = 0
    val c = -(a * imageCenterX + b * imageCenterY) - rhoPixels.toDouble()
    return Triple(a, b, c)
}
