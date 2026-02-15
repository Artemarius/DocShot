package com.docshot.cv

import org.opencv.core.Mat
import org.opencv.core.Point
import kotlin.math.roundToInt

private const val TAG = "DocShot:QuadValid"

/** Number of evenly-spaced sample points along each quad edge. */
private const val SAMPLES_PER_EDGE = 20

/** Default perpendicular search radius in pixels for edge support lookup. */
private const val DEFAULT_SEARCH_RADIUS = 3

/**
 * Validates detected quadrilaterals by checking how well their edges
 * align with actual Canny edge pixels in the image.
 *
 * A real document boundary produces strong, continuous edges that closely
 * follow the detected quad perimeter. False-positive detections — caused
 * by noise, texture, or low-contrast regions — will have little to no
 * supporting edge evidence.
 */
object QuadValidator {

    /**
     * Measures what fraction of a quadrilateral's perimeter is supported by
     * actual Canny edge pixels. A real document boundary will have high edge
     * density (>0.7), while a false detection in empty space will score low.
     *
     * The algorithm samples [SAMPLES_PER_EDGE] points along each of the 4 quad
     * edges (80 total) using linear interpolation. For each sample point it
     * checks a small square region of radius [searchRadius] in the binary edge
     * image for any non-zero (edge) pixel. The returned score is the fraction
     * of sample points that have at least one supporting edge pixel nearby.
     *
     * @param edges Binary edge image from Canny (CV_8UC1, values 0 or 255).
     * @param quad Four corners in order [TL, TR, BR, BL].
     * @param searchRadius How many pixels around each sample point to search
     *   for edge support. Larger values are more tolerant of slight
     *   misalignment between the detected quad and the true edge. Default is 3.
     * @return Score in [0.0, 1.0] — fraction of sampled points that have a
     *   nearby edge pixel. Returns 0.0 if the quad is empty or entirely
     *   outside the image.
     */
    fun edgeDensityScore(
        edges: Mat,
        quad: List<Point>,
        searchRadius: Int = DEFAULT_SEARCH_RADIUS
    ): Double {
        require(quad.size == 4) { "Expected 4 corners, got ${quad.size}" }

        val rows = edges.rows()
        val cols = edges.cols()
        if (rows == 0 || cols == 0) return 0.0

        var supported = 0
        var total = 0

        // Walk each of the 4 edges: TL->TR, TR->BR, BR->BL, BL->TL
        for (edgeIdx in 0 until 4) {
            val start = quad[edgeIdx]
            val end = quad[(edgeIdx + 1) % 4]

            for (s in 0..SAMPLES_PER_EDGE) {
                val t = s.toDouble() / SAMPLES_PER_EDGE
                val px = start.x + t * (end.x - start.x)
                val py = start.y + t * (end.y - start.y)

                val cx = px.roundToInt()
                val cy = py.roundToInt()

                // Skip sample points that fall entirely outside the image
                if (cx + searchRadius < 0 || cx - searchRadius >= cols) continue
                if (cy + searchRadius < 0 || cy - searchRadius >= rows) continue

                total++

                if (hasEdgePixelNearby(edges, cx, cy, searchRadius, cols, rows)) {
                    supported++
                }
            }
        }

        return if (total == 0) 0.0 else supported.toDouble() / total
    }

    /**
     * Checks whether any pixel in the square region
     * `[cx - radius, cx + radius] x [cy - radius, cy + radius]`
     * is non-zero in the edge image. Coordinates are clamped to image bounds.
     *
     * Uses [Mat.get] with (row, col) = (y, x) ordering.
     */
    private fun hasEdgePixelNearby(
        edges: Mat,
        cx: Int,
        cy: Int,
        radius: Int,
        cols: Int,
        rows: Int
    ): Boolean {
        val yMin = (cy - radius).coerceAtLeast(0)
        val yMax = (cy + radius).coerceAtMost(rows - 1)
        val xMin = (cx - radius).coerceAtLeast(0)
        val xMax = (cx + radius).coerceAtMost(cols - 1)

        for (y in yMin..yMax) {
            for (x in xMin..xMax) {
                // Mat.get(row, col) — row = y, col = x
                val pixel = edges.get(y, x)
                if (pixel != null && pixel[0] > 0.0) {
                    return true
                }
            }
        }
        return false
    }
}
