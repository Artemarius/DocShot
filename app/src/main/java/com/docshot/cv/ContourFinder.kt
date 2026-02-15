package com.docshot.cv

import android.util.Log
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

private const val TAG = "DocShot:Contour"

/** Minimum contour area as a fraction of image area. Lowered to 2% to support receipts/cards. */
private const val MIN_AREA_RATIO = 0.02

/** Epsilon for polygon approximation as a fraction of arc length. */
private const val APPROX_EPSILON_RATIO = 0.03

/**
 * Minimum area ratio for partial document detection. A contour must be at least
 * this large to be considered a potential partial document (prevents small noise
 * contours that happen to touch frame edges from triggering the hint).
 */
private const val PARTIAL_DOC_MIN_AREA_RATIO = 0.08

/** Distance in pixels from the image edge to consider a point "touching" the frame. */
private const val EDGE_PROXIMITY_PX = 5.0

/**
 * Result of contour analysis including both valid quads and partial-document detection.
 * @param quads List of quadrilateral contours, each as 4 Points.
 * @param hasPartialDocument True if a large contour touching 2+ frame edges was
 *   found but could not be approximated as a valid quad.
 */
data class ContourAnalysis(
    val quads: List<List<Point>>,
    val hasPartialDocument: Boolean
)

/**
 * Finds quadrilateral contours in a binary edge image.
 * Filters by minimum area (2% of image) and 4-vertex polygon approximation.
 * Returns a list of quadrilaterals, each as 4 Points.
 */
fun findQuadrilaterals(edges: Mat, imageSize: Size): List<List<Point>> {
    return analyzeContours(edges, imageSize).quads
}

/**
 * Analyzes contours in a binary edge image, finding both valid quadrilaterals
 * and detecting partial documents (large contours touching frame edges).
 *
 * @param edges Binary edge image from Canny.
 * @param imageSize Size of the original image (for area calculations and edge detection).
 * @return [ContourAnalysis] with valid quads and partial-document flag.
 */
fun analyzeContours(edges: Mat, imageSize: Size): ContourAnalysis {
    val start = System.nanoTime()

    val contours = mutableListOf<MatOfPoint>()
    val hierarchy = Mat()
    Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
    hierarchy.release()

    val totalArea = imageSize.width * imageSize.height
    val minArea = totalArea * MIN_AREA_RATIO
    val partialDocMinArea = totalArea * PARTIAL_DOC_MIN_AREA_RATIO
    val quads = mutableListOf<List<Point>>()
    var hasPartialDocument = false

    for (contour in contours) {
        val area = Imgproc.contourArea(contour)
        if (area < minArea) {
            contour.release()
            continue
        }

        val contour2f = MatOfPoint2f(*contour.toArray())
        val points = contour.toList()
        contour.release()

        val peri = Imgproc.arcLength(contour2f, true)
        val approx = MatOfPoint2f()
        Imgproc.approxPolyDP(contour2f, approx, APPROX_EPSILON_RATIO * peri, true)
        contour2f.release()

        val approxPoints = approx.toList()
        approx.release()

        if (approxPoints.size == 4) {
            quads.add(approxPoints)
        } else if (!hasPartialDocument && area >= partialDocMinArea) {
            // Check if this large non-quad contour touches 2+ frame edges,
            // indicating a document that extends beyond the frame
            if (touchesFrameEdges(points, imageSize) >= 2) {
                hasPartialDocument = true
            }
        }
    }

    val ms = (System.nanoTime() - start) / 1_000_000.0
    Log.d(TAG, "analyzeContours: %.1f ms (contours=%d, quads=%d, partial=%s)".format(
        ms, contours.size, quads.size, hasPartialDocument))
    return ContourAnalysis(quads = quads, hasPartialDocument = hasPartialDocument)
}

/**
 * Counts how many distinct frame edges (top, right, bottom, left) the contour touches.
 * A point is considered "touching" an edge if it's within [EDGE_PROXIMITY_PX] pixels.
 *
 * @return Number of distinct edges touched (0-4).
 */
internal fun touchesFrameEdges(points: List<Point>, imageSize: Size): Int {
    var touchesTop = false
    var touchesRight = false
    var touchesBottom = false
    var touchesLeft = false

    val maxX = imageSize.width - 1
    val maxY = imageSize.height - 1

    for (pt in points) {
        if (pt.y <= EDGE_PROXIMITY_PX) touchesTop = true
        if (pt.x >= maxX - EDGE_PROXIMITY_PX) touchesRight = true
        if (pt.y >= maxY - EDGE_PROXIMITY_PX) touchesBottom = true
        if (pt.x <= EDGE_PROXIMITY_PX) touchesLeft = true
    }

    var count = 0
    if (touchesTop) count++
    if (touchesRight) count++
    if (touchesBottom) count++
    if (touchesLeft) count++
    return count
}
