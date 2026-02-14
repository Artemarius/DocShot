package com.docshot.cv

import android.util.Log
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

private const val TAG = "DocShot:Contour"

/** Minimum contour area as a fraction of image area. */
private const val MIN_AREA_RATIO = 0.10

/** Epsilon for polygon approximation as a fraction of arc length. */
private const val APPROX_EPSILON_RATIO = 0.03

/**
 * Finds quadrilateral contours in a binary edge image.
 * Filters by minimum area (10% of image) and 4-vertex polygon approximation.
 * Returns a list of quadrilaterals, each as 4 Points.
 */
fun findQuadrilaterals(edges: Mat, imageSize: Size): List<List<Point>> {
    val start = System.nanoTime()

    val contours = mutableListOf<MatOfPoint>()
    val hierarchy = Mat()
    Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
    hierarchy.release()

    val minArea = imageSize.width * imageSize.height * MIN_AREA_RATIO
    val quads = mutableListOf<List<Point>>()

    for (contour in contours) {
        val area = Imgproc.contourArea(contour)
        if (area < minArea) {
            contour.release()
            continue
        }

        val contour2f = MatOfPoint2f(*contour.toArray())
        contour.release()

        val peri = Imgproc.arcLength(contour2f, true)
        val approx = MatOfPoint2f()
        Imgproc.approxPolyDP(contour2f, approx, APPROX_EPSILON_RATIO * peri, true)
        contour2f.release()

        val points = approx.toList()
        approx.release()

        if (points.size == 4) {
            quads.add(points)
        }
    }

    val ms = (System.nanoTime() - start) / 1_000_000.0
    Log.d(TAG, "findQuadrilaterals: %.1f ms (contours=%d, quads=%d)".format(ms, contours.size, quads.size))
    return quads
}
