package com.docshot.cv

import android.util.Log
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.core.TermCriteria
import org.opencv.imgproc.Imgproc

private const val TAG = "DocShot:CornerRefine"

/**
 * Refines detected document corners to sub-pixel accuracy using [Imgproc.cornerSubPix].
 * This improves warp quality by placing corners precisely on actual edges rather than
 * relying on the contour approximation grid.
 *
 * @param gray Grayscale image (same resolution as the corners).
 * @param corners 4 corner points [TL, TR, BR, BL] from [detectDocument].
 * @return Refined corner points at sub-pixel positions.
 */
fun refineCorners(gray: Mat, corners: List<Point>): List<Point> {
    require(corners.size == 4) { "Expected 4 corners, got ${corners.size}" }
    val start = System.nanoTime()

    val winSize = 5.0
    // Clamp corners to be within the image bounds so cornerSubPix doesn't assert.
    // Corners from normalized preview coordinates can land exactly on the edge
    // (e.g., x=width) which is outside valid pixel range [0, width-1].
    val maxX = gray.cols() - 1 - winSize
    val maxY = gray.rows() - 1 - winSize
    val clamped = corners.map { pt ->
        Point(pt.x.coerceIn(winSize, maxX), pt.y.coerceIn(winSize, maxY))
    }

    val cornersMat = MatOfPoint2f(*clamped.toTypedArray())

    // 11x11 search window (winSize=5 means 2*5+1=11)
    // 30 iterations or 0.01 pixel accuracy, whichever comes first
    Imgproc.cornerSubPix(
        gray,
        cornersMat,
        Size(winSize, winSize),
        Size(-1.0, -1.0),
        TermCriteria(
            TermCriteria.COUNT + TermCriteria.EPS,
            30,
            0.01
        )
    )

    val refined = cornersMat.toList()
    cornersMat.release()

    val ms = (System.nanoTime() - start) / 1_000_000.0
    Log.d(TAG, "refineCorners: %.1f ms".format(ms))
    return refined
}
