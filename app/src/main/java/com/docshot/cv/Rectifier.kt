package com.docshot.cv

import android.util.Log
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.sqrt

private const val TAG = "DocShot:Rectify"

/**
 * Applies a perspective warp to extract a rectangular document from the source image.
 * Corners must be ordered [TL, TR, BR, BL].
 * Output dimensions are derived from the longest edge pairs to preserve aspect ratio.
 * Uses INTER_CUBIC for final quality (use INTER_LINEAR for preview).
 * Caller must release the returned Mat.
 */
fun rectify(
    source: Mat,
    corners: List<Point>,
    interpolation: Int = Imgproc.INTER_CUBIC
): Mat {
    require(corners.size == 4) { "Expected 4 ordered corners [TL, TR, BR, BL]" }
    val start = System.nanoTime()

    val (tl, tr, br, bl) = corners

    // Derive output dimensions from the longest edge pair to preserve document aspect ratio
    val widthTop = distance(tl, tr)
    val widthBottom = distance(bl, br)
    val outWidth = maxOf(widthTop, widthBottom).toInt()

    val heightLeft = distance(tl, bl)
    val heightRight = distance(tr, br)
    val outHeight = maxOf(heightLeft, heightRight).toInt()

    check(outWidth > 0 && outHeight > 0) { "Invalid output dimensions: ${outWidth}x${outHeight}" }

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
    Imgproc.warpPerspective(source, output, transform, Size(outWidth.toDouble(), outHeight.toDouble()), interpolation)
    transform.release()

    val ms = (System.nanoTime() - start) / 1_000_000.0
    Log.d(TAG, "rectify: %.1f ms (output=%dx%d)".format(ms, outWidth, outHeight))
    return output
}

private fun distance(a: Point, b: Point): Double {
    val dx = a.x - b.x
    val dy = a.y - b.y
    return sqrt(dx * dx + dy * dy)
}
