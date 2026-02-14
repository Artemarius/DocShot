package com.docshot.camera

import android.graphics.Bitmap
import android.util.Log
import androidx.camera.core.ImageProxy
import com.docshot.cv.detectDocument
import com.docshot.cv.rectify
import com.docshot.cv.refineCorners
import com.docshot.util.matToBitmap
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

private const val TAG = "DocShot:Capture"

/**
 * Result of a full-resolution capture pipeline.
 * Caller owns the bitmaps and should recycle them when no longer needed.
 */
data class CaptureResult(
    val originalBitmap: Bitmap,
    val rectifiedBitmap: Bitmap,
    val pipelineMs: Double
)

/**
 * Processes a full-resolution [ImageProxy] through the complete document pipeline:
 * YUV→BGR → rotate → detect → refine corners → rectify → bitmaps.
 *
 * All intermediate Mats are released in a finally block.
 * The caller must close the [ImageProxy] after this returns.
 *
 * @return [CaptureResult] with original and rectified bitmaps, or null if no document found.
 */
fun processCapture(imageProxy: ImageProxy): CaptureResult? {
    val start = System.nanoTime()
    var bgrMat: Mat? = null
    var rotatedMat: Mat? = null
    var grayMat: Mat? = null
    var rectifiedMat: Mat? = null

    try {
        bgrMat = yuvImageProxyToBgrMat(imageProxy)
        Log.d(TAG, "YUV→BGR: %dx%d".format(bgrMat.cols(), bgrMat.rows()))

        // Apply rotation to match display orientation
        val rotation = imageProxy.imageInfo.rotationDegrees
        rotatedMat = rotateMat(bgrMat, rotation)
        bgrMat.release()
        bgrMat = null

        Log.d(TAG, "Rotated %d°: %dx%d".format(rotation, rotatedMat.cols(), rotatedMat.rows()))

        // Detect document
        val detection = detectDocument(rotatedMat)
        if (detection == null) {
            Log.d(TAG, "processCapture: no document found")
            rotatedMat.release()
            return null
        }

        // Sub-pixel corner refinement on grayscale
        grayMat = Mat()
        Imgproc.cvtColor(rotatedMat, grayMat, Imgproc.COLOR_BGR2GRAY)
        val refinedCorners = refineCorners(grayMat, detection.corners)
        grayMat.release()
        grayMat = null

        // Rectify with high-quality interpolation
        rectifiedMat = rectify(rotatedMat, refinedCorners, Imgproc.INTER_CUBIC)

        // Convert to bitmaps
        val originalBitmap = matToBitmap(rotatedMat)
        val rectifiedBitmap = matToBitmap(rectifiedMat)

        val ms = (System.nanoTime() - start) / 1_000_000.0
        Log.d(TAG, "processCapture: %.1f ms total".format(ms))

        return CaptureResult(
            originalBitmap = originalBitmap,
            rectifiedBitmap = rectifiedBitmap,
            pipelineMs = ms
        )
    } finally {
        bgrMat?.release()
        rotatedMat?.release()
        grayMat?.release()
        rectifiedMat?.release()
    }
}

/**
 * Converts a YUV_420_888 [ImageProxy] to a BGR [Mat].
 *
 * Handles the common NV21 interleaved layout (pixelStride=2 on U/V planes)
 * as well as planar I420 (pixelStride=1). Builds a contiguous YUV byte array
 * and converts via [Imgproc.cvtColor].
 */
private fun yuvImageProxyToBgrMat(imageProxy: ImageProxy): Mat {
    val width = imageProxy.width
    val height = imageProxy.height
    val yPlane = imageProxy.planes[0]
    val uPlane = imageProxy.planes[1]
    val vPlane = imageProxy.planes[2]

    val yRowStride = yPlane.rowStride
    val uvRowStride = uPlane.rowStride
    val uvPixelStride = uPlane.pixelStride

    val yBuffer = yPlane.buffer
    val uBuffer = uPlane.buffer
    val vBuffer = vPlane.buffer
    yBuffer.rewind()
    uBuffer.rewind()
    vBuffer.rewind()

    // NV21 layout: Y plane followed by interleaved VU
    val yuvSize = width * height * 3 / 2
    val yuvBytes = ByteArray(yuvSize)

    // Copy Y plane
    if (yRowStride == width) {
        yBuffer.get(yuvBytes, 0, width * height)
    } else {
        for (row in 0 until height) {
            yBuffer.position(row * yRowStride)
            yBuffer.get(yuvBytes, row * width, width)
        }
    }

    // Copy UV planes into NV21 interleaved VU format
    val uvHeight = height / 2
    val uvWidth = width / 2
    var uvOffset = width * height

    if (uvPixelStride == 2) {
        // Most devices: U/V planes are interleaved (NV21-like)
        // V plane already has interleaved VU data in many implementations
        for (row in 0 until uvHeight) {
            for (col in 0 until uvWidth) {
                val vIdx = row * uvRowStride + col * uvPixelStride
                val uIdx = row * uvRowStride + col * uvPixelStride
                yuvBytes[uvOffset++] = vBuffer.get(vIdx)
                yuvBytes[uvOffset++] = uBuffer.get(uIdx)
            }
        }
    } else {
        // Planar I420: pixelStride == 1
        for (row in 0 until uvHeight) {
            for (col in 0 until uvWidth) {
                val idx = row * uvRowStride + col
                yuvBytes[uvOffset++] = vBuffer.get(idx)
                yuvBytes[uvOffset++] = uBuffer.get(idx)
            }
        }
    }

    val yuvMat = Mat(height + height / 2, width, CvType.CV_8UC1)
    yuvMat.put(0, 0, yuvBytes)

    val bgrMat = Mat()
    Imgproc.cvtColor(yuvMat, bgrMat, Imgproc.COLOR_YUV2BGR_NV21)
    yuvMat.release()

    return bgrMat
}

/**
 * Rotates a Mat by the given degrees (0, 90, 180, 270) to match display orientation.
 * Returns a new Mat; the caller should release the original if no longer needed.
 */
private fun rotateMat(mat: Mat, rotationDegrees: Int): Mat {
    return when (rotationDegrees) {
        90 -> {
            val rotated = Mat()
            Core.rotate(mat, rotated, Core.ROTATE_90_CLOCKWISE)
            rotated
        }
        180 -> {
            val rotated = Mat()
            Core.rotate(mat, rotated, Core.ROTATE_180)
            rotated
        }
        270 -> {
            val rotated = Mat()
            Core.rotate(mat, rotated, Core.ROTATE_90_COUNTERCLOCKWISE)
            rotated
        }
        else -> {
            val copy = Mat()
            mat.copyTo(copy)
            copy
        }
    }
}
