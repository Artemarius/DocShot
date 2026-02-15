package com.docshot.camera

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.util.Log
import androidx.camera.core.ImageProxy
import com.docshot.cv.detectAndCorrect
import com.docshot.cv.detectDocument
import com.docshot.cv.rectify
import com.docshot.cv.refineCorners
import com.docshot.util.bitmapToMat
import com.docshot.util.matToBitmap
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

private const val TAG = "DocShot:Capture"
private const val MAX_DETECTION_WIDTH = 1000

/**
 * Result of a full-resolution capture pipeline.
 * Caller owns the bitmaps and should recycle them when no longer needed.
 */
data class CaptureResult(
    val originalBitmap: Bitmap,
    val rectifiedBitmap: Bitmap,
    val pipelineMs: Double,
    val confidence: Double = 0.0
)

/**
 * Processes a full-resolution [ImageProxy] through the complete document pipeline:
 * decode (JPEG or YUV) → rotate → detect → refine corners → rectify → bitmaps.
 *
 * Handles both JPEG format (common on ImageCapture with MAXIMIZE_QUALITY)
 * and YUV_420_888 format.
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

    var stage = "image decode"
    try {
        val rotation = imageProxy.imageInfo.rotationDegrees

        bgrMat = when (imageProxy.format) {
            ImageFormat.JPEG -> {
                Log.d(TAG, "Capture format: JPEG, rotation=%d°".format(rotation))
                jpegImageProxyToBgrMat(imageProxy, rotation).also {
                    // Rotation handled during JPEG decode, so skip rotateMat
                }
            }
            ImageFormat.YUV_420_888 -> {
                Log.d(TAG, "Capture format: YUV_420_888, rotation=%d°".format(rotation))
                yuvImageProxyToBgrMat(imageProxy)
            }
            else -> error("Unsupported ImageProxy format: ${imageProxy.format}")
        }
        Log.d(TAG, "Decoded: %dx%d".format(bgrMat.cols(), bgrMat.rows()))

        // Apply rotation — for JPEG this was already done during decode, skip it
        stage = "rotation"
        if (imageProxy.format == ImageFormat.JPEG) {
            rotatedMat = bgrMat
            bgrMat = null // Prevent double-release
        } else {
            rotatedMat = rotateMat(bgrMat, rotation)
            bgrMat.release()
            bgrMat = null
        }
        Log.d(TAG, "Rotated: %dx%d".format(rotatedMat.cols(), rotatedMat.rows()))

        // Detect document — downscale for detection since kernels are tuned for ~640px
        stage = "detection"
        val detectionMat: Mat
        val scaleFactor: Double
        if (rotatedMat.cols() > MAX_DETECTION_WIDTH) {
            val scale = MAX_DETECTION_WIDTH.toDouble() / rotatedMat.cols()
            val newH = (rotatedMat.rows() * scale).toInt()
            detectionMat = Mat()
            Imgproc.resize(rotatedMat, detectionMat,
                org.opencv.core.Size(MAX_DETECTION_WIDTH.toDouble(), newH.toDouble()))
            scaleFactor = 1.0 / scale
        } else {
            detectionMat = rotatedMat
            scaleFactor = 1.0
        }
        val detection = detectDocument(detectionMat)
        if (detectionMat !== rotatedMat) detectionMat.release()

        if (detection == null) {
            Log.d(TAG, "processCapture: no document found")
            rotatedMat.release()
            rotatedMat = null
            return null
        }

        // Scale corners back to full image coordinates
        val fullResCorners = detection.corners.map { pt ->
            org.opencv.core.Point(pt.x * scaleFactor, pt.y * scaleFactor)
        }

        // Sub-pixel corner refinement on grayscale
        stage = "corner refinement"
        grayMat = Mat()
        val colorConversion = if (rotatedMat.channels() == 4)
            Imgproc.COLOR_RGBA2GRAY else Imgproc.COLOR_BGR2GRAY
        Imgproc.cvtColor(rotatedMat, grayMat, colorConversion)
        val refinedCorners = refineCorners(grayMat, fullResCorners)
        grayMat.release()
        grayMat = null

        // Rectify with high-quality interpolation
        stage = "rectification"
        rectifiedMat = rectify(rotatedMat, refinedCorners, Imgproc.INTER_CUBIC)

        // Detect and correct document orientation (upside-down, sideways)
        stage = "orientation correction"
        val (orientedMat, orientation) = detectAndCorrect(rectifiedMat)
        if (orientedMat !== rectifiedMat) {
            rectifiedMat.release()
            rectifiedMat = orientedMat
        }
        Log.d(TAG, "Orientation: %s".format(orientation.name))

        // Convert to bitmaps
        stage = "bitmap conversion"
        val originalBitmap = matToBitmap(rotatedMat)
        val rectifiedBitmap = matToBitmap(rectifiedMat)

        val ms = (System.nanoTime() - start) / 1_000_000.0
        Log.d(TAG, "processCapture: %.1f ms total, confidence: %.2f".format(
            ms, detection.confidence))

        return CaptureResult(
            originalBitmap = originalBitmap,
            rectifiedBitmap = rectifiedBitmap,
            pipelineMs = ms,
            confidence = detection.confidence
        )
    } catch (e: Exception) {
        throw RuntimeException("Capture failed at $stage: ${e.message}", e)
    } finally {
        bgrMat?.release()
        rotatedMat?.release()
        grayMat?.release()
        rectifiedMat?.release()
    }
}

/**
 * Decodes a JPEG [ImageProxy] to a BGR Mat with rotation already applied.
 * ImageCapture with CAPTURE_MODE_MAXIMIZE_QUALITY often returns JPEG on modern devices.
 */
private fun jpegImageProxyToBgrMat(imageProxy: ImageProxy, rotationDegrees: Int): Mat {
    val buffer = imageProxy.planes[0].buffer
    val bytes = ByteArray(buffer.remaining())
    buffer.get(bytes)

    val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        ?: error("Failed to decode JPEG from ImageCapture")

    // Apply rotation if needed
    val rotated = if (rotationDegrees != 0) {
        val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
        val r = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        if (r !== bitmap) bitmap.recycle()
        r
    } else {
        bitmap
    }

    val mat = bitmapToMat(rotated)
    rotated.recycle()
    return mat
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
