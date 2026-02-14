package com.docshot.ui

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.docshot.cv.detectDocument
import com.docshot.cv.rectify
import com.docshot.cv.refineCorners
import com.docshot.util.bitmapToMat
import com.docshot.util.loadGalleryImage
import com.docshot.util.matToBitmap
import com.docshot.util.saveBitmapToGallery
import com.docshot.util.shareImage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

private const val TAG = "DocShot:GalleryVM"

/** Detection kernels are tuned for ~640px frames; downscale large images for reliable detection. */
private const val MAX_DETECTION_WIDTH = 1000

sealed class GalleryUiState {
    data object Idle : GalleryUiState()
    data object Loading : GalleryUiState()

    data class Detected(
        val bitmap: Bitmap,
        val corners: List<Point>,
        val detectionMs: Double
    ) : GalleryUiState()

    data class ManualAdjust(
        val bitmap: Bitmap,
        val corners: List<Point>
    ) : GalleryUiState()

    data object Rectifying : GalleryUiState()

    data class Result(val data: CaptureResultData) : GalleryUiState()

    data class Error(val message: String) : GalleryUiState()
}

class GalleryViewModel : ViewModel() {

    private val _state = MutableStateFlow<GalleryUiState>(GalleryUiState.Idle)
    val state: StateFlow<GalleryUiState> = _state

    /** Holds the loaded bitmap across state transitions (Detected / ManualAdjust / Rectifying). */
    private var loadedBitmap: Bitmap? = null

    fun loadAndDetect(context: Context, uri: Uri) {
        if (_state.value !is GalleryUiState.Idle) return

        _state.value = GalleryUiState.Loading

        viewModelScope.launch(Dispatchers.Default) {
            try {
                val bitmap = loadGalleryImage(context, uri)
                loadedBitmap = bitmap

                val mat = bitmapToMat(bitmap)

                // Downscale for detection — kernels (9x9 blur, 3x3 morph) are
                // tuned for ~640px camera frames and don't bridge gaps at 4000px
                val detectionMat: Mat
                val scaleFactor: Double
                if (mat.cols() > MAX_DETECTION_WIDTH) {
                    val scale = MAX_DETECTION_WIDTH.toDouble() / mat.cols()
                    val newH = (mat.rows() * scale).toInt()
                    detectionMat = Mat()
                    Imgproc.resize(mat, detectionMat, Size(MAX_DETECTION_WIDTH.toDouble(), newH.toDouble()))
                    scaleFactor = 1.0 / scale
                } else {
                    detectionMat = mat
                    scaleFactor = 1.0
                }

                val detection = detectDocument(detectionMat)
                if (detectionMat !== mat) detectionMat.release()
                mat.release()

                if (detection != null) {
                    // Scale corners back to full image coordinates
                    val fullResCorners = detection.corners.map { pt ->
                        Point(pt.x * scaleFactor, pt.y * scaleFactor)
                    }
                    _state.value = GalleryUiState.Detected(
                        bitmap = bitmap,
                        corners = fullResCorners,
                        detectionMs = detection.detectionMs
                    )
                } else {
                    // Auto-detection failed — go to manual adjustment with default corners
                    val defaultCorners = defaultCorners(bitmap.width, bitmap.height)
                    _state.value = GalleryUiState.ManualAdjust(
                        bitmap = bitmap,
                        corners = defaultCorners
                    )
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load/detect gallery image", e)
                loadedBitmap?.recycle()
                loadedBitmap = null
                _state.value = GalleryUiState.Error("Failed to load image: ${e.message}")
                resetAfterDelay()
            }
        }
    }

    fun acceptDetection() {
        val current = _state.value
        if (current !is GalleryUiState.Detected) return
        rectifyWithCorners(current.bitmap, current.corners)
    }

    fun enterManualAdjust() {
        val current = _state.value
        if (current !is GalleryUiState.Detected) return
        _state.value = GalleryUiState.ManualAdjust(
            bitmap = current.bitmap,
            corners = current.corners
        )
    }

    fun updateCorner(index: Int, point: Point) {
        val current = _state.value
        if (current !is GalleryUiState.ManualAdjust) return
        require(index in 0..3) { "Corner index must be 0-3, got $index" }

        val updatedCorners = current.corners.toMutableList()
        updatedCorners[index] = point
        _state.value = current.copy(corners = updatedCorners)
    }

    fun applyManualCorners() {
        val current = _state.value
        if (current !is GalleryUiState.ManualAdjust) return
        rectifyWithCorners(current.bitmap, current.corners)
    }

    fun saveResult(context: Context, onResult: (Boolean) -> Unit) {
        val current = _state.value
        if (current !is GalleryUiState.Result) return

        viewModelScope.launch {
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val displayName = "DocShot_$timestamp"
            val uri = saveBitmapToGallery(context, current.data.rectifiedBitmap, displayName)
            onResult(uri != null)
        }
    }

    fun shareResult(context: Context) {
        val current = _state.value
        if (current !is GalleryUiState.Result) return
        shareImage(context, current.data.rectifiedBitmap)
    }

    fun reset() {
        val current = _state.value
        if (current is GalleryUiState.Result) {
            current.data.originalBitmap.recycle()
            current.data.rectifiedBitmap.recycle()
        }
        loadedBitmap = null
        _state.value = GalleryUiState.Idle
    }

    override fun onCleared() {
        super.onCleared()
        val current = _state.value
        when (current) {
            is GalleryUiState.Result -> {
                current.data.originalBitmap.recycle()
                current.data.rectifiedBitmap.recycle()
            }
            is GalleryUiState.Detected,
            is GalleryUiState.ManualAdjust -> {
                loadedBitmap?.recycle()
            }
            else -> {}
        }
        loadedBitmap = null
    }

    private fun rectifyWithCorners(bitmap: Bitmap, corners: List<Point>) {
        _state.value = GalleryUiState.Rectifying

        viewModelScope.launch(Dispatchers.Default) {
            try {
                val start = System.nanoTime()
                val mat = bitmapToMat(bitmap)

                // Convert to grayscale for corner refinement
                val gray = org.opencv.core.Mat()
                Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGBA2GRAY)
                val refined = refineCorners(gray, corners)
                gray.release()

                val rectified = rectify(mat, refined, Imgproc.INTER_CUBIC)
                mat.release()

                val rectifiedBitmap = matToBitmap(rectified)
                rectified.release()

                val ms = (System.nanoTime() - start) / 1_000_000.0
                Log.d(TAG, "Gallery rectify: %.1f ms".format(ms))

                _state.value = GalleryUiState.Result(
                    CaptureResultData(
                        originalBitmap = bitmap,
                        rectifiedBitmap = rectifiedBitmap,
                        pipelineMs = ms
                    )
                )
                // Don't null loadedBitmap — it's now owned by CaptureResultData as originalBitmap
            } catch (e: Exception) {
                Log.e(TAG, "Rectification failed", e)
                _state.value = GalleryUiState.Error("Rectification failed: ${e.message}")
                resetAfterDelay()
            }
        }
    }

    private fun resetAfterDelay() {
        viewModelScope.launch {
            withContext(Dispatchers.Main) {
                delay(2000)
                if (_state.value is GalleryUiState.Error) {
                    _state.value = GalleryUiState.Idle
                }
            }
        }
    }
}

/** Default corners inset 10% from image edges, used when auto-detection fails. */
private fun defaultCorners(width: Int, height: Int): List<Point> {
    val margin = 0.1
    val left = width * margin
    val right = width * (1 - margin)
    val top = height * margin
    val bottom = height * (1 - margin)
    return listOf(
        Point(left, top),      // TL
        Point(right, top),     // TR
        Point(right, bottom),  // BR
        Point(left, bottom)    // BL
    )
}
