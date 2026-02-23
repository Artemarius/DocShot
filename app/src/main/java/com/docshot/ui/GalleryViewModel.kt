package com.docshot.ui

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import java.io.File
import com.docshot.cv.detectDocument
import com.docshot.cv.estimateAspectRatio
import com.docshot.cv.rectify
import com.docshot.cv.rectifyWithAspectRatio
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

/** Minimum confidence from the dual-regime AR estimator to use its ratio for gallery rectification. */
private const val GALLERY_AR_MIN_CONFIDENCE = 0.35

sealed class GalleryUiState {
    data object Idle : GalleryUiState()
    data object Loading : GalleryUiState()

    data class ManualAdjust(
        val bitmap: Bitmap,
        val corners: List<Point>
    ) : GalleryUiState()

    data object Rectifying : GalleryUiState()

    data class Result(val data: CaptureResultData) : GalleryUiState()

    data class Error(val message: String) : GalleryUiState()
}

class GalleryViewModel : ViewModel() {

    companion object {
        private const val RESULT_CACHE_DIR = "gallery_result_cache"
        private const val RESULT_CACHE_FILE = "gallery_result.jpg"
        private const val ORIGINAL_CACHE_FILE = "gallery_original.jpg"
    }

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
                    // Always go to corner adjustment — lets user verify/fix detection
                    _state.value = GalleryUiState.ManualAdjust(
                        bitmap = bitmap,
                        corners = fullResCorners
                    )
                } else {
                    // Auto-detection failed — manual adjustment with default corners
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

    /** Saves the displayed bitmap (with WB + filter applied) to the device gallery. */
    fun saveResult(context: Context, bitmap: Bitmap, onResult: (Boolean) -> Unit) {
        viewModelScope.launch {
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val displayName = "DocShot_$timestamp"
            val uri = saveBitmapToGallery(context, bitmap, displayName)
            onResult(uri != null)
        }
    }

    /** Shares the displayed bitmap (with WB + filter applied) via the system share sheet. */
    fun shareResult(context: Context, bitmap: Bitmap) {
        if (_state.value !is GalleryUiState.Result) return
        cacheResultForRestore(context)
        shareImage(context, bitmap)
    }

    fun reset() {
        val current = _state.value
        if (current is GalleryUiState.Result) {
            current.data.originalBitmap.recycle()
            current.data.rectifiedBitmap.recycle()
        }
        loadedBitmap = null
        // Cache will be cleared by GalleryScreen via clearResultCache when context is available
        _state.value = GalleryUiState.Idle
    }

    /**
     * Rotates the rectified bitmap 90 degrees clockwise and emits an updated Result.
     */
    fun rotateResult() {
        val current = _state.value
        if (current !is GalleryUiState.Result) return
        val data = current.data
        val oldBitmap = data.rectifiedBitmap

        val matrix = android.graphics.Matrix().apply { postRotate(90f) }
        val rotated = Bitmap.createBitmap(
            oldBitmap, 0, 0, oldBitmap.width, oldBitmap.height, matrix, true
        )
        if (rotated !== oldBitmap) oldBitmap.recycle()

        _state.value = GalleryUiState.Result(
            data.copy(
                rectifiedBitmap = rotated,
                manualRotationSteps = (data.manualRotationSteps + 1) % 4
            )
        )
    }

    /**
     * Transitions from Result back to ManualAdjust so the user can re-adjust corners.
     */
    fun adjustFromResult() {
        val current = _state.value
        if (current !is GalleryUiState.Result) return
        val data = current.data
        if (data.corners.isEmpty()) return

        // Recycle the rectified bitmap; user will re-rectify after adjustment
        data.rectifiedBitmap.recycle()

        _state.value = GalleryUiState.ManualAdjust(
            bitmap = data.originalBitmap,
            corners = data.corners
        )
    }

    /**
     * Re-warps the document from the original image with an adjusted aspect ratio.
     * Gallery imports have no camera intrinsics.
     */
    fun reWarpWithAspectRatio(targetRatio: Double) {
        val current = _state.value
        if (current !is GalleryUiState.Result) return
        val data = current.data
        if (data.corners.size != 4) return

        viewModelScope.launch(Dispatchers.Default) {
            var mat: Mat? = null
            var rectifiedMat: Mat? = null
            try {
                mat = bitmapToMat(data.originalBitmap)
                rectifiedMat = rectifyWithAspectRatio(
                    mat, data.corners, targetRatio, Imgproc.INTER_CUBIC
                )

                var newBitmap = matToBitmap(rectifiedMat)

                // Apply stored rotation: auto-orientation + manual rotation
                val totalSteps = (data.autoRotationSteps + data.manualRotationSteps) % 4
                if (totalSteps > 0) {
                    val degrees = totalSteps * 90f
                    val rotMatrix = android.graphics.Matrix().apply { postRotate(degrees) }
                    val rotated = Bitmap.createBitmap(
                        newBitmap, 0, 0, newBitmap.width, newBitmap.height, rotMatrix, true
                    )
                    if (rotated !== newBitmap) newBitmap.recycle()
                    newBitmap = rotated
                }

                val oldBitmap = data.rectifiedBitmap

                _state.value = GalleryUiState.Result(
                    data.copy(rectifiedBitmap = newBitmap)
                )
                oldBitmap.recycle()
            } catch (e: Exception) {
                Log.e(TAG, "reWarpWithAspectRatio failed: ${e.message}")
            } finally {
                mat?.release()
                rectifiedMat?.release()
            }
        }
    }

    // ── Result bitmap caching for process death recovery ────────────────

    /**
     * Saves both the rectified and original bitmaps to cacheDir/gallery_result_cache/ as JPEG files.
     * Called before sharing so that if the ViewModel is destroyed while the user is in the
     * share app, we can restore the result screen.
     */
    private fun cacheResultForRestore(context: Context) {
        val current = _state.value
        if (current !is GalleryUiState.Result) return

        viewModelScope.launch(Dispatchers.IO) {
            try {
                val cacheDir = File(context.cacheDir, RESULT_CACHE_DIR)
                if (!cacheDir.exists()) cacheDir.mkdirs()

                File(cacheDir, RESULT_CACHE_FILE).outputStream().use {
                    current.data.rectifiedBitmap.compress(Bitmap.CompressFormat.JPEG, 95, it)
                }
                File(cacheDir, ORIGINAL_CACHE_FILE).outputStream().use {
                    current.data.originalBitmap.compress(Bitmap.CompressFormat.JPEG, 95, it)
                }
                Log.d(TAG, "Gallery result cached for restore")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to cache gallery result: ${e.message}")
            }
        }
    }

    /**
     * Attempts to restore the result screen from cached bitmap files.
     * Returns true if restore succeeded (state transitions to Result), false otherwise.
     */
    fun restoreFromCache(context: Context): Boolean {
        val cacheDir = File(context.cacheDir, RESULT_CACHE_DIR)
        val rectifiedFile = File(cacheDir, RESULT_CACHE_FILE)
        val originalFile = File(cacheDir, ORIGINAL_CACHE_FILE)

        if (!rectifiedFile.exists() || !originalFile.exists()) return false

        try {
            val rectifiedBitmap = android.graphics.BitmapFactory.decodeFile(rectifiedFile.absolutePath)
            val originalBitmap = android.graphics.BitmapFactory.decodeFile(originalFile.absolutePath)

            if (rectifiedBitmap == null || originalBitmap == null) {
                rectifiedBitmap?.recycle()
                originalBitmap?.recycle()
                return false
            }

            _state.value = GalleryUiState.Result(
                CaptureResultData(
                    originalBitmap = originalBitmap,
                    rectifiedBitmap = rectifiedBitmap,
                    pipelineMs = 0.0,
                    confidence = 0.0,
                    corners = emptyList(),
                    normalizedCorners = floatArrayOf()
                )
            )
            Log.d(TAG, "Gallery result restored from cache")
            return true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to restore gallery result from cache: ${e.message}")
            return false
        }
    }

    /** Deletes cached result bitmap files. Call when the user explicitly leaves the result screen. */
    fun clearResultCache(context: Context) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val cacheDir = File(context.cacheDir, RESULT_CACHE_DIR)
                cacheDir.listFiles()?.forEach { it.delete() }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to clear gallery result cache: ${e.message}")
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        val current = _state.value
        when (current) {
            is GalleryUiState.Result -> {
                current.data.originalBitmap.recycle()
                current.data.rectifiedBitmap.recycle()
            }
            is GalleryUiState.ManualAdjust -> {
                loadedBitmap?.recycle()
            }
            else -> {}
        }
        loadedBitmap = null
        // Do NOT delete cache files here — they must survive ViewModel death
        // so restoreFromCache() can recover the result screen
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

                // Apply dual-regime aspect ratio estimation (angular correction for
                // mild perspective, projective for heavy skew). Gallery images have no
                // camera intrinsics, so the estimator falls back to the no-intrinsics
                // path (angular correction or closest-format snap without homography
                // disambiguation).
                val arEstimate = estimateAspectRatio(
                    corners = refined,
                    intrinsics = null
                )
                Log.d(TAG, "Gallery AR estimate: ratio=%.4f, format=%s, conf=%.3f".format(
                    arEstimate.estimatedRatio,
                    arEstimate.matchedFormat?.name ?: "Custom",
                    arEstimate.confidence
                ))

                // Use estimated ratio when confidence is sufficient (>= 0.35);
                // otherwise fall back to raw edge-length rectification.
                val rectified = if (arEstimate.estimationConfidence >= GALLERY_AR_MIN_CONFIDENCE) {
                    rectifyWithAspectRatio(
                        source = mat,
                        corners = refined,
                        targetRatio = arEstimate.estimatedRatio,
                        interpolation = Imgproc.INTER_CUBIC
                    )
                } else {
                    Log.d(TAG, "Gallery AR confidence too low (%.3f), using raw edge-length rectify"
                        .format(arEstimate.confidence))
                    rectify(mat, refined, Imgproc.INTER_CUBIC)
                }
                mat.release()

                val rectifiedBitmap = matToBitmap(rectified)
                rectified.release()

                val ms = (System.nanoTime() - start) / 1_000_000.0
                Log.d(TAG, "Gallery rectify: %.1f ms".format(ms))

                _state.value = GalleryUiState.Result(
                    CaptureResultData(
                        originalBitmap = bitmap,
                        rectifiedBitmap = rectifiedBitmap,
                        pipelineMs = ms,
                        corners = refined,
                        normalizedCorners = cornersToNormalized(refined, bitmap.width, bitmap.height)
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

/**
 * Converts full-res pixel corners to a flat FloatArray of 8 normalized [0,1] values.
 * Layout: [x0, y0, x1, y1, x2, y2, x3, y3]
 */
private fun cornersToNormalized(
    corners: List<Point>,
    imageWidth: Int,
    imageHeight: Int
): FloatArray {
    if (corners.size != 4 || imageWidth <= 0 || imageHeight <= 0) return floatArrayOf()
    val w = imageWidth.toFloat()
    val h = imageHeight.toFloat()
    return FloatArray(8) { i ->
        val corner = corners[i / 2]
        if (i % 2 == 0) (corner.x / w).toFloat() else (corner.y / h).toFloat()
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
