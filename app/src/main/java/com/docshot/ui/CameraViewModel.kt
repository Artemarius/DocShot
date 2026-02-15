package com.docshot.ui

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.docshot.camera.FrameAnalyzer
import com.docshot.camera.processCapture
import com.docshot.cv.detectAndCorrect
import com.docshot.cv.rectify
import com.docshot.cv.refineCorners
import com.docshot.util.bitmapToMat
import com.docshot.util.matToBitmap
import com.docshot.util.saveBitmapToGallery
import com.docshot.util.shareImage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.lang.ref.WeakReference
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

private const val TAG = "DocShot:ViewModel"
private const val LOW_CONFIDENCE_FALLBACK_THRESHOLD = 0.65

/**
 * UI state for the detected document overlay.
 * Corners are in normalized [0,1] display coordinates (rotation already applied).
 */
data class DetectionUiState(
    val normalizedCorners: List<FloatArray>? = null,
    val sourceWidth: Int = 0,
    val sourceHeight: Int = 0,
    val detectionMs: Double = 0.0,
    val isStable: Boolean = false,
    val stabilityProgress: Float = 0f,
    val isPartialDocument: Boolean = false
)

/** State machine for the capture -> result flow. */
sealed class CameraUiState {
    data object Idle : CameraUiState()
    data object Capturing : CameraUiState()
    data class Processing(val message: String) : CameraUiState()
    data class Result(val data: CaptureResultData) : CameraUiState()
    data class LowConfidence(
        val originalBitmap: Bitmap,
        val corners: List<org.opencv.core.Point>,
        val confidence: Double
    ) : CameraUiState()
    data class Error(val message: String) : CameraUiState()
}

data class CaptureResultData(
    val originalBitmap: Bitmap,
    val rectifiedBitmap: Bitmap,
    val pipelineMs: Double,
    val confidence: Double = 0.0
)

class CameraViewModel : ViewModel() {

    private val _detectionState = MutableStateFlow(DetectionUiState())
    val detectionState: StateFlow<DetectionUiState> = _detectionState

    private val _cameraState = MutableStateFlow<CameraUiState>(CameraUiState.Idle)
    val cameraState: StateFlow<CameraUiState> = _cameraState

    /** Whether auto-capture triggers when document is stably detected. */
    private val _autoCapEnabled = MutableStateFlow(true)
    val autoCapEnabled: StateFlow<Boolean> = _autoCapEnabled

    /** Emitted when capture triggers (auto or manual) so CameraScreen can fire haptic. */
    private val _hapticEvent = MutableSharedFlow<Unit>(extraBufferCapacity = 1)
    val hapticEvent: SharedFlow<Unit> = _hapticEvent

    /** Weak reference to activity context for auto-capture. Set via [setContext]. */
    private var contextRef: WeakReference<Context>? = null

    /** Set by CameraScreen when binding the camera. */
    var imageCapture: ImageCapture? = null

    /**
     * Store a context reference for auto-capture. Must be called from CameraScreen
     * during composition (e.g., in a LaunchedEffect or remember block).
     * Uses a WeakReference to avoid leaking the Activity.
     */
    fun setContext(context: Context) {
        contextRef = WeakReference(context)
    }

    fun toggleAutoCap() {
        _autoCapEnabled.value = !_autoCapEnabled.value
        Log.d(TAG, "Auto-capture toggled: ${_autoCapEnabled.value}")
    }

    val frameAnalyzer = FrameAnalyzer { result ->
        _detectionState.value = DetectionUiState(
            normalizedCorners = result.normalizedCorners,
            sourceWidth = result.sourceWidth,
            sourceHeight = result.sourceHeight,
            detectionMs = result.detectionMs,
            isStable = result.isStable,
            stabilityProgress = result.stabilityProgress,
            isPartialDocument = result.isPartialDocument
        )

        // Auto-capture: trigger when stable, high confidence, enabled, and idle
        if (result.isStable
            && result.confidence >= LOW_CONFIDENCE_FALLBACK_THRESHOLD
            && _autoCapEnabled.value
            && _cameraState.value is CameraUiState.Idle
        ) {
            val ctx = contextRef?.get()
            if (ctx != null) {
                Log.d(TAG, "Auto-capture triggered (stable detection)")
                captureDocument(ctx)
            }
        }
    }

    /**
     * Triggers the full capture pipeline: take photo -> detect -> refine -> rectify -> result.
     */
    fun captureDocument(context: Context) {
        val capture = imageCapture ?: run {
            Log.e(TAG, "ImageCapture not initialized")
            return
        }
        if (_cameraState.value !is CameraUiState.Idle) return

        _cameraState.value = CameraUiState.Capturing
        _hapticEvent.tryEmit(Unit)

        capture.takePicture(
            ContextCompat.getMainExecutor(context),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(imageProxy: ImageProxy) {
                    viewModelScope.launch(Dispatchers.Default) {
                        _cameraState.value = CameraUiState.Processing("Processing document...")
                        try {
                            val result = processCapture(imageProxy)
                            imageProxy.close()

                            if (result != null) {
                                if (result.confidence >= LOW_CONFIDENCE_FALLBACK_THRESHOLD) {
                                    Log.d(TAG, "High confidence (%.2f) — routing to Result".format(
                                        result.confidence))
                                    _cameraState.value = CameraUiState.Result(
                                        CaptureResultData(
                                            originalBitmap = result.originalBitmap,
                                            rectifiedBitmap = result.rectifiedBitmap,
                                            pipelineMs = result.pipelineMs,
                                            confidence = result.confidence
                                        )
                                    )
                                } else {
                                    Log.d(TAG, "Low confidence (%.2f) — routing to LowConfidence".format(
                                        result.confidence))
                                    // Recycle the rectified bitmap; user will re-rectify after adjustment
                                    result.rectifiedBitmap.recycle()
                                    _cameraState.value = CameraUiState.LowConfidence(
                                        originalBitmap = result.originalBitmap,
                                        corners = result.corners,
                                        confidence = result.confidence
                                    )
                                }
                            } else {
                                _cameraState.value = CameraUiState.Error("No document detected")
                                resetAfterDelay()
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "Capture processing failed", e)
                            imageProxy.close()
                            _cameraState.value = CameraUiState.Error("Processing failed: ${e.message}")
                            resetAfterDelay()
                        }
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    Log.e(TAG, "Image capture failed", exception)
                    _cameraState.value = CameraUiState.Error("Capture failed: ${exception.message}")
                    viewModelScope.launch { resetAfterDelay() }
                }
            }
        )
    }

    /** Saves the rectified bitmap to the device gallery. Returns true on success. */
    fun saveResult(context: Context, onResult: (Boolean) -> Unit) {
        val state = _cameraState.value
        if (state !is CameraUiState.Result) return

        viewModelScope.launch {
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val displayName = "DocShot_$timestamp"
            val uri = saveBitmapToGallery(context, state.data.rectifiedBitmap, displayName)
            onResult(uri != null)
        }
    }

    /** Shares the rectified bitmap via the system share sheet. */
    fun shareResult(context: Context) {
        val state = _cameraState.value
        if (state !is CameraUiState.Result) return
        shareImage(context, state.data.rectifiedBitmap)
    }

    /** Returns to camera preview and recycles bitmaps. */
    fun resetToCamera() {
        val state = _cameraState.value
        when (state) {
            is CameraUiState.Result -> {
                state.data.originalBitmap.recycle()
                state.data.rectifiedBitmap.recycle()
            }
            is CameraUiState.LowConfidence -> {
                state.originalBitmap.recycle()
            }
            else -> { /* no bitmaps to recycle */ }
        }
        _cameraState.value = CameraUiState.Idle
    }

    /**
     * Accepts user-adjusted corners from the LowConfidence screen,
     * re-runs corner refinement + rectification, and transitions to Result.
     */
    fun acceptLowConfidenceCorners(adjustedCorners: List<org.opencv.core.Point>) {
        val state = _cameraState.value
        check(state is CameraUiState.LowConfidence) {
            "acceptLowConfidenceCorners called in wrong state: ${state::class.simpleName}"
        }

        val originalBitmap = state.originalBitmap
        val confidence = state.confidence

        viewModelScope.launch(Dispatchers.Default) {
            _cameraState.value = CameraUiState.Processing("Rectifying document...")
            val start = System.nanoTime()
            var mat: Mat? = null
            var grayMat: Mat? = null
            var rectifiedMat: Mat? = null

            try {
                mat = bitmapToMat(originalBitmap)

                // Sub-pixel corner refinement on grayscale
                grayMat = Mat()
                val colorConversion = if (mat.channels() == 4)
                    Imgproc.COLOR_RGBA2GRAY else Imgproc.COLOR_BGR2GRAY
                Imgproc.cvtColor(mat, grayMat, colorConversion)
                val refinedCorners = refineCorners(grayMat, adjustedCorners)
                grayMat.release()
                grayMat = null

                // Rectify with high-quality interpolation
                rectifiedMat = rectify(mat, refinedCorners, Imgproc.INTER_CUBIC)

                // Detect and correct document orientation
                val (orientedMat, orientation) = detectAndCorrect(rectifiedMat)
                if (orientedMat !== rectifiedMat) {
                    rectifiedMat.release()
                    rectifiedMat = orientedMat
                }
                Log.d(TAG, "LowConfidence orientation: ${orientation.name}")

                // Convert to bitmap
                val rectifiedBitmap = matToBitmap(rectifiedMat)

                val pipelineMs = (System.nanoTime() - start) / 1_000_000.0
                Log.d(TAG, "acceptLowConfidenceCorners: %.1f ms".format(pipelineMs))

                _cameraState.value = CameraUiState.Result(
                    CaptureResultData(
                        originalBitmap = originalBitmap,
                        rectifiedBitmap = rectifiedBitmap,
                        pipelineMs = pipelineMs,
                        confidence = confidence
                    )
                )
            } catch (e: Exception) {
                Log.e(TAG, "Low-confidence rectification failed", e)
                _cameraState.value = CameraUiState.Error("Rectification failed: ${e.message}")
                resetAfterDelay()
            } finally {
                mat?.release()
                grayMat?.release()
                rectifiedMat?.release()
            }
        }
    }

    /** Cancels the low-confidence adjustment flow and returns to camera preview. */
    fun cancelLowConfidence() {
        val state = _cameraState.value
        if (state is CameraUiState.LowConfidence) {
            state.originalBitmap.recycle()
        }
        _cameraState.value = CameraUiState.Idle
    }

    private suspend fun resetAfterDelay() {
        withContext(Dispatchers.Main) {
            delay(2000)
            if (_cameraState.value is CameraUiState.Error) {
                _cameraState.value = CameraUiState.Idle
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        val state = _cameraState.value
        when (state) {
            is CameraUiState.Result -> {
                state.data.originalBitmap.recycle()
                state.data.rectifiedBitmap.recycle()
            }
            is CameraUiState.LowConfidence -> {
                state.originalBitmap.recycle()
            }
            else -> { /* no bitmaps to recycle */ }
        }
    }
}
