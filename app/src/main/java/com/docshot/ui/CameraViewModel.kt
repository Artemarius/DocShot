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
import java.lang.ref.WeakReference
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

private const val TAG = "DocShot:ViewModel"

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
    val stabilityProgress: Float = 0f
)

/** State machine for the capture -> result flow. */
sealed class CameraUiState {
    data object Idle : CameraUiState()
    data object Capturing : CameraUiState()
    data class Processing(val message: String) : CameraUiState()
    data class Result(val data: CaptureResultData) : CameraUiState()
    data class Error(val message: String) : CameraUiState()
}

data class CaptureResultData(
    val originalBitmap: Bitmap,
    val rectifiedBitmap: Bitmap,
    val pipelineMs: Double
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
            stabilityProgress = result.stabilityProgress
        )

        // Auto-capture: trigger when stable, enabled, and idle
        if (result.isStable
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
                                _cameraState.value = CameraUiState.Result(
                                    CaptureResultData(
                                        originalBitmap = result.originalBitmap,
                                        rectifiedBitmap = result.rectifiedBitmap,
                                        pipelineMs = result.pipelineMs
                                    )
                                )
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
        if (state is CameraUiState.Result) {
            state.data.originalBitmap.recycle()
            state.data.rectifiedBitmap.recycle()
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
        if (state is CameraUiState.Result) {
            state.data.originalBitmap.recycle()
            state.data.rectifiedBitmap.recycle()
        }
    }
}
