package com.docshot.ui

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import androidx.camera.core.Camera
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.SurfaceOrientedMeteringPointFactory
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.docshot.camera.FrameAnalyzer
import com.docshot.camera.processCapture
import com.docshot.cv.CameraIntrinsics
import com.docshot.cv.MultiFrameEstimate
import com.docshot.cv.detectAndCorrect
import com.docshot.cv.rectify
import com.docshot.cv.rectifyWithAspectRatio
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
// Minimum confidence for auto-capture to fire (stability + this threshold)
private const val AUTO_CAPTURE_CONFIDENCE_THRESHOLD = 0.65
// Minimum multi-frame estimate confidence to propagate to ResultScreen
private const val MULTI_FRAME_CONFIDENCE_THRESHOLD = 0.7
// Minimum confidence to route directly to Result; below this goes to corner adjustment.
// Matches AUTO_CAPTURE_CONFIDENCE_THRESHOLD — if we trust it enough to auto-capture,
// we trust it enough to show the result (user can still hit Adjust).
private const val RESULT_CONFIDENCE_THRESHOLD = 0.65
// Suppress auto-capture for this many ms after entering Idle state.
// Gives the user time to frame the document and lets detection settle.
private const val AUTO_CAPTURE_WARMUP_MS = 1500L
// Trigger AF lock at this fraction of stability (50% = 10/20 stable frames).
// Gives AF ~333ms (10 frames at 30fps) to settle before auto-capture fires at frame 20.
private const val AF_LOCK_STABILITY_THRESHOLD = 0.5f

/**
 * UI state for the detected document overlay.
 * Corners are in normalized [0,1] display coordinates (rotation already applied).
 */
data class DetectionUiState(
    val normalizedCorners: List<FloatArray>? = null,
    val sourceWidth: Int = 0,
    val sourceHeight: Int = 0,
    val detectionMs: Double = 0.0,
    val confidence: Double = 0.0,
    val isStable: Boolean = false,
    val stabilityProgress: Float = 0f,
    val isPartialDocument: Boolean = false,
    /** True if corners came from KLT tracking rather than full detection. */
    val isTracked: Boolean = false
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
    val confidence: Double = 0.0,
    val corners: List<org.opencv.core.Point> = emptyList(),
    val normalizedCorners: FloatArray = floatArrayOf(),
    val cameraIntrinsics: CameraIntrinsics? = null,
    val autoRotationSteps: Int = 0,  // 0-3: auto-orientation from detectAndCorrect
    val manualRotationSteps: Int = 0,  // 0-3: additional manual rotations by user
    /** Multi-frame estimated aspect ratio (min/max, <= 1.0) from stabilization window.
     *  Non-null only when confidence >= 0.7. Used by ResultScreen as initial ratio. */
    val estimatedAspectRatio: Float? = null
)

class CameraViewModel : ViewModel() {

    private val _detectionState = MutableStateFlow(DetectionUiState())
    val detectionState: StateFlow<DetectionUiState> = _detectionState

    private val _cameraState = MutableStateFlow<CameraUiState>(CameraUiState.Idle)
    val cameraState: StateFlow<CameraUiState> = _cameraState

    /** Whether auto-capture triggers when document is stably detected. */
    private val _autoCapEnabled = MutableStateFlow(true)
    val autoCapEnabled: StateFlow<Boolean> = _autoCapEnabled

    /** Whether the camera torch is enabled. */
    private val _flashEnabled = MutableStateFlow(false)
    val flashEnabled: StateFlow<Boolean> = _flashEnabled

    /** Emitted when capture triggers (auto or manual) so CameraScreen can fire haptic. */
    private val _hapticEvent = MutableSharedFlow<Unit>(extraBufferCapacity = 1)
    val hapticEvent: SharedFlow<Unit> = _hapticEvent

    /** Weak reference to activity context for auto-capture. Set via [setContext]. */
    private var contextRef: WeakReference<Context>? = null

    /** Set by CameraScreen when binding the camera. */
    var imageCapture: ImageCapture? = null

    /** Set by CameraScreen after bindToLifecycle returns the Camera object. */
    var camera: Camera? = null

    /**
     * Timestamp (SystemClock.elapsedRealtime) when Idle state was entered.
     * Auto-capture is suppressed for [AUTO_CAPTURE_WARMUP_MS] after this to give
     * the user time to frame the document and let detection settle.
     */
    private var idleEnteredAt: Long = android.os.SystemClock.elapsedRealtime()

    /** Whether AF has confirmed focus lock (ready for sharp capture). */
    private var _isAfLocked = false

    /** Whether an AF trigger request is in flight. */
    private var _isAfTriggering = false

    /** Camera intrinsics for homography-based aspect ratio verification. */
    private var _cameraIntrinsics: CameraIntrinsics? = null

    /**
     * Latest multi-frame aspect ratio estimate from the stabilization window.
     * Updated each frame when stable; snapshotted at auto-capture time for B8.
     */
    private var _latestMultiFrameEstimate: MultiFrameEstimate? = null

    /** Read-only accessor for the capture pipeline (task B8). */
    val latestMultiFrameEstimate: MultiFrameEstimate? get() = _latestMultiFrameEstimate

    /** Called from CameraScreen after binding camera to extract lens calibration. */
    fun setCameraIntrinsics(intrinsics: CameraIntrinsics) {
        _cameraIntrinsics = intrinsics
    }

    /** Read-only accessor for debug overlay. */
    val isAfLocked: Boolean get() = _isAfLocked

    /** Read-only accessor for debug overlay. */
    val isAfTriggering: Boolean get() = _isAfTriggering

    // ── Debug overlay: AR estimation state ──────────────────────────────

    /**
     * Perspective severity (max corner angle deviation from 90deg).
     * TODO: Wire from FrameAnalyzer when B7 integration lands --
     * call perspectiveSeverity() on detected corners each frame and emit here.
     */
    private val _perspectiveSeverity = MutableStateFlow(0.0)
    val perspectiveSeverity: StateFlow<Double> = _perspectiveSeverity

    /**
     * Current single-frame aspect ratio estimate (min/max, <= 1.0).
     * TODO: Wire from FrameAnalyzer when B7 integration lands --
     * call estimateAspectRatioDualRegime() on detected corners and emit here.
     */
    private val _estimatedAspectRatio = MutableStateFlow(0.0)
    val estimatedAspectRatio: StateFlow<Double> = _estimatedAspectRatio

    /**
     * Name of matched known format (e.g., "A4", "US Letter"), or null if no snap.
     * TODO: Wire from FrameAnalyzer when B7 integration lands.
     */
    private val _matchedFormatName = MutableStateFlow<String?>(null)
    val matchedFormatName: StateFlow<String?> = _matchedFormatName

    /**
     * Number of frames accumulated by MultiFrameAspectEstimator.
     * TODO: Wire from FrameAnalyzer when B7 integration lands --
     * expose MultiFrameAspectEstimator.frameCount during stabilization window.
     */
    private val _multiFrameCount = MutableStateFlow(0)
    val multiFrameCount: StateFlow<Int> = _multiFrameCount

    /**
     * Store a context reference for auto-capture. Must be called from CameraScreen
     * during composition (e.g., in a LaunchedEffect or remember block).
     * Uses a WeakReference to avoid leaking the Activity.
     */
    fun setContext(context: Context) {
        contextRef = WeakReference(context)
    }

    /** Returns ms remaining in auto-capture warmup suppression, or 0 if elapsed. */
    fun warmupRemainingMs(): Long {
        val elapsed = android.os.SystemClock.elapsedRealtime() - idleEnteredAt
        return maxOf(0L, AUTO_CAPTURE_WARMUP_MS - elapsed)
    }

    fun toggleAutoCap() {
        _autoCapEnabled.value = !_autoCapEnabled.value
        Log.d(TAG, "Auto-capture toggled: ${_autoCapEnabled.value}")
    }

    fun toggleFlash() {
        _flashEnabled.value = !_flashEnabled.value
        camera?.cameraControl?.enableTorch(_flashEnabled.value)
        Log.d(TAG, "Flash toggled: ${_flashEnabled.value}")
    }

    /** Initialize flash state from persisted settings (called once on composition). */
    fun setFlashFromSettings(enabled: Boolean) {
        _flashEnabled.value = enabled
        camera?.cameraControl?.enableTorch(enabled)
    }

    val frameAnalyzer = FrameAnalyzer { result ->
        // Skip detection state updates during capture/processing — freezes the quad overlay
        if (_cameraState.value !is CameraUiState.Idle) return@FrameAnalyzer

        _detectionState.value = DetectionUiState(
            normalizedCorners = result.normalizedCorners,
            sourceWidth = result.sourceWidth,
            sourceHeight = result.sourceHeight,
            detectionMs = result.detectionMs,
            confidence = result.confidence,
            isStable = result.isStable,
            stabilityProgress = result.stabilityProgress,
            isPartialDocument = result.isPartialDocument,
            isTracked = result.isTracked
        )

        // Store multi-frame estimate when available (becomes non-null at stability)
        result.multiFrameEstimate?.let { _latestMultiFrameEstimate = it }

        // AF lock: trigger at 50% stability so AF has time to settle before auto-capture fires
        if (result.stabilityProgress >= AF_LOCK_STABILITY_THRESHOLD
            && !_isAfLocked
            && !_isAfTriggering
            && _autoCapEnabled.value
        ) {
            triggerAfLock()
        }

        // Cancel AF lock when quad is lost (no detection) so continuous AF resumes
        if (result.normalizedCorners == null && (_isAfLocked || _isAfTriggering)) {
            cancelAfLock()
        }

        // Auto-capture: trigger when stable, high confidence, enabled, idle, past warmup, and AF locked
        val warmupElapsed = android.os.SystemClock.elapsedRealtime() - idleEnteredAt
        if (result.isStable
            && result.confidence >= AUTO_CAPTURE_CONFIDENCE_THRESHOLD
            && _autoCapEnabled.value
            && _cameraState.value is CameraUiState.Idle
            && warmupElapsed >= AUTO_CAPTURE_WARMUP_MS
            && _isAfLocked
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

        // Snapshot preview state before capture
        val previewCorners = _detectionState.value.normalizedCorners
        val previewConfidence = _detectionState.value.confidence

        _cameraState.value = CameraUiState.Capturing
        _hapticEvent.tryEmit(Unit)

        // Turn off torch physically after shutter to save battery,
        // but preserve the logical state so flash re-enables on return to camera
        if (_flashEnabled.value) {
            camera?.cameraControl?.enableTorch(false)
            Log.d(TAG, "Torch off for capture (flash state preserved)")
        }

        capture.takePicture(
            ContextCompat.getMainExecutor(context),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(imageProxy: ImageProxy) {
                    viewModelScope.launch(Dispatchers.Default) {
                        _cameraState.value = CameraUiState.Processing("Processing document...")
                        try {
                            val result = processCapture(imageProxy, previewCorners, previewConfidence)
                            imageProxy.close()

                            if (result != null) {
                                if (result.confidence >= RESULT_CONFIDENCE_THRESHOLD) {
                                    Log.d(TAG, "High confidence (%.2f) — routing to Result".format(
                                        result.confidence))
                                    val normCorners = cornersToNormalized(
                                        result.corners,
                                        result.originalBitmap.width,
                                        result.originalBitmap.height
                                    )
                                    // Snapshot multi-frame AR estimate if confidence is sufficient
                                    val mfEstimate = _latestMultiFrameEstimate
                                    val estimatedAR = if (mfEstimate != null
                                        && mfEstimate.confidence >= MULTI_FRAME_CONFIDENCE_THRESHOLD
                                    ) {
                                        Log.d(TAG, "Multi-frame AR: ratio=%.4f, confidence=%.2f, frames=%d — using as initial".format(
                                            mfEstimate.estimatedRatio, mfEstimate.confidence, mfEstimate.frameCount))
                                        mfEstimate.estimatedRatio.toFloat()
                                    } else {
                                        Log.d(TAG, "Multi-frame AR: %s — falling back to default".format(
                                            if (mfEstimate == null) "not available"
                                            else "low confidence (%.2f < %.2f)".format(
                                                mfEstimate.confidence, MULTI_FRAME_CONFIDENCE_THRESHOLD)))
                                        null
                                    }
                                    _cameraState.value = CameraUiState.Result(
                                        CaptureResultData(
                                            originalBitmap = result.originalBitmap,
                                            rectifiedBitmap = result.rectifiedBitmap,
                                            pipelineMs = result.pipelineMs,
                                            confidence = result.confidence,
                                            corners = result.corners,
                                            normalizedCorners = normCorners,
                                            cameraIntrinsics = _cameraIntrinsics,
                                            autoRotationSteps = result.autoRotationSteps,
                                            estimatedAspectRatio = estimatedAR
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
        enterIdle()
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

                val normCorners = cornersToNormalized(
                    adjustedCorners,
                    originalBitmap.width,
                    originalBitmap.height
                )
                val autoSteps = when (orientation) {
                    com.docshot.cv.DocumentOrientation.CORRECT -> 0
                    com.docshot.cv.DocumentOrientation.ROTATE_90 -> 1
                    com.docshot.cv.DocumentOrientation.ROTATE_180 -> 2
                    com.docshot.cv.DocumentOrientation.ROTATE_270 -> 3
                }
                // Snapshot multi-frame AR estimate (may still be available from
                // the stabilization window before this low-confidence capture)
                val mfEstimate = _latestMultiFrameEstimate
                val estimatedAR = if (mfEstimate != null
                    && mfEstimate.confidence >= MULTI_FRAME_CONFIDENCE_THRESHOLD
                ) {
                    mfEstimate.estimatedRatio.toFloat()
                } else {
                    null
                }
                _cameraState.value = CameraUiState.Result(
                    CaptureResultData(
                        originalBitmap = originalBitmap,
                        rectifiedBitmap = rectifiedBitmap,
                        pipelineMs = pipelineMs,
                        confidence = confidence,
                        corners = adjustedCorners,
                        normalizedCorners = normCorners,
                        cameraIntrinsics = _cameraIntrinsics,
                        autoRotationSteps = autoSteps,
                        estimatedAspectRatio = estimatedAR
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

    /**
     * Transitions from Result to LowConfidence (corner adjustment) so the user
     * can manually adjust corners on the original image.
     */
    fun adjustFromResult() {
        val state = _cameraState.value
        check(state is CameraUiState.Result) {
            "adjustFromResult called in wrong state: ${state::class.simpleName}"
        }
        val data = state.data
        if (data.corners.isEmpty()) return

        // Recycle the rectified bitmap; user will re-rectify after adjustment
        data.rectifiedBitmap.recycle()

        _cameraState.value = CameraUiState.LowConfidence(
            originalBitmap = data.originalBitmap,
            corners = data.corners,
            confidence = data.confidence
        )
    }

    /**
     * Re-warps the document from the original image with an adjusted aspect ratio.
     * Re-applies orientation correction. Filter re-application is handled by
     * ResultScreen observing the bitmap change.
     */
    fun reWarpWithAspectRatio(targetRatio: Double) {
        val state = _cameraState.value
        if (state !is CameraUiState.Result) return
        val data = state.data
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

                _cameraState.value = CameraUiState.Result(
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

    /**
     * Rotates the rectified bitmap 90 degrees clockwise and emits an updated Result.
     * Corners are unchanged since they reference the original image.
     */
    fun rotateResult() {
        val state = _cameraState.value
        check(state is CameraUiState.Result) {
            "rotateResult called in wrong state: ${state::class.simpleName}"
        }
        val data = state.data
        val oldBitmap = data.rectifiedBitmap

        val matrix = android.graphics.Matrix().apply { postRotate(90f) }
        val rotated = Bitmap.createBitmap(
            oldBitmap, 0, 0, oldBitmap.width, oldBitmap.height, matrix, true
        )
        if (rotated !== oldBitmap) oldBitmap.recycle()

        _cameraState.value = CameraUiState.Result(
            data.copy(
                rectifiedBitmap = rotated,
                manualRotationSteps = (data.manualRotationSteps + 1) % 4
            )
        )
    }

    /** Cancels the low-confidence adjustment flow and returns to camera preview. */
    fun cancelLowConfidence() {
        val state = _cameraState.value
        if (state is CameraUiState.LowConfidence) {
            state.originalBitmap.recycle()
        }
        enterIdle()
    }

    private suspend fun resetAfterDelay() {
        withContext(Dispatchers.Main) {
            delay(2000)
            if (_cameraState.value is CameraUiState.Error) {
                enterIdle()
            }
        }
    }

    /**
     * Triggers a one-shot AF lock at the center of the frame.
     * Uses FLAG_AF only with auto-cancel disabled so focus stays locked
     * until we explicitly cancel via [cancelAfLock].
     */
    private fun triggerAfLock() {
        val cam = camera ?: return
        val ctx = contextRef?.get() ?: return
        _isAfTriggering = true
        val factory = SurfaceOrientedMeteringPointFactory(1f, 1f)
        val centerPoint = factory.createPoint(0.5f, 0.5f)
        val action = FocusMeteringAction.Builder(centerPoint, FocusMeteringAction.FLAG_AF)
            .disableAutoCancel()
            .build()
        val future = cam.cameraControl.startFocusAndMetering(action)
        future.addListener({
            try {
                val result = future.get()
                if (result.isFocusSuccessful) {
                    _isAfLocked = true
                    Log.d(TAG, "AF lock successful")
                } else {
                    Log.d(TAG, "AF lock failed — focus not achieved")
                }
            } catch (e: Exception) {
                Log.d(TAG, "AF lock error: ${e.message}")
            } finally {
                _isAfTriggering = false
            }
        }, ContextCompat.getMainExecutor(ctx))
    }

    /**
     * Cancels any active AF lock and returns the camera to continuous autofocus.
     */
    private fun cancelAfLock() {
        camera?.cameraControl?.cancelFocusAndMetering()
        _isAfLocked = false
        _isAfTriggering = false
    }

    /**
     * Centralized transition to Idle: resets warmup timer and clears the smoother
     * so stability must build from scratch (prevents instant re-fire of auto-capture).
     */
    private fun enterIdle() {
        cancelAfLock()
        idleEnteredAt = android.os.SystemClock.elapsedRealtime()
        frameAnalyzer.resetSmoothing()  // Also resets multi-frame estimator
        _latestMultiFrameEstimate = null
        _cameraState.value = CameraUiState.Idle
        // Re-enable torch if flash was logically on (it was turned off during capture)
        if (_flashEnabled.value) {
            camera?.cameraControl?.enableTorch(true)
        }
    }

    /**
     * Converts full-res pixel corners to a flat FloatArray of 8 normalized [0,1] values.
     * Layout: [x0, y0, x1, y1, x2, y2, x3, y3]
     */
    private fun cornersToNormalized(
        corners: List<org.opencv.core.Point>,
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

    override fun onCleared() {
        super.onCleared()
        // Release native Mat resources (CornerTracker + MultiFrameAspectEstimator)
        frameAnalyzer.releaseNativeResources()
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
