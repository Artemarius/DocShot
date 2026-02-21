package com.docshot.ui

import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCharacteristics
import android.os.Build
import android.util.Log
import android.util.Size
import android.view.HapticFeedbackConstants
import android.widget.Toast
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import com.docshot.cv.CameraIntrinsics
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.FlashOff
import androidx.compose.material.icons.filled.FlashOn
import androidx.compose.material.icons.outlined.AutoAwesome
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.FloatingActionButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Snackbar
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Fill
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipPath
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.viewmodel.compose.viewModel
import com.docshot.R
import com.docshot.util.UserPreferencesRepository
import com.docshot.util.rememberCameraPermissionState
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

private const val TAG = "DocShot:CameraScreen"

@Composable
fun CameraPermissionScreen(
    onOpenGallery: () -> Unit = {},
    preferencesRepository: UserPreferencesRepository? = null,
    onShowingResult: (Boolean) -> Unit = {}
) {
    val context = LocalContext.current
    val hasCamera = remember {
        context.packageManager.hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY)
    }

    if (!hasCamera) {
        NoCameraMessage()
        return
    }

    val permissionState = rememberCameraPermissionState()

    when {
        permissionState.hasPermission -> CameraPreview(
            onOpenGallery = onOpenGallery,
            preferencesRepository = preferencesRepository,
            onShowingResult = onShowingResult
        )
        permissionState.permissionRequested -> CameraDeniedMessage()
    }
}

@Composable
fun CameraPreview(
    viewModel: CameraViewModel = viewModel(),
    onOpenGallery: () -> Unit = {},
    preferencesRepository: UserPreferencesRepository? = null,
    onShowingResult: (Boolean) -> Unit = {}
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val view = LocalView.current
    val detectionState by viewModel.detectionState.collectAsState()
    val cameraState by viewModel.cameraState.collectAsState()
    val autoCapEnabled by viewModel.autoCapEnabled.collectAsState()
    val flashEnabled by viewModel.flashEnabled.collectAsState()
    val severity by viewModel.perspectiveSeverity.collectAsState()
    val estimatedAR by viewModel.estimatedAspectRatio.collectAsState()
    val matchedFormat by viewModel.matchedFormatName.collectAsState()
    val multiFrameCount by viewModel.multiFrameCount.collectAsState()
    val analysisExecutor = remember { Executors.newSingleThreadExecutor() }
    val scope = rememberCoroutineScope()

    // Collect aspect ratio lock settings continuously
    val settings by preferencesRepository?.settings?.collectAsState(
        initial = com.docshot.util.DocShotSettings()
    ) ?: remember { androidx.compose.runtime.mutableStateOf(com.docshot.util.DocShotSettings()) }

    // Notify parent when showing/hiding result screen
    LaunchedEffect(cameraState) {
        onShowingResult(cameraState is CameraUiState.Result)
    }

    // Provide context to ViewModel for auto-capture
    LaunchedEffect(context) {
        viewModel.setContext(context)
    }

    // Sync persisted flash setting into ViewModel on first composition
    LaunchedEffect(preferencesRepository) {
        if (preferencesRepository != null) {
            val settings = preferencesRepository.settings.first()
            viewModel.setFlashFromSettings(settings.flashEnabled)
        }
    }

    // Collect haptic events and perform haptic feedback
    // CONFIRM requires API 30; fall back to LONG_PRESS on older devices
    LaunchedEffect(Unit) {
        viewModel.hapticEvent.collect {
            val feedbackConstant = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                HapticFeedbackConstants.CONFIRM
            } else {
                HapticFeedbackConstants.LONG_PRESS
            }
            view.performHapticFeedback(feedbackConstant)
        }
    }

    DisposableEffect(Unit) {
        onDispose { analysisExecutor.shutdown() }
    }

    // If showing low-confidence adjustment, render CornerAdjustScreen
    if (cameraState is CameraUiState.LowConfidence) {
        val lowConf = cameraState as CameraUiState.LowConfidence
        CornerAdjustScreen(
            bitmap = lowConf.originalBitmap,
            corners = lowConf.corners,
            onApply = { adjustedCorners ->
                viewModel.acceptLowConfidenceCorners(adjustedCorners)
            },
            onCancel = { viewModel.cancelLowConfidence() }
        )
        return
    }

    // If showing result, render ResultScreen instead of camera
    if (cameraState is CameraUiState.Result) {
        val resultData = (cameraState as CameraUiState.Result).data
        ResultScreen(
            data = resultData,
            onSave = {
                viewModel.saveResult(context) { success ->
                    val msg = if (success) "Saved to gallery" else "Save failed"
                    Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
                }
            },
            onShare = { viewModel.shareResult(context) },
            onRetake = { viewModel.resetToCamera() },
            onAdjust = { viewModel.adjustFromResult() },
            onRotate = { viewModel.rotateResult() },
            onAspectRatioChange = { viewModel.reWarpWithAspectRatio(it) },
            isAspectRatioLocked = settings.aspectRatioLocked,
            lockedAspectRatio = settings.lockedAspectRatio,
            onToggleAspectRatioLock = { locked, ratio ->
                scope.launch {
                    preferencesRepository?.setAspectRatioLocked(locked)
                    preferencesRepository?.setLockedAspectRatio(ratio)
                }
            },
            aspectRatioAutoEstimate = settings.aspectRatioAutoEstimate
        )
        return
    }

    Box(modifier = Modifier.fillMaxSize()) {
        val isIdle = cameraState is CameraUiState.Idle
        val isBusy = cameraState is CameraUiState.Capturing || cameraState is CameraUiState.Processing

        var previewViewRef by remember { mutableStateOf<PreviewView?>(null) }

        // Snapshot the preview bitmap exactly when the freeze overlay appears
        val frozenPreviewBitmap = remember(isBusy) {
            if (isBusy) previewViewRef?.getBitmap()?.asImageBitmap() else null
        }

        AndroidView(
            factory = { ctx ->
                val previewView = PreviewView(ctx).apply {
                    implementationMode = PreviewView.ImplementationMode.PERFORMANCE
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                }

                bindCamera(ctx, lifecycleOwner, previewView, viewModel, analysisExecutor)
                previewViewRef = previewView

                previewView
            },
            modifier = Modifier.fillMaxSize()
        )

        // Quad overlay with stability visual feedback (live detection)
        if (!isBusy) {
            QuadOverlay(
                detectionState = detectionState,
                modifier = Modifier.fillMaxSize()
            )
        }

        // Freeze overlay: dark scrim + last quad when capturing/processing.
        // Prevents ambiguity between live preview and the captured frame.
        if (isBusy) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.Black.copy(alpha = 0.75f))
            )
            // Show the frozen quad from the last detection state
            QuadOverlay(
                detectionState = detectionState,
                frozenPreviewBitmap = frozenPreviewBitmap,
                modifier = Modifier.fillMaxSize()
            )
            Text(
                text = if (cameraState is CameraUiState.Processing)
                    (cameraState as CameraUiState.Processing).message
                else "Capturing...",
                color = Color.White,
                style = MaterialTheme.typography.bodyLarge,
                modifier = Modifier
                    .align(Alignment.Center)
                    .padding(bottom = 64.dp)
            )
        }

        // Flash toggle button (top-right)
        IconButton(
            onClick = {
                viewModel.toggleFlash()
                scope.launch { preferencesRepository?.setFlashEnabled(viewModel.flashEnabled.value) }
            },
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(top = 16.dp, end = 16.dp)
                .size(40.dp)
                .background(
                    color = Color.Black.copy(alpha = 0.4f),
                    shape = CircleShape
                )
        ) {
            Icon(
                imageVector = if (flashEnabled) Icons.Filled.FlashOn
                    else Icons.Filled.FlashOff,
                contentDescription = if (flashEnabled) "Turn off flash"
                    else "Turn on flash",
                modifier = Modifier.size(22.dp),
                tint = if (flashEnabled) Color(0xFFFFC107) else Color.White
            )
        }

        // Debug overlay: detection stats + auto-capture readiness + AR estimation
        if (detectionState.normalizedCorners != null) {
            val warmupRemaining = viewModel.warmupRemainingMs()
            val stableFrames = (detectionState.stabilityProgress * 20).toInt() // stableThreshold=20
            val conf = detectionState.confidence
            val afLocked = viewModel.isAfLocked
            val afTriggering = viewModel.isAfTriggering
            val autoReady = detectionState.isStable
                    && conf >= 0.65
                    && autoCapEnabled
                    && warmupRemaining <= 0
                    && afLocked
            Text(
                text = buildString {
                    // Line 1: timing, stability, confidence, AF, readiness
                    append("%.0f ms | ".format(detectionState.detectionMs))
                    append("stable: $stableFrames/20 | ")
                    append("conf: %.2f".format(conf))
                    if (afTriggering) append(" | AF")
                    else if (afLocked) append(" | AF OK")
                    if (warmupRemaining > 0) append(" | warmup: ${warmupRemaining}ms")
                    if (autoReady) append(" | READY")

                    // Line 2: KLT tracking state + AR estimation info
                    append("\n")
                    // Tracking state from isTracked flag
                    val kltLabel = if (detectionState.isTracked) "TRACKING" else "DETECT"
                    append("KLT: $kltLabel")

                    // Perspective severity and regime label
                    val severityLabel = when {
                        severity < 15.0 -> "LOW"
                        severity > 20.0 -> "HIGH"
                        else -> "TRANSITION"
                    }
                    if (severity > 0.0) {
                        append(" | sev: %.0f\u00B0 %s".format(severity, severityLabel))
                    }

                    // Estimated aspect ratio with format name
                    if (estimatedAR > 0.0) {
                        append(" | AR: %.3f".format(estimatedAR))
                        matchedFormat?.let { append(" ($it)") }
                    }

                    // Multi-frame accumulation count
                    if (multiFrameCount > 0) {
                        append(" | MF: $multiFrameCount/20")
                    }
                },
                color = Color.White,
                style = MaterialTheme.typography.labelSmall,
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(16.dp)
                    .background(
                        color = Color.Black.copy(alpha = 0.5f),
                        shape = RoundedCornerShape(4.dp)
                    )
                    .padding(horizontal = 6.dp, vertical = 2.dp)
            )
        }

        // Hint when no detection and camera is idle
        if (detectionState.normalizedCorners == null && cameraState is CameraUiState.Idle) {
            val hintText = if (detectionState.isPartialDocument) {
                "Move back to fit document"
            } else {
                "Tap to capture manually"
            }
            Text(
                text = hintText,
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f),
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 120.dp)
                    .background(
                        color = MaterialTheme.colorScheme.surface.copy(alpha = 0.5f),
                        shape = RoundedCornerShape(50)
                    )
                    .padding(horizontal = 24.dp, vertical = 8.dp)
            )
        }

        // Capture FAB
        FloatingActionButton(
            onClick = {
                if (isIdle) viewModel.captureDocument(context)
            },
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 32.dp)
                .size(72.dp),
            shape = CircleShape,
            containerColor = if (isIdle) MaterialTheme.colorScheme.primary
                else MaterialTheme.colorScheme.surfaceVariant,
            elevation = FloatingActionButtonDefaults.elevation(defaultElevation = 6.dp)
        ) {
            if (isBusy) {
                CircularProgressIndicator(
                    modifier = Modifier.size(32.dp),
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    strokeWidth = 3.dp
                )
            } else {
                Icon(
                    painter = painterResource(id = R.drawable.ic_camera),
                    contentDescription = "Capture document",
                    modifier = Modifier.size(32.dp)
                )
            }
        }

        // Gallery button
        FloatingActionButton(
            onClick = onOpenGallery,
            modifier = Modifier
                .align(Alignment.BottomStart)
                .padding(start = 24.dp, bottom = 40.dp)
                .size(48.dp),
            containerColor = MaterialTheme.colorScheme.surfaceVariant,
            elevation = FloatingActionButtonDefaults.elevation(defaultElevation = 4.dp)
        ) {
            Icon(
                painter = painterResource(id = R.drawable.ic_photo_library),
                contentDescription = "Import from gallery",
                modifier = Modifier.size(24.dp)
            )
        }

        // Auto-capture toggle button
        FloatingActionButton(
            onClick = { viewModel.toggleAutoCap() },
            modifier = Modifier
                .align(Alignment.BottomEnd)
                .padding(end = 24.dp, bottom = 40.dp)
                .size(48.dp),
            containerColor = if (autoCapEnabled) MaterialTheme.colorScheme.primaryContainer
                else MaterialTheme.colorScheme.surfaceVariant,
            elevation = FloatingActionButtonDefaults.elevation(defaultElevation = 4.dp)
        ) {
            Icon(
                imageVector = if (autoCapEnabled) Icons.Filled.AutoAwesome
                    else Icons.Outlined.AutoAwesome,
                contentDescription = if (autoCapEnabled) "Disable auto-capture"
                    else "Enable auto-capture",
                modifier = Modifier.size(24.dp),
                tint = if (autoCapEnabled) MaterialTheme.colorScheme.onPrimaryContainer
                    else MaterialTheme.colorScheme.onSurfaceVariant
            )
        }

        // Error snackbar
        if (cameraState is CameraUiState.Error) {
            Snackbar(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 120.dp, start = 16.dp, end = 16.dp)
            ) {
                Text((cameraState as CameraUiState.Error).message)
            }
        }
    }
}

private fun bindCamera(
    context: android.content.Context,
    lifecycleOwner: androidx.lifecycle.LifecycleOwner,
    previewView: PreviewView,
    viewModel: CameraViewModel,
    analysisExecutor: ExecutorService
) {
    val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
    cameraProviderFuture.addListener({
        val cameraProvider = cameraProviderFuture.get()

        val preview = Preview.Builder().build().also {
            it.surfaceProvider = previewView.surfaceProvider
        }

        val resolutionSelector = ResolutionSelector.Builder()
            .setResolutionStrategy(
                ResolutionStrategy(
                    Size(640, 480),
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER
                )
            )
            .build()

        val imageAnalysis = ImageAnalysis.Builder()
            .setResolutionSelector(resolutionSelector)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .build()
            .also { it.setAnalyzer(analysisExecutor, viewModel.frameAnalyzer) }

        val imageCapture = ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_ZERO_SHUTTER_LAG)
            .build()

        viewModel.imageCapture = imageCapture

        cameraProvider.unbindAll()
        try {
            val camera = cameraProvider.bindToLifecycle(
                lifecycleOwner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                imageAnalysis,
                imageCapture
            )
            viewModel.camera = camera
            Log.d(TAG, "ZSL supported: ${camera.cameraInfo.isZslSupported}")
            camera.cameraControl.enableTorch(viewModel.flashEnabled.value)
            extractCameraIntrinsics(camera, viewModel)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to bind 3 use cases, retrying without ImageCapture", e)
            cameraProvider.unbindAll()
            val camera = cameraProvider.bindToLifecycle(
                lifecycleOwner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                imageAnalysis
            )
            viewModel.camera = camera
            camera.cameraControl.enableTorch(viewModel.flashEnabled.value)
            viewModel.imageCapture = null
            extractCameraIntrinsics(camera, viewModel)
        }
    }, ContextCompat.getMainExecutor(context))
}

/**
 * Extracts camera intrinsics for homography-based aspect ratio verification.
 * Tries LENS_INTRINSIC_CALIBRATION first (API 28+, LEVEL_3 devices),
 * falls back to computing from sensor physical size + focal length.
 */
@SuppressLint("RestrictedApi")
private fun extractCameraIntrinsics(
    camera: androidx.camera.core.Camera,
    viewModel: CameraViewModel
) {
    try {
        val camera2Info = Camera2CameraInfo.from(camera.cameraInfo)
        val chars = camera2Info.getCameraCharacteristic(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)

        val pixelArray = camera2Info.getCameraCharacteristic(
            CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE
        )

        if (chars != null && chars.size >= 4) {
            // LENS_INTRINSIC_CALIBRATION: [fx, fy, cx, cy, s]
            val intrinsics = CameraIntrinsics(
                fx = chars[0].toDouble(),
                fy = chars[1].toDouble(),
                cx = chars[2].toDouble(),
                cy = chars[3].toDouble(),
                sensorWidth = pixelArray?.width ?: 0,
                sensorHeight = pixelArray?.height ?: 0
            )
            viewModel.setCameraIntrinsics(intrinsics)
            Log.d(TAG, "Camera intrinsics from LENS_INTRINSIC_CALIBRATION: fx=%.1f fy=%.1f sensor=%dx%d".format(
                intrinsics.fx, intrinsics.fy, intrinsics.sensorWidth, intrinsics.sensorHeight))
            return
        }

        // Fallback: compute from focal length + sensor size + pixel array
        val focalLengths = camera2Info.getCameraCharacteristic(
            CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS
        )
        val sensorSize = camera2Info.getCameraCharacteristic(
            CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE
        )

        if (focalLengths != null && focalLengths.isNotEmpty() && sensorSize != null && pixelArray != null) {
            val focalLength = focalLengths[0].toDouble()
            val sensorWidth = sensorSize.width.toDouble()
            val sensorHeight = sensorSize.height.toDouble()
            val pixelW = pixelArray.width.toDouble()
            val pixelH = pixelArray.height.toDouble()

            val fx = focalLength * pixelW / sensorWidth
            val fy = focalLength * pixelH / sensorHeight
            val cx = pixelW / 2.0
            val cy = pixelH / 2.0

            val intrinsics = CameraIntrinsics(
                fx = fx, fy = fy, cx = cx, cy = cy,
                sensorWidth = pixelArray.width, sensorHeight = pixelArray.height
            )
            viewModel.setCameraIntrinsics(intrinsics)
            Log.d(TAG, "Camera intrinsics from sensor size: fx=%.1f fy=%.1f sensor=%dx%d".format(
                fx, fy, pixelArray.width, pixelArray.height))
        } else {
            Log.d(TAG, "Could not extract camera intrinsics — aspect ratio estimation will use distance-only snapping")
        }
    } catch (e: Exception) {
        Log.w(TAG, "Failed to extract camera intrinsics: ${e.message}")
    }
}

/**
 * Draws the detected document quadrilateral over the camera preview.
 * Maps normalized [0,1] detection coordinates to canvas space using FILL_CENTER
 * scaling to match PreviewView's coordinate system.
 *
 * Visual feedback based on stability:
 * - Detection, not stable: Green quad stroke
 * - Progress >= 0.5: gradually increasing fill opacity (0% to 15%)
 * - Stable (progress >= 1.0): Cyan/Teal quad with thicker stroke ("ready to capture")
 */
@Composable
private fun QuadOverlay(
    detectionState: DetectionUiState,
    modifier: Modifier = Modifier,
    frozenPreviewBitmap: ImageBitmap? = null
) {
    val corners = detectionState.normalizedCorners ?: return
    val srcW = detectionState.sourceWidth
    val srcH = detectionState.sourceHeight
    if (srcW == 0 || srcH == 0) return

    val progress = detectionState.stabilityProgress
    val isStable = detectionState.isStable

    // Color transitions based on stability
    val quadColor = if (isStable) Color.Cyan else Color.Green
    val strokeWidth = if (isStable) 5f else 3f // dp, applied below

    // Fill opacity ramps from 0% at progress=0.5 to 15% at progress=1.0
    val fillAlpha = if (progress > 0.5f) {
        ((progress - 0.5f) / 0.5f * 0.15f).coerceIn(0f, 0.15f)
    } else {
        0f
    }

    Canvas(modifier = modifier) {
        // Compute FILL_CENTER transform to match PreviewView's scaling
        val srcAspect = srcW.toFloat() / srcH.toFloat()
        val canvasAspect = size.width / size.height

        val scale: Float
        val offsetX: Float
        val offsetY: Float

        if (srcAspect > canvasAspect) {
            // Source is wider -- scale by height, sides may be cropped
            scale = size.height / srcH.toFloat()
            offsetX = (size.width - srcW * scale) / 2f
            offsetY = 0f
        } else {
            // Source is taller -- scale by width, top/bottom may be cropped
            scale = size.width / srcW.toFloat()
            offsetX = 0f
            offsetY = (size.height - srcH * scale) / 2f
        }

        fun mapPoint(normalized: FloatArray): Offset {
            val x = normalized[0] * srcW * scale + offsetX
            val y = normalized[1] * srcH * scale + offsetY
            return Offset(x, y)
        }

        // Build quad path
        val path = Path().apply {
            val p0 = mapPoint(corners[0])
            moveTo(p0.x, p0.y)
            for (i in 1..3) {
                val p = mapPoint(corners[i])
                lineTo(p.x, p.y)
            }
            close()
        }

        // Draw frozen preview clipped to quad (capture freeze only)
        if (frozenPreviewBitmap != null) {
            clipPath(path) {
                drawImage(image = frozenPreviewBitmap, alpha = 0.7f)
            }
        }

        // Draw fill (builds up as stability progresses)
        if (fillAlpha > 0f) {
            drawPath(
                path = path,
                color = quadColor.copy(alpha = fillAlpha),
                style = Fill
            )
        }

        // Draw quad outline
        drawPath(
            path = path,
            color = quadColor,
            style = Stroke(width = strokeWidth.dp.toPx())
        )

        // Draw corner dots
        val dotRadius = if (isStable) 8f else 6f
        for (corner in corners) {
            drawCircle(
                color = quadColor,
                radius = dotRadius.dp.toPx(),
                center = mapPoint(corner)
            )
        }
    }
}

@Composable
fun CameraDeniedMessage() {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = "Camera permission is required to scan documents.",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onBackground
        )
    }
}

@Composable
private fun NoCameraMessage() {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = "No camera available on this device.",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onBackground
        )
    }
}
