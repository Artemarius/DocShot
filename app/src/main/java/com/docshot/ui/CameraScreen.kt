package com.docshot.ui

import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.FloatingActionButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Snackbar
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.viewmodel.compose.viewModel
import com.docshot.R
import com.docshot.util.rememberCameraPermissionState
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

private const val TAG = "DocShot:CameraScreen"

@Composable
fun CameraPermissionScreen() {
    val permissionState = rememberCameraPermissionState()

    when {
        permissionState.hasPermission -> CameraPreview()
        permissionState.permissionRequested -> CameraDeniedMessage()
    }
}

@Composable
fun CameraPreview(viewModel: CameraViewModel = viewModel()) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val detectionState by viewModel.detectionState.collectAsState()
    val cameraState by viewModel.cameraState.collectAsState()
    val analysisExecutor = remember { Executors.newSingleThreadExecutor() }

    DisposableEffect(Unit) {
        onDispose { analysisExecutor.shutdown() }
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
            onRetake = { viewModel.resetToCamera() }
        )
        return
    }

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            factory = { ctx ->
                val previewView = PreviewView(ctx).apply {
                    implementationMode = PreviewView.ImplementationMode.PERFORMANCE
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                }

                bindCamera(ctx, lifecycleOwner, previewView, viewModel, analysisExecutor)

                previewView
            },
            modifier = Modifier.fillMaxSize()
        )

        // Quad overlay
        QuadOverlay(
            detectionState = detectionState,
            modifier = Modifier.fillMaxSize()
        )

        // Debug: detection latency
        if (detectionState.normalizedCorners != null) {
            Text(
                text = "%.0f ms".format(detectionState.detectionMs),
                color = Color.White,
                style = MaterialTheme.typography.labelSmall,
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(16.dp)
            )
        }

        // Capture FAB
        val isIdle = cameraState is CameraUiState.Idle
        val isBusy = cameraState is CameraUiState.Capturing || cameraState is CameraUiState.Processing

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
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
            .build()

        viewModel.imageCapture = imageCapture

        cameraProvider.unbindAll()
        try {
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                imageAnalysis,
                imageCapture
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to bind 3 use cases, retrying without ImageCapture", e)
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                imageAnalysis
            )
            viewModel.imageCapture = null
        }
    }, ContextCompat.getMainExecutor(context))
}

/**
 * Draws the detected document quadrilateral over the camera preview.
 * Maps normalized [0,1] detection coordinates to canvas space using FILL_CENTER
 * scaling to match PreviewView's coordinate system.
 */
@Composable
private fun QuadOverlay(
    detectionState: DetectionUiState,
    modifier: Modifier = Modifier
) {
    val corners = detectionState.normalizedCorners ?: return
    val srcW = detectionState.sourceWidth
    val srcH = detectionState.sourceHeight
    if (srcW == 0 || srcH == 0) return

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

        // Draw quad outline
        val path = Path().apply {
            val p0 = mapPoint(corners[0])
            moveTo(p0.x, p0.y)
            for (i in 1..3) {
                val p = mapPoint(corners[i])
                lineTo(p.x, p.y)
            }
            close()
        }

        drawPath(
            path = path,
            color = Color.Green,
            style = Stroke(width = 3.dp.toPx())
        )

        // Draw corner dots
        for (corner in corners) {
            drawCircle(
                color = Color.Green,
                radius = 6.dp.toPx(),
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
