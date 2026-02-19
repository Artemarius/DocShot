package com.docshot.ui

import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Snackbar
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.docshot.R
import com.docshot.util.DocShotSettings
import com.docshot.util.UserPreferencesRepository
import kotlinx.coroutines.launch

@Composable
fun GalleryScreen(
    viewModel: GalleryViewModel = viewModel(),
    onShowingResult: (Boolean) -> Unit = {},
    preferencesRepository: UserPreferencesRepository? = null
) {
    val state by viewModel.state.collectAsState()
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val settings by preferencesRepository?.settings?.collectAsState(
        initial = DocShotSettings()
    ) ?: androidx.compose.runtime.remember { androidx.compose.runtime.mutableStateOf(DocShotSettings()) }

    // Notify parent when showing/hiding result screen
    LaunchedEffect(state) {
        onShowingResult(state is GalleryUiState.Result)
    }

    val photoPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        if (uri != null) {
            viewModel.loadAndDetect(context, uri)
        }
    }

    when (val current = state) {
        is GalleryUiState.Idle -> {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Icon(
                        painter = painterResource(id = R.drawable.ic_photo_library),
                        contentDescription = null,
                        modifier = Modifier.size(64.dp),
                        tint = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Spacer(modifier = Modifier.height(16.dp))
                    Button(onClick = {
                        photoPickerLauncher.launch(
                            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                        )
                    }) {
                        Text("Select Photo")
                    }
                }
            }
        }

        is GalleryUiState.Loading -> {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CircularProgressIndicator()
                    Spacer(modifier = Modifier.height(16.dp))
                    Text("Loading image...")
                }
            }
        }

        is GalleryUiState.Detected -> {
            DetectedPreview(
                state = current,
                onAccept = { viewModel.acceptDetection() },
                onAdjust = { viewModel.enterManualAdjust() },
                onPickAnother = {
                    viewModel.reset()
                    photoPickerLauncher.launch(
                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                    )
                }
            )
        }

        is GalleryUiState.ManualAdjust -> {
            CornerAdjustScreen(
                bitmap = current.bitmap,
                corners = current.corners,
                onApply = { adjustedCorners ->
                    for ((i, corner) in adjustedCorners.withIndex()) {
                        viewModel.updateCorner(i, corner)
                    }
                    viewModel.applyManualCorners()
                },
                onCancel = { viewModel.reset() }
            )
        }

        is GalleryUiState.Rectifying -> {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CircularProgressIndicator()
                    Spacer(modifier = Modifier.height(16.dp))
                    Text("Rectifying document...")
                }
            }
        }

        is GalleryUiState.Result -> {
            ResultScreen(
                data = current.data,
                onSave = {
                    viewModel.saveResult(context) { success ->
                        val msg = if (success) "Saved to gallery" else "Save failed"
                        Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
                    }
                },
                onShare = { viewModel.shareResult(context) },
                onRetake = { viewModel.reset() },
                onAspectRatioChange = { viewModel.reWarpWithAspectRatio(it) },
                isAspectRatioLocked = settings.aspectRatioLocked,
                lockedAspectRatio = settings.lockedAspectRatio,
                onToggleAspectRatioLock = { locked, ratio ->
                    scope.launch {
                        preferencesRepository?.setAspectRatioLocked(locked)
                        preferencesRepository?.setLockedAspectRatio(ratio)
                    }
                }
            )
        }

        is GalleryUiState.Error -> {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                Snackbar(
                    modifier = Modifier.padding(16.dp)
                ) {
                    Text(current.message)
                }
            }
        }
    }
}

/**
 * Preview of the detected document with quad overlay.
 * Shows Accept / Adjust Corners / Pick Another buttons.
 */
@Composable
private fun DetectedPreview(
    state: GalleryUiState.Detected,
    onAccept: () -> Unit,
    onAdjust: () -> Unit,
    onPickAnother: () -> Unit
) {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "Document Detected (%.0f ms)".format(state.detectionMs),
            style = MaterialTheme.typography.titleMedium,
            modifier = Modifier.padding(16.dp)
        )

        // Image with quad overlay
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .padding(horizontal = 16.dp)
        ) {
            Image(
                bitmap = state.bitmap.asImageBitmap(),
                contentDescription = "Detected document",
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Fit
            )

            // Overlay the detected quad
            val imgW = state.bitmap.width.toFloat()
            val imgH = state.bitmap.height.toFloat()

            androidx.compose.foundation.Canvas(modifier = Modifier.fillMaxSize()) {
                val transform = computeFitTransform(
                    containerWidth = size.width,
                    containerHeight = size.height,
                    imageWidth = imgW,
                    imageHeight = imgH
                )

                val path = androidx.compose.ui.graphics.Path().apply {
                    val p0 = imageToScreenOffset(state.corners[0], transform)
                    moveTo(p0.x, p0.y)
                    for (i in 1..3) {
                        val p = imageToScreenOffset(state.corners[i], transform)
                        lineTo(p.x, p.y)
                    }
                    close()
                }

                drawPath(
                    path = path,
                    color = androidx.compose.ui.graphics.Color.Green.copy(alpha = 0.2f),
                    style = androidx.compose.ui.graphics.drawscope.Fill
                )
                drawPath(
                    path = path,
                    color = androidx.compose.ui.graphics.Color.Green,
                    style = androidx.compose.ui.graphics.drawscope.Stroke(width = 3.dp.toPx())
                )

                for (corner in state.corners) {
                    val p = imageToScreenOffset(corner, transform)
                    drawCircle(
                        color = androidx.compose.ui.graphics.Color.Green,
                        radius = 6.dp.toPx(),
                        center = p
                    )
                }
            }
        }

        // Action buttons
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            OutlinedButton(onClick = onPickAnother) {
                Text("Pick Another")
            }
            OutlinedButton(onClick = onAdjust) {
                Text("Adjust")
            }
            Button(onClick = onAccept) {
                Text("Accept")
            }
        }
    }
}

/** Helper to convert image Point to screen Offset using FitTransform. */
private fun imageToScreenOffset(point: org.opencv.core.Point, transform: FitTransform): androidx.compose.ui.geometry.Offset {
    return androidx.compose.ui.geometry.Offset(
        x = point.x.toFloat() * transform.scale + transform.offsetX,
        y = point.y.toFloat() * transform.scale + transform.offsetY
    )
}
