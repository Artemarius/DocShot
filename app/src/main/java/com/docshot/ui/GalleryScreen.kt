package com.docshot.ui

import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Snackbar
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
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

    // Restore result screen from cache after process death (share intent killed Activity)
    LaunchedEffect(Unit) {
        if (state is GalleryUiState.Idle) {
            viewModel.restoreFromCache(context)
        }
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
                onSave = { bitmap ->
                    viewModel.saveResult(context, bitmap) { success ->
                        val msg = if (success) "Saved to gallery" else "Save failed"
                        Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
                    }
                },
                onShare = { bitmap -> viewModel.shareResult(context, bitmap) },
                onRetake = {
                    viewModel.clearResultCache(context)
                    viewModel.reset()
                },
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
                autoWhiteBalanceEnabled = settings.autoWhiteBalance,
                onToggleWhiteBalance = { /* local toggle only, no persistence needed */ }
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

