package com.docshot.ui

import android.graphics.Bitmap
import androidx.compose.animation.AnimatedVisibility
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
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.BottomAppBar
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import com.docshot.cv.PostProcessFilter
import com.docshot.cv.applyFilter
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ResultScreen(
    data: CaptureResultData,
    onSave: () -> Unit,
    onShare: () -> Unit,
    onRetake: () -> Unit
) {
    var showRectified by rememberSaveable { mutableStateOf(true) }
    var selectedFilter by rememberSaveable { mutableStateOf(PostProcessFilter.NONE.name) }
    var processedBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var isProcessing by remember { mutableStateOf(false) }

    // Apply filter off the main thread whenever the selection changes
    LaunchedEffect(selectedFilter, showRectified) {
        if (!showRectified || selectedFilter == PostProcessFilter.NONE.name) {
            processedBitmap?.recycle()
            processedBitmap = null
            return@LaunchedEffect
        }
        isProcessing = true
        val result = withContext(Dispatchers.Default) {
            applyFilter(
                source = data.rectifiedBitmap,
                filter = PostProcessFilter.valueOf(selectedFilter)
            )
        }
        processedBitmap?.recycle()
        processedBitmap = result
        isProcessing = false
    }

    // Clean up processed bitmap when leaving the screen
    DisposableEffect(Unit) {
        onDispose {
            processedBitmap?.recycle()
            processedBitmap = null
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Document Captured") },
                navigationIcon = {
                    IconButton(onClick = onRetake) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Retake"
                        )
                    }
                }
            )
        },
        bottomBar = {
            BottomAppBar {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp),
                    horizontalArrangement = Arrangement.SpaceEvenly
                ) {
                    OutlinedButton(
                        onClick = onShare,
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("Share")
                    }
                    Spacer(modifier = Modifier.width(16.dp))
                    Button(
                        onClick = onSave,
                        modifier = Modifier.weight(1f)
                    ) {
                        // TODO: Wire save/share to use processedBitmap when filter is active
                        //  (requires ViewModel changes in integration phase)
                        Text("Save")
                    }
                }
            }
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(horizontal = 16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Original / Rectified toggle
            Row(
                modifier = Modifier.padding(vertical = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                FilterChip(
                    selected = !showRectified,
                    onClick = { showRectified = false },
                    label = { Text("Original") }
                )
                FilterChip(
                    selected = showRectified,
                    onClick = { showRectified = true },
                    label = { Text("Rectified") }
                )
            }

            // Post-processing filter chips — only visible when viewing rectified image
            AnimatedVisibility(visible = showRectified) {
                Row(
                    modifier = Modifier.padding(bottom = 8.dp),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    FilterChip(
                        selected = selectedFilter == PostProcessFilter.NONE.name,
                        onClick = { selectedFilter = PostProcessFilter.NONE.name },
                        label = { Text("None") }
                    )
                    FilterChip(
                        selected = selectedFilter == PostProcessFilter.BLACK_WHITE.name,
                        onClick = { selectedFilter = PostProcessFilter.BLACK_WHITE.name },
                        label = { Text("B&W") }
                    )
                    FilterChip(
                        selected = selectedFilter == PostProcessFilter.CONTRAST.name,
                        onClick = { selectedFilter = PostProcessFilter.CONTRAST.name },
                        label = { Text("Contrast") }
                    )
                    FilterChip(
                        selected = selectedFilter == PostProcessFilter.COLOR_CORRECT.name,
                        onClick = { selectedFilter = PostProcessFilter.COLOR_CORRECT.name },
                        label = { Text("Color Fix") }
                    )
                }
            }

            // Image display with optional processing spinner
            val displayBitmap = when {
                !showRectified -> data.originalBitmap
                processedBitmap != null -> processedBitmap!!
                else -> data.rectifiedBitmap
            }

            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
                contentAlignment = Alignment.Center
            ) {
                Image(
                    bitmap = displayBitmap.asImageBitmap(),
                    contentDescription = when {
                        !showRectified -> "Original photo"
                        selectedFilter != PostProcessFilter.NONE.name -> "Filtered document"
                        else -> "Rectified document"
                    },
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Fit
                )

                if (isProcessing) {
                    CircularProgressIndicator()
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = "Pipeline: %.0f ms".format(data.pipelineMs),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Spacer(modifier = Modifier.height(8.dp))
        }
    }
}
