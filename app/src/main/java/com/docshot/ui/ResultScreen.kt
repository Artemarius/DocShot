package com.docshot.ui

import android.graphics.Bitmap
import android.util.Log
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.RotateRight
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.LockOpen
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Fill
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import com.docshot.cv.KNOWN_FORMATS
import com.docshot.cv.PostProcessFilter
import com.docshot.cv.applyFilter
import com.docshot.cv.estimateAspectRatio
import kotlin.math.abs
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext

private const val TAG = "DocShot:ResultScreen"

/** Snap threshold for format label reactivity (matches AspectRatioEstimator). */
private const val FORMAT_SNAP_THRESHOLD = 0.06f

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ResultScreen(
    data: CaptureResultData,
    onSave: () -> Unit,
    onShare: () -> Unit,
    onRetake: () -> Unit,
    onAdjust: () -> Unit = {},
    onRotate: () -> Unit = {},
    onAspectRatioChange: (Double) -> Unit = {},
    isAspectRatioLocked: Boolean = false,
    lockedAspectRatio: Float = 0.707f,
    onToggleAspectRatioLock: (Boolean, Float) -> Unit = { _, _ -> },
    aspectRatioAutoEstimate: Boolean = true
) {
    var showRectified by rememberSaveable { mutableStateOf(true) }
    var selectedFilter by rememberSaveable { mutableStateOf(PostProcessFilter.NONE.name) }
    var processedBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var isProcessing by remember { mutableStateOf(false) }

    // Aspect ratio estimation (runs once per corner set)
    val estimate = remember(data.corners, data.cameraIntrinsics) {
        if (data.corners.size == 4) estimateAspectRatio(data.corners, data.cameraIntrinsics) else null
    }

    // Raw ratio from current rectified bitmap (fallback when no estimation)
    val bitmapRatio = remember(data.rectifiedBitmap) {
        val w = data.rectifiedBitmap.width.toFloat()
        val h = data.rectifiedBitmap.height.toFloat()
        minOf(w, h) / maxOf(w, h)
    }

    // Initial aspect ratio priority: (1) locked > (2) multi-frame estimate > (3) A4 default
    val defaultRatio = remember(isAspectRatioLocked, lockedAspectRatio, data.estimatedAspectRatio, aspectRatioAutoEstimate) {
        when {
            isAspectRatioLocked -> {
                Log.d(TAG, "Initial ratio: %.4f (locked)".format(lockedAspectRatio))
                lockedAspectRatio
            }
            aspectRatioAutoEstimate && data.estimatedAspectRatio != null -> {
                Log.d(TAG, "Initial ratio: %.4f (multi-frame estimate)".format(data.estimatedAspectRatio))
                data.estimatedAspectRatio
            }
            else -> {
                Log.d(TAG, "Initial ratio: 0.7070 (A4 default)")
                0.707f
            }
        }
    }

    // Track whether the initial ratio was set from auto-estimation
    val wasAutoEstimated = remember(isAspectRatioLocked, data.estimatedAspectRatio, aspectRatioAutoEstimate) {
        !isAspectRatioLocked && aspectRatioAutoEstimate && data.estimatedAspectRatio != null
    }
    // User has manually moved the slider or picked from dropdown — suppress "(auto)" label
    var userAdjustedRatio by rememberSaveable { mutableStateOf(false) }
    var currentRatio by rememberSaveable {
        mutableFloatStateOf(defaultRatio)
    }

    // Reactive format label: updates as slider moves, appends "(auto)" when auto-estimated
    val currentFormatLabel = remember(currentRatio, wasAutoEstimated, userAdjustedRatio) {
        val baseName = KNOWN_FORMATS
            .filter { abs(currentRatio - it.ratio.toFloat()) <= FORMAT_SNAP_THRESHOLD }
            .minByOrNull { abs(currentRatio - it.ratio.toFloat()) }
            ?.name ?: "Custom"
        if (wasAutoEstimated && !userAdjustedRatio) "$baseName (auto)" else baseName
    }

    // Format dropdown state
    var showFormatMenu by remember { mutableStateOf(false) }

    // Debounced re-warp when slider changes; initialized to default to trigger first warp
    var pendingRatio by remember { mutableStateOf<Float?>(currentRatio) }
    LaunchedEffect(pendingRatio) {
        val ratio = pendingRatio ?: return@LaunchedEffect
        delay(300) // debounce
        onAspectRatioChange(ratio.toDouble())
    }

    // Apply filter off the main thread whenever the selection or bitmap changes
    LaunchedEffect(selectedFilter, showRectified, data.rectifiedBitmap) {
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

    val hasCorners = data.normalizedCorners.size == 8

    Column(modifier = Modifier.fillMaxSize()) {
        // Top bar
        TopAppBar(
            title = { Text("Result") },
            navigationIcon = {
                IconButton(onClick = onRetake) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                        contentDescription = "Retake"
                    )
                }
            },
            actions = {
                IconButton(onClick = onRotate) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.RotateRight,
                        contentDescription = "Rotate"
                    )
                }
                if (hasCorners) {
                    IconButton(onClick = onAdjust) {
                        Icon(
                            imageVector = Icons.Filled.Tune,
                            contentDescription = "Adjust corners"
                        )
                    }
                }
            }
        )

        // Combined view toggle + filter chips in one scrollable row
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState())
                .padding(horizontal = 16.dp, vertical = 2.dp),
            horizontalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            FilterChip(
                selected = !showRectified,
                onClick = { showRectified = false },
                label = { Text("Original") }
            )
            FilterChip(
                selected = showRectified && selectedFilter == PostProcessFilter.NONE.name,
                onClick = {
                    showRectified = true
                    selectedFilter = PostProcessFilter.NONE.name
                },
                label = { Text("Rectified") }
            )
            FilterChip(
                selected = showRectified && selectedFilter == PostProcessFilter.BLACK_WHITE.name,
                onClick = {
                    showRectified = true
                    selectedFilter = PostProcessFilter.BLACK_WHITE.name
                },
                label = { Text("B&W") }
            )
            FilterChip(
                selected = showRectified && selectedFilter == PostProcessFilter.CONTRAST.name,
                onClick = {
                    showRectified = true
                    selectedFilter = PostProcessFilter.CONTRAST.name
                },
                label = { Text("Contrast") }
            )
            FilterChip(
                selected = showRectified && selectedFilter == PostProcessFilter.COLOR_CORRECT.name,
                onClick = {
                    showRectified = true
                    selectedFilter = PostProcessFilter.COLOR_CORRECT.name
                },
                label = { Text("Even Light") }
            )
        }

        // Aspect ratio: lock button + clickable format label (dropdown) + slider
        AnimatedVisibility(visible = showRectified && data.corners.isNotEmpty()) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Lock/unlock aspect ratio button
                IconButton(
                    onClick = {
                        onToggleAspectRatioLock(!isAspectRatioLocked, currentRatio)
                    },
                    modifier = Modifier.size(32.dp)
                ) {
                    Icon(
                        imageVector = if (isAspectRatioLocked) Icons.Filled.Lock
                            else Icons.Filled.LockOpen,
                        contentDescription = if (isAspectRatioLocked) "Unlock aspect ratio"
                            else "Lock aspect ratio",
                        modifier = Modifier.size(18.dp),
                        tint = if (isAspectRatioLocked) MaterialTheme.colorScheme.primary
                            else MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                // Clickable format label with dropdown
                Box {
                    Row(
                        modifier = Modifier.clickable(enabled = !isAspectRatioLocked) {
                            showFormatMenu = true
                        },
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = currentFormatLabel,
                            style = MaterialTheme.typography.labelSmall,
                            color = if (isAspectRatioLocked) MaterialTheme.colorScheme.onSurfaceVariant
                                else MaterialTheme.colorScheme.primary
                        )
                        Icon(
                            imageVector = Icons.Filled.ArrowDropDown,
                            contentDescription = "Select format",
                            modifier = Modifier.size(16.dp),
                            tint = if (isAspectRatioLocked) MaterialTheme.colorScheme.onSurfaceVariant
                                else MaterialTheme.colorScheme.primary
                        )
                    }
                    DropdownMenu(
                        expanded = showFormatMenu,
                        onDismissRequest = { showFormatMenu = false }
                    ) {
                        KNOWN_FORMATS.forEach { format ->
                            DropdownMenuItem(
                                text = { Text(format.name) },
                                onClick = {
                                    currentRatio = format.ratio.toFloat()
                                    pendingRatio = currentRatio
                                    userAdjustedRatio = true
                                    showFormatMenu = false
                                }
                            )
                        }
                    }
                }

                Spacer(modifier = Modifier.width(8.dp))

                // Slider fills remaining space (disabled when locked)
                Slider(
                    value = currentRatio,
                    onValueChange = {
                        currentRatio = it
                        userAdjustedRatio = true
                    },
                    onValueChangeFinished = { pendingRatio = currentRatio },
                    valueRange = 0.25f..1.0f,
                    enabled = !isAspectRatioLocked,
                    modifier = Modifier.weight(1f)
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
                .weight(1f)
                .padding(horizontal = 16.dp),
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

            // Quad overlay on the original image showing detected document borders
            if (!showRectified && hasCorners) {
                val imgW = data.originalBitmap.width.toFloat()
                val imgH = data.originalBitmap.height.toFloat()
                val nc = data.normalizedCorners

                Canvas(modifier = Modifier.fillMaxSize()) {
                    val transform = computeFitTransform(
                        containerWidth = size.width,
                        containerHeight = size.height,
                        imageWidth = imgW,
                        imageHeight = imgH
                    )

                    fun mapCorner(index: Int): androidx.compose.ui.geometry.Offset {
                        val px = nc[index * 2] * imgW
                        val py = nc[index * 2 + 1] * imgH
                        return androidx.compose.ui.geometry.Offset(
                            x = px * transform.scale + transform.offsetX,
                            y = py * transform.scale + transform.offsetY
                        )
                    }

                    val path = Path().apply {
                        val p0 = mapCorner(0)
                        moveTo(p0.x, p0.y)
                        for (i in 1..3) {
                            val p = mapCorner(i)
                            lineTo(p.x, p.y)
                        }
                        close()
                    }

                    drawPath(
                        path = path,
                        color = Color.Green.copy(alpha = 0.15f),
                        style = Fill
                    )
                    drawPath(
                        path = path,
                        color = Color.Green,
                        style = Stroke(width = 3.dp.toPx())
                    )
                    for (i in 0..3) {
                        drawCircle(
                            color = Color.Green,
                            radius = 6.dp.toPx(),
                            center = mapCorner(i)
                        )
                    }
                }
            }

            if (isProcessing) {
                CircularProgressIndicator()
            }
        }

        // Slim bottom action row
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp),
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
                Text("Save")
            }
        }
    }
}
