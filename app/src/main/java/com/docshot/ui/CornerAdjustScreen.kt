package com.docshot.ui

import android.graphics.Bitmap
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.BottomAppBar
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Fill
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import org.opencv.core.Point
import kotlin.math.roundToInt
import kotlin.math.sqrt

private const val HANDLE_RADIUS_DP = 12f
private const val HIT_RADIUS_DP = 24f
private const val LOUPE_ZOOM = 3f
private const val LOUPE_SIZE_DP = 140f

/**
 * Manual corner adjustment screen. Displays the image with four draggable corner handles.
 * When dragging a handle, a magnifier loupe appears in the opposite corner of the screen
 * showing a zoomed-in view around the active handle so the finger doesn't block the view.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CornerAdjustScreen(
    bitmap: Bitmap,
    corners: List<Point>,
    onApply: (List<Point>) -> Unit,
    onCancel: () -> Unit
) {
    var containerSize by remember { mutableStateOf(IntSize.Zero) }
    var activeHandle by remember { mutableIntStateOf(-1) }

    // Local mutable copy of corners in image coordinates
    var currentCorners by remember(corners) { mutableStateOf(corners.toList()) }

    val imageBitmap = remember(bitmap) { bitmap.asImageBitmap() }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Adjust Corners") },
                navigationIcon = {
                    IconButton(onClick = onCancel) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Cancel"
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
                        onClick = onCancel,
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("Cancel")
                    }
                    Spacer(modifier = Modifier.padding(horizontal = 8.dp))
                    Button(
                        onClick = { onApply(currentCorners) },
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("Apply")
                    }
                }
            }
        }
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
        ) {
            // Image layer
            Image(
                bitmap = imageBitmap,
                contentDescription = "Document to adjust",
                modifier = Modifier
                    .fillMaxSize()
                    .onSizeChanged { containerSize = it },
                contentScale = ContentScale.Fit
            )

            // Canvas overlay for handles + quad
            if (containerSize.width > 0 && containerSize.height > 0) {
                val imgW = bitmap.width.toFloat()
                val imgH = bitmap.height.toFloat()
                val transform = remember(containerSize, imgW, imgH) {
                    computeFitTransform(
                        containerWidth = containerSize.width.toFloat(),
                        containerHeight = containerSize.height.toFloat(),
                        imageWidth = imgW,
                        imageHeight = imgH
                    )
                }

                Canvas(
                    modifier = Modifier
                        .fillMaxSize()
                        .pointerInput(Unit) {
                            detectDragGestures(
                                onDragStart = { offset ->
                                    activeHandle = findNearestHandle(
                                        offset,
                                        currentCorners,
                                        transform,
                                        hitRadiusPx = HIT_RADIUS_DP * density
                                    )
                                },
                                onDrag = { change, _ ->
                                    if (activeHandle >= 0) {
                                        change.consume()
                                        val imagePoint = screenToImage(
                                            change.position,
                                            transform,
                                            imgW,
                                            imgH
                                        )
                                        val updated = currentCorners.toMutableList()
                                        updated[activeHandle] = imagePoint
                                        currentCorners = updated
                                    }
                                },
                                onDragEnd = { activeHandle = -1 },
                                onDragCancel = { activeHandle = -1 }
                            )
                        }
                ) {
                    drawQuadOverlay(currentCorners, transform, activeHandle)
                }

                // Magnifier loupe — shown while dragging, positioned opposite to active handle
                if (activeHandle >= 0) {
                    val activeCorner = currentCorners[activeHandle]
                    val loupeAlignment = loupeAlignment(activeHandle, activeCorner, imgW, imgH)

                    MagnifierLoupe(
                        imageBitmap = imageBitmap,
                        center = activeCorner,
                        imageWidth = bitmap.width,
                        imageHeight = bitmap.height,
                        modifier = Modifier
                            .align(loupeAlignment)
                            .padding(12.dp)
                    )
                }
            }
        }
    }
}

/**
 * Renders a zoomed-in crop of the image around [center], with a crosshair at the center.
 */
@Composable
private fun MagnifierLoupe(
    imageBitmap: ImageBitmap,
    center: Point,
    imageWidth: Int,
    imageHeight: Int,
    modifier: Modifier = Modifier
) {
    val loupeSizeDp = LOUPE_SIZE_DP.dp
    val shape = RoundedCornerShape(12.dp)

    Canvas(
        modifier = modifier
            .size(loupeSizeDp)
            .clip(shape)
            .border(2.dp, MaterialTheme.colorScheme.outline, shape)
    ) {
        val loupePx = size.width

        // Source region in image pixels: area around the corner at LOUPE_ZOOM magnification
        val srcSizePx = (loupePx / LOUPE_ZOOM)
        val halfSrc = srcSizePx / 2f

        val srcX = (center.x.toFloat() - halfSrc).coerceIn(0f, (imageWidth - srcSizePx).coerceAtLeast(0f))
        val srcY = (center.y.toFloat() - halfSrc).coerceIn(0f, (imageHeight - srcSizePx).coerceAtLeast(0f))

        drawImage(
            image = imageBitmap,
            srcOffset = IntOffset(srcX.roundToInt(), srcY.roundToInt()),
            srcSize = IntSize(srcSizePx.roundToInt().coerceAtLeast(1), srcSizePx.roundToInt().coerceAtLeast(1)),
            dstSize = IntSize(loupePx.roundToInt(), loupePx.roundToInt())
        )

        // Crosshair at center of loupe
        val cx = loupePx / 2f
        val cy = loupePx / 2f
        val crossLen = 12.dp.toPx()
        val crossStroke = 2.dp.toPx()

        drawLine(Color.Yellow, Offset(cx - crossLen, cy), Offset(cx + crossLen, cy), crossStroke)
        drawLine(Color.Yellow, Offset(cx, cy - crossLen), Offset(cx, cy + crossLen), crossStroke)

        // Border circle around crosshair
        drawCircle(Color.Yellow, radius = crossLen, center = Offset(cx, cy), style = Stroke(crossStroke))
    }
}

/**
 * Determines where to place the loupe so it's opposite to the active corner.
 * TL corner (index 0) → loupe at BottomEnd, TR (1) → BottomStart, etc.
 * Falls back based on which half the corner is in when it's been dragged.
 */
private fun loupeAlignment(
    handleIndex: Int,
    cornerPoint: Point,
    imageWidth: Float,
    imageHeight: Float
): Alignment {
    // Use corner position to decide — more robust when corners are dragged
    val inLeftHalf = cornerPoint.x < imageWidth / 2
    val inTopHalf = cornerPoint.y < imageHeight / 2

    return when {
        inTopHalf && inLeftHalf -> Alignment.BottomEnd
        inTopHalf && !inLeftHalf -> Alignment.BottomStart
        !inTopHalf && inLeftHalf -> Alignment.TopEnd
        else -> Alignment.TopStart
    }
}

/**
 * Describes the ContentScale.Fit transform: scale factor and offset
 * to map image coordinates to screen coordinates.
 */
data class FitTransform(
    val scale: Float,
    val offsetX: Float,
    val offsetY: Float
)

fun computeFitTransform(
    containerWidth: Float,
    containerHeight: Float,
    imageWidth: Float,
    imageHeight: Float
): FitTransform {
    val scaleX = containerWidth / imageWidth
    val scaleY = containerHeight / imageHeight
    val scale = minOf(scaleX, scaleY)
    val offsetX = (containerWidth - imageWidth * scale) / 2f
    val offsetY = (containerHeight - imageHeight * scale) / 2f
    return FitTransform(scale, offsetX, offsetY)
}

private fun imageToScreen(point: Point, transform: FitTransform): Offset {
    return Offset(
        x = point.x.toFloat() * transform.scale + transform.offsetX,
        y = point.y.toFloat() * transform.scale + transform.offsetY
    )
}

private fun screenToImage(
    offset: Offset,
    transform: FitTransform,
    imageWidth: Float,
    imageHeight: Float
): Point {
    val x = ((offset.x - transform.offsetX) / transform.scale)
        .coerceIn(0f, imageWidth - 1f)
    val y = ((offset.y - transform.offsetY) / transform.scale)
        .coerceIn(0f, imageHeight - 1f)
    return Point(x.toDouble(), y.toDouble())
}

private fun findNearestHandle(
    tapPosition: Offset,
    corners: List<Point>,
    transform: FitTransform,
    hitRadiusPx: Float
): Int {
    var bestIndex = -1
    var bestDist = Float.MAX_VALUE

    for (i in corners.indices) {
        val screenPos = imageToScreen(corners[i], transform)
        val dx = tapPosition.x - screenPos.x
        val dy = tapPosition.y - screenPos.y
        val dist = sqrt(dx * dx + dy * dy)
        if (dist < hitRadiusPx && dist < bestDist) {
            bestDist = dist
            bestIndex = i
        }
    }
    return bestIndex
}

private fun DrawScope.drawQuadOverlay(
    corners: List<Point>,
    transform: FitTransform,
    activeHandle: Int
) {
    val screenCorners = corners.map { imageToScreen(it, transform) }

    // Semi-transparent green fill
    val fillPath = Path().apply {
        moveTo(screenCorners[0].x, screenCorners[0].y)
        for (i in 1..3) {
            lineTo(screenCorners[i].x, screenCorners[i].y)
        }
        close()
    }
    drawPath(
        path = fillPath,
        color = Color.Green.copy(alpha = 0.15f),
        style = Fill
    )

    // Green quad outline
    drawPath(
        path = fillPath,
        color = Color.Green,
        style = Stroke(width = 3.dp.toPx())
    )

    // Corner handles
    val handleRadius = HANDLE_RADIUS_DP.dp.toPx()
    for (i in screenCorners.indices) {
        val color = if (i == activeHandle) Color.Yellow else Color.Green
        // Outer circle
        drawCircle(
            color = color,
            radius = handleRadius,
            center = screenCorners[i],
            style = Stroke(width = 3.dp.toPx())
        )
        // Inner fill
        drawCircle(
            color = color.copy(alpha = 0.4f),
            radius = handleRadius,
            center = screenCorners[i]
        )
    }
}
