package com.docshot.util

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.util.Log
import androidx.exifinterface.media.ExifInterface
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.max

private const val TAG = "DocShot:GalleryLoader"

/**
 * Loads a gallery image from a content URI with EXIF rotation handling
 * and downscaling to keep the longest edge within [maxDimension].
 *
 * Opens the URI twice: once for EXIF (ExifInterface consumes its stream)
 * and once for decoding.
 */
suspend fun loadGalleryImage(
    context: Context,
    uri: Uri,
    maxDimension: Int = 4000
): Bitmap = withContext(Dispatchers.IO) {
    val start = System.nanoTime()

    // 1. Read EXIF orientation
    val rotation = context.contentResolver.openInputStream(uri)?.use { stream ->
        val exif = ExifInterface(stream)
        exifToDegrees(exif.getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_NORMAL
        ))
    } ?: 0

    // 2. Decode bounds only to compute inSampleSize
    val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    context.contentResolver.openInputStream(uri)?.use { stream ->
        BitmapFactory.decodeStream(stream, null, opts)
    }
    require(opts.outWidth > 0 && opts.outHeight > 0) { "Failed to decode image bounds from URI" }

    // Account for rotation when computing effective dimensions
    val effectiveWidth = if (rotation == 90 || rotation == 270) opts.outHeight else opts.outWidth
    val effectiveHeight = if (rotation == 90 || rotation == 270) opts.outWidth else opts.outHeight
    val longestEdge = max(effectiveWidth, effectiveHeight)

    // 3. Compute inSampleSize (power of 2 downscale)
    var sampleSize = 1
    while (longestEdge / (sampleSize * 2) >= maxDimension) {
        sampleSize *= 2
    }

    // 4. Decode with inSampleSize
    val decodeOpts = BitmapFactory.Options().apply { inSampleSize = sampleSize }
    val raw = context.contentResolver.openInputStream(uri)?.use { stream ->
        BitmapFactory.decodeStream(stream, null, decodeOpts)
    }
    requireNotNull(raw) { "Failed to decode image from URI" }

    // 5. Apply EXIF rotation + further downscale if still over maxDimension
    val currentLongest = max(raw.width, raw.height).toFloat()
    val needsRotation = rotation != 0
    val needsScale = currentLongest > maxDimension

    val result = if (needsRotation || needsScale) {
        val matrix = Matrix()
        if (needsScale) {
            val scale = maxDimension / currentLongest
            matrix.postScale(scale, scale)
        }
        if (needsRotation) {
            matrix.postRotate(rotation.toFloat())
        }
        val transformed = Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, matrix, true)
        if (transformed !== raw) raw.recycle()
        transformed
    } else {
        raw
    }

    val ms = (System.nanoTime() - start) / 1_000_000.0
    Log.d(TAG, "loadGalleryImage: %.1f ms (%dx%d, sampleSize=%d, rotation=%d°)"
        .format(ms, result.width, result.height, sampleSize, rotation))

    result
}

private fun exifToDegrees(orientation: Int): Int = when (orientation) {
    ExifInterface.ORIENTATION_ROTATE_90 -> 90
    ExifInterface.ORIENTATION_ROTATE_180 -> 180
    ExifInterface.ORIENTATION_ROTATE_270 -> 270
    else -> 0
}
