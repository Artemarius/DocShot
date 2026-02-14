package com.docshot.util

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

private const val TAG = "DocShot:MediaStore"
private const val JPEG_QUALITY = 95
private const val SUBFOLDER = "DocShot"

/**
 * Saves a bitmap as JPEG to the device gallery under DCIM/DocShot/.
 *
 * Uses scoped storage (MediaStore with IS_PENDING) on API 29+ and falls back
 * to direct file access with MediaScanner on API 24–28.
 *
 * @return Content URI of the saved image, or null on failure.
 */
suspend fun saveBitmapToGallery(
    context: Context,
    bitmap: Bitmap,
    displayName: String
): Uri? = withContext(Dispatchers.IO) {
    try {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            saveScopedStorage(context, bitmap, displayName)
        } else {
            saveLegacy(context, bitmap, displayName)
        }
    } catch (e: Exception) {
        Log.e(TAG, "Failed to save to gallery", e)
        null
    }
}

private fun saveScopedStorage(
    context: Context,
    bitmap: Bitmap,
    displayName: String
): Uri? {
    val values = ContentValues().apply {
        put(MediaStore.Images.Media.DISPLAY_NAME, "$displayName.jpg")
        put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
        put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_DCIM}/$SUBFOLDER")
        put(MediaStore.Images.Media.IS_PENDING, 1)
    }

    val resolver = context.contentResolver
    val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
        ?: return null

    resolver.openOutputStream(uri)?.use { stream ->
        bitmap.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, stream)
    } ?: run {
        resolver.delete(uri, null, null)
        return null
    }

    values.clear()
    values.put(MediaStore.Images.Media.IS_PENDING, 0)
    resolver.update(uri, values, null, null)

    Log.d(TAG, "Saved to gallery (scoped): $uri")
    return uri
}

@Suppress("DEPRECATION")
private fun saveLegacy(
    context: Context,
    bitmap: Bitmap,
    displayName: String
): Uri? {
    val dcim = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM)
    val dir = File(dcim, SUBFOLDER)
    if (!dir.exists()) dir.mkdirs()

    val file = File(dir, "$displayName.jpg")
    file.outputStream().use { stream ->
        bitmap.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, stream)
    }

    // Notify MediaScanner so the file appears in gallery apps
    MediaScannerConnection.scanFile(context, arrayOf(file.absolutePath), arrayOf("image/jpeg"), null)

    Log.d(TAG, "Saved to gallery (legacy): ${file.absolutePath}")
    return Uri.fromFile(file)
}
