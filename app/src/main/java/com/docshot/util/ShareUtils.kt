package com.docshot.util

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.util.Log
import androidx.core.content.FileProvider
import java.io.File

private const val TAG = "DocShot:Share"
private const val AUTHORITY = "com.docshot.fileprovider"
private const val JPEG_QUALITY = 95

/**
 * Shares a bitmap via the system share sheet using [Intent.ACTION_SEND].
 * Writes the bitmap to a temp JPEG in cacheDir/shared/ and generates a
 * content:// URI via [FileProvider].
 */
fun shareImage(context: Context, bitmap: Bitmap) {
    val shareDir = File(context.cacheDir, "shared")
    if (!shareDir.exists()) shareDir.mkdirs()

    val file = File(shareDir, "docshot_share.jpg")
    file.outputStream().use { stream ->
        bitmap.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, stream)
    }

    val uri = FileProvider.getUriForFile(context, AUTHORITY, file)

    val intent = Intent(Intent.ACTION_SEND).apply {
        type = "image/jpeg"
        putExtra(Intent.EXTRA_STREAM, uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }

    val chooser = Intent.createChooser(intent, null)
    chooser.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
    context.startActivity(chooser)

    Log.d(TAG, "Share intent launched")
}
