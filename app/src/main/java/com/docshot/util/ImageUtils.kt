package com.docshot.util

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import org.opencv.android.Utils
import org.opencv.core.Mat

/**
 * Loads an image from the app's assets folder as an OpenCV Mat in BGR format.
 * Caller is responsible for releasing the returned Mat.
 */
fun loadMatFromAsset(context: Context, fileName: String): Mat {
    val bitmap = context.assets.open(fileName).use { stream ->
        BitmapFactory.decodeStream(stream)
    }
    requireNotNull(bitmap) { "Failed to decode asset: $fileName" }
    val mat = Mat()
    Utils.bitmapToMat(bitmap, mat)
    bitmap.recycle()
    return mat
}

/** Converts an OpenCV Mat (BGR or RGBA) to an Android Bitmap. */
fun matToBitmap(mat: Mat): Bitmap {
    val bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
    Utils.matToBitmap(mat, bitmap)
    return bitmap
}

/** Converts an Android Bitmap to an OpenCV Mat (RGBA). Caller must release the returned Mat. */
fun bitmapToMat(bitmap: Bitmap): Mat {
    val mat = Mat()
    Utils.bitmapToMat(bitmap, mat)
    return mat
}
