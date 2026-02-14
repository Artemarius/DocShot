package com.docshot

import android.app.Application
import android.util.Log
import org.opencv.android.OpenCVLoader

class DocShotApplication : Application() {

    override fun onCreate() {
        super.onCreate()
        if (OpenCVLoader.initLocal()) {
            Log.d(TAG, "OpenCV loaded: ${OpenCVLoader.OPENCV_VERSION}")
        } else {
            Log.e(TAG, "OpenCV initialization failed")
        }
    }

    companion object {
        private const val TAG = "DocShot"
    }
}
