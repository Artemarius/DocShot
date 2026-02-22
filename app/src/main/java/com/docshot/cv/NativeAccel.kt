package com.docshot.cv

import android.util.Log

private const val TAG = "DocShot:NativeAccel"

/**
 * JNI bridge to native C++ acceleration for hot-path CV operations.
 *
 * Loads `libdocshot_native.so` on first access. If the native library
 * is unavailable (e.g., unsupported ABI), [isAvailable] returns false
 * and callers fall back to the equivalent Kotlin implementation.
 */
object NativeAccel {

    val isAvailable: Boolean = try {
        System.loadLibrary("docshot_native")
        Log.d(TAG, "Native acceleration loaded successfully")
        true
    } catch (_: UnsatisfiedLinkError) {
        Log.w(TAG, "Native acceleration unavailable — using Kotlin fallback")
        false
    }

    /**
     * Steps 4-6 of the DIRECTIONAL_GRADIENT preprocessing strategy:
     * accumulate directional responses across 5 tilt angles, normalize,
     * and threshold to a binary edge map.
     *
     * @param gyData     |Gy| gradient image as flat ByteArray (rows * cols)
     * @param gxData     |Gx| gradient image as flat ByteArray (rows * cols)
     * @param resultData Output buffer, pre-allocated by caller (rows * cols)
     * @param rows       Image height
     * @param cols       Image width
     * @param hOffsets   Flat offset table for H-edge accumulation (numAngles * kernelLength)
     * @param vOffsets   Flat offset table for V-edge accumulation (numAngles * kernelLength)
     * @param numAngles  Number of tilt angles (5)
     * @param kernelLength Length of 1D smoothing kernel (21)
     * @param marginY    Vertical margin to skip
     * @param marginX    Horizontal margin to skip
     * @param thresholdPercentile Percentile for binary threshold (0.90)
     */
    @JvmStatic
    external fun nativeDirectionalGradient(
        gyData: ByteArray, gxData: ByteArray,
        resultData: ByteArray,
        rows: Int, cols: Int,
        hOffsets: IntArray, vOffsets: IntArray,
        numAngles: Int, kernelLength: Int,
        marginY: Int, marginX: Int,
        thresholdPercentile: Float
    )
}
