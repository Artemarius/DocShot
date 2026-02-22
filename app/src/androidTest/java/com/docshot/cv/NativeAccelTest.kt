package com.docshot.cv

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin

private const val TAG = "DocShot:NativeAccelTest"

/**
 * Differential correctness test: verifies that native C++ and Kotlin fallback
 * paths for DIRECTIONAL_GRADIENT steps 4-6 produce identical (or near-identical)
 * results. Also benchmarks both paths.
 */
@RunWith(AndroidJUnit4::class)
class NativeAccelTest {

    companion object {
        private const val NUM_ANGLES = 5
        private const val KERNEL_LENGTH = 21
        private val TILT_ANGLES_DEG = doubleArrayOf(-10.0, -5.0, 0.0, 5.0, 10.0)

        @JvmStatic
        @BeforeClass
        fun initOpenCV() {
            val success = OpenCVLoader.initLocal()
            check(success) { "Failed to initialize OpenCV" }
            Log.d(TAG, "OpenCV initialized, NativeAccel.isAvailable=${NativeAccel.isAvailable}")
        }
    }

    /**
     * Prepares gradient data from a synthetic image, mimicking the real pipeline
     * steps 0-3 (downsample, blur, Sobel, ByteArray extraction).
     */
    private data class GradientData(
        val gyData: ByteArray,
        val gxData: ByteArray,
        val rows: Int,
        val cols: Int,
        val flatH: IntArray,
        val flatV: IntArray,
        val marginY: Int,
        val marginX: Int
    )

    private fun prepareGradientData(image: Mat): GradientData {
        // Convert to gray
        val gray = Mat()
        Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY)

        // Downsample 2x
        val small = Mat()
        Imgproc.resize(gray, small, Size(gray.cols() / 2.0, gray.rows() / 2.0),
            0.0, 0.0, Imgproc.INTER_AREA)
        gray.release()

        // Blur
        val blurred = Mat()
        Imgproc.GaussianBlur(small, blurred, Size(5.0, 5.0), 1.4)
        small.release()

        val rows = blurred.rows()
        val cols = blurred.cols()

        // Sobel
        val gradX16 = Mat()
        val gradY16 = Mat()
        Imgproc.Sobel(blurred, gradX16, CvType.CV_16S, 1, 0, 3)
        Imgproc.Sobel(blurred, gradY16, CvType.CV_16S, 0, 1, 3)
        blurred.release()

        val absGx = Mat()
        val absGy = Mat()
        Core.convertScaleAbs(gradX16, absGx)
        Core.convertScaleAbs(gradY16, absGy)
        gradX16.release()
        gradY16.release()

        val gyData = ByteArray(rows * cols)
        absGy.get(0, 0, gyData)
        val gxData = ByteArray(rows * cols)
        absGx.get(0, 0, gxData)
        absGx.release()
        absGy.release()

        // Build offset tables (same logic as Preprocessor.kt)
        val halfLen = KERNEL_LENGTH / 2
        val hOffsets = Array(NUM_ANGLES) { IntArray(KERNEL_LENGTH) }
        val vOffsets = Array(NUM_ANGLES) { IntArray(KERNEL_LENGTH) }
        var hMaxDy = 0; var hMaxDx = 0
        var vMaxDy = 0; var vMaxDx = 0

        for (a in 0 until NUM_ANGLES) {
            val angleRad = Math.toRadians(TILT_ANGLES_DEG[a])
            val cosA = cos(angleRad)
            val sinA = sin(angleRad)
            for (k in 0 until KERNEL_LENGTH) {
                val t = (k - halfLen).toDouble()
                val hDx = (t * cosA).roundToInt()
                val hDy = (t * sinA).roundToInt()
                hOffsets[a][k] = hDy * cols + hDx
                if (abs(hDx) > hMaxDx) hMaxDx = abs(hDx)
                if (abs(hDy) > hMaxDy) hMaxDy = abs(hDy)
                val vDx = (t * -sinA).roundToInt()
                val vDy = (t * cosA).roundToInt()
                vOffsets[a][k] = vDy * cols + vDx
                if (abs(vDx) > vMaxDx) vMaxDx = abs(vDx)
                if (abs(vDy) > vMaxDy) vMaxDy = abs(vDy)
            }
        }

        val marginY = maxOf(hMaxDy, vMaxDy)
        val marginX = maxOf(hMaxDx, vMaxDx)

        // Flatten for JNI
        val flatH = IntArray(NUM_ANGLES * KERNEL_LENGTH)
        val flatV = IntArray(NUM_ANGLES * KERNEL_LENGTH)
        for (a in 0 until NUM_ANGLES) {
            System.arraycopy(hOffsets[a], 0, flatH, a * KERNEL_LENGTH, KERNEL_LENGTH)
            System.arraycopy(vOffsets[a], 0, flatV, a * KERNEL_LENGTH, KERNEL_LENGTH)
        }

        return GradientData(gyData, gxData, rows, cols, flatH, flatV, marginY, marginX)
    }

    /**
     * Kotlin-only implementation of steps 4-6 for comparison.
     * Mirrors the logic in Preprocessor.directionalGradientKotlinFallback().
     */
    private fun runKotlinPath(data: GradientData): ByteArray {
        val totalPixels = data.rows * data.cols
        val hResponse = IntArray(totalPixels)
        val vResponse = IntArray(totalPixels)

        for (a in 0 until NUM_ANGLES) {
            val hOff = IntArray(KERNEL_LENGTH)
            val vOff = IntArray(KERNEL_LENGTH)
            System.arraycopy(data.flatH, a * KERNEL_LENGTH, hOff, 0, KERNEL_LENGTH)
            System.arraycopy(data.flatV, a * KERNEL_LENGTH, vOff, 0, KERNEL_LENGTH)

            for (y in data.marginY until data.rows - data.marginY) {
                val rowBase = y * data.cols
                for (x in data.marginX until data.cols - data.marginX) {
                    val baseIdx = rowBase + x
                    var sumH = 0
                    var sumV = 0
                    for (k in 0 until KERNEL_LENGTH) {
                        sumH += data.gyData[baseIdx + hOff[k]].toInt() and 0xFF
                        sumV += data.gxData[baseIdx + vOff[k]].toInt() and 0xFF
                    }
                    if (sumH > hResponse[baseIdx]) hResponse[baseIdx] = sumH
                    if (sumV > vResponse[baseIdx]) vResponse[baseIdx] = sumV
                }
            }
        }

        var globalMax = 1
        for (i in 0 until totalPixels) {
            val combined = maxOf(hResponse[i], vResponse[i])
            hResponse[i] = combined
            if (combined > globalMax) globalMax = combined
        }

        val resultData = ByteArray(totalPixels)
        val histogram = IntArray(256)
        for (i in 0 until totalPixels) {
            val normalized = (hResponse[i] * 255L / globalMax).toInt().coerceIn(0, 255)
            resultData[i] = normalized.toByte()
            histogram[normalized]++
        }

        val target = (totalPixels * 0.90).toInt()
        var cumSum = 0
        var thresholdVal = 255
        for (i in 0 until 256) {
            cumSum += histogram[i]
            if (cumSum >= target) {
                thresholdVal = i
                break
            }
        }

        for (i in 0 until totalPixels) {
            val v = resultData[i].toInt() and 0xFF
            resultData[i] = if (v > thresholdVal) 255.toByte() else 0
        }

        return resultData
    }

    /**
     * Differential correctness: native and Kotlin paths must produce byte-for-byte
     * identical output on the same input data.
     */
    @Test
    fun nativeMatchesKotlinFallback() {
        assertTrue("Native acceleration must be available for this test", NativeAccel.isAvailable)

        val images = listOf(
            "5unit" to SyntheticImageFactory.ultraLowContrast5Unit(),
            "3unit" to SyntheticImageFactory.ultraLowContrast3Unit(),
            "tilted" to SyntheticImageFactory.ultraLowContrastTilted8deg(),
            "whiteOnWhite" to Pair(SyntheticImageFactory.whiteOnWhite(), SyntheticImageFactory.defaultA4Corners())
        )

        for ((name, imageAndCorners) in images) {
            val (image, _) = imageAndCorners
            val data = prepareGradientData(image)
            image.release()

            // Run both paths
            val kotlinResult = runKotlinPath(data)

            val nativeResult = ByteArray(data.rows * data.cols)
            NativeAccel.nativeDirectionalGradient(
                data.gyData, data.gxData, nativeResult,
                data.rows, data.cols, data.flatH, data.flatV,
                NUM_ANGLES, KERNEL_LENGTH,
                data.marginY, data.marginX, 0.90f
            )

            // Compare byte-for-byte
            var mismatches = 0
            var maxDiff = 0
            for (i in kotlinResult.indices) {
                val kVal = kotlinResult[i].toInt() and 0xFF
                val nVal = nativeResult[i].toInt() and 0xFF
                val diff = abs(kVal - nVal)
                if (diff > 0) {
                    mismatches++
                    if (diff > maxDiff) maxDiff = diff
                }
            }

            val totalPixels = data.rows * data.cols
            val mismatchPct = mismatches * 100.0 / totalPixels
            Log.d(TAG, "$name: $mismatches/$totalPixels mismatches (%.2f%%), maxDiff=$maxDiff".format(mismatchPct))

            // Binary output should be identical (both are 0 or 255).
            // Allow <=0.1% mismatches for potential rounding differences at threshold boundary.
            assertTrue(
                "$name: too many mismatches: $mismatches/$totalPixels (${mismatchPct}%), maxDiff=$maxDiff",
                mismatchPct < 0.1
            )
        }
    }

    /**
     * Benchmark: measures native vs Kotlin execution time.
     * Runs 10 iterations each, reports median and speedup.
     */
    @Test
    fun benchmarkNativeVsKotlin() {
        assertTrue("Native acceleration must be available for this test", NativeAccel.isAvailable)

        val (image, _) = SyntheticImageFactory.ultraLowContrast5Unit()
        val data = prepareGradientData(image)
        image.release()

        val warmupIterations = 3
        val benchIterations = 10

        // Warmup
        repeat(warmupIterations) {
            val result = ByteArray(data.rows * data.cols)
            NativeAccel.nativeDirectionalGradient(
                data.gyData, data.gxData, result,
                data.rows, data.cols, data.flatH, data.flatV,
                NUM_ANGLES, KERNEL_LENGTH,
                data.marginY, data.marginX, 0.90f
            )
            runKotlinPath(data)
        }

        // Benchmark native
        val nativeTimes = mutableListOf<Double>()
        repeat(benchIterations) {
            val result = ByteArray(data.rows * data.cols)
            val start = System.nanoTime()
            NativeAccel.nativeDirectionalGradient(
                data.gyData, data.gxData, result,
                data.rows, data.cols, data.flatH, data.flatV,
                NUM_ANGLES, KERNEL_LENGTH,
                data.marginY, data.marginX, 0.90f
            )
            nativeTimes.add((System.nanoTime() - start) / 1_000_000.0)
        }

        // Benchmark Kotlin
        val kotlinTimes = mutableListOf<Double>()
        repeat(benchIterations) {
            val start = System.nanoTime()
            runKotlinPath(data)
            kotlinTimes.add((System.nanoTime() - start) / 1_000_000.0)
        }

        nativeTimes.sort()
        kotlinTimes.sort()

        val nativeMedian = nativeTimes[nativeTimes.size / 2]
        val kotlinMedian = kotlinTimes[kotlinTimes.size / 2]
        val speedup = kotlinMedian / nativeMedian

        Log.d(TAG, "=".repeat(60))
        Log.d(TAG, "Benchmark: steps 4-6 on ${data.rows}x${data.cols} image")
        Log.d(TAG, "  Native:  %.2fms median (range %.2f-%.2f)".format(
            nativeMedian, nativeTimes.first(), nativeTimes.last()))
        Log.d(TAG, "  Kotlin:  %.2fms median (range %.2f-%.2f)".format(
            kotlinMedian, kotlinTimes.first(), kotlinTimes.last()))
        Log.d(TAG, "  Speedup: %.1fx".format(speedup))
        Log.d(TAG, "=".repeat(60))

        // Native should be faster than Kotlin (at least 2x)
        assertTrue("Native should be at least 2x faster, got ${speedup}x", speedup >= 2.0)
    }
}
