package com.docshot.cv

import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader

private const val TAG = "DocShot:PostProcTest"

/**
 * Instrumented tests for the [PostProcessFilter.COLOR_CORRECT] lighting gradient
 * correction filter. Uses synthetic Bitmaps with known illumination patterns
 * to verify correctness, mean preservation, passthrough, and performance.
 *
 * These are instrumented tests because [applyFilter] uses OpenCV + android.util.Log.
 */
@RunWith(AndroidJUnit4::class)
class PostProcessorTest {

    companion object {
        @JvmStatic
        @BeforeClass
        fun initOpenCV() {
            val success = OpenCVLoader.initLocal()
            check(success) { "Failed to initialize OpenCV" }
            Log.d(TAG, "OpenCV initialized successfully")
        }

        /** Creates a Bitmap with a left-to-right luminance gradient (dark left, bright right). */
        private fun createGradientBitmap(width: Int, height: Int, lMin: Int, lMax: Int): Bitmap {
            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            val pixels = IntArray(width * height)
            for (y in 0 until height) {
                for (x in 0 until width) {
                    val t = x.toFloat() / (width - 1).coerceAtLeast(1)
                    val gray = (lMin + t * (lMax - lMin)).toInt().coerceIn(0, 255)
                    pixels[y * width + x] = Color.rgb(gray, gray, gray)
                }
            }
            bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
            return bitmap
        }

        /** Creates a Bitmap with uniform brightness. */
        private fun createUniformBitmap(width: Int, height: Int, gray: Int): Bitmap {
            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            val color = Color.rgb(gray, gray, gray)
            val pixels = IntArray(width * height) { color }
            bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
            return bitmap
        }

        /** Measures the average brightness (R channel, since images are gray) of
         *  a vertical strip: columns [xStart, xEnd). */
        private fun avgBrightnessStrip(bitmap: Bitmap, xStart: Int, xEnd: Int): Double {
            var sum = 0L
            var count = 0
            val pixels = IntArray(bitmap.width)
            for (y in 0 until bitmap.height) {
                bitmap.getPixels(pixels, 0, bitmap.width, 0, y, bitmap.width, 1)
                for (x in xStart until xEnd) {
                    sum += Color.red(pixels[x])
                    count++
                }
            }
            return sum.toDouble() / count
        }

        /** Measures the overall mean brightness (R channel). */
        private fun meanBrightness(bitmap: Bitmap): Double {
            return avgBrightnessStrip(bitmap, 0, bitmap.width)
        }
    }

    /**
     * A left-to-right gradient (L=60→120) should be significantly flattened.
     * Verify that the left-right brightness difference is reduced by >75%.
     */
    @Test
    fun gradientReduction_reducesLeftRightDifference() {
        val source = createGradientBitmap(width = 800, height = 600, lMin = 60, lMax = 120)
        try {
            val stripW = source.width / 10  // 10% strips on each edge
            val leftBefore = avgBrightnessStrip(source, 0, stripW)
            val rightBefore = avgBrightnessStrip(source, source.width - stripW, source.width)
            val diffBefore = kotlin.math.abs(rightBefore - leftBefore)

            val result = applyFilter(source, PostProcessFilter.COLOR_CORRECT)
            try {
                val leftAfter = avgBrightnessStrip(result, 0, stripW)
                val rightAfter = avgBrightnessStrip(result, result.width - stripW, result.width)
                val diffAfter = kotlin.math.abs(rightAfter - leftAfter)

                val reduction = 1.0 - diffAfter / diffBefore
                Log.d(TAG, "Gradient reduction: before=%.1f after=%.1f reduction=%.1f%%".format(
                    diffBefore, diffAfter, reduction * 100
                ))
                assertTrue(
                    "Gradient should be reduced by >75%%, got %.1f%% (before=%.1f, after=%.1f)".format(
                        reduction * 100, diffBefore, diffAfter
                    ),
                    reduction > 0.75
                )
            } finally {
                result.recycle()
            }
        } finally {
            source.recycle()
        }
    }

    /**
     * Overall mean brightness should stay within 10% of the original.
     */
    @Test
    fun meanPreservation_staysWithin10Percent() {
        val source = createGradientBitmap(width = 800, height = 600, lMin = 80, lMax = 180)
        try {
            val meanBefore = meanBrightness(source)

            val result = applyFilter(source, PostProcessFilter.COLOR_CORRECT)
            try {
                val meanAfter = meanBrightness(result)
                val deviation = kotlin.math.abs(meanAfter - meanBefore) / meanBefore

                Log.d(TAG, "Mean preservation: before=%.1f after=%.1f deviation=%.1f%%".format(
                    meanBefore, meanAfter, deviation * 100
                ))
                assertTrue(
                    "Mean brightness should stay within 10%%, deviated %.1f%% (before=%.1f, after=%.1f)".format(
                        deviation * 100, meanBefore, meanAfter
                    ),
                    deviation < 0.10
                )
            } finally {
                result.recycle()
            }
        } finally {
            source.recycle()
        }
    }

    /**
     * A uniformly-lit image should not be degraded (mean stays within 5%,
     * no excessive per-pixel deviation).
     */
    @Test
    fun uniformPassthrough_doesNotDegrade() {
        val gray = 140
        val source = createUniformBitmap(width = 800, height = 600, gray = gray)
        try {
            val result = applyFilter(source, PostProcessFilter.COLOR_CORRECT)
            try {
                val meanAfter = meanBrightness(result)
                val deviation = kotlin.math.abs(meanAfter - gray.toDouble()) / gray

                Log.d(TAG, "Uniform passthrough: expected=%d actual=%.1f deviation=%.1f%%".format(
                    gray, meanAfter, deviation * 100
                ))
                assertTrue(
                    "Uniform image mean should stay within 5%%, deviated %.1f%%".format(deviation * 100),
                    deviation < 0.05
                )
            } finally {
                result.recycle()
            }
        } finally {
            source.recycle()
        }
    }

    /**
     * Filter should complete in <50ms on a 3000x4000 image.
     */
    @Test
    fun performance_under50msOn3000x4000() {
        val source = createGradientBitmap(width = 3000, height = 4000, lMin = 80, lMax = 180)
        try {
            // Warm-up run (JIT, OpenCV lazy init)
            applyFilter(source, PostProcessFilter.COLOR_CORRECT).recycle()

            val start = System.nanoTime()
            val result = applyFilter(source, PostProcessFilter.COLOR_CORRECT)
            val ms = (System.nanoTime() - start) / 1_000_000.0
            result.recycle()

            Log.d(TAG, "Performance: %.1f ms on 3000x4000".format(ms))
            assertTrue(
                "Filter should complete in <50ms on 3000x4000, took %.1f ms".format(ms),
                ms < 50.0
            )
        } finally {
            source.recycle()
        }
    }
}
