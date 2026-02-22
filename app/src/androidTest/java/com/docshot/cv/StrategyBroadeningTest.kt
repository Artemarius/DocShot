package com.docshot.cv

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader

private const val TAG = "DocShot:StrategyBroadening"

/**
 * Tests for WP-C strategy broadening: low-contrast non-white scenes
 * (mean 120-180, stddev < 40) now get DOG and GRADIENT_MAGNITUDE strategies
 * in addition to the standard ones.
 *
 * Split into two groups:
 * - Strategy list assertion tests: verify [analyzeScene] returns the correct
 *   strategy lists for different scene types.
 * - Integration detection tests: verify the full pipeline detects documents
 *   in the new low-contrast non-white synthetic images.
 */
@RunWith(AndroidJUnit4::class)
class StrategyBroadeningTest {

    companion object {
        @JvmStatic
        @BeforeClass
        fun initOpenCV() {
            val success = OpenCVLoader.initLocal()
            check(success) { "Failed to initialize OpenCV" }
            Log.d(TAG, "OpenCV initialized successfully")
        }
    }

    // ----------------------------------------------------------------
    // Strategy list assertion tests
    // ----------------------------------------------------------------

    /**
     * Low-contrast non-white scene (gray surface, mean ~175, stddev ~20) should get
     * DOG and GRADIENT_MAGNITUDE via the isLowContrast path.
     */
    @Test
    fun lowContrastNonWhite_getsAdvancedStrategies() {
        // docOnGraySurfaceLowContrast: doc gray 200 on surface gray 160 → mean ~175, stddev ~20
        // Solidly in isLowContrast (stddev < 40) and NOT isWhiteOnWhite (mean < 180).
        val (image, _) = SyntheticImageFactory.docOnGraySurfaceLowContrast()
        try {
            invalidateSceneCache()
            val start = System.nanoTime()
            val scene = analyzeScene(image, useCache = false)
            val ms = (System.nanoTime() - start) / 1_000_000.0
            Log.d(TAG, "analyzeScene(docOnGraySurfaceLowContrast): %.1f ms, mean=%.0f, stddev=%.0f, strategies=%s".format(
                ms, scene.meanIntensity, scene.stddevIntensity, scene.strategies))

            scene.grayMat?.release()

            // Key assertion: DOG and GRADIENT_MAGNITUDE must be present (isLowContrast path)
            assertTrue(
                "Low-contrast non-white scene should include DOG strategy, got: ${scene.strategies}",
                scene.strategies.contains(PreprocessStrategy.DOG)
            )
            assertTrue(
                "Low-contrast non-white scene should include GRADIENT_MAGNITUDE strategy, got: ${scene.strategies}",
                scene.strategies.contains(PreprocessStrategy.GRADIENT_MAGNITUDE)
            )
            // Standard strategies should also be present
            assertTrue(
                "Low-contrast non-white scene should include STANDARD strategy, got: ${scene.strategies}",
                scene.strategies.contains(PreprocessStrategy.STANDARD)
            )
            assertTrue(
                "Low-contrast non-white scene should include CLAHE_ENHANCED strategy, got: ${scene.strategies}",
                scene.strategies.contains(PreprocessStrategy.CLAHE_ENHANCED)
            )
        } finally {
            image.release()
        }
    }

    /**
     * High-contrast grout scene (tan surface, stddev > 40 due to grout+doc range)
     * gets standard strategies only — NOT the isLowContrast extras.
     * This is correct: the grout scene relies on line suppression (WP-D),
     * not strategy broadening (WP-C).
     */
    @Test
    fun highContrastGroutScene_getsStandardStrategies() {
        // docOnTanSurfaceWithGroutLines: doc 230, surface ~148, grout 120
        // Range 120-230 → stddev > 40 → neither isLowContrast nor isWhiteOnWhite.
        val (image, _) = SyntheticImageFactory.docOnTanSurfaceWithGroutLines()
        try {
            invalidateSceneCache()
            val start = System.nanoTime()
            val scene = analyzeScene(image, useCache = false)
            val ms = (System.nanoTime() - start) / 1_000_000.0
            Log.d(TAG, "analyzeScene(docOnTanSurfaceWithGroutLines): %.1f ms, mean=%.0f, stddev=%.0f, strategies=%s".format(
                ms, scene.meanIntensity, scene.stddevIntensity, scene.strategies))

            scene.grayMat?.release()

            // High stddev → standard pipeline (grout detection relies on WP-D line suppression)
            assertTrue(
                "Grout scene (stddev > 40) should include STANDARD, got: ${scene.strategies}",
                scene.strategies.contains(PreprocessStrategy.STANDARD)
            )
            assertTrue(
                "Grout scene (stddev > 40) should include CLAHE_ENHANCED, got: ${scene.strategies}",
                scene.strategies.contains(PreprocessStrategy.CLAHE_ENHANCED)
            )
            // Should NOT get low-contrast extras
            assertTrue(
                "Grout scene (stddev > 40) should NOT include DOG, got: ${scene.strategies}",
                !scene.strategies.contains(PreprocessStrategy.DOG)
            )
            assertTrue(
                "Grout scene (stddev > 40) should NOT include GRADIENT_MAGNITUDE, got: ${scene.strategies}",
                !scene.strategies.contains(PreprocessStrategy.GRADIENT_MAGNITUDE)
            )
        } finally {
            image.release()
        }
    }

    /**
     * White-on-white scenes should still get the specialized white-on-white
     * strategy list (DOG first, DIRECTIONAL_GRADIENT, etc.) -- regression guard.
     */
    @Test
    fun whiteOnWhite_stillGetsWhiteOnWhiteStrategies() {
        val image = SyntheticImageFactory.whiteOnWhite()
        try {
            invalidateSceneCache()
            val start = System.nanoTime()
            val scene = analyzeScene(image, useCache = false)
            val ms = (System.nanoTime() - start) / 1_000_000.0
            Log.d(TAG, "analyzeScene(whiteOnWhite): %.1f ms, mean=%.0f, stddev=%.0f, isWoW=%s, strategies=%s".format(
                ms, scene.meanIntensity, scene.stddevIntensity, scene.isWhiteOnWhite, scene.strategies))

            scene.grayMat?.release()

            // Must be classified as white-on-white
            assertTrue(
                "whiteOnWhite scene should be classified as isWhiteOnWhite=true",
                scene.isWhiteOnWhite
            )

            // White-on-white strategy list should start with DOG
            assertTrue(
                "White-on-white strategies should start with DOG, got: ${scene.strategies}",
                scene.strategies.isNotEmpty() && scene.strategies[0] == PreprocessStrategy.DOG
            )
            // Should include DIRECTIONAL_GRADIENT (position #2)
            assertTrue(
                "White-on-white strategies should include DIRECTIONAL_GRADIENT, got: ${scene.strategies}",
                scene.strategies.contains(PreprocessStrategy.DIRECTIONAL_GRADIENT)
            )
            // Should include other white-on-white strategies
            assertTrue(
                "White-on-white strategies should include GRADIENT_MAGNITUDE, got: ${scene.strategies}",
                scene.strategies.contains(PreprocessStrategy.GRADIENT_MAGNITUDE)
            )
            assertTrue(
                "White-on-white strategies should include LAB_CLAHE, got: ${scene.strategies}",
                scene.strategies.contains(PreprocessStrategy.LAB_CLAHE)
            )
            assertTrue(
                "White-on-white strategies should include CLAHE_ENHANCED, got: ${scene.strategies}",
                scene.strategies.contains(PreprocessStrategy.CLAHE_ENHANCED)
            )
            assertTrue(
                "White-on-white strategies should include MULTICHANNEL_FUSION, got: ${scene.strategies}",
                scene.strategies.contains(PreprocessStrategy.MULTICHANNEL_FUSION)
            )
            assertTrue(
                "White-on-white strategies should include ADAPTIVE_THRESHOLD, got: ${scene.strategies}",
                scene.strategies.contains(PreprocessStrategy.ADAPTIVE_THRESHOLD)
            )
            // Should NOT include standard-pipeline-only strategies
            assertTrue(
                "White-on-white strategies should NOT include STANDARD, got: ${scene.strategies}",
                !scene.strategies.contains(PreprocessStrategy.STANDARD)
            )
        } finally {
            image.release()
        }
    }

    /**
     * Normal-contrast scene (stddev >= 40) should get standard strategies
     * without DOG or GRADIENT_MAGNITUDE being added by the low-contrast path.
     * Note: DOG/GRADIENT_MAGNITUDE should only appear for isLowContrast (stddev < 40)
     * or isWhiteOnWhite scenes.
     */
    @Test
    fun normalContrast_getsStandardStrategies() {
        // High-contrast scene: white doc on dark background -> stddev well above 40
        val image = SyntheticImageFactory.whiteDocOnSolidBg()
        try {
            invalidateSceneCache()
            val start = System.nanoTime()
            val scene = analyzeScene(image, useCache = false)
            val ms = (System.nanoTime() - start) / 1_000_000.0
            Log.d(TAG, "analyzeScene(whiteDocOnSolidBg): %.1f ms, mean=%.0f, stddev=%.0f, isWoW=%s, strategies=%s".format(
                ms, scene.meanIntensity, scene.stddevIntensity, scene.isWhiteOnWhite, scene.strategies))

            scene.grayMat?.release()

            // Should NOT be white-on-white
            assertTrue(
                "Normal contrast scene should NOT be isWhiteOnWhite",
                !scene.isWhiteOnWhite
            )
            // Should include STANDARD
            assertTrue(
                "Normal contrast scene should include STANDARD, got: ${scene.strategies}",
                scene.strategies.contains(PreprocessStrategy.STANDARD)
            )
            // DOG and GRADIENT_MAGNITUDE should NOT be in the list for normal contrast
            // (they are added only for isLowContrast or isWhiteOnWhite)
            assertTrue(
                "Normal contrast (stddev >= 40) should NOT include DOG, got: ${scene.strategies}",
                !scene.strategies.contains(PreprocessStrategy.DOG)
            )
            assertTrue(
                "Normal contrast (stddev >= 40) should NOT include GRADIENT_MAGNITUDE, got: ${scene.strategies}",
                !scene.strategies.contains(PreprocessStrategy.GRADIENT_MAGNITUDE)
            )
        } finally {
            image.release()
        }
    }

    // ----------------------------------------------------------------
    // Integration detection tests
    // ----------------------------------------------------------------

    /**
     * Document on beige surface (~50 unit gradient) should be detected
     * with high confidence via the broadened strategy list.
     */
    @Test
    fun docOnBeigeSurface_detected() {
        val (image, _) = SyntheticImageFactory.docOnBeigeSurface()
        try {
            invalidateSceneCache()
            val start = System.nanoTime()
            val status = detectDocumentWithStatus(image)
            val ms = (System.nanoTime() - start) / 1_000_000.0
            Log.d(TAG, "detectDocumentWithStatus(docOnBeigeSurface): %.1f ms, " +
                "detected=${status.result != null}, confidence=${status.result?.confidence ?: "N/A"}".format(ms))

            assertNotNull(
                "docOnBeigeSurface should be detected (low-contrast non-white scene)",
                status.result
            )
            assertTrue(
                "docOnBeigeSurface confidence should be >= 0.65, got ${status.result!!.confidence}",
                status.result.confidence >= 0.65
            )
        } finally {
            image.release()
        }
    }

    /**
     * Document on gray surface with low contrast (~40 unit gradient, achromatic)
     * should be detected with high confidence.
     */
    @Test
    fun docOnGraySurfaceLowContrast_detected() {
        val (image, _) = SyntheticImageFactory.docOnGraySurfaceLowContrast()
        try {
            invalidateSceneCache()
            val start = System.nanoTime()
            val status = detectDocumentWithStatus(image)
            val ms = (System.nanoTime() - start) / 1_000_000.0
            Log.d(TAG, "detectDocumentWithStatus(docOnGraySurfaceLowContrast): %.1f ms, " +
                "detected=${status.result != null}, confidence=${status.result?.confidence ?: "N/A"}".format(ms))

            assertNotNull(
                "docOnGraySurfaceLowContrast should be detected (gray low-contrast scene)",
                status.result
            )
            assertTrue(
                "docOnGraySurfaceLowContrast confidence should be >= 0.65, got ${status.result!!.confidence}",
                status.result.confidence >= 0.65
            )
        } finally {
            image.release()
        }
    }

    /**
     * Document on tan surface with grout lines -- the key regression test.
     * Grout lines fragment contours, so detection may be harder.
     * Minimum bar: detected at any confidence >= 0.35 (manual corner adjustment).
     * Ideal: confidence >= 0.65 (auto-capture eligible).
     */
    @Test
    fun docOnTanSurfaceWithGroutLines_detected() {
        val (image, _) = SyntheticImageFactory.docOnTanSurfaceWithGroutLines()
        try {
            invalidateSceneCache()
            val start = System.nanoTime()
            val status = detectDocumentWithStatus(image)
            val ms = (System.nanoTime() - start) / 1_000_000.0
            Log.d(TAG, "detectDocumentWithStatus(docOnTanSurfaceWithGroutLines): %.1f ms, " +
                "detected=${status.result != null}, confidence=${status.result?.confidence ?: "N/A"}".format(ms))

            assertNotNull(
                "docOnTanSurfaceWithGroutLines should be detected (grout lines should not prevent detection)",
                status.result
            )
            assertTrue(
                "docOnTanSurfaceWithGroutLines confidence should be >= 0.35, got ${status.result!!.confidence}",
                status.result.confidence >= 0.35
            )

            // Log whether it reaches auto-capture threshold
            if (status.result.confidence >= 0.65) {
                Log.d(TAG, "docOnTanSurfaceWithGroutLines: auto-capture eligible (confidence >= 0.65)")
            } else {
                Log.d(TAG, "docOnTanSurfaceWithGroutLines: manual adjustment (confidence < 0.65)")
            }
        } finally {
            image.release()
        }
    }

    /**
     * Document on tile floor (checkerboard tiles with dense grout grid).
     * Tests detection on a patterned low-contrast surface.
     */
    @Test
    fun docOnTileFloor_detected() {
        val (image, _) = SyntheticImageFactory.docOnTileFloor()
        try {
            invalidateSceneCache()
            val start = System.nanoTime()
            val status = detectDocumentWithStatus(image)
            val ms = (System.nanoTime() - start) / 1_000_000.0
            Log.d(TAG, "detectDocumentWithStatus(docOnTileFloor): %.1f ms, " +
                "detected=${status.result != null}, confidence=${status.result?.confidence ?: "N/A"}".format(ms))

            assertNotNull(
                "docOnTileFloor should be detected (tile pattern should not prevent detection)",
                status.result
            )

            // Log confidence level for diagnostics
            Log.d(TAG, "docOnTileFloor confidence: ${status.result!!.confidence}")
        } finally {
            image.release()
        }
    }
}
