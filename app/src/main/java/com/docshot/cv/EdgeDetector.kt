package com.docshot.cv

import android.util.Log
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.sqrt

private const val TAG = "DocShot:Edge"

// Cached structuring kernels — immutable, created once and reused across calls.
// Avoids per-frame allocation of these tiny Mats.
private val kernel3x3: Mat by lazy {
    Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
}
private val kernel5x5: Mat by lazy {
    Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
}

/**
 * Runs Canny edge detection with automatic or explicit threshold selection.
 * When [thresholdLow] and [thresholdHigh] are <= 0 (default), thresholds are
 * derived from the median intensity: low = 0.67 * median, high = 1.33 * median.
 * Explicit thresholds are useful for CLAHE-preprocessed images where the
 * auto-threshold formula overestimates edge gradients.
 * Caller must release the returned Mat.
 */
fun detectEdges(
    grayscale: Mat,
    thresholdLow: Double = -1.0,
    thresholdHigh: Double = -1.0
): Mat {
    val start = System.nanoTime()

    val median = computeMedian(grayscale)
    val low = if (thresholdLow > 0) thresholdLow else (0.67 * median).coerceIn(10.0, 200.0)
    val high = if (thresholdHigh > 0) thresholdHigh else (1.33 * median).coerceIn(30.0, 250.0)

    val edges = Mat()
    Imgproc.Canny(grayscale, edges, low, high)

    // Morphological close (dilate then erode) bridges gaps in document edges
    // without bloating them — dilate-only tends to thicken text edges too much
    Imgproc.morphologyEx(edges, edges, Imgproc.MORPH_CLOSE, kernel3x3)

    val ms = (System.nanoTime() - start) / 1_000_000.0
    Log.d(TAG, "detectEdges: %.1f ms (median=%.0f, low=%.0f, high=%.0f)".format(ms, median, low, high))
    return edges
}

/**
 * Runs Canny edge detection with a heavier morphological close (5x5 kernel).
 * The larger kernel bridges wider gaps caused by patterned/textured surfaces
 * where document edges get fragmented by background texture.
 * Caller must release the returned Mat.
 */
fun detectEdgesHeavyMorph(grayscale: Mat): Mat {
    val start = System.nanoTime()

    val median = computeMedian(grayscale)
    val low = (0.67 * median).coerceIn(10.0, 200.0)
    val high = (1.33 * median).coerceIn(30.0, 250.0)

    val edges = Mat()
    Imgproc.Canny(grayscale, edges, low, high)

    // 5x5 morph close: bridges wider gaps in edges caused by texture interference
    Imgproc.morphologyEx(edges, edges, Imgproc.MORPH_CLOSE, kernel5x5)

    val ms = (System.nanoTime() - start) / 1_000_000.0
    Log.d(TAG, "detectEdgesHeavyMorph: %.1f ms (median=%.0f, low=%.0f, high=%.0f)".format(ms, median, low, high))
    return edges
}

internal fun computeMedian(gray: Mat): Double {
    val mean = MatOfDouble()
    val stddev = MatOfDouble()
    Core.meanStdDev(gray, mean, stddev)
    // Use mean as a fast proxy for median — close enough for threshold selection
    val result = mean.get(0, 0)[0]
    mean.release()
    stddev.release()
    return result
}

// --- Spanning line suppression (WP-D) ---

/** Minimum line length as fraction of max(W, H) to qualify as "spanning". */
private const val MIN_SPAN_FRACTION = 0.70

/** HoughLinesP accumulator threshold — roughly 33% of a spanning line's pixels must be edges.
 *  150 is high enough to reject collinear noise patterns in noisy backgrounds
 *  while still catching real grout/seam lines (>80% edge density when continuous). */
private const val HOUGH_THRESHOLD = 150

/** HoughLinesP max gap — bridges 15px gaps within a single grout/seam line. */
private const val HOUGH_MAX_GAP = 15.0

/** Line-zeroing thickness — Canny grout edges are ~1-2px; 3px covers with margin.
 *  Narrower than the 5x5 morph-close bridge distance (4px) so junction gaps are healed. */
private const val MASKING_THICKNESS = 3

/** Both endpoints must be within this many pixels of the image border. */
private const val BORDER_MARGIN = 15.0

/**
 * Detects and suppresses image-spanning lines (tile grout, table seams) from an edge image.
 *
 * Surface features that run the full width or height of the frame create strong Canny edges.
 * Where these cross a document boundary, `findContours` fuses them into a composite contour
 * (8-12 vertices) instead of a clean 4-point quad. This function removes those spanning lines
 * and heals the resulting junction gaps.
 *
 * Algorithm:
 * 1. HoughLinesP detects line segments in [edges]
 * 2. Filter: keep only lines spanning >70% of max(W,H) whose BOTH endpoints are within
 *    15px of an image border (document corners are typically 50-100px inside the frame)
 * 3. Zero a 5px-thick strip along each qualifying line
 * 4. Morph-close (5x5 kernel) heals the 3-5px junction gaps left by zeroing
 *
 * Modifies [edges] in-place. All intermediate Mats are released.
 *
 * @param edges Single-channel binary edge image (modified in place).
 * @param imageSize Width and height of the image.
 * @return Number of suppressed lines (for debug logging).
 */
fun suppressSpanningLines(edges: Mat, imageSize: Size): Int {
    val w = imageSize.width
    val h = imageSize.height
    val maxDim = max(w, h)
    val minLength = MIN_SPAN_FRACTION * maxDim

    // HoughLinesP: minLineLength set to the spanning threshold so short segments are
    // never even returned — saves filtering work.
    val lines = Mat()
    try {
        Imgproc.HoughLinesP(
            edges,
            lines,
            1.0,                         // rho resolution: 1 pixel
            Math.PI / 180.0,             // theta resolution: 1 degree
            HOUGH_THRESHOLD,
            minLength,                   // minLineLength — pre-filters short segments
            HOUGH_MAX_GAP               // maxLineGap — bridges grout gaps
        )

        if (lines.rows() == 0) return 0

        var suppressedCount = 0
        for (i in 0 until lines.rows()) {
            val data = lines.get(i, 0) // [x1, y1, x2, y2]
            val x1 = data[0]
            val y1 = data[1]
            val x2 = data[2]
            val y2 = data[3]

            // Verify actual segment length (HoughLinesP minLineLength should handle this,
            // but double-check since the parameter semantics can vary by OpenCV build).
            val dx = x2 - x1
            val dy = y2 - y1
            val length = sqrt(dx * dx + dy * dy)
            if (length < minLength) continue

            // The line must span from one border to the OPPOSITE border:
            //   Horizontal-ish: one endpoint near left (x≤margin), other near right (x≥w-margin)
            //   Vertical-ish: one endpoint near top (y≤margin), other near bottom (y≥h-margin)
            // This prevents suppressing document edges that run ALONG a single border
            // (e.g., a partial document's top edge at y=3 from x=100 to x=797 — both
            // endpoints are "near a border" but on the SAME side, not opposite sides).
            if (!spansOppositeBorders(x1, y1, x2, y2, w, h)) continue

            // Zero out the line in the edge image
            Imgproc.line(
                edges,
                Point(x1, y1),
                Point(x2, y2),
                Scalar(0.0),
                MASKING_THICKNESS
            )
            suppressedCount++
        }

        // Morph-close to heal junction gaps where the suppressed line crossed document edges.
        // Uses the cached 5x5 kernel (already defined at module level).
        if (suppressedCount > 0) {
            Imgproc.morphologyEx(edges, edges, Imgproc.MORPH_CLOSE, kernel5x5)
        }

        return suppressedCount
    } finally {
        lines.release()
    }
}

/**
 * Returns true if the line from (x1,y1) to (x2,y2) spans from one image border
 * to the opposite border. Checks horizontal spanning (left↔right) and vertical
 * spanning (top↔bottom). Diagonal lines that cross opposite corners are also caught.
 *
 * This is stricter than "both endpoints near any border" — it rejects document edges
 * that run along a single border (e.g., partial document top edge at y=3).
 */
private fun spansOppositeBorders(
    x1: Double, y1: Double, x2: Double, y2: Double,
    w: Double, h: Double
): Boolean {
    val m = BORDER_MARGIN
    // Horizontal spanning: one endpoint near left border, other near right border
    val horizontalSpan = (x1 <= m && x2 >= w - m) || (x2 <= m && x1 >= w - m)
    // Vertical spanning: one endpoint near top border, other near bottom border
    val verticalSpan = (y1 <= m && y2 >= h - m) || (y2 <= m && y1 >= h - m)
    return horizontalSpan || verticalSpan
}
