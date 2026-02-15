package com.docshot.cv

import android.util.Log
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc

private const val TAG = "DocShot:Orientation"

/**
 * Detected orientation of a rectified document image.
 */
enum class DocumentOrientation {
    CORRECT,       // No rotation needed
    ROTATE_90,     // Needs 90 degrees CW rotation
    ROTATE_180,    // Upside down
    ROTATE_270     // Needs 90 degrees CCW rotation (270 CW)
}

/**
 * Detects the orientation of an already-rectified document image and optionally
 * corrects it. Uses gradient analysis (Sobel) to determine text direction and
 * a top-heavy ink-density heuristic to distinguish right-side-up from upside-down.
 *
 * This is a best-effort heuristic tuned for typical document scans (white paper,
 * black text). For ambiguous cases it returns [DocumentOrientation.CORRECT].
 */

// Ratio threshold: when sumX/sumY (or vice versa) exceeds this, we consider
// the dominant direction to be significant. 1.4 means 40% more gradient energy
// in one axis than the other.
private const val GRADIENT_RATIO_THRESHOLD = 1.4

// Minimum absolute difference in mean intensity between top and bottom halves
// to make a top-heavy decision. Below this, the image is considered ambiguous.
private const val INK_DENSITY_MIN_DIFF = 3.0

/**
 * Detects the orientation of a rectified BGR document image.
 *
 * @param rectifiedMat The perspective-corrected BGR image.
 * @return The detected [DocumentOrientation].
 */
fun detectOrientation(rectifiedMat: Mat): DocumentOrientation {
    require(!rectifiedMat.empty()) { "Input Mat must not be empty" }
    val start = System.nanoTime()

    val rows = rectifiedMat.rows()
    val cols = rectifiedMat.cols()

    // --- Step 1: Convert to grayscale ---
    val gray = Mat()
    Imgproc.cvtColor(rectifiedMat, gray, Imgproc.COLOR_BGR2GRAY)

    // --- Step 2: Compute Sobel gradients ---
    val sobelX = Mat()
    val sobelY = Mat()
    // depth CV_32F avoids overflow; ksize=3 is the standard 3x3 Sobel kernel
    Imgproc.Sobel(gray, sobelX, CvType.CV_32F, /* dx = */ 1, /* dy = */ 0, /* ksize = */ 3)
    Imgproc.Sobel(gray, sobelY, CvType.CV_32F, /* dx = */ 0, /* dy = */ 1, /* ksize = */ 3)

    val absSobelX = Mat()
    val absSobelY = Mat()
    Core.absdiff(sobelX, Scalar(0.0), absSobelX)
    Core.absdiff(sobelY, Scalar(0.0), absSobelY)
    sobelX.release()
    sobelY.release()

    val sumX = Core.sumElems(absSobelX).`val`[0]
    val sumY = Core.sumElems(absSobelY).`val`[0]
    absSobelX.release()
    absSobelY.release()

    Log.d(TAG, "Gradient sums: sumX=%.0f  sumY=%.0f  ratio=%.2f".format(
        sumX, sumY, if (sumY > 0) sumX / sumY else Double.MAX_VALUE
    ))

    // --- Step 3: Determine dominant text direction ---
    // For horizontal text, vertical edges of characters produce strong X gradients,
    // and horizontal baselines produce Y gradients. In practice sumX > sumY for
    // horizontal text because character strokes have more vertical edges than
    // horizontal ones.
    val isHorizontalText = sumX > sumY * GRADIENT_RATIO_THRESHOLD
    val isVerticalText = sumY > sumX * GRADIENT_RATIO_THRESHOLD

    // --- Step 4: Top-heavy heuristic (ink density) ---
    // Split grayscale image into top and bottom halves; darker = more ink.
    val topHalf = gray.submat(0, rows / 2, 0, cols)
    val bottomHalf = gray.submat(rows / 2, rows, 0, cols)
    val topMean = Core.mean(topHalf).`val`[0]
    val bottomMean = Core.mean(bottomHalf).`val`[0]
    topHalf.release()
    bottomHalf.release()

    // Also check left vs right for sideways orientation disambiguation
    val leftHalf = gray.submat(0, rows, 0, cols / 2)
    val rightHalf = gray.submat(0, rows, cols / 2, cols)
    val leftMean = Core.mean(leftHalf).`val`[0]
    val rightMean = Core.mean(rightHalf).`val`[0]
    leftHalf.release()
    rightHalf.release()

    gray.release()

    Log.d(TAG, "Ink density: topMean=%.1f  bottomMean=%.1f  leftMean=%.1f  rightMean=%.1f".format(
        topMean, bottomMean, leftMean, rightMean
    ))

    // Lower mean intensity = more ink (darker). A document that starts with a
    // title/header at the top typically has slightly more ink (lower intensity)
    // in the top region, or at least comparable density.
    val topDarker = topMean < bottomMean - INK_DENSITY_MIN_DIFF
    val bottomDarker = bottomMean < topMean - INK_DENSITY_MIN_DIFF
    val leftDarker = leftMean < rightMean - INK_DENSITY_MIN_DIFF
    val rightDarker = rightMean < leftMean - INK_DENSITY_MIN_DIFF

    // --- Step 5: Decide orientation ---
    val orientation = when {
        isHorizontalText -> {
            // Text runs horizontally. Is it right-side-up or upside-down?
            if (bottomDarker) {
                // More ink at the bottom is unusual for a correctly oriented
                // document — likely upside-down.
                Log.d(TAG, "Horizontal text, bottom-heavy -> ROTATE_180")
                DocumentOrientation.ROTATE_180
            } else {
                // Top-heavy or ambiguous — assume correct
                Log.d(TAG, "Horizontal text, top-heavy or ambiguous -> CORRECT")
                DocumentOrientation.CORRECT
            }
        }
        isVerticalText -> {
            // Text runs vertically. Determine if 90 or 270 rotation is needed.
            // When text is sideways-right (needs 90 CW to fix), the "top" of
            // the text (where the title is) is on the left side of the image.
            if (leftDarker) {
                // More ink on the left = title is on the left = text reads top-to-bottom
                // from left side. Rotating 90 CW brings left to top.
                Log.d(TAG, "Vertical text, left-heavy -> ROTATE_90")
                DocumentOrientation.ROTATE_90
            } else if (rightDarker) {
                Log.d(TAG, "Vertical text, right-heavy -> ROTATE_270")
                DocumentOrientation.ROTATE_270
            } else {
                // Ambiguous sideways — default to CORRECT to avoid incorrect rotation
                Log.d(TAG, "Vertical text, ambiguous -> CORRECT")
                DocumentOrientation.CORRECT
            }
        }
        else -> {
            // Gradient ratio is inconclusive — image might not be a text document,
            // or it's a mix. Fall back to CORRECT to avoid making things worse.
            Log.d(TAG, "Ambiguous gradient ratio -> CORRECT")
            DocumentOrientation.CORRECT
        }
    }

    val ms = (System.nanoTime() - start) / 1_000_000.0
    Log.d(TAG, "detectOrientation: %.1f ms -> %s".format(ms, orientation.name))
    return orientation
}

/**
 * Rotates the rectified image to correct the detected orientation.
 *
 * If [orientation] is [DocumentOrientation.CORRECT], returns the input Mat directly
 * (no copy). Caller must handle the lifecycle: if the returned Mat differs from the
 * input, the caller owns and must release it.
 *
 * @param rectifiedMat The perspective-corrected BGR image.
 * @param orientation The detected orientation to correct.
 * @return A (possibly new) Mat in the corrected orientation.
 */
fun correctOrientation(rectifiedMat: Mat, orientation: DocumentOrientation): Mat {
    require(!rectifiedMat.empty()) { "Input Mat must not be empty" }

    if (orientation == DocumentOrientation.CORRECT) {
        return rectifiedMat
    }

    val start = System.nanoTime()
    val rotateCode = when (orientation) {
        DocumentOrientation.ROTATE_90 -> Core.ROTATE_90_CLOCKWISE
        DocumentOrientation.ROTATE_180 -> Core.ROTATE_180
        DocumentOrientation.ROTATE_270 -> Core.ROTATE_90_COUNTERCLOCKWISE
        DocumentOrientation.CORRECT -> error("Unreachable") // Already handled above
    }

    val rotated = Mat()
    Core.rotate(rectifiedMat, rotated, rotateCode)

    val ms = (System.nanoTime() - start) / 1_000_000.0
    Log.d(TAG, "correctOrientation: %.1f ms (%s -> %dx%d)".format(
        ms, orientation.name, rotated.cols(), rotated.rows()
    ))
    return rotated
}

/**
 * Convenience: detects and corrects orientation in a single call.
 *
 * @param rectifiedMat The perspective-corrected BGR image.
 * @return A [Pair] of the (possibly rotated) Mat and the detected orientation.
 *         If orientation is [DocumentOrientation.CORRECT], the returned Mat IS the
 *         input Mat (no copy). Otherwise, it is a new Mat that the caller must release.
 */
fun detectAndCorrect(rectifiedMat: Mat): Pair<Mat, DocumentOrientation> {
    val orientation = detectOrientation(rectifiedMat)
    val corrected = correctOrientation(rectifiedMat, orientation)
    return corrected to orientation
}
