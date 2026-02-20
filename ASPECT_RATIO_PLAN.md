# DocShot -- Aspect Ratio Recovery: Technical Design

> This document describes the approach for recovering the correct document aspect ratio
> from a detected quadrilateral. Read alongside [CLAUDE.md](CLAUDE.md) and [PROJECT.md](PROJECT.md).

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Edge-length ratio (`computeRawRatio`) | Implemented (Phase 11B) | No angular correction yet |
| Format snapping (6 formats) | Implemented (Phase 11B) | A4, US Letter, ID Card, Business Card, Receipt, Square |
| Homography error disambiguation | Implemented (Phase 11B) | Uses camera intrinsics when available |
| Camera intrinsics extraction | Implemented (Phase 11B) | LENS_INTRINSIC_CALIBRATION + focal-length fallback |
| Aspect ratio slider + lock | Implemented (UX Polish) | 0.25-1.0 range, 300ms debounce, DataStore persistence |
| Hartley normalization | **Implemented (v1.2.0 B1)** | Centroid + sqrt(2) scaling, returns 3x3 transform Mat |
| Perspective severity classifier | **Implemented (v1.2.0 B2)** | Max corner angle deviation from 90deg, thresholds 15/20deg |
| Angular correction (low-severity) | **Implemented (v1.2.0 B3)** | cos(alpha_v/2) / cos(alpha_h/2) correction factor |
| Projective VP estimation (high-severity) | **Implemented (v1.2.0 B4)** | K_inv * H decomposition, homogeneous coords throughout |
| Transition-zone blending | **Implemented (v1.2.0 B5)** | Linear blend 15-20deg, `estimateAspectRatio()` delegates to dual-regime |
| Multi-frame refinement | **Implemented (v1.2.0 B6)** | Zhang's method + intrinsics path, median aggregation, variance confidence |
| KLT corner tracking (`CornerTracker`) | **Implemented (v1.2.0 A1-A4)** | State machine, KLT flow, hybrid detect+track in FrameAnalyzer, lifecycle wired |
| Multi-frame integration | **Implemented (v1.2.0 B7)** | FrameAnalyzer accumulates KLT-tracked corners during stabilization |
| Capture pipeline wiring | **Implemented (v1.2.0 B8)** | Multi-frame estimate flows through CaptureResultData to ResultScreen |
| Gallery single-frame AR | **Implemented (v1.2.0 B9)** | Dual-regime estimation for gallery imports (no multi-frame) |
| Debug overlay | **Implemented (v1.2.0 C2)** | KLT state, severity, AR estimate, multi-frame count |
| Auto vs A4 setting | **Implemented (v1.2.0 C3)** | DataStore toggle, Settings UI |
| Format label + "(auto)" | **Implemented (v1.2.0 C1)** | ResultScreen shows auto-estimated indicator, settings wired |

---

## Problem Statement

When we detect a document quadrilateral and call `warpPerspective`, we must choose a
destination rectangle. The homography maps any quad to any rectangle -- but only one
rectangle has the correct aspect ratio matching the real-world document. A naive approach
(using detected edge lengths) works for near-frontal shots but fails under perspective.
A projective approach (vanishing points + intrinsics) works for skewed views but is
numerically unstable when the document appears nearly rectangular.

We need an approach that is accurate and numerically stable across the full range from
near-frontal to heavily skewed views.

## Key Insight: Two Regimes

| Regime | Vanishing Points | Edge-Length Ratio | Best Approach |
|--------|-----------------|-------------------|---------------|
| **Near-frontal** (quad ~ rectangle) | Unstable -- nearly parallel lines intersect at infinity | Accurate -- perspective distortion is small | Edge-length ratio + angular correction |
| **Heavily skewed** (visible trapezoid) | Stable -- lines converge visibly | Inaccurate -- foreshortening distorts lengths | Full projective method with intrinsics |

## Architecture Overview

Five components working together:

1. **Hartley normalization** -- always applied, stabilizes all downstream math
2. **Perspective severity classifier** -- determines which regime
3. **Regime-specific estimation** -- edge-length ratio OR projective reconstruction
4. **Multi-frame refinement** -- accumulated stabilization frames improve accuracy
5. **KLT corner tracking** -- sub-pixel consistent corners feed multi-frame refinement

### Component 1: Hartley Normalization

**Always apply before any projective computation.**

Translate corner coordinates so centroid is at origin, scale so average distance from
origin is sqrt(2). Standard conditioning technique (Hartley & Zisserman, ch4.4).

```kotlin
fun hartleyNormalize(corners: List<Point>): Pair<List<Point>, Mat> {
    val centroid = Point(corners.map { it.x }.average(), corners.map { it.y }.average())
    val centered = corners.map { Point(it.x - centroid.x, it.y - centroid.y) }
    val avgDist = centered.map { sqrt(it.x * it.x + it.y * it.y) }.average()
    val scale = sqrt(2.0) / avgDist
    // T = [[scale, 0, -scale*cx], [0, scale, -scale*cy], [0, 0, 1]]
    val normalized = centered.map { Point(it.x * scale, it.y * scale) }
    return Pair(normalized, T)
}
```

### Component 2: Perspective Severity Metric

**Metric:** Maximum corner angle deviation from 90deg.

```
severity = max(|angle_i - 90|) for i in {TL, TR, BR, BL}
```

| Severity | Threshold | Action |
|----------|-----------|--------|
| Low | < 15deg | Edge-length ratio + angular correction |
| High | > 20deg | Full projective reconstruction |
| Transition | 15-20deg | Weighted blend |

Thresholds are initial estimates -- tune on real test images.

### Component 3a: Low-Severity (Edge-Length + Angular Correction)

```
width_est  = (len(top) + len(bottom)) / 2
height_est = (len(left) + len(right)) / 2
raw_ratio  = width_est / height_est

alpha_h = angle between lines(TL->TR) and (BL->BR)  // horizontal convergence
alpha_v = angle between lines(TL->BL) and (TR->BR)  // vertical convergence

corrected_ratio = raw_ratio * cos(alpha_v / 2) / cos(alpha_h / 2)
```

Accurate to ~1% for severity < 15deg.

### Component 3b: High-Severity (Projective Reconstruction)

**Critical: work in homogeneous coordinates throughout. Never convert VPs to Euclidean.**

```
v_h = cross(line(TL, TR), line(BL, BR))  // horizontal vanishing point (homogeneous)
v_v = cross(line(TL, BL), line(TR, BR))  // vertical vanishing point (homogeneous)

// Image of the Absolute Conic from intrinsics: omega = (K K^T)^{-1}
// Orthogonality constraint: v_h^T * omega * v_v = 0
// Recover aspect ratio from homography satisfying the orthogonality constraint
```

**Camera intrinsics source (priority order):**
1. `LENS_INTRINSIC_CALIBRATION` (Camera2 API) -- already implemented
2. EXIF focal length + `SENSOR_INFO_PHYSICAL_SIZE` + `SENSOR_INFO_PIXEL_ARRAY_SIZE` -- already implemented
3. Fallback: principal point at center, focal = `max(w, h) * 0.9`

### Component 4: Multi-Frame Refinement

**Key differentiator.** During auto-capture stabilization (20 frames, ~667ms at 30fps),
natural hand tremor provides micro-viewpoint variation.

Each homography `H_i = K * [r1_i  r2_i  t_i]` must satisfy:
- `r1^T * r2 = 0` (orthogonality)
- `||r1|| = ||r2||` (equal norm)

With N frames, stack N constraint pairs into a least-squares system:

```
For each frame i:
    h1_i = H_i[:, 0]
    h2_i = H_i[:, 1]
    Constraint 1: h1_i^T * omega * h2_i = 0
    Constraint 2: h1_i^T * omega * h1_i = h2_i^T * omega * h2_i
```

**Confidence:** Residual of the solve. Small = all frames agree (high confidence).

```kotlin
class MultiFrameAspectEstimator {
    private val homographies = mutableListOf<Mat>()

    fun addFrame(corners: List<Point>) {
        val H = computeHomographyToCanonical(corners)
        homographies.add(H)
    }

    fun estimateAspectRatio(K: Mat): AspectRatioEstimate {
        // Stack constraints, solve least-squares, return ratio + confidence
    }

    fun reset() { homographies.forEach { it.release() }; homographies.clear() }
}
```

### Component 5: KLT Corner Tracking

**New for v1.2.0.** Replaces per-frame re-detection during stabilization with optical
flow tracking, providing sub-pixel consistent corners for multi-frame accumulation.

#### Why KLT?

Currently, `FrameAnalyzer` re-detects the full document quad from scratch every frame
(Canny + contours + polygon approximation). This introduces 1-2px jitter from contour
quantization. KLT tracking on 4 points provides ~0.03px inter-frame consistency.

| Method | Latency (4 pts) | Sub-pixel | Failure detection | Recommendation |
|--------|-----------------|-----------|-------------------|----------------|
| **KLT (pyramidal LK)** | **0.3-0.5ms** | **Yes** | **Yes (status+error)** | **Use this** |
| Template matching | 4-8ms | With extra work | Partial | No |
| ORB features | 2-5ms | No | Partial | No |
| Linear extrapolation | ~0ms | N/A | No | No |
| Phase correlation | 2-5ms | Yes (translation only) | No | No |

#### API: `Video.calcOpticalFlowPyrLK`

```kotlin
Video.calcOpticalFlowPyrLK(
    prevImg,            // 8-bit grayscale (previous frame)
    nextImg,            // 8-bit grayscale (current frame)
    prevPts,            // MatOfPoint2f (4 corners to track)
    nextPts,            // MatOfPoint2f (OUTPUT: tracked positions)
    status,             // MatOfByte (OUTPUT: 1=found, 0=lost per point)
    err,                // MatOfFloat (OUTPUT: error per point)
    Size(15.0, 15.0),   // search window
    2,                  // pyramid levels
    TermCriteria(COUNT + EPS, 20, 0.03)  // convergence
)
```

#### Hybrid Detection+Tracking Strategy

```
State: DETECT_ONLY (default)
  |-- High-confidence detection (>= 0.65) -->
State: TRACKING
  |-- KLT fails / correction diverges -->
State: DETECT_ONLY (reset)
```

During TRACKING:
- **Every frame:** Run KLT (0.5ms) to track 4 corners
- **Every 3rd frame:** Also run full detection as correction step
- If KLT-tracked corners diverge > 8px from re-detected corners: reset tracking
- Validated: tracked quad must remain convex with area > 100px^2

```
Frame-by-frame during stabilization:
  Frame 1: DETECT_ONLY -> detects quad (conf=0.72) -> TRACKING
  Frame 2: KLT only  (~0.5ms)
  Frame 3: KLT + detection correction -> compare, trust KLT if drift < 8px
  Frame 4: KLT only  (~0.5ms)
  Frame 5: KLT only  (~0.5ms)
  Frame 6: KLT + detection correction
  ...
  Frame 20: stable -> auto-capture fires
```

**CPU savings:** ~65% reduction during stabilization (from ~360ms to ~126ms total).
**Memory cost:** +300KB (one 640x480 grayscale Mat for previous frame).

#### Integration with QuadSmoother

Feed KLT-tracked corners into QuadSmoother as-is. The smoother's 5-frame averaging
and jump detection (10% of diagonal) still provide safety against scene changes.
Pass `isTracked: Boolean` through `FrameDetectionResult` so downstream consumers
(MultiFrameAspectEstimator) can distinguish tracked vs re-detected frames.

#### Corner Refinement Synergy

- **Preview pipeline (640px, 30fps):** Use KLT-tracked corners directly. No cornerSubPix.
  KLT already provides sub-pixel accuracy (0.03px convergence).
- **Capture pipeline (4032px, single frame):** Re-detect at full res + cornerSubPix (unchanged).
- **Homography accumulation:** Use KLT-tracked corners. Sub-pixel consistency -> cleaner
  homographies -> smaller least-squares residual.

---

## Format Snapping (Implemented)

| Format | Aspect Ratio | Snap Threshold |
|--------|-------------|----------------|
| A4 / A-series | 1 : 1.4142 (1/sqrt(2)) | +/-6% |
| US Letter | 1 : 1.2941 | +/-6% |
| ID Card | 1 : 1.586 | +/-6% |
| Business Card | 1 : 1.75 | +/-6% |
| Square | 1 : 1.0 | +/-6% |
| Receipt | 1 : ~3.0+ | +/-6% |

Confidence scoring: Gaussian falloff (`sigma = 0.04`) from matched format ratio.
When ambiguous (2+ formats within threshold), homography error disambiguates if intrinsics available.

## Testing Strategy

**Ground truth generation:**
- Print known-ratio documents (A4, Letter) on a flat surface
- Photograph at various angles: 0deg, 15deg, 30deg, 45deg, 60deg from vertical
- Measure recovered aspect ratio vs ground truth
- Acceptable error: < 2% for near-frontal, < 5% for heavy skew

**Numerical stability tests:**
- Near-rectangular quads (89deg-91deg corners) -- verify no NaN/Inf
- Heavily skewed quads -- verify projective method gives reasonable results
- Jittered corner sequences -- verify multi-frame averaging reduces variance

**KLT tracking tests:**
- Static corners (no motion) -- tracked positions identical
- Uniform translation -- tracked shift matches known amount
- Tracking failure (blank image) -- graceful reset to DETECT_ONLY
- Correction drift > threshold -- re-anchoring from detection
- Non-convex tracked quad -- rejection and reset

## References

- Hartley & Zisserman, *Multiple View Geometry in Computer Vision*, ch4.4 (normalization), ch8 (homography decomposition)
- Hartley, "In defense of the eight-point algorithm" (IEEE TPAMI 1997)
- Zhang, "A flexible new technique for camera calibration" (IEEE TPAMI 2000)
- Liebowitz & Zisserman, "Metric Rectification for Perspective Images of Planes" (CVPR 1998)
- Malis & Vargas, "Deeper understanding of the homography decomposition for vision-based control" (INRIA 2007)
- OpenCV calcOpticalFlowPyrLK: https://docs.opencv.org/4.x/dc/d6b/group__video__track.html
