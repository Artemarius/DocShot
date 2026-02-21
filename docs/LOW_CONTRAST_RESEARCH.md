# Ultra-Low-Contrast Detection — Research & Design

> Research conducted Feb 2026. Evaluates line-based methods to extend document detection below the current ~10-unit gradient floor to ~3 units.

## Problem Statement

v1.2.4's DOG strategy detects 6/6 synthetic white-on-white images at ~3ms. But real-world extreme cases (glossy paper on white desk under even lighting) can have gradients below 10 units — below the DOG+Canny(10/30) floor. The root cause is architectural: **Canny binarizes per-pixel before any geometric reasoning**. A 10-unit gradient at a single pixel is indistinguishable from noise, but accumulated along a 600-pixel straight line the SNR is ~49:1.

## Theoretical Foundation

From Ofir et al. (IEEE TPAMI 2020), the detection threshold for a straight edge of length L pixels with noise sigma is:

```
c_min ~ sigma * sqrt(log(N_search) / L)
```

For L=600, sigma=5 (typical sensor noise in bright scenes): c_min ~ 1-2 units. Our current pipeline fails at ~10 units — there is a 5-10x gap between the theoretical limit and what we achieve. The gap is entirely due to the Canny binarization bottleneck.

## Methods Evaluated

### Tier 1: Methods that work within the existing contour pipeline

| Method | Principle | Min gradient | Time (640x480) | Verdict |
|--------|-----------|-------------|-----------------|---------|
| **DIRECTIONAL_GRADIENT** | Smooth Sobel Gy/Gx along tilted 1D kernels (5 angles), accumulate coherent gradient, threshold → binary edges | ~5 units | ~3ms | **RECOMMENDED** |
| Perona-Malik diffusion | Iterative anisotropic smoothing, edge-preserving | ~8 units | ~20ms (8 iter) | Too slow, kappa gap too narrow |
| Oriented bilateral filter | Per-pixel varying elliptical kernel | ~5 units | ~300ms+ | Far too expensive |
| Coherence-Enhanced Diffusion | Weickert's tensor-driven diffusion | ~5 units | ~50ms | Too expensive |
| Higher-order steerable filters | 4th-order edge detection | ~7 units | ~15ms | Marginal gain over Sobel |

### Tier 2: Methods that bypass contours (line-based detection)

| Method | Principle | Min gradient | Time (640x480) | Verdict |
|--------|-----------|-------------|-----------------|---------|
| **LSD (Line Segment Detector)** | Gradient direction field + NFA validation | ~2.6 units (quant=1.0) | ~2-3ms | **RECOMMENDED** |
| **Restricted Radon / projection profiles** | Accumulate perpendicular gradient along tilted scan lines | ~3 units | ~1-4ms | **RECOMMENDED** (with LSD) |
| HoughLinesP on gradient image | Standard probabilistic Hough on thresholded gradient | ~8 units | ~4-5ms | Decent but not as good as LSD |
| Gradient-direction-aware Hough | Custom accumulator weighted by gradient direction | ~5 units | ~7-9ms | Good theory, too slow in Kotlin |
| Parametric line search (coarse-to-fine) | Exhaustive search + gradient agreement scoring | ~5 units | ~5-10ms | Works but complex |
| RANSAC line search | Random sample + gradient agreement | ~8 units | ~2-4ms | Fast but unreliable |
| EDLines/ELSED | Anchor-based line drawing | ~8 units | ~2-4ms | Needs JNI, anchor weakness for faint edges |
| Phase congruency | Contrast-invariant feature detection | ~1 unit | ~100ms+ | Perfect in theory, impractical |

## Selected Approach: Two Complementary Additions

### 1. `DIRECTIONAL_GRADIENT` PreprocessStrategy (~3ms)

**What:** 5-angle tilted 1D kernels smooth Sobel gradient components along candidate edge directions. Coherent edge gradients accumulate while noise cancels. Binary output feeds existing `findContours` pipeline.

**How it works:**
```
Sobel Gx, Gy
  → For each of 5 angles [-10, -5, 0, +5, +10] deg:
    → filter2D(|Gy|, tilted 21x1 kernel) → horizontal edge response
    → filter2D(|Gx|, tilted 1x21 kernel) → vertical edge response
  → Per-pixel max across 5 angles
  → max(H response, V response) → normalize → threshold at 90th percentile → morph close
```

**Tilt tolerance:** With 5-degree angle spacing, worst-case residual is 2.5 degrees, giving <0.2dB signal loss. Effective coverage: ±12.5 degrees.

**SNR improvement:** 21-pixel directional averaging reduces noise by sqrt(21) = 4.6x while preserving coherent edge signal. A 10-unit gradient goes from SNR ~2 (marginal) to SNR ~9 (strong).

**Mathematical equivalence:** This IS a local restricted Radon transform, but producing a 2D edge map (pipeline-friendly) rather than 1D projection profiles.

### 2. LSD + Restricted Radon Cascade (~3-10ms)

**What:** Three-tier cascade that bypasses Canny/contours entirely. Detects line segments (LSD), clusters them into document edges, assembles a rectangle, and uses corner consistency + Radon search to rescue partial detections.

**Tier 1 — LSD Fast Path (~2.5ms):**
- `Imgproc.createLineSegmentDetector(quant=1.0)` — detects gradients down to ~2.6 units
- Cluster segments by angle (8 deg tolerance) + perpendicular distance (15px)
- Form rectangle from best 2H + 2V edges, validate geometry

**Tier 2 — Corner-Constrained Radon (+2ms):**
- When LSD finds 2-3 edges, corner constraints determine where missing edges must be
- 3 known edges → missing edge is a 1D search (must intersect 2 known perpendicular edges)
- 2 perpendicular edges → constrained 2D search (known corner + aspect ratio bounds)
- 2 parallel edges → two independent 1D searches
- Restricted Radon: accumulate perpendicular gradient only (|Gy| for H, |Gx| for V), ±12 degrees, coarse-to-fine

**Tier 3 — Joint Radon Rectangle Fit (+4ms):**
- Full restricted Radon scan decomposed by shared theta (9 angles)
- Independent H/V peak search per angle
- Combination scoring with geometric priors (area, centering, aspect ratio)
- Last resort for gradient < 3 units

**Key innovations:**
- Corner consistency exploits geometric dependencies between the 4 sides
- Perpendicular-gradient-only Radon naturally suppresses text and texture
- Gradient density verification replaces edge density (no Canny edges to validate against)
- LSD's NFA criterion naturally rewards long lines with weak but consistent gradients

## Detection Floor Comparison

| Method | Min detectable gradient (600px line, noise sigma=5) |
|--------|--------------------------------------------------|
| Canny (auto-threshold) | ~30 units |
| DOG + Canny(10/30) | ~10 units |
| DIRECTIONAL_GRADIENT | ~5 units |
| LSD (quant=1.0) | ~2.6 units |
| Restricted Radon | ~3 units |
| Phase congruency | ~1 unit (theoretical, impractical) |

## Methods Explicitly Rejected

- **Oriented bilateral filter** (~300ms+): per-pixel varying kernel fundamentally incompatible with mobile
- **Coherence-Enhanced Diffusion** (~50ms): ~30 intermediate Mats per iteration, beautiful math but impractical
- **Custom Hough accumulator in Kotlin** (10-18ms): 5-10x slower than native OpenCV HoughLinesP
- **Perona-Malik diffusion** (~20ms): kappa must sit between noise (~5) and signal (~10), almost no gap
- **Higher-order steerable filters**: 1st-order is just Sobel (already have it), higher orders cost too much
- **Full Radon transform**: restricted version with geometric priors is 10x faster

## References

- Ofir, Galun, Alpert, Basri — "On Detection of Faint Edges in Noisy Images" (IEEE TPAMI 2020)
- Desolneux, Moisan, Morel — Helmholtz principle / NFA framework (foundation of LSD)
- LSD: `Imgproc.createLineSegmentDetector()` in OpenCV 4.12 (restored MIT license in 4.5.4+)
- Freeman & Adelson — "The Design and Use of Steerable Filters" (IEEE TPAMI 1991)
- Weickert — "Coherence-Enhancing Diffusion Filtering" (IJCV 1999)
