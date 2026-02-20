# DocShot — Phase History (Reference)

> Completed development phases, preserved for reference. See [PROJECT.md](../PROJECT.md) for current roadmap.

## Phase Summary

| Phase | Name | Status | Tests Added |
|-------|------|--------|-------------|
| 1 | Project Skeleton & Build System | Complete | — |
| 2 | Static Image Processing Pipeline | Complete | 10 unit |
| 3 | Real-Time Camera Detection | Complete | — |
| 4 | Capture & Rectify | Complete | — |
| 5 | Gallery Import | Complete | — |
| 6 | Auto-Capture & UX Polish | Complete | — |
| 7 | Robustness & Edge Cases | Complete | 32 unit + 21 instrumented = 53 |
| 8 | Performance Optimization | Complete | 35 unit + 21 instrumented = 56 |
| 9 | Flash Support + Capture Quality | Complete | — (count unchanged) |
| 10 | Lighting Gradient Correction | Complete | 36 unit + 25 instrumented = 61 |
| 11 | Capture UX & Auto-Capture Quality | Complete | 47 unit + 27 instrumented = 74 |
| 12 | Google Play Release | In Progress | — |
| v1.1.0 | UX Polish (flash, A4 default, AR lock) | Complete | — |
| v1.1.2 | No-Camera Fallback | Complete | — |
| v1.2.0 | KLT Tracking + Aspect Ratio Recovery | Complete | ~124 unit + 27 instrumented = ~151 |
| v1.2.1 | AR Estimation Fixes & Robustness | Complete | ~124 unit + 27 instrumented = ~151 |
| v1.2.2 | Manual Capture Path | Complete | ~124 unit + 27 instrumented = ~151 |

---

## Phase 1: Project Skeleton & Build System

**Goal:** Empty Android app that builds, installs, and displays a camera preview.

- Created Android project with Kotlin DSL Gradle build
- Configured AGP 8.x, min SDK 24, target SDK 34
- Integrated OpenCV Android SDK via CMake/ndk-build
- Set up CameraX with basic preview
- Camera permissions handling
- Verified OpenCV loads successfully on device

---

## Phase 2: Static Image Processing Pipeline

**Goal:** Process a hardcoded test image through the full detection -> rectification pipeline.

- Grayscale conversion + blur preprocessing
- Canny edge detection with auto-threshold
- Contour detection + polygon approximation
- Quadrilateral scoring (area, convexity, angle regularity)
- Corner ordering (TL, TR, BR, BL)
- getPerspectiveTransform + warpPerspective
- Input/rectified output side by side display
- Unit tests for corner ordering and quad scoring (10 tests)

---

## Phase 3: Real-Time Camera Detection

**Goal:** Live document boundary overlay on camera preview.

- CameraX ImageAnalysis with STRATEGY_KEEP_ONLY_LATEST
- FrameAnalyzer running detection per frame, downscaled to 640px
- Quadrilateral overlay on Compose Canvas with coordinate mapping
- Temporal smoothing (rolling average of last N detections)
- Per-frame latency: 5-15ms on S21
- Gaussian blur kernel 5x5 -> 9x9, morphological closing for gap bridging

**Known limitations (resolved in Phase 7):** Classical Canny pipeline needed sufficient contrast. Cluttered/low-contrast backgrounds caused detection failures.

---

## Phase 4: Capture & Rectify

**Goal:** Full shutter-to-saved-image pipeline.

- Tap shutter FAB -> full-res pipeline (YUV->BGR -> rotate -> detect -> sub-pixel refine -> rectify)
- Result screen with original/rectified toggle
- Save to gallery (MediaStore), share via FileProvider
- CameraX 3-use-case binding (Preview + ImageAnalysis + ImageCapture) with graceful fallback
- Full pipeline latency < 200ms

---

## Phase 5: Gallery Import

**Goal:** Process existing photos from the device gallery.

- System photo picker (PickVisualMedia, no permissions needed)
- Manual corner adjustment with draggable handles and magnifier loupe
- EXIF rotation handling, downscale to 4000px max
- Detection downscaled to ~1000px for reliable kernel matching
- Fixed capture crash on S21/Android 15: ImageCapture returns JPEG (1 plane), not YUV; handles both formats
- Tab 1 replaced from Pipeline Test to Import; gallery button on CameraScreen

---

## Phase 6: Auto-Capture & UX Polish

**Goal:** Zero-tap experience and polished UI.

- Auto-capture: stability detection in QuadSmoother, 15 consecutive frames with <2% corner drift
- Visual feedback: green->cyan quad with progressive fill opacity
- Haptic on capture (CONFIRM on API 30+, LONG_PRESS fallback)
- Auto-capture toggle FAB
- Post-processing filters on ResultScreen (B&W adaptive threshold, CLAHE contrast, gray-world white balance)
- Document orientation detection (Sobel gradient analysis + ink-density heuristic)
- Settings screen with DataStore (auto-capture, haptic, debug overlay, output format/quality, default filter)
- Custom DocShot brand color scheme (teal/cyan primary, dark theme)

---

## Phase 7: Robustness & Edge Cases

**Goal:** Handle real-world conditions reliably.

### Group A: Detection Confidence
- Per-detection confidence scoring (60% quad score + 20% area ratio + 20% edge density)
- Edge-density validation via QuadValidator
- Suppression threshold at 0.35

### Group B: UX Fallbacks & Multi-Candidate Handling
- Low-confidence fallback to manual corner adjustment (0.35-0.65)
- "Point at a document" hint
- Auto-capture gated to high-confidence (>=0.65)
- Multi-candidate score-margin penalty

### Group C: Multi-Strategy Preprocessing & Edge Cases
- 5 preprocessing strategies (STANDARD, CLAHE_ENHANCED, SATURATION_CHANNEL, BILATERAL, HEAVY_MORPH) with scene analysis and 25ms time budget
- Small document support (2% min area, aspect-ratio scoring)
- Partial document detection with "Move back" hint

### Group D: Testing
- 21 instrumented regression tests (SyntheticImageFactory with 7 corner presets + 10 image generators)
- 17 QuadScoringTest unit tests

---

## Phase 8: Performance Optimization

**Goal:** Optimize for speed and battery efficiency.

- ABI splits (arm64/armv7) + R8 minification + resource shrinking: arm64 release APK 26MB (from 59MB)
- MatPool utility for detection hot path, cached structuring kernels in EdgeDetector
- Shared grayscale conversion between analyzeScene and preprocessing
- Adaptive frame skipping (every other frame after 5 misses, 2-of-3 after 15 misses)
- Scene analysis caching across 10 frames
- Reduced intermediate Mats in PostProcessor, try/finally exception safety

---

## Phase 9: Flash Support + Capture Quality & UX Fixes

**Goal:** Torch/flash control for low-light scanning, capture pipeline reliability.

### Flash
- Torch toggle (CameraX camera.cameraControl.enableTorch), persisted in DataStore
- Torch off during capture but logical state preserved, re-enables on return to camera

### Capture Pipeline Rework
- Always re-detects on capture frame, validates against preview corners
- orderCorners fix for rotation-induced ordering
- Prefers preview corners when re-detection deviates >5%
- Trusts max(preview, redetect) confidence when quads agree

### Detection Tuning
- OrientationDetector: ambiguous vertical text defaults to CORRECT
- QuadSmoother: stableThreshold 8->10, average corner drift instead of max
- Split confidence thresholds: auto-capture at >=0.65, result routing at >=0.65

### UX
- ResultScreen: quad overlay on original view, Adjust/Rotate buttons, filter reset on data change
- CameraScreen: freeze overlay (dark scrim + frozen quad + status text) during Capturing/Processing

---

## Phase 10: Lighting Gradient Correction

**Goal:** Replace gray-world white balance with intelligent brightness gradient correction.

- Algorithm: LAB color space -> downsample L by 8x -> heavy Gaussian blur (51x51) -> upsample -> divide original L by estimated illumination -> rescale to original mean
- Filter renamed from "Color Fix" to "Even Light"
- Bugfixes: cornerSubPix bounds clamping, full-frame quad rejection in ContourFinder
- 4 new instrumented tests + 1 new unit test

---

## Phase 11: Capture UX & Auto-Capture Quality

**Goal:** Capture freeze overlay, auto-capture hardening, aspect ratio adjustment, AF lock.

### 11A: Capture Freeze Overlay
- Guard in frameAnalyzer callback prevents state updates during Capturing/Processing
- Scrim opacity 50% -> 75%

### 11A+: Auto-Capture Stability Hardening
- 1.5s warmup suppression after entering Idle
- stableThreshold 10->20 (~667ms at 30fps)
- Three-tier drift response (increment/halve/hard-reset)
- Pre-smoothing jump detection (>10% deviation clears buffer)
- Confidence formula rebalanced: `(0.6 * quadScore + 0.4 * edgeDensity) * marginFactor`
- Debug overlay (stable N/20, conf, warmup, AF state, READY)

### 11B: Aspect Ratio Adjustment
- Auto-detection via known-format snapping (A4, US Letter, ID Card, Business Card, Receipt, Square)
- Homography decomposition verification using camera intrinsics (Camera2 interop)
- Manual slider (0.25-1.0) with 300ms debounce re-warp
- Format label reactive to slider + dropdown for pre-selection
- Camera intrinsics extracted via LENS_INTRINSIC_CALIBRATION (API 28+) with focal-length fallback
- Rotation preserved across re-warps (autoRotationSteps + manualRotationSteps)

### 11C: Autofocus Lock During Auto-Capture
- At 50% stability (10/20 frames): one-shot center AF via FocusMeteringAction (FLAG_AF + disableAutoCancel)
- Auto-capture gated on AF lock confirmation
- AF cancelled on quad loss / enterIdle to restore continuous AF

### 11D: Gradient-Based Quad Tracking -- Deprioritized
Smoother hardening sufficient for v1. Superseded by v1.2.0 KLT corner tracking plan.

---

## UX Polish (v1.1.0)

- Flash persistence: torch off physically during capture, logical state preserved, re-enables via enterIdle()
- A4 default: ResultScreen initializes currentRatio to 0.707f instead of per-scan auto-detection
- Aspect ratio lock: DataStore keys (aspectRatioLocked, lockedAspectRatio), Lock/LockOpen icon, slider disabled when locked, persisted across captures and imports

---

## No-Camera Fallback (v1.1.2)

- Manifest camera feature marked optional (`required="false"`)
- Auto-selects Import tab when no camera detected
- Camera tab hidden, defensive guard in CameraPermissionScreen
- Enables Play Store screenshot capture on no-camera emulators

---

## v1.2.0: KLT Corner Tracking + Aspect Ratio Recovery

**Goal:** Accurate automatic aspect ratio estimation using dual-regime analysis and multi-frame refinement, powered by KLT optical flow corner tracking.

### WP-A: KLT Corner Tracking
- `CornerTracker`: pyramidal Lucas-Kanade optical flow state machine (DETECT_ONLY/TRACKING), 15x15 window, 2 pyramid levels, status+error validation, convexity guard
- Hybrid detect+track in `FrameAnalyzer`: KLT on most frames, full detection every 3rd frame during tracking, 8px correction drift threshold
- Wired through `QuadSmoother` with `isTracked` flag on `FrameDetectionResult`
- Lifecycle: reset on enterIdle, release on ViewModel clear
- ~65% CPU reduction during 20-frame stabilization window, +300KB memory

### WP-B: Multi-Frame Aspect Ratio Estimation
- Hartley normalization for numerical stability
- Perspective severity classifier (max corner angle deviation from 90deg)
- Dual-regime estimation: angular correction (<15deg) + projective decomposition (>20deg) + transition blending (15-20deg)
- `MultiFrameAspectEstimator`: accumulates homographies during stabilization, Zhang's method solve, median aggregation, variance-based confidence
- Integration: `FrameAnalyzer` accumulates KLT-tracked corners, estimate flows through `CaptureResultData` to `ResultScreen`
- Gallery imports use single-frame dual-regime estimation
- 75 new tests (severity classifier, angular correction, format snapping, multi-frame variance reduction)

### WP-C: Integration & Polish
- ResultScreen: three-tier initial ratio (locked > multi-frame estimate > A4 fallback), "(auto)" format label suffix
- Debug overlay: KLT tracking state, perspective severity, estimated AR, multi-frame count
- Settings: "Aspect ratio default" toggle (Auto estimated vs Always A4)
- Version bump to 1.2.0

### C4: On-Device Performance Validation (S21, Snapdragon 888)
- KLT-only frames: 2.1ms median, 4.8ms P95
- KLT + correction detection: 11.8ms median, 36.1ms P95
- 97.2% of tracked frames under 30ms budget
- No-doc detection: 30.4ms median (exhausts 4 strategies; non-critical path)
- Post-profiling fixes: `DetectionStatus.detectionMs` logging corrected (was 0.0 for no-doc frames), `MultiFrameAspectEstimator` result cached after first solve (eliminated ~2,400 redundant SVD solves per session)

### Post-v1.2.0 Hotfixes (On-Device Testing)
- **Auto-capture KLT confidence fix:** KLT-only frames injected `confidence=0.0` into smoother, keeping average at ~0.32 (below 0.65 threshold). Auto-capture could never fire during KLT tracking. Fix: carry forward last detection confidence for tracked frames.
- **Single-frame AR estimation:** Multi-frame Zhang's method degenerate during stabilization (stationary camera → identical homographies → garbage ratios 0.01-0.85). Switched to single-frame dual-regime estimation on full-res capture corners with camera intrinsics. Validated on 11×22cm letter: estimates within 2% of true ratio (0.500).

---

## Post-v1.2.0: Capture Preview Overlay

**Goal:** Show the actual document image inside the frozen quad overlay during capture, giving instant visual feedback of what was captured.

- `PreviewView.getBitmap()` snapshot captured at freeze start via `remember(isBusy)`
- Bitmap drawn clipped to quad path at 70% alpha, layered under existing cyan fill
- Layering (bottom to top): dark scrim (75%) → document texture (70% alpha) → cyan fill (15% alpha) → stroke + corner dots
- Single-file change (`CameraScreen.kt`): added `previewViewRef` state, `frozenPreviewBitmap` snapshot, `frozenPreviewBitmap` parameter on `QuadOverlay`, `clipPath`+`drawImage` in Canvas
- No ViewModel modifications, no new tests (pure UI composable change)
- Graceful fallback: if `getBitmap()` returns null (preview not ready), overlay renders as before (cyan fill only)

---

## v1.2.1: AR Estimation Fixes & Robustness

**Goal:** Fix broken projective decomposition and angular correction paths, make aspect ratio estimation robust at all document orientations and viewing angles.

**Fixes:**

- **Projective decomposition (critical):** Two bugs — (1) H direction was inverted (image→world instead of world→image), producing near-zero rotation column norms (~0.0003). Fixed: `getPerspectiveTransform(square → corners)` (world→image). (2) Camera intrinsics not scaled from sensor-native landscape coords (4032x3024 on S21) to capture frame coords (3024x4032 portrait). Fixed: `CameraIntrinsics.forCaptureFrame()` with rotation-aware fx/fy and cx/cy swapping. Added sanity checks: norm ratio in [0.2, 5.0], orthogonality dot product < 0.3.
- **Angular correction (orientation bias):** `rawRatio = min/max` lost track of which dimension is horizontal vs vertical, then applied `cos(alphaV/2)/cos(alphaH/2)` correction that assumed a fixed relationship. Fixed: correct each dimension independently (`trueH = sideH / cos(alphaV/2)`, `trueV = sideV / cos(alphaH/2)`), then take min/max. Now orientation-invariant.
- **Severity thresholds lowered:** 15/20 deg → 5/10 deg so projective decomposition kicks in earlier (most real-world captures are 10-25 deg).
- **Estimation confidence separated from snap confidence:** New `estimationConfidence` field based on severity regime (0.85 angular, 0.75 projective with intrinsics, 0.4 fallback). Previously used format-snap Gaussian distance as confidence, which dropped below threshold for valid custom ratios → A4 fallback.
- **Format list simplified:** Removed Business Card (0.571) and Receipt (0.600) — too close to common custom ratios, caused aggressive false snapping. Tightened SNAP_THRESHOLD from 0.06 to 0.035, SNAP_SIGMA from 0.04 to 0.025.
- **Raw ratio tracking:** Added `rawRatio` field to `AspectRatioEstimate` (pre-snap value) and `rawEstimatedRatio` to `CaptureResultData`. Debug label shows `[raw X | slider Y]`.
- **ResultScreen UI:** Split aspect ratio controls into two rows — row 1: lock button + format label (11sp); row 2: full-width slider. Previously slider was cramped.

**Validation:** Tested on S21 with 11x22cm envelope (true ratio 0.50). Projective path produces 0.497 at 18.4 deg severity. Robust across portrait, landscape, and angled orientations.

**Files changed:** `AspectRatioEstimator.kt`, `CameraScreen.kt`, `CameraViewModel.kt`, `GalleryViewModel.kt`, `ResultScreen.kt`, `AspectRatioEstimatorTest.kt`

---

## v1.2.2: Manual Capture Path

**Goal:** Always allow the user to take a photo, even when no document quad is detected. Never block the manual path.

**Problem:** When detection failed (low contrast, cluttered background, unusual document), `processCapture()` returned `null`, ViewModel showed "No document detected" error, and auto-reset after 2s. User was completely blocked.

**Changes:**

- **`CaptureProcessor`:** The null-return branch now returns a `CaptureResult` with the original bitmap and `corners = emptyList()`. `rectifiedBitmap` shares the same reference (no warp performed). The `rotatedMat` is still released by the existing `finally` block after `matToBitmap` creates a copy.
- **`CameraViewModel`:** New routing branch before existing confidence checks: when `result.corners.isEmpty()`, routes to `LowConfidence` state with `defaultCorners()` — a 10%-inset rectangle (TL, TR, BR, BL). No `rectifiedBitmap.recycle()` since it's the same reference as `originalBitmap`. Added `defaultCorners()` helper (same pattern as `GalleryViewModel`).
- **`CameraScreen`:** Hint text changed from "Point at a document" to "Tap to capture manually" when no detection and camera is idle.

**What already worked (no changes needed):**
- Shutter FAB always visible and clickable when Idle
- `CornerAdjustScreen` accepts any bitmap + 4 corners
- `acceptLowConfidenceCorners()` flow: sub-pixel refine → rectify → orientation → AR estimate → Result
- Auto-capture path unchanged (requires confidence >= 0.65 + stability + AF lock)
- Gallery no-detection path already used default corners

**Files changed:** `CaptureProcessor.kt`, `CameraViewModel.kt`, `CameraScreen.kt` (~25 lines)

---

## Dropped: ML-Enhanced Detection (originally planned Phase 9)

The classical CV pipeline with Phase 7 robustness improvements handles real-world documents reliably. ML segmentation would add complexity (TFLite dependency, model size, maintenance) without meaningful improvement. Revisitable if edge cases arise.

## Dropped: Blur Detection (originally planned v1.2)

Laplacian variance sharpness scoring. Deprioritized in favor of aspect ratio recovery and KLT tracking for v1.2.0.
