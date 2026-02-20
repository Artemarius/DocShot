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

## Dropped: ML-Enhanced Detection (originally planned Phase 9)

The classical CV pipeline with Phase 7 robustness improvements handles real-world documents reliably. ML segmentation would add complexity (TFLite dependency, model size, maintenance) without meaningful improvement. Revisitable if edge cases arise.

## Dropped: Blur Detection (originally planned v1.2)

Laplacian variance sharpness scoring. Deprioritized in favor of aspect ratio recovery and KLT tracking for v1.2.0.
