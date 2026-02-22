# DocShot — CLAUDE.md

## Project Overview
Android document rectification app. Camera input -> automatic document boundary detection -> perspective correction -> clean rectangular output. Zero user interaction beyond shutter tap.

## Tech Stack
- **Language:** Kotlin (idiomatic, no Java fallbacks)
- **UI:** Jetpack Compose + Material 3
- **Camera:** CameraX with Camera2 backend
- **CV:** OpenCV Android SDK 4.10+
- **Build:** Gradle Kotlin DSL, AGP 8.x
- **Min SDK:** 24 / Target SDK: 34
- **Test device:** Samsung Galaxy S21 (Snapdragon 888)

## Architecture Rules
- MVVM with ViewModels for camera and processing state
- CV pipeline runs entirely off the main thread (coroutines + Dispatchers.Default)
- OpenCV Mat objects must be explicitly released -- no relying on GC
- CameraX ImageAnalysis runs at STRATEGY_KEEP_ONLY_LATEST to avoid frame queue buildup
- Separate detection (runs every frame, must be <30ms) from rectification (runs once on capture, can take up to 200ms)
- All OpenCV calls go through Kotlin wrapper functions in the `cv/` package -- no raw OpenCV calls in UI or camera code

## Build & Run
```bash
./gradlew assembleDebug
./gradlew installDebug
./gradlew testDebugUnitTest
```

## Code Style
- Kotlin conventions: named parameters for functions with >2 params, data classes for value types
- No wildcard imports
- Use `require()` / `check()` for preconditions, not if-throw
- Document non-obvious OpenCV parameter choices with comments (e.g., why a specific Canny threshold ratio)
- Prefer `FloatArray` / `DoubleArray` over `List<Float>` in hot paths
- Log processing times with `android.util.Log.d` in debug builds for performance tracking

## CV Pipeline Specifics
- Canny thresholds: auto-compute from median intensity (0.67*median, 1.33*median)
- Contour filtering: reject contours < 10% of image area, require 4-point polygon approximation with epsilon = 2-5% of arc length
- Quadrilateral scoring: `(0.6 * quadScore + 0.4 * edgeDensity) * marginFactor`
- Corner ordering: consistent TL, TR, BR, BL based on sum/difference of coordinates
- Output dimensions: derive from longest edge pairs to preserve document aspect ratio
- Always use `Imgproc.INTER_CUBIC` for the final warp, `INTER_LINEAR` for preview
- 11 preprocessing strategies: 5 original (STANDARD, CLAHE_ENHANCED, SATURATION_CHANNEL, BILATERAL, HEAVY_MORPH) + 6 low-contrast (ADAPTIVE_THRESHOLD, LAB_CLAHE, GRADIENT_MAGNITUDE, DOG, MULTICHANNEL_FUSION, DIRECTIONAL_GRADIENT) with scene analysis, white-on-white detection, and 25ms time budget + LSD+Radon cascade fallback

## File Structure
```
app/src/main/java/com/docshot/
├── ui/          # Compose screens + navigation
├── camera/      # CameraX setup, frame analysis, QuadSmoother, CornerTracker
├── cv/          # Document detection, rectification, post-processing, LSD+Radon detector
└── util/        # Permissions, image I/O, gallery save, DataStore prefs
```

## Current State (v1.2.5-dev)
- **Phases 1-11 complete.** Full classical CV pipeline, auto-capture with AF lock, aspect ratio slider with format snapping, flash, gallery import, post-processing filters (B&W, Contrast, Even Light). See [docs/PHASE_HISTORY.md](docs/PHASE_HISTORY.md) for detailed phase-by-phase history.
- **Phase 12 (Play Store release) in progress.** App submitted to testers (14-day testing period). App icon, splash screen, signing, privacy policy, store listing done.
- **v1.2.4 complete.** Low-contrast / white-on-white detection: 5 new preprocessing strategies (ADAPTIVE_THRESHOLD, LAB_CLAHE, GRADIENT_MAGNITUDE, DOG, MULTICHANNEL_FUSION) with scene-aware strategy selection. White-on-white scenes (mean > 180, stddev < 35) use specialized strategies that handle 5-35 unit gradients where auto-Canny saturates. Binary-output strategies bypass Canny entirely. Zero performance regression for non-white-on-white scenes.
- **Capture preview overlay:** During capture freeze, quad overlay fills with the actual preview frame (70% alpha) clipped to the quad path, giving instant visual confirmation of what was captured.
- **v1.2.5 implementation complete, awaiting on-device validation.** Ultra-low-contrast detection — two new detection paths for gradients down to ~3 units:
  - `DIRECTIONAL_GRADIENT` strategy: 5-angle tilted 1D kernel smoothing of Sobel gradients, 90th percentile threshold, binary output. Position #2 in white-on-white strategy list after DOG.
  - `LsdRadonDetector`: entirely new detection path bypassing Canny/contours. Three-tier cascade: LSD segment detection (quant=1.0, ~2.6 unit threshold) → corner-constrained Radon → joint Radon rectangle fit. Invoked as fallback after strategy loop for white-on-white scenes.
  - Test infrastructure: 5 ultra-low-contrast synthetic generators (3-unit, 5-unit, noisy, tilted, warm), `UltraLowContrastBenchmarkTest`, `LsdRadonBenchmarkTest` (7 test methods including false positive guards).
  - **Next:** On-device S21 benchmarks + real-world validation (A3, A4, B9).

## Key Architecture Details (for current work)

### Auto-Capture Pipeline
- `QuadSmoother`: buffers last 5 detections, 20-frame stability threshold, three-tier drift response (<2.5% increment, 2.5-10% halve, >10% hard reset), pre-smoothing jump detection at 10%
- `FrameAnalyzer`: hybrid detect+track via `CornerTracker` (KLT on most frames, full detection every 3rd frame during tracking), adaptive frame skipping disabled during tracking, KLT-only frames carry forward last detection confidence (v1.2.0 fix: previously injected 0.0 → broke auto-capture)
- AF lock triggers at 50% stability (10/20 frames), auto-capture fires at 100% + confidence >= 0.65 + 1.5s warmup
- `CaptureProcessor`: re-detects on full-res capture frame, validates against preview corners (5% drift tolerance). ZSL mode (`CAPTURE_MODE_ZERO_SHUTTER_LAG`) ensures captured frame matches preview (ring buffer selects past frame closest to trigger timestamp). Falls back to `MINIMIZE_LATENCY` with flash or on unsupported devices.

### Aspect Ratio (single-frame dual-regime)
- `AspectRatioEstimator`: dual-regime estimation — angular correction (<5deg severity) + projective decomposition (>10deg) + transition blending (5-10deg). Format snapping (A4, US Letter, ID Card, Square) with SNAP_THRESHOLD=0.035, SNAP_SIGMA=0.025 + homography error disambiguation when intrinsics available
- Camera intrinsics: `LENS_INTRINSIC_CALIBRATION` (API 28+) or sensor-size fallback. `CameraIntrinsics.forCaptureFrame()` handles rotation-aware scaling from sensor coords to capture frame coords (fx/fy and cx/cy swap on 90/270 rotation)
- Projective path: `H = getPerspectiveTransform(square → corners)` (world→image direction), `M = K_inv * H = [r1 r2 t]`, ratio = `||r1||/||r2||` with sanity checks (norm ratio in [0.2, 5.0], orthogonality dot < 0.3)
- Angular path: corrects each dimension independently (`trueH = sideH / cos(alphaV/2)`, `trueV = sideV / cos(alphaH/2)`), orientation-invariant
- Estimation confidence (0.85 angular, 0.75 projective with intrinsics, 0.4 fallback) separate from format snap confidence
- AR estimated at capture time from full-res corners + intrinsics (single-frame). Validated on S21: robust at all orientations, envelope (0.5 true) estimated at 0.497
- `MultiFrameAspectEstimator` exists but accumulation disabled — Zhang's method is degenerate during stabilization (camera stationary → near-identical homographies → garbage SVD). Code preserved for future revisit
- ResultScreen initial ratio: locked ratio (priority) > single-frame estimate (estConf >= 0.5, if auto-estimate enabled) > A4 fallback (0.707). Format label shows "(auto)" suffix when auto-estimated, debug info shows raw and slider ratios. Slider 0.25-1.0 with 300ms debounce re-warp, two-row layout (lock+label / full-width slider), aspect ratio lock persisted in DataStore
- Gallery imports use single-frame dual-regime estimation
- Settings toggle: "Aspect ratio default" — Auto (estimated) vs Always A4, persisted in DataStore

### Low-Contrast / White-on-White Detection (v1.2.4)
- Root cause: auto-Canny thresholds `(0.67*median, 1.33*median)` saturate for bright scenes (median ~220 → thresholds 147/293, clamped 147/250), missing 5-35 unit boundary gradients
- Scene analysis: `isWhiteOnWhite = meanVal > 180 && stddevVal < 35` triggers specialized strategy pipeline
- Strategy order (benchmark-driven, DOG first — see [docs/LOW_CONTRAST_BENCHMARK.md](docs/LOW_CONTRAST_BENCHMARK.md)):
  1. `DOG`: GaussianBlur(3x3) - GaussianBlur(21x21). Bandpass filter isolates edge-scale features, suppresses texture + illumination. 6/6 detected, 2.9-3.8ms, 0.0px error. Grayscale output → Canny 10/30.
  2. `GRADIENT_MAGNITUDE`: Sobel X/Y → magnitude → 95th percentile threshold (histogram-based). Document boundary = strongest relative gradient. 5/6 detected. Binary output bypasses Canny.
  3. `LAB_CLAHE`: BGR→LAB, L-channel + CLAHE(clipLimit=6.0, tileSize=2x2). Amplifies micro-contrast from warmth/coolness differences. 5/6 detected. Grayscale output → Canny 30/60.
  4. `CLAHE_ENHANCED`: existing strategy, reliable fallback. 6/6 detected.
  5. `MULTICHANNEL_FUSION`: Per-channel Canny(20/50) + bitwise OR. Captures color differences invisible in grayscale. 5/6 detected. Binary output bypasses Canny.
  6. `ADAPTIVE_THRESHOLD`: blockSize=51, C=5 → binary segmentation → morph gradient → edge image. 3/6 detected, last resort.
- `PreprocessStrategy.isBinaryOutput`: ADAPTIVE_THRESHOLD, GRADIENT_MAGNITUDE, MULTICHANNEL_FUSION, DIRECTIONAL_GRADIENT bypass Canny in `detectWithStrategy()` — preprocessed output is the edge map
- Non-white-on-white scenes keep existing 5-strategy pipeline — zero performance regression
- Pipeline efficiency: DOG short-circuits on first attempt (~3ms) vs old pipeline STANDARD(fail, ~8ms) → CLAHE(~5ms) = ~13ms. ~4x faster for white-on-white.
- Test infrastructure: 6 synthetic generators (whiteOnNearWhite..glossyPaper) + 5 ultra-low-contrast generators (3-unit, 5-unit, noisy, tilted, warm), `LowContrastBenchmarkTest` + `UltraLowContrastBenchmarkTest` per-strategy harness, false positive guards

### Ultra-Low-Contrast Detection — LSD+Radon Cascade (v1.2.5)
- `DIRECTIONAL_GRADIENT` strategy: 5-angle tilted 21px kernels smooth Sobel Gx/Gy along candidate edge directions, per-pixel max across angles, 90th percentile threshold → binary. ~5 unit detection floor. Position #2 in white-on-white list (after DOG).
- `LsdRadonDetector`: separate detection path (not a PreprocessStrategy), invoked as fallback after strategy loop exhaustion for white-on-white scenes only. Runs outside 25ms time budget (~10ms worst case).
  - Tier 1 (LSD fast path): `createLineSegmentDetector(quant=1.0)` → segment clustering (8deg/15px tolerance) → 2H×2V rectangle formation. ~2.6 unit detection floor, ~2.5ms.
  - Tier 2 (corner-constrained Radon): rescues 2-3 edge partial detections via coarse-to-fine restricted Radon search (perpendicular gradient only, ±12deg). +2ms.
  - Tier 3 (joint Radon rectangle fit): full restricted Radon scan, 9 angles [-8..+8 deg], independent H/V peak search, geometric priors. Last resort for <3 unit gradients. +4ms.
- Gradient density verification: perpendicular Sobel sampling along quad sides (replaces `edgeDensityScore` for LSD path). 3-of-4 sides minimum, reference gradient 20.0.
- Confidence ranges: Tier 1 [0.50, 0.85], Tier 2 [0.45, 0.75], Tier 3 [0.40, 0.65]

### Confidence Thresholds
- No corners (empty list): routes to manual corner placement with 10%-inset defaults
- < 0.35: suppressed (no detection returned)
- 0.35-0.65: routes to manual corner adjustment with detected corners pre-filled
- >= 0.65: auto-capture eligible, result shown directly

## Performance Budget
- Detection per frame: < 30ms (for real-time preview overlay)
- Full capture pipeline: < 200ms (detect + warp + save)
- Memory: < 150MB including camera buffers
- Profile with `System.nanoTime()` around key operations, log in debug
- **Validated on S21 (Snapdragon 888):** KLT-only 2.1ms median, KLT+correction 11.8ms median, 97.2% of tracked frames under 30ms. No-doc scanning ~30ms (exhausts all strategies). Capture pipeline 478ms total (307ms JPEG decode, ~88ms CV).

## Key Dependencies (versions pinned in gradle/libs.versions.toml)
- AGP: 8.7.3
- Kotlin: 2.0.21
- OpenCV Android SDK: 4.12.0
- CameraX: 1.4.1
- Compose BOM: 2024.12.01

## Local Development Environment (Windows)
- **OS:** Windows 10 Pro (10.0.19045)
- **IDE:** Android Studio (installed at `D:\Program Files\Android\Android Studio`)
- **JDK:** JetBrains Runtime 21 (bundled with Android Studio at `D:\Program Files\Android\Android Studio\jbr`)
- **Android SDK:** `D:\Android\Sdk` (SDK 34, NDK, CMake installed)
- **OpenCV Android SDK:** `D:\OpenCV-android-sdk` (v4.12.0)
- **Gradle cache:** `D:\Android\.gradle`
- **Project repo:** `E:\Repos\DocShot`

### Environment Variables (user-level)
| Variable | Value |
|----------|-------|
| `ANDROID_HOME` | `D:\Android\Sdk` |
| `JAVA_HOME` | `D:\Program Files\Android\Android Studio\jbr` |
| `GRADLE_USER_HOME` | `D:\Android\.gradle` |
| `OPENCV_ANDROID_SDK` | `D:\OpenCV-android-sdk` |
| `PATH` (appended) | `D:\Android\Sdk\platform-tools` |

### Notes
- C: drive has limited space -- all SDK/build artifacts go on D:
- No standalone JDK installed; using Android Studio's bundled JBR 21
- Test device: Samsung Galaxy S21 via USB debugging
