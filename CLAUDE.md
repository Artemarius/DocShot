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
- 5 preprocessing strategies (STANDARD, CLAHE_ENHANCED, SATURATION_CHANNEL, BILATERAL, HEAVY_MORPH) with scene analysis and 25ms time budget

## File Structure
```
app/src/main/java/com/docshot/
├── ui/          # Compose screens + navigation
├── camera/      # CameraX setup, frame analysis, QuadSmoother
├── cv/          # Document detection, rectification, post-processing
└── util/        # Permissions, image I/O, gallery save, DataStore prefs
```

## Current State (v1.1.2)
- **Phases 1-11 complete.** Full classical CV pipeline, auto-capture with AF lock, aspect ratio slider with format snapping, flash, gallery import, post-processing filters (B&W, Contrast, Even Light), 74 tests (47 unit + 27 instrumented). See [docs/PHASE_HISTORY.md](docs/PHASE_HISTORY.md) for detailed phase-by-phase history.
- **Phase 12 (Play Store release) in progress.** App icon, splash screen, signing, privacy policy, store listing done. Remaining: Play Console forms, screenshots, submit for review.
- **v1.2.0 planned.** KLT corner tracking + multi-frame aspect ratio estimation. See [PROJECT.md](PROJECT.md) for roadmap and [ASPECT_RATIO_PLAN.md](ASPECT_RATIO_PLAN.md) for technical design.

## Key Architecture Details (for current work)

### Auto-Capture Pipeline
- `QuadSmoother`: buffers last 5 detections, 20-frame stability threshold, three-tier drift response (<2.5% increment, 2.5-10% halve, >10% hard reset), pre-smoothing jump detection at 10%
- `FrameAnalyzer`: runs `DocumentDetector` per frame at 640px, adaptive frame skipping (tiers at 5 and 15 consecutive misses)
- AF lock triggers at 50% stability (10/20 frames), auto-capture fires at 100% + confidence >= 0.65 + 1.5s warmup
- `CaptureProcessor`: re-detects on full-res capture frame, validates against preview corners (5% drift tolerance)

### Aspect Ratio (current -- single-frame only)
- `AspectRatioEstimator`: edge-length ratio + format snapping (A4, US Letter, ID Card, Business Card, Receipt, Square) + homography error disambiguation when intrinsics available
- Camera intrinsics: `LENS_INTRINSIC_CALIBRATION` (API 28+) or sensor-size fallback
- ResultScreen defaults to A4 (0.707), slider 0.25-1.0 with 300ms debounce re-warp, aspect ratio lock persisted in DataStore

### Confidence Thresholds
- < 0.35: suppressed (no detection returned)
- 0.35-0.65: routes to manual corner adjustment
- >= 0.65: auto-capture eligible, result shown directly

## Performance Budget
- Detection per frame: < 30ms (for real-time preview overlay)
- Full capture pipeline: < 200ms (detect + warp + save)
- Memory: < 150MB including camera buffers
- Profile with `System.nanoTime()` around key operations, log in debug

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
