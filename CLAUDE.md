# DocShot — CLAUDE.md

## Project Overview
Android document rectification app. Camera input → automatic document boundary detection → perspective correction → clean rectangular output. Zero user interaction beyond shutter tap.

## Tech Stack
- **Language:** Kotlin (idiomatic, no Java fallbacks)
- **UI:** Jetpack Compose + Material 3
- **Camera:** CameraX with Camera2 backend
- **CV:** OpenCV Android SDK 4.10+
- **ML (Phase 2):** TensorFlow Lite
- **Build:** Gradle Kotlin DSL, AGP 8.x
- **Min SDK:** 24 / Target SDK: 34
- **Test device:** Samsung Galaxy S21 (Snapdragon 888)

## Architecture Rules
- MVVM with ViewModels for camera and processing state
- CV pipeline runs entirely off the main thread (coroutines + Dispatchers.Default)
- OpenCV Mat objects must be explicitly released — no relying on GC
- CameraX ImageAnalysis runs at STRATEGY_KEEP_ONLY_LATEST to avoid frame queue buildup
- Separate detection (runs every frame, must be <30ms) from rectification (runs once on capture, can take up to 200ms)
- All OpenCV calls go through Kotlin wrapper functions in the `cv/` package — no raw OpenCV calls in UI or camera code

## Build & Run
```bash
./gradlew assembleDebug
./gradlew installDebug
# Run tests:
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
- Quadrilateral scoring: weighted combination of area (largest preferred), convexity (must be convex), angle regularity (prefer ~90° corners)
- Corner ordering: consistent top-left, top-right, bottom-right, bottom-left based on sum/difference of coordinates
- Output dimensions: derive from longest edge pairs to preserve document aspect ratio
- Always use `Imgproc.INTER_CUBIC` for the final warp, `INTER_LINEAR` for preview

## File Structure
```
app/src/main/java/com/docshot/
├── ui/          # Compose screens + navigation
├── camera/      # CameraX setup, frame analysis
├── cv/          # Document detection, rectification, post-processing
├── ml/          # TFLite segmentation (Phase 2)
└── util/        # Permissions, image I/O, gallery save
```

## Phase Boundaries
- **Phase 1-5:** Classical CV only, no ML dependencies
- **Phase 6+:** Add TensorFlow Lite, ML model, A/B comparison with classical
- Keep classical pipeline fully functional even after ML is added (user toggle or fallback)

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
- TFLite (Phase 2): 2.16.x

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
- C: drive has limited space — all SDK/build artifacts go on D:
- No standalone JDK installed; using Android Studio's bundled JBR 21
- Test device: Samsung Galaxy S21 via USB debugging
- Phase 1 complete: project skeleton builds, CameraX preview + OpenCV init working
- Phase 2 complete: full classical CV pipeline (detect + rectify) working on static images, 10 unit tests passing
- Phase 3 complete: real-time detection works well on dark/uniform backgrounds; fails on cluttered or low-contrast (light) backgrounds — deferred to Phase 7
- Phase 4 complete: capture & rectify flow — tap shutter FAB → full-res pipeline (YUV→BGR → rotate → detect → sub-pixel refine → rectify) → result screen with original/rectified toggle, save to gallery (MediaStore), share via FileProvider. CameraX 3-use-case binding (Preview + ImageAnalysis + ImageCapture) with graceful fallback.
- Phase 5 complete: gallery import via system photo picker (PickVisualMedia, no permissions needed) + manual corner adjustment with draggable handles and magnifier loupe. EXIF rotation handling, downscale to 4000px max. Detection downscaled to ~1000px for reliable kernel matching. Fixed capture crash on S21/Android 15 — ImageCapture returns JPEG (1 plane), not YUV; now handles both formats. Tab 1 replaced from Pipeline Test to Import; gallery button added on CameraScreen.
- Phase 6 complete: auto-capture (stability detection in QuadSmoother, 15 consecutive frames with <2% corner drift), visual feedback (green→cyan quad with progressive fill opacity), haptic on capture (CONFIRM on API 30+, LONG_PRESS fallback), auto-capture toggle FAB. Post-processing filters on ResultScreen (B&W adaptive threshold, CLAHE contrast, gray-world white balance) with async processing. Document orientation detection (Sobel gradient analysis + ink-density heuristic) integrated into capture pipeline. Settings screen with DataStore (auto-capture, haptic, debug overlay, output format/quality, default filter). Custom DocShot brand color scheme (teal/cyan primary, dark theme optimized for camera). App icon and splash screen deferred to Phase 10.
- Phase 7 complete: robustness & edge case handling. Group A: per-detection confidence scoring (60% quad score + 20% area ratio + 20% edge density), edge-density validation via QuadValidator, suppression threshold at 0.35. Group B: low-confidence fallback to manual corner adjustment (0.35–0.65), "Point at a document" hint, auto-capture gated to high-confidence (>=0.65), multi-candidate score-margin penalty. Group C: multi-strategy preprocessing (STANDARD, CLAHE_ENHANCED, SATURATION_CHANNEL, BILATERAL, HEAVY_MORPH) with scene analysis and 25ms budget, small document support (2% min area, aspect-ratio scoring), partial document detection with "Move back" hint. Group D: expanded test coverage — 21 instrumented regression tests (SyntheticImageFactory with 7 corner presets + 10 image generators), 17 QuadScoringTest unit tests. All Phase 7 roadmap items complete. Total: 32 unit tests + 21 instrumented tests = 53 tests.
