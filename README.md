# DocShot

Zero-tap document rectification for Android. Point your camera at a document — DocShot automatically detects the boundary, corrects perspective distortion, and saves a clean rectangular image. No corner dragging, no multi-step workflows.

## Motivation

Every phone has a document scanner buried 3 taps deep in the camera app, and it still asks you to adjust corners manually. I wanted something faster: open the app, point at a document, get a clean scan. The kind of tool I'd actually use daily instead of just knowing it exists somewhere in my phone's settings.

## How It Works

**Classical CV pipeline (Phase 1):**
1. **Preprocessing** — Adaptive Gaussian blur + bilateral filtering to handle varying lighting
2. **Edge detection** — Canny with automatic threshold selection based on image statistics
3. **Contour extraction** — Find external contours, approximate to polygons, score quadrilateral candidates by area, convexity, and aspect ratio
4. **Corner refinement** — Sub-pixel corner detection on the best quadrilateral
5. **Homography & warp** — `getPerspectiveTransform` → `warpPerspective` to a properly-sized output rectangle
6. **Post-processing** — Optional adaptive thresholding for black-and-white document mode

## Features

- **Single-tap capture** — Tap shutter or let auto-capture trigger when a stable document is detected
- **Gallery import** — Rectify existing photos from your gallery
- **Real-time preview** — Live quadrilateral overlay showing detected document boundary
- **Auto-crop** — Output contains only the document, properly oriented
- **Post-processing** — B&W adaptive threshold, CLAHE contrast enhancement, white balance correction
- **Manual fallback** — Draggable corner handles with magnifier loupe when auto-detection needs help

## Tech Stack

- **Language:** Kotlin
- **Camera:** CameraX (Camera2 backend)
- **Computer Vision:** OpenCV Android SDK 4.x
- **Build:** Gradle with Kotlin DSL
- **Min SDK:** 24 (Android 7.0) / **Target SDK:** 34

## Building

```bash
# Clone
git clone https://github.com/Artemarius/DocShot.git
cd DocShot

# Open in Android Studio and sync Gradle
# Or build from command line:
./gradlew assembleDebug

# Install on connected device
./gradlew installDebug
```

**Requirements:**
- Android Studio Hedgehog (2023.1.1) or later
- Android SDK 34
- NDK (for OpenCV native libs)
- Physical device recommended (camera features don't work well in emulator)

## Architecture

```
app/src/main/java/com/docshot/
├── ui/                     # Compose UI + camera preview
│   ├── CameraScreen.kt     # Main camera view with overlay
│   ├── GalleryScreen.kt    # Image picker flow
│   └── ResultScreen.kt     # Before/after comparison + save
├── camera/                 # CameraX setup and image analysis
│   ├── CameraManager.kt    # Camera lifecycle and configuration
│   └── FrameAnalyzer.kt    # Real-time document detection on preview frames
├── cv/                     # Computer vision pipeline
│   ├── DocumentDetector.kt # Edge detection + contour scoring
│   ├── QuadRanker.kt       # Quadrilateral candidate evaluation
│   ├── Rectifier.kt        # Homography computation + warp
│   └── PostProcessor.kt    # Enhancement filters (B&W, contrast)
└── util/                   # Image I/O, permissions, gallery save
```

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Detection latency | < 30ms per frame | Real-time preview at 30fps on mid-range devices |
| Full pipeline (detect + warp + save) | < 200ms | From shutter press to saved image |
| APK size | < 25MB | OpenCV contributes ~15MB for arm64 |
| Memory usage | < 150MB | Including camera preview buffers |

## Development Device

Samsung Galaxy S21 (Snapdragon 888, 8GB RAM, Android 14)

## References

- Javed et al., *Real-Time Document Localization in Natural Images by Recursive Application of a CNN* (2017)
- OpenCV `findContours`, `approxPolyDP`, `getPerspectiveTransform` documentation
- CameraX documentation: https://developer.android.com/media/camera/camerax

## License

MIT
