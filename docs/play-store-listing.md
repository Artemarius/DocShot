# Play Store Listing — DocShot

## Short Description (80 chars max)

Zero-tap document scanner. Point, auto-capture, get a perfect rectangle.

## Full Description (4000 chars max)

DocShot scans documents instantly — no corner dragging, no multi-step wizards. Open the app, point your camera at a document, and DocShot does the rest: detects the boundary in real time, waits for a stable frame, locks autofocus, captures automatically, corrects perspective, and saves a clean rectangular image.

HOW IT WORKS

DocShot uses classical computer vision running entirely on your device:

• Real-time document detection at 30fps with live boundary overlay
• Auto-capture triggers after 20 stable frames with confirmed autofocus lock
• Sub-pixel corner refinement for precise perspective correction
• Automatic orientation detection so your scan is always right-side up
• Smart aspect ratio snapping to common formats (A4, US Letter, ID card, business card)

FEATURES

• Zero-tap scanning — Auto-capture fires when the document is stable. Just point and wait.
• Manual capture — Tap the shutter anytime if you prefer full control.
• Aspect ratio lock — Set your ratio once (e.g., A4) and it persists across scans. Perfect for batch scanning same-format documents.
• Gallery import — Rectify photos you already have. Auto-detection or manual corner adjustment with draggable handles and magnifier loupe.
• Post-processing filters — Black & White (adaptive threshold), High Contrast (CLAHE), Even Light (corrects brightness gradients from angled lighting).
• Flash control — Torch toggle persists across captures, re-enables automatically after each scan.
• Confidence-gated quality — High-confidence detections go straight to the result. Low-confidence routes to manual corner adjustment so you always get a good scan.
• Multi-strategy detection — Five preprocessing strategies adapt to varied backgrounds: cluttered desks, colored surfaces, low contrast, shadows.
• Share & save — Save to gallery or share directly from the result screen.

PRIVACY

DocShot processes everything on-device. No internet connection required, no accounts, no analytics, no data collection. Your documents stay on your phone.

PERFORMANCE

• Detection: ~15ms per frame
• Full capture pipeline: ~120ms
• App size: ~26MB installed

Built with OpenCV, CameraX, and Jetpack Compose.
