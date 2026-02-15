# Add project specific ProGuard rules here.

# OpenCV: keep all JNI classes and native method bindings
-keep class org.opencv.** { *; }
-dontwarn org.opencv.**

# Keep Android entry points
-keep class com.docshot.** { *; }
