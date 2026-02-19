pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "DocShot"

include(":app")
include(":opencv")
val opencvSdkDir = System.getenv("OPENCV_ANDROID_SDK") ?: "D:/OpenCV-android-sdk"
project(":opencv").projectDir = file("$opencvSdkDir/sdk")
