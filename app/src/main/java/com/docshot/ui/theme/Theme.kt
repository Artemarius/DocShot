package com.docshot.ui.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext

// Teal/Cyan primary — evokes precision, scanning, technology
// Dark theme is the primary theme (camera app works better in dark)

private val DocShotDarkColorScheme = darkColorScheme(
    primary = Color(0xFF4DD0E1),          // Cyan 300 — primary actions
    onPrimary = Color(0xFF003738),        // Dark teal on primary
    primaryContainer = Color(0xFF004F50), // Teal dark container
    onPrimaryContainer = Color(0xFF97F0FF), // Light cyan on container
    secondary = Color(0xFFB0BEC5),        // Blue Grey 200
    onSecondary = Color(0xFF1B2B31),
    secondaryContainer = Color(0xFF324047),
    onSecondaryContainer = Color(0xFFCCD8DE),
    tertiary = Color(0xFFA5D6A7),         // Green 200 — success/detection
    onTertiary = Color(0xFF0E3913),
    error = Color(0xFFFFB4AB),
    background = Color(0xFF121212),       // Near-black background
    onBackground = Color(0xFFE0E0E0),
    surface = Color(0xFF1E1E1E),
    onSurface = Color(0xFFE0E0E0),
    surfaceVariant = Color(0xFF2C2C2C),
    onSurfaceVariant = Color(0xFFBDBDBD),
    outline = Color(0xFF757575)
)

private val DocShotLightColorScheme = lightColorScheme(
    primary = Color(0xFF00838F),          // Cyan 800
    onPrimary = Color(0xFFFFFFFF),
    primaryContainer = Color(0xFFB2EBF2), // Cyan 100
    onPrimaryContainer = Color(0xFF003D42),
    secondary = Color(0xFF546E7A),        // Blue Grey 600
    onSecondary = Color(0xFFFFFFFF),
    secondaryContainer = Color(0xFFCFD8DC),
    onSecondaryContainer = Color(0xFF0D1F28),
    tertiary = Color(0xFF2E7D32),         // Green 800
    onTertiary = Color(0xFFFFFFFF),
    error = Color(0xFFB3261E),
    background = Color(0xFFFAFAFA),
    onBackground = Color(0xFF1C1C1C),
    surface = Color(0xFFFAFAFA),
    onSurface = Color(0xFF1C1C1C),
    surfaceVariant = Color(0xFFE0E0E0),
    onSurfaceVariant = Color(0xFF49454F),
    outline = Color(0xFF9E9E9E)
)

@Composable
fun DocShotTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = true,
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context)
            else dynamicLightColorScheme(context)
        }
        darkTheme -> DocShotDarkColorScheme
        else -> DocShotLightColorScheme
    }

    MaterialTheme(
        colorScheme = colorScheme,
        content = content
    )
}
