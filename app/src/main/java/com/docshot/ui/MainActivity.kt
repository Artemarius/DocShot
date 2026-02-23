package com.docshot.ui

import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.docshot.ui.theme.DocShotTheme
import com.docshot.util.UserPreferencesRepository

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        installSplashScreen()
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            DocShotTheme {
                MainContent()
            }
        }
    }
}

@Composable
private fun MainContent() {
    val context = LocalContext.current
    val hasCamera = remember {
        context.packageManager.hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY)
    }
    var selectedTab by rememberSaveable { mutableIntStateOf(if (hasCamera) 0 else 1) }
    var showSettings by rememberSaveable { mutableStateOf(false) }
    var showingResult by rememberSaveable { mutableStateOf(false) }
    val preferencesRepository = remember { UserPreferencesRepository(context) }

    if (showSettings) {
        SettingsScreen(
            onBack = { showSettings = false },
            preferencesRepository = preferencesRepository
        )
        return
    }

    Scaffold { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
        ) {
            // Hide the tab row when showing a result screen to maximize image space
            if (!showingResult) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    if (hasCamera) {
                        FilterChip(
                            selected = selectedTab == 0,
                            onClick = { selectedTab = 0 },
                            label = { Text("Camera") }
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                    }
                    FilterChip(
                        selected = selectedTab == 1,
                        onClick = { selectedTab = 1 },
                        label = { Text("Import") },
                        modifier = if (hasCamera) Modifier.padding(start = 8.dp) else Modifier
                    )
                    Spacer(modifier = Modifier.weight(1f))
                    IconButton(onClick = { showSettings = true }) {
                        Icon(
                            imageVector = Icons.Filled.Settings,
                            contentDescription = "Settings",
                            tint = MaterialTheme.colorScheme.onSurface
                        )
                    }
                }
            }

            when (selectedTab) {
                0 -> CameraPermissionScreen(
                    onOpenGallery = { selectedTab = 1 },
                    preferencesRepository = preferencesRepository,
                    onShowingResult = { showingResult = it }
                )
                1 -> GalleryScreen(
                    onShowingResult = { showingResult = it },
                    preferencesRepository = preferencesRepository
                )
            }
        }
    }
}
