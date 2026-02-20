package com.docshot.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.BugReport
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.AspectRatio
import androidx.compose.material.icons.filled.FilterBAndW
import androidx.compose.material.icons.filled.HighQuality
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.FlashOn
import androidx.compose.material.icons.filled.Vibration
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import com.docshot.util.DocShotSettings
import com.docshot.util.UserPreferencesRepository
import kotlinx.coroutines.launch
import kotlin.math.roundToInt

@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun SettingsScreen(
    onBack: () -> Unit,
    preferencesRepository: UserPreferencesRepository
) {
    val settings by preferencesRepository.settings.collectAsState(initial = DocShotSettings())
    val scope = rememberCoroutineScope()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Settings") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back"
                        )
                    }
                }
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .verticalScroll(rememberScrollState())
        ) {
            // ── Capture section ──────────────────────────────────────────
            SectionHeader(text = "Capture")

            SettingsToggleItem(
                icon = Icons.Filled.CameraAlt,
                title = "Auto-capture",
                description = "Automatically capture when a stable document is detected",
                checked = settings.autoCaptureEnabled,
                onCheckedChange = { scope.launch { preferencesRepository.setAutoCaptureEnabled(it) } }
            )

            SettingsToggleItem(
                icon = Icons.Filled.Vibration,
                title = "Haptic feedback",
                description = "Vibrate on capture",
                checked = settings.hapticFeedback,
                onCheckedChange = { scope.launch { preferencesRepository.setHapticFeedback(it) } }
            )

            SettingsToggleItem(
                icon = Icons.Filled.FlashOn,
                title = "Flash",
                description = "Enable camera torch for low-light scanning",
                checked = settings.flashEnabled,
                onCheckedChange = { scope.launch { preferencesRepository.setFlashEnabled(it) } }
            )

            SettingsToggleItem(
                icon = Icons.Filled.BugReport,
                title = "Debug overlay",
                description = "Show detection latency on camera preview",
                checked = settings.showDebugOverlay,
                onCheckedChange = { scope.launch { preferencesRepository.setShowDebugOverlay(it) } }
            )

            HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))

            // ── Aspect Ratio section ─────────────────────────────────────
            SectionHeader(text = "Aspect Ratio")

            SettingsToggleItem(
                icon = Icons.Filled.AspectRatio,
                title = "Aspect ratio default",
                description = "When enabled, automatically estimates the document's aspect ratio. When disabled, defaults to A4.",
                checked = settings.aspectRatioAutoEstimate,
                onCheckedChange = { scope.launch { preferencesRepository.setAspectRatioAutoEstimate(it) } }
            )

            HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))

            // ── Output section ───────────────────────────────────────────
            SectionHeader(text = "Output")

            // Format chips
            ListItem(
                leadingContent = {
                    Icon(
                        imageVector = Icons.Filled.Image,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                },
                headlineContent = { Text("Format") },
                supportingContent = {
                    FlowRow(
                        modifier = Modifier.padding(top = 4.dp),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        listOf("JPEG", "PNG").forEach { format ->
                            FilterChip(
                                selected = settings.outputFormat == format,
                                onClick = { scope.launch { preferencesRepository.setOutputFormat(format) } },
                                label = { Text(format) }
                            )
                        }
                    }
                }
            )

            // Quality slider (only visible for JPEG)
            if (settings.outputFormat == "JPEG") {
                ListItem(
                    leadingContent = {
                        Icon(
                            imageVector = Icons.Filled.HighQuality,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    },
                    headlineContent = { Text("Quality") },
                    supportingContent = {
                        Column {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Slider(
                                    value = settings.outputQuality.toFloat(),
                                    onValueChange = { newValue ->
                                        scope.launch {
                                            preferencesRepository.setOutputQuality(newValue.roundToInt())
                                        }
                                    },
                                    valueRange = 50f..100f,
                                    steps = 9,
                                    modifier = Modifier.weight(1f)
                                )
                                Spacer(modifier = Modifier.width(12.dp))
                                Text(
                                    text = "${settings.outputQuality}%",
                                    style = MaterialTheme.typography.bodyMedium
                                )
                            }
                        }
                    }
                )
            }

            // Default filter chips
            ListItem(
                leadingContent = {
                    Icon(
                        imageVector = Icons.Filled.FilterBAndW,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                },
                headlineContent = { Text("Default filter") },
                supportingContent = {
                    FlowRow(
                        modifier = Modifier.padding(top = 4.dp),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        val filters = listOf(
                            "NONE" to "None",
                            "BLACK_WHITE" to "B&W",
                            "CONTRAST" to "Contrast",
                            "COLOR_CORRECT" to "Even Light"
                        )
                        filters.forEach { (key, label) ->
                            FilterChip(
                                selected = settings.defaultFilter == key,
                                onClick = { scope.launch { preferencesRepository.setDefaultFilter(key) } },
                                label = { Text(label) }
                            )
                        }
                    }
                }
            )

            Spacer(modifier = Modifier.height(16.dp))
        }
    }
}

@Composable
private fun SectionHeader(text: String) {
    Text(
        text = text,
        style = MaterialTheme.typography.labelLarge,
        color = MaterialTheme.colorScheme.primary,
        modifier = Modifier.padding(start = 16.dp, top = 16.dp, bottom = 4.dp)
    )
}

@Composable
private fun SettingsToggleItem(
    icon: ImageVector,
    title: String,
    description: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit
) {
    ListItem(
        leadingContent = {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant
            )
        },
        headlineContent = { Text(title) },
        supportingContent = { Text(description) },
        trailingContent = {
            Switch(
                checked = checked,
                onCheckedChange = onCheckedChange
            )
        }
    )
}
