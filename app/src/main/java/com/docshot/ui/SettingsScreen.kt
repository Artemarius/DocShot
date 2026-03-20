package com.docshot.ui

import androidx.compose.foundation.clickable
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
import androidx.compose.material.icons.filled.WbSunny
import androidx.compose.material.icons.outlined.Email
import androidx.compose.material.icons.outlined.HelpOutline
import androidx.compose.material.icons.outlined.Info
import androidx.compose.material.icons.outlined.Shield
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import android.content.Intent
import android.net.Uri
import com.docshot.R
import com.docshot.util.DocShotSettings
import com.docshot.util.UserPreferencesRepository
import kotlinx.coroutines.launch
import kotlin.math.roundToInt

@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun SettingsScreen(
    onBack: () -> Unit,
    preferencesRepository: UserPreferencesRepository,
    onShowOnboarding: () -> Unit = {}
) {
    val settings by preferencesRepository.settings.collectAsState(initial = DocShotSettings())
    val scope = rememberCoroutineScope()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.title_settings)) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = stringResource(R.string.cd_back)
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
            SectionHeader(text = stringResource(R.string.section_capture))

            SettingsToggleItem(
                icon = Icons.Filled.CameraAlt,
                title = stringResource(R.string.setting_auto_capture),
                description = stringResource(R.string.setting_auto_capture_desc),
                checked = settings.autoCaptureEnabled,
                onCheckedChange = { scope.launch { preferencesRepository.setAutoCaptureEnabled(it) } }
            )

            SettingsToggleItem(
                icon = Icons.Filled.Vibration,
                title = stringResource(R.string.setting_haptic),
                description = stringResource(R.string.setting_haptic_desc),
                checked = settings.hapticFeedback,
                onCheckedChange = { scope.launch { preferencesRepository.setHapticFeedback(it) } }
            )

            SettingsToggleItem(
                icon = Icons.Filled.FlashOn,
                title = stringResource(R.string.setting_flash),
                description = stringResource(R.string.setting_flash_desc),
                checked = settings.flashEnabled,
                onCheckedChange = { scope.launch { preferencesRepository.setFlashEnabled(it) } }
            )

            SettingsToggleItem(
                icon = Icons.Filled.BugReport,
                title = stringResource(R.string.setting_debug),
                description = stringResource(R.string.setting_debug_desc),
                checked = settings.showDebugOverlay,
                onCheckedChange = { scope.launch { preferencesRepository.setShowDebugOverlay(it) } }
            )

            HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))

            // ── Aspect Ratio section ─────────────────────────────────────
            SectionHeader(text = stringResource(R.string.section_aspect_ratio))

            SettingsToggleItem(
                icon = Icons.Filled.AspectRatio,
                title = stringResource(R.string.setting_ar_auto),
                description = stringResource(R.string.setting_ar_auto_desc),
                checked = settings.aspectRatioAutoEstimate,
                onCheckedChange = { scope.launch { preferencesRepository.setAspectRatioAutoEstimate(it) } }
            )

            HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))

            // ── Output section ───────────────────────────────────────────
            SectionHeader(text = stringResource(R.string.section_output))

            SettingsToggleItem(
                icon = Icons.Filled.WbSunny,
                title = stringResource(R.string.setting_wb),
                description = stringResource(R.string.setting_wb_desc),
                checked = settings.autoWhiteBalance,
                onCheckedChange = { scope.launch { preferencesRepository.setAutoWhiteBalance(it) } }
            )

            // Format chips
            ListItem(
                leadingContent = {
                    Icon(
                        imageVector = Icons.Filled.Image,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                },
                headlineContent = { Text(stringResource(R.string.setting_format)) },
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
                    headlineContent = { Text(stringResource(R.string.setting_quality)) },
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
                                    text = stringResource(R.string.setting_quality_value, settings.outputQuality),
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
                headlineContent = { Text(stringResource(R.string.setting_default_filter)) },
                supportingContent = {
                    FlowRow(
                        modifier = Modifier.padding(top = 4.dp),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        val filters = listOf(
                            "NONE" to R.string.filter_none,
                            "BLACK_WHITE" to R.string.filter_bw,
                            "CONTRAST" to R.string.filter_contrast,
                            "COLOR_CORRECT" to R.string.filter_even_light
                        )
                        filters.forEach { (key, labelRes) ->
                            FilterChip(
                                selected = settings.defaultFilter == key,
                                onClick = { scope.launch { preferencesRepository.setDefaultFilter(key) } },
                                label = { Text(stringResource(labelRes)) }
                            )
                        }
                    }
                }
            )

            HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))

            // ── About section ─────────────────────────────────────────
            SectionHeader(text = stringResource(R.string.section_about))

            val aboutContext = LocalContext.current

            ListItem(
                leadingContent = {
                    Icon(
                        imageVector = Icons.Outlined.HelpOutline,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                },
                headlineContent = { Text(stringResource(R.string.setting_show_tutorial)) },
                supportingContent = { Text(stringResource(R.string.setting_show_tutorial_desc)) },
                modifier = Modifier.clickable { onShowOnboarding() }
            )

            ListItem(
                leadingContent = {
                    Icon(
                        imageVector = Icons.Outlined.Shield,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                },
                headlineContent = { Text(stringResource(R.string.setting_privacy_policy)) },
                supportingContent = { Text(stringResource(R.string.setting_privacy_policy_desc)) },
                modifier = Modifier.clickable {
                    val intent = Intent(
                        Intent.ACTION_VIEW,
                        Uri.parse("https://artemarius.github.io/DocShot/privacy-policy.html")
                    )
                    aboutContext.startActivity(intent)
                }
            )

            ListItem(
                leadingContent = {
                    Icon(
                        imageVector = Icons.Outlined.Email,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                },
                headlineContent = { Text(stringResource(R.string.setting_send_feedback)) },
                supportingContent = { Text(stringResource(R.string.setting_send_feedback_desc)) },
                modifier = Modifier.clickable {
                    val intent = Intent(
                        Intent.ACTION_VIEW,
                        Uri.parse("https://github.com/Artemarius/DocShot/issues")
                    )
                    aboutContext.startActivity(intent)
                }
            )

            ListItem(
                leadingContent = {
                    Icon(
                        imageVector = Icons.Outlined.Info,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                },
                headlineContent = { Text(stringResource(R.string.setting_version)) },
                supportingContent = {
                    val packageInfo = aboutContext.packageManager
                        .getPackageInfo(aboutContext.packageName, 0)
                    Text("${packageInfo.versionName}")
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
