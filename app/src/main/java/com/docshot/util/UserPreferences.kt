package com.docshot.util

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "docshot_settings")

/**
 * All user-configurable settings with their defaults.
 */
data class DocShotSettings(
    val autoCaptureEnabled: Boolean = true,
    val outputQuality: Int = 95,
    val outputFormat: String = "JPEG",
    val defaultFilter: String = "NONE",
    val showDebugOverlay: Boolean = false,
    val hapticFeedback: Boolean = true,
    val flashEnabled: Boolean = false
)

/**
 * Reads and writes user preferences via Jetpack DataStore.
 *
 * Each setter performs a single atomic edit on the backing DataStore file.
 * The [settings] flow emits a new [DocShotSettings] whenever any key changes.
 */
class UserPreferencesRepository(private val context: Context) {

    private companion object {
        val KEY_AUTO_CAPTURE = booleanPreferencesKey("auto_capture_enabled")
        val KEY_OUTPUT_QUALITY = intPreferencesKey("output_quality")
        val KEY_OUTPUT_FORMAT = stringPreferencesKey("output_format")
        val KEY_DEFAULT_FILTER = stringPreferencesKey("default_filter")
        val KEY_SHOW_DEBUG_OVERLAY = booleanPreferencesKey("show_debug_overlay")
        val KEY_HAPTIC_FEEDBACK = booleanPreferencesKey("haptic_feedback")
        val KEY_FLASH_ENABLED = booleanPreferencesKey("flash_enabled")
    }

    /**
     * Observable stream of the current settings. Missing keys fall back to
     * the defaults defined in [DocShotSettings].
     */
    val settings: Flow<DocShotSettings> = context.dataStore.data.map { prefs ->
        DocShotSettings(
            autoCaptureEnabled = prefs[KEY_AUTO_CAPTURE] ?: true,
            outputQuality = prefs[KEY_OUTPUT_QUALITY] ?: 95,
            outputFormat = prefs[KEY_OUTPUT_FORMAT] ?: "JPEG",
            defaultFilter = prefs[KEY_DEFAULT_FILTER] ?: "NONE",
            showDebugOverlay = prefs[KEY_SHOW_DEBUG_OVERLAY] ?: false,
            hapticFeedback = prefs[KEY_HAPTIC_FEEDBACK] ?: true,
            flashEnabled = prefs[KEY_FLASH_ENABLED] ?: false
        )
    }

    suspend fun setAutoCaptureEnabled(enabled: Boolean) {
        context.dataStore.edit { prefs -> prefs[KEY_AUTO_CAPTURE] = enabled }
    }

    suspend fun setOutputQuality(quality: Int) {
        require(quality in 1..100) { "JPEG quality must be 1-100, got $quality" }
        context.dataStore.edit { prefs -> prefs[KEY_OUTPUT_QUALITY] = quality }
    }

    suspend fun setOutputFormat(format: String) {
        require(format == "JPEG" || format == "PNG") { "Format must be JPEG or PNG, got $format" }
        context.dataStore.edit { prefs -> prefs[KEY_OUTPUT_FORMAT] = format }
    }

    suspend fun setDefaultFilter(filter: String) {
        require(filter in listOf("NONE", "BLACK_WHITE", "CONTRAST", "COLOR_CORRECT")) {
            "Unknown filter: $filter"
        }
        context.dataStore.edit { prefs -> prefs[KEY_DEFAULT_FILTER] = filter }
    }

    suspend fun setShowDebugOverlay(show: Boolean) {
        context.dataStore.edit { prefs -> prefs[KEY_SHOW_DEBUG_OVERLAY] = show }
    }

    suspend fun setHapticFeedback(enabled: Boolean) {
        context.dataStore.edit { prefs -> prefs[KEY_HAPTIC_FEEDBACK] = enabled }
    }

    suspend fun setFlashEnabled(enabled: Boolean) {
        context.dataStore.edit { prefs -> prefs[KEY_FLASH_ENABLED] = enabled }
    }
}
