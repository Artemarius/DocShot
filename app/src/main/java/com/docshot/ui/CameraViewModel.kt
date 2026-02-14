package com.docshot.ui

import androidx.lifecycle.ViewModel
import com.docshot.camera.FrameAnalyzer
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

/**
 * UI state for the detected document overlay.
 * Corners are in normalized [0,1] display coordinates (rotation already applied).
 */
data class DetectionUiState(
    val normalizedCorners: List<FloatArray>? = null,
    val sourceWidth: Int = 0,
    val sourceHeight: Int = 0,
    val detectionMs: Double = 0.0
)

class CameraViewModel : ViewModel() {

    private val _detectionState = MutableStateFlow(DetectionUiState())
    val detectionState: StateFlow<DetectionUiState> = _detectionState

    val frameAnalyzer = FrameAnalyzer { result ->
        _detectionState.value = DetectionUiState(
            normalizedCorners = result.normalizedCorners,
            sourceWidth = result.sourceWidth,
            sourceHeight = result.sourceHeight,
            detectionMs = result.detectionMs
        )
    }
}
