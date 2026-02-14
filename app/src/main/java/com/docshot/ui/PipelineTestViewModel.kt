package com.docshot.ui

import android.app.Application
import android.graphics.Bitmap
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.docshot.cv.detectAndRectify
import com.docshot.util.loadMatFromAsset
import com.docshot.util.matToBitmap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

private const val TAG = "DocShot:TestVM"
private const val TEST_IMAGE = "test_document.png"

data class PipelineTestState(
    val inputBitmap: Bitmap? = null,
    val outputBitmap: Bitmap? = null,
    val pipelineMs: Double = 0.0,
    val status: String = "Idle",
    val error: String? = null
)

class PipelineTestViewModel(application: Application) : AndroidViewModel(application) {

    private val _state = MutableStateFlow(PipelineTestState())
    val state: StateFlow<PipelineTestState> = _state

    fun runPipeline() {
        _state.value = PipelineTestState(status = "Processing...")
        viewModelScope.launch(Dispatchers.Default) {
            try {
                val context = getApplication<Application>()
                val inputMat = loadMatFromAsset(context, TEST_IMAGE)
                val inputBmp = matToBitmap(inputMat)

                val result = detectAndRectify(inputMat)
                inputMat.release()

                if (result != null) {
                    val outputBmp = matToBitmap(result.rectified)
                    result.rectified.release()
                    _state.value = PipelineTestState(
                        inputBitmap = inputBmp,
                        outputBitmap = outputBmp,
                        pipelineMs = result.pipelineMs,
                        status = "Done (%.1f ms)".format(result.pipelineMs)
                    )
                } else {
                    _state.value = PipelineTestState(
                        inputBitmap = inputBmp,
                        status = "No document detected"
                    )
                }
            } catch (e: Exception) {
                Log.e(TAG, "Pipeline failed", e)
                _state.value = PipelineTestState(
                    status = "Error",
                    error = e.message
                )
            }
        }
    }
}
