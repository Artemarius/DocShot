#include <jni.h>
#include "directional_gradient.h"

extern "C" JNIEXPORT void JNICALL
Java_com_docshot_cv_NativeAccel_nativeDirectionalGradient(
    JNIEnv* env, jclass /* clazz */,
    jbyteArray j_gy, jbyteArray j_gx,
    jbyteArray j_result,
    jint rows, jint cols,
    jintArray j_h_offsets, jintArray j_v_offsets,
    jint num_angles, jint kernel_length,
    jint margin_y, jint margin_x,
    jfloat threshold_percentile
) {
    // Pin arrays for zero-copy access (no GC during this ~5ms call)
    auto* gy_data = static_cast<uint8_t*>(
        env->GetPrimitiveArrayCritical(j_gy, nullptr));
    auto* gx_data = static_cast<uint8_t*>(
        env->GetPrimitiveArrayCritical(j_gx, nullptr));
    auto* result_data = static_cast<uint8_t*>(
        env->GetPrimitiveArrayCritical(j_result, nullptr));
    auto* h_offsets = static_cast<int32_t*>(
        env->GetPrimitiveArrayCritical(j_h_offsets, nullptr));
    auto* v_offsets = static_cast<int32_t*>(
        env->GetPrimitiveArrayCritical(j_v_offsets, nullptr));

    if (gy_data && gx_data && result_data && h_offsets && v_offsets) {
        directional_gradient_accumulate(
            gy_data, gx_data, result_data,
            rows, cols,
            h_offsets, v_offsets,
            num_angles, kernel_length,
            margin_y, margin_x,
            threshold_percentile
        );
    }

    // Release in reverse order — mode 0 copies back changes
    if (v_offsets) env->ReleasePrimitiveArrayCritical(j_v_offsets, v_offsets, JNI_ABORT);
    if (h_offsets) env->ReleasePrimitiveArrayCritical(j_h_offsets, h_offsets, JNI_ABORT);
    if (result_data) env->ReleasePrimitiveArrayCritical(j_result, result_data, 0);
    if (gx_data) env->ReleasePrimitiveArrayCritical(j_gx, gx_data, JNI_ABORT);
    if (gy_data) env->ReleasePrimitiveArrayCritical(j_gy, gy_data, JNI_ABORT);
}
