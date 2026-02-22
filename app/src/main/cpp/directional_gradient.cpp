#include "directional_gradient.h"
#include <cstdlib>
#include <cstring>
#include <algorithm>

void directional_gradient_accumulate(
    const uint8_t* gy_data,
    const uint8_t* gx_data,
    uint8_t* result_data,
    int rows, int cols,
    const int32_t* h_offsets,
    const int32_t* v_offsets,
    int num_angles,
    int kernel_length,
    int margin_y, int margin_x,
    float threshold_percentile
) {
    const int total_pixels = rows * cols;

    // Allocate accumulation buffers (heap — up to ~960KB for 400x300 * 2 * 4 bytes)
    int32_t* h_response = static_cast<int32_t*>(calloc(total_pixels, sizeof(int32_t)));
    int32_t* v_response = static_cast<int32_t*>(calloc(total_pixels, sizeof(int32_t)));

    if (!h_response || !v_response) {
        free(h_response);
        free(v_response);
        memset(result_data, 0, total_pixels);
        return;
    }

    // Step 4: 5-angle accumulation — track per-pixel max across angles.
    // The inner k-loop (21 iterations) auto-vectorizes well on ARM64.
    for (int a = 0; a < num_angles; a++) {
        const int32_t* h_off = h_offsets + a * kernel_length;
        const int32_t* v_off = v_offsets + a * kernel_length;

        for (int y = margin_y; y < rows - margin_y; y++) {
            const int row_base = y * cols;
            for (int x = margin_x; x < cols - margin_x; x++) {
                const int base_idx = row_base + x;
                int32_t sum_h = 0;
                int32_t sum_v = 0;

                for (int k = 0; k < kernel_length; k++) {
                    sum_h += static_cast<uint32_t>(gy_data[base_idx + h_off[k]]);
                    sum_v += static_cast<uint32_t>(gx_data[base_idx + v_off[k]]);
                }

                if (sum_h > h_response[base_idx]) h_response[base_idx] = sum_h;
                if (sum_v > v_response[base_idx]) v_response[base_idx] = sum_v;
            }
        }
    }

    // Step 5: Combine H and V, find global max
    int32_t global_max = 1;
    for (int i = 0; i < total_pixels; i++) {
        int32_t combined = h_response[i] > v_response[i] ? h_response[i] : v_response[i];
        h_response[i] = combined;
        if (combined > global_max) global_max = combined;
    }

    // Normalize to 0-255 and build histogram in one pass
    int histogram[256];
    memset(histogram, 0, sizeof(histogram));

    for (int i = 0; i < total_pixels; i++) {
        int normalized = static_cast<int>(
            static_cast<int64_t>(h_response[i]) * 255 / global_max
        );
        if (normalized < 0) normalized = 0;
        if (normalized > 255) normalized = 255;
        result_data[i] = static_cast<uint8_t>(normalized);
        histogram[normalized]++;
    }

    free(h_response);
    free(v_response);

    // Step 6: Compute threshold from percentile via histogram
    const int target = static_cast<int>(total_pixels * threshold_percentile);
    int cum_sum = 0;
    int threshold_val = 255;
    for (int i = 0; i < 256; i++) {
        cum_sum += histogram[i];
        if (cum_sum >= target) {
            threshold_val = i;
            break;
        }
    }

    // Apply binary threshold in-place
    for (int i = 0; i < total_pixels; i++) {
        result_data[i] = (result_data[i] > threshold_val) ? 255 : 0;
    }
}
