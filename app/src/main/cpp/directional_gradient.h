#ifndef DOCSHOT_DIRECTIONAL_GRADIENT_H
#define DOCSHOT_DIRECTIONAL_GRADIENT_H

#include <cstdint>

/**
 * Steps 4-6 of the DIRECTIONAL_GRADIENT preprocessing strategy:
 * accumulate directional responses, normalize, and threshold to binary.
 *
 * @param gy_data     |Gy| gradient image, rows*cols bytes (CV_8UC1)
 * @param gx_data     |Gx| gradient image, rows*cols bytes (CV_8UC1)
 * @param result_data Output: normalized + thresholded binary, rows*cols bytes (pre-allocated by caller)
 * @param rows        Image height
 * @param cols        Image width
 * @param h_offsets   Flat array of linear offsets for horizontal edge accumulation (num_angles * kernel_length)
 * @param v_offsets   Flat array of linear offsets for vertical edge accumulation (num_angles * kernel_length)
 * @param num_angles  Number of tilt angles (5)
 * @param kernel_length Length of 1D smoothing kernel (21)
 * @param margin_y    Vertical margin to skip (avoids out-of-bounds offset access)
 * @param margin_x    Horizontal margin to skip
 * @param threshold_percentile Percentile for binary threshold (0.90 = 90th percentile)
 */
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
);

#endif // DOCSHOT_DIRECTIONAL_GRADIENT_H
