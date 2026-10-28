#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

inline constexpr int MaxAttentionPartitions = 16;
inline constexpr int PrepareThreads = 256;

// One entry is shared by every head. An invalid row marks an unused grid slot.
// KV bounds are token positions and always start at a page boundary.
struct alignas(16) AttentionWork {
    int row, request, part, partitions;
    int begin, stop, padding[2];
};
static_assert(sizeof(AttentionWork) == 8 * sizeof(int));

// C++17 has no constexpr sqrt. Evaluate Newton's method at compile time,
// then round the inverse square root once to FP32.
template <int HeadDim>
inline constexpr float AttentionScale = [] {
    static_assert(HeadDim > 0);
    double root = HeadDim;
    for (int iteration = 0; iteration < 32; ++iteration)
        root = 0.5 * (root + HeadDim / root);
    return static_cast<float>(1.0 / root);
}();

// Validated pointers and row strides passed by value to attention kernels.
struct LiveAttention {
    const __nv_bfloat16* query;
    const __nv_bfloat16* key;
    const __nv_bfloat16* value;
    __nv_bfloat16* output;
    const __half* value_rotation;
    const __half* key_rotation;
    const float* key_mean;
    int64_t query_stride, key_stride, value_stride, output_stride;
};
