#pragma once


#include "rotation.cuh"

// Final history is in FP32. Inverse rotation uses TF32 products with FP32
// accumulation; the original live V is added afterwards, then stored as BF16.
// Shared: scratch (read/write history vector, reused for inverse rotation).
template <int HeadDim, int Threads>
__device__ __forceinline__ void write_attention_output(
    const __nv_bfloat16* value, __nv_bfloat16* output, const __half* rotation,
    int row, int head, int64_t value_stride, int64_t output_stride,
    float history, float live_weight, float* scratch
) {
    const int channel = threadIdx.x;
    if (rotation) {
        if (channel < HeadDim) scratch[channel] = history;
        __syncthreads();
        inverse_rotate_vector<HeadDim, Threads>(
            scratch, rotation + head * (HeadDim * HeadDim), scratch
        );
        if (channel < HeadDim) history = scratch[channel];
    }
    if (channel < HeadDim) {
        float current = __bfloat162float(
            value[row * value_stride + head * HeadDim + channel]
        );
        output[row * output_stride + head * HeadDim + channel] =
            __float2bfloat16_rn(history + current * live_weight);
    }
}
