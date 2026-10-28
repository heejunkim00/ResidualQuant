#pragma once

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/types.h>

#include <utility>
#include <vector>

#include "kernel.cuh"

// Select 1, 2, 4 or 8 tokens per block from the input token count.
inline int quantize_tile_tokens(int64_t tokens) {
    if (tokens >= 256) return 8;
    if (tokens >= 32) return 4;
    if (tokens >= 16) return 2;
    return 1;
}

template <
    int Heads, int HeadDim, typename Layout, CoefficientMethod Coefficient,
    int Tokens>
void launch_quantize_tile(
    quantization::QuantizeInputs<Layout::RecordCount> inputs,
    torch::Tensor kv_cache, torch::Tensor slot_mapping,
    const __half* key_rotation, const __half* value_rotation,
    const float* key_mean
) {
    constexpr int ScratchBytes =
        Layout::RecordCount * Tokens * quantization::RotationStride * sizeof(float);
    quantization::quantize_kernel<Heads, HeadDim, Layout, Coefficient, Tokens>
        <<<dim3((slot_mapping.numel() + Tokens - 1) / Tokens, Heads, 2),
           128, ScratchBytes, c10::cuda::getCurrentCUDAStream()>>>(
            inputs, kv_cache.data_ptr<unsigned char>(),
            slot_mapping.data_ptr<int64_t>(), key_rotation, value_rotation,
            key_mean, kv_cache.stride(0), slot_mapping.numel()
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Layout fixes the record count, reference mode and anchor at compile time.
// A single-record group writes only its Base.
template <
    int Heads, int HeadDim, typename Layout, CoefficientMethod Coefficient>
void launch_quantize(
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& key_value_pairs,
    torch::Tensor kv_cache, torch::Tensor slot_mapping,
    const __half* key_rotation, const __half* value_rotation,
    const float* key_mean
) {
    TORCH_CHECK(
        kv_cache.size(4) == Layout::PageWidth(),
        "Packed cache width differs from records"
    );
    if (slot_mapping.numel() == 0) return;

    quantization::QuantizeInputs<Layout::RecordCount> inputs{};
    for (int loop = 0; loop < Layout::RecordCount; ++loop) {
        for (int plane = 0; plane < 2; ++plane) {
            const torch::Tensor& tensor = plane == 0
                                              ? key_value_pairs[loop].first
                                              : key_value_pairs[loop].second;
            inputs.pointers[plane][loop] =
                reinterpret_cast<const __nv_bfloat16*>(tensor.data_ptr());
            inputs.strides[plane][loop] = tensor.stride(0);
        }
    }
    // Neither dispatch nor grid depends on device lengths or slot contents.
    switch (quantize_tile_tokens(slot_mapping.numel())) {
        case 1:
            launch_quantize_tile<Heads, HeadDim, Layout, Coefficient, 1>(
                inputs, kv_cache, slot_mapping, key_rotation, value_rotation, key_mean);
            break;
        case 2:
            launch_quantize_tile<Heads, HeadDim, Layout, Coefficient, 2>(
                inputs, kv_cache, slot_mapping, key_rotation, value_rotation, key_mean);
            break;
        case 4:
            launch_quantize_tile<Heads, HeadDim, Layout, Coefficient, 4>(
                inputs, kv_cache, slot_mapping, key_rotation, value_rotation, key_mean);
            break;
        default:
            launch_quantize_tile<Heads, HeadDim, Layout, Coefficient, 8>(
                inputs, kv_cache, slot_mapping, key_rotation, value_rotation, key_mean);
    }
}
