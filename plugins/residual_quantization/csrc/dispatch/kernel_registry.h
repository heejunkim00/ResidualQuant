#pragma once

#include <cuda_fp16.h>
#include <torch/types.h>

#include <array>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "common/types.h"

// Key: heads, head dimension, records, anchor, reference, mean, page tokens.
using KernelKey = std::tuple<
    int, int, std::vector<std::vector<int>>, int, ReferenceMode, bool, int>;

struct LiveAttention;

using AttentionLauncher = void (*)(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int, LiveAttention
);

using QuantizeLauncher = void (*)(
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>&, torch::Tensor,
    torch::Tensor, const __half*, const __half*, const float*
);

struct KernelLaunchers {
    AttentionLauncher attention;
    std::array<QuantizeLauncher, 3> quantize;
};

inline std::map<KernelKey, KernelLaunchers>& registered_kernels() {
    static std::map<KernelKey, KernelLaunchers> implementations;
    return implementations;
}

inline void register_kernels(KernelKey key, KernelLaunchers kernels) {
    TORCH_CHECK(
        registered_kernels().emplace(std::move(key), kernels).second,
        "Duplicate packed CUDA implementation"
    );
}

inline KernelLaunchers select_kernels(
    int heads, int head_dim, const std::vector<std::vector<int>>& records,
    const std::string& reference_mode, int anchor_index, bool has_mean,
    int page_tokens
) {
    TORCH_CHECK(
        reference_mode == "anchor" || reference_mode == "adjacent",
        "Unknown packed reference mode: ", reference_mode
    );

    const ReferenceMode mode = reference_mode == "anchor"
                                   ? ReferenceMode::Anchor
                                   : ReferenceMode::Adjacent;
    KernelKey key{heads, head_dim, records,    anchor_index,
                  mode,  has_mean, page_tokens};

    auto implementation = registered_kernels().find(key);

    TORCH_CHECK(
        implementation != registered_kernels().end(),
        "No compiled packed CUDA kernel for heads=", heads,
        ", head_dim=", head_dim, ", record count=", records.size(),
        ", reference_mode=", reference_mode, ", anchor_index=", anchor_index,
        ", has_mean=", has_mean, ", page_tokens=", page_tokens
    );

    return implementation->second;
}
