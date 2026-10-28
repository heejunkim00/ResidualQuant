#pragma once

#include <cuda_fp16.h>
#include <torch/types.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

void attention(
    torch::Tensor query, torch::Tensor key, torch::Tensor value,
    torch::Tensor kv_cache, torch::Tensor output, torch::Tensor query_start_loc,
    torch::Tensor seq_lens, torch::Tensor block_table,
    std::optional<torch::Tensor> key_rotation,
    std::optional<torch::Tensor> value_rotation,
    std::optional<torch::Tensor> key_mean,
    const std::vector<std::vector<int>>& records,
    const std::string& reference_mode, int anchor_index, int loop_index
);

void quantize(
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& key_value_pairs,
    torch::Tensor kv_cache, torch::Tensor slot_mapping,
    std::optional<torch::Tensor> key_rotation,
    std::optional<torch::Tensor> value_rotation,
    std::optional<torch::Tensor> key_mean,
    const std::vector<std::vector<int>>& records,
    const std::string& reference_mode, int anchor_index,
    const std::string& coefficient_method
);

inline void validate_bf16_token_tensor(const torch::Tensor& tensor) {
    TORCH_CHECK(
        tensor.is_cuda() && tensor.scalar_type() == at::kBFloat16 &&
            tensor.dim() == 3 && tensor.stride(1) == tensor.size(2) &&
            tensor.stride(2) == 1,
        "BF16 [tokens, heads, channels] with packed trailing strides required"
    );
}

inline const __half* validate_and_get_rotation_pointer(
    const std::optional<torch::Tensor>& rotation, c10::Device device, int heads,
    int head_dim
) {
    if (!rotation) return nullptr;
    TORCH_CHECK(
        rotation->device() == device && rotation->is_contiguous() &&
            rotation->scalar_type() == at::kHalf && rotation->dim() == 3 &&
            rotation->size(0) == heads && rotation->size(1) == head_dim &&
            rotation->size(2) == head_dim,
        "Contiguous FP16 [heads, channels, channels] rotation required"
    );
    return reinterpret_cast<const __half*>(rotation->data_ptr<at::Half>());
}
