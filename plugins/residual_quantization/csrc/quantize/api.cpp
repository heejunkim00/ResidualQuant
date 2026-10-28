#include "common/api.h"

#include <c10/cuda/CUDAGuard.h>

#include "dispatch/kernel_registry.h"
#include "types.h"

void quantize(
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& key_value_pairs,
    torch::Tensor kv_cache, torch::Tensor slot_mapping,
    std::optional<torch::Tensor> key_rotation,
    std::optional<torch::Tensor> value_rotation,
    std::optional<torch::Tensor> key_mean,
    const std::vector<std::vector<int>>& records,
    const std::string& reference_mode, int anchor_index,
    const std::string& coefficient_method
) {
    TORCH_CHECK(
        !records.empty() && key_value_pairs.size() == records.size(),
        "A nonempty group with one K/V input per record required"
    );

    TORCH_CHECK(
        kv_cache.is_cuda() && kv_cache.scalar_type() == at::kByte &&
            kv_cache.is_contiguous() && kv_cache.dim() == 5 &&
            kv_cache.size(0) == 2,
        "Contiguous CUDA UINT8 [2, blocks, heads, page tokens, page width] "
        "cache required"
    );

    const int heads = kv_cache.size(2);
    // Cache width is packed bytes; read the logical channel count from K once.
    const int head_dim = key_value_pairs.front().first.size(-1);

    TORCH_CHECK(
        slot_mapping.device() == kv_cache.device() &&
            slot_mapping.scalar_type() == at::kLong &&
            slot_mapping.dim() == 1 && slot_mapping.is_contiguous(),
        "Contiguous INT64 slot mapping on the cache device required"
    );

    for (const auto& pair : key_value_pairs) {
        for (const torch::Tensor& tensor : {pair.first, pair.second}) {
            validate_bf16_token_tensor(tensor);
            TORCH_CHECK(
                tensor.device() == kv_cache.device() &&
                    tensor.size(1) == heads && tensor.size(2) == head_dim,
                "K/V must match the cache device/head count and logical "
                "channel count"
            );
            TORCH_CHECK(
                tensor.size(0) >= slot_mapping.numel(),
                "Input has fewer rows than slot mapping"
            );
        }
    }

    if (key_mean)
        TORCH_CHECK(
            key_mean->device() == kv_cache.device() &&
                key_mean->scalar_type() == at::kFloat &&
                key_mean->is_contiguous() && key_mean->dim() == 3 &&
                key_mean->size(0) == records.size() &&
                key_mean->size(1) == heads && key_mean->size(2) == head_dim,
            "Contiguous FP32 [records, heads, channels] key mean required"
        );

    const __half* key_rotation_ptr = validate_and_get_rotation_pointer(
        key_rotation, kv_cache.device(), heads, head_dim
    );
    const __half* value_rotation_ptr = validate_and_get_rotation_pointer(
        value_rotation, kv_cache.device(), heads, head_dim
    );
    const float* key_mean_ptr =
        key_mean ? key_mean->data_ptr<float>() : nullptr;

    TORCH_CHECK(
        coefficient_method == "unit" || coefficient_method == "norm" ||
            coefficient_method == "ls",
        "Unsupported coefficient"
    );
    const CoefficientMethod coefficient =
        coefficient_method == "unit"   ? CoefficientMethod::Unit
        : coefficient_method == "norm" ? CoefficientMethod::Norm
                                       : CoefficientMethod::LeastSquares;

    c10::cuda::CUDAGuard guard(kv_cache.device());
    const KernelLaunchers kernels = select_kernels(
        heads, head_dim, records, reference_mode, anchor_index,
        key_mean.has_value(), kv_cache.size(3)
    );
    kernels.quantize[static_cast<int>(coefficient)](
        key_value_pairs, kv_cache, slot_mapping, key_rotation_ptr, value_rotation_ptr, key_mean_ptr
    );
}
