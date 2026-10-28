#include "common/api.h"

#include <c10/cuda/CUDAGuard.h>

#include "dispatch/kernel_registry.h"
#include "types.h"

void attention(
    torch::Tensor query, torch::Tensor key, torch::Tensor value,
    torch::Tensor kv_cache, torch::Tensor output, torch::Tensor query_start_loc,
    torch::Tensor seq_lens, torch::Tensor block_table,
    std::optional<torch::Tensor> key_rotation,
    std::optional<torch::Tensor> value_rotation,
    std::optional<torch::Tensor> key_mean,
    const std::vector<std::vector<int>>& records,
    const std::string& reference_mode, int anchor_index, int loop_index
) {
    validate_bf16_token_tensor(query);
    const int heads = query.size(1), head_dim = query.size(2);

    for (const torch::Tensor& tensor : {key, value, output}) {
        validate_bf16_token_tensor(tensor);
        TORCH_CHECK(
            tensor.device() == query.device() &&
                tensor.sizes() == query.sizes(),
            "Q/K/V/output must have matching shapes and devices"
        );
    }

    for (const torch::Tensor& tensor : {query, key}) {
        TORCH_CHECK(
            reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 8 == 0 &&
                tensor.stride(0) % 4 == 0,
            "Q/K addresses and token strides must be 8-byte aligned"
        );
    }

    TORCH_CHECK(
        kv_cache.device() == query.device() &&
            kv_cache.scalar_type() == at::kByte && kv_cache.is_contiguous() &&
            kv_cache.dim() == 5 && kv_cache.size(0) == 2 &&
            kv_cache.size(2) == heads,
        "Contiguous UINT8 KV cache on the query device required"
    );

    if (key_mean)
        TORCH_CHECK(
            key_mean->device() == query.device() &&
                key_mean->scalar_type() == at::kFloat &&
                key_mean->is_contiguous() &&
                reinterpret_cast<uintptr_t>(key_mean->data_ptr()) % 16 == 0 &&
                key_mean->numel() == records.size() * heads * head_dim,
            "16-byte aligned contiguous FP32 key mean with matching dimensions "
            "required"
        );

    TORCH_CHECK(
        query_start_loc.device() == query.device() &&
            query_start_loc.scalar_type() == at::kInt &&
            query_start_loc.dim() == 1 && query_start_loc.is_contiguous() &&
            query_start_loc.numel() == seq_lens.numel() + 1,
        "Contiguous INT32 query starts with one final boundary required"
    );

    TORCH_CHECK(
        seq_lens.device() == query.device() &&
            seq_lens.scalar_type() == at::kInt && seq_lens.dim() == 1 &&
            seq_lens.is_contiguous(),
        "Contiguous INT32 sequence lengths required"
    );

    TORCH_CHECK(
        block_table.device() == query.device() &&
            block_table.scalar_type() == at::kInt && block_table.dim() == 2 &&
            block_table.stride(1) == 1 &&
            block_table.size(0) >= seq_lens.numel(),
        "INT32 page table with packed rows required"
    );

    LiveAttention live{reinterpret_cast<const __nv_bfloat16*>(query.data_ptr()),
                       reinterpret_cast<const __nv_bfloat16*>(key.data_ptr()),
                       reinterpret_cast<const __nv_bfloat16*>(value.data_ptr()),
                       reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                       validate_and_get_rotation_pointer(
                           value_rotation, query.device(), heads, head_dim
                       ),
                       validate_and_get_rotation_pointer(
                           key_rotation, query.device(), heads, head_dim
                       ),
                       key_mean ? key_mean->data_ptr<float>() : nullptr,
                       query.stride(0),
                       key.stride(0),
                       value.stride(0),
                       output.stride(0)};

    TORCH_CHECK(
        !live.key_rotation ||
            reinterpret_cast<uintptr_t>(live.key_rotation) % 16 == 0,
        "16-byte aligned key rotation required for asynchronous copies"
    );

    TORCH_CHECK(
        loop_index >= 0 && loop_index < records.size(), "Invalid loop index"
    );

    const KernelLaunchers kernel = select_kernels(
        heads, head_dim, records, reference_mode, anchor_index,
        key_mean.has_value(), kv_cache.size(3)
    );
    c10::cuda::CUDAGuard guard(query.device());
    kernel.attention(
        query, kv_cache, query_start_loc, seq_lens, block_table, loop_index,
        live
    );
}
