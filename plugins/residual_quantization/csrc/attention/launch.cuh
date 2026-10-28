#pragma once

#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/types.h>

#include <map>
#include <utility>

#include "compute.cuh"
#include "merge.cuh"
#include "prepare.cuh"

// Expand only the loop indices present in this record schedule.
template <int Heads, int HeadDim, typename Options, typename LoopIndices>
struct AttentionLoopTable;

template <int Heads, int HeadDim, typename Options, size_t... LoopIndices>
struct AttentionLoopTable<
    Heads, HeadDim, Options, std::index_sequence<LoopIndices...>> {
    using PrepareKernel =
        decltype(&attention_prepare_kernel<Heads, HeadDim, Options, 0>);
    static constexpr PrepareKernel PrepareKernels[] = {
        attention_prepare_kernel<Heads, HeadDim, Options, LoopIndices>...};
    using Kernel = decltype(&attention_kernel<Heads, HeadDim, Options, 0>);
    static constexpr Kernel Kernels[] = {
        attention_kernel<Heads, HeadDim, Options, LoopIndices>...};
};

// Record layout and loop index remain compile-time specializations.
// The registry selects the layout; this launcher selects the current loop.
template <int Heads, int HeadDim, typename Options>
void launch_attention(
    torch::Tensor query, torch::Tensor kv_cache, torch::Tensor query_start_loc,
    torch::Tensor seq_lens, torch::Tensor block_table, int loop_index,
    LiveAttention live
) {
    using Layout = PackedLayout<HeadDim, Options>;
    TORCH_CHECK(
        kv_cache.size(4) == Layout::PageWidth(),
        "Packed cache width differs from records"
    );
    const int query_token_count = query.size(0);
    if (query_token_count == 0) return;

    using LoopTable = AttentionLoopTable<
        Heads, HeadDim, Options,
        std::make_index_sequence<Options::Records::Count>>;
    constexpr int Threads = Options::PageTokens * 8;
    // Cache occupancy for this device and concrete loop kernel. Static shared
    // memory and registers constrain residency; dynamic allocation is zero.
    static thread_local std::map<std::pair<int, int>, int> concurrency;
    const std::pair<int, int> device_loop{kv_cache.get_device(), loop_index};
    std::map<std::pair<int, int>, int>::iterator found =
        concurrency.find(device_loop);
    if (found == concurrency.end()) {
        int blocks_per_sm = 0;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, LoopTable::Kernels[loop_index], Threads,
            0
        ));
        const int concurrent_blocks =
            blocks_per_sm * at::cuda::getDeviceProperties(kv_cache.get_device())
                                ->multiProcessorCount;
        found = concurrency.emplace(device_loop, concurrent_blocks).first;
    }
    // Reserve four additional resident waves, shared by all requests. Small
    // batches remain bounded by the existing sixteen-partition limit.
    const int spare_work = std::min<int64_t>(
        int64_t(MaxAttentionPartitions - 1) * query_token_count,
        (4 * found->second + Heads - 1) / Heads
    );
    const int max_partitions = std::min<int64_t>(
        MaxAttentionPartitions, std::max<int64_t>(1, block_table.size(1))
    );
    const int work_capacity = std::min<int64_t>(
        int64_t(query_token_count) * max_partitions,
        int64_t(query_token_count) + spare_work
    );

    // Store maxima, denominators, then partition-major V sums per row/head.
    // Reserve the maximum partitions; device lengths choose how many are
    // active; allocation and grid stay unchanged throughout CUDA graph replay.
    torch::Tensor partition_results = torch::empty(
        {query_token_count, Heads, MaxAttentionPartitions * (HeadDim + 2)},
        query.options().dtype(at::kFloat)
    );
    torch::Tensor live_scores = torch::empty(
        {query_token_count, Heads}, query.options().dtype(at::kFloat)
    );
    torch::Tensor prepared_query = torch::empty(
        {query_token_count, Heads, HeadDim}, query.options().dtype(at::kFloat)
    );
    torch::Tensor partition_counts =
        torch::empty({query_token_count}, query_start_loc.options());
    torch::Tensor work_list =
        torch::empty({work_capacity, 8}, query_start_loc.options());
    LoopTable::PrepareKernels[loop_index]<<<
        1 + ((query_token_count + PrepareQueryRows - 1) / PrepareQueryRows) * Heads,
        PrepareThreads,
        2 * 32 * (HeadDim + PrepareMatrixPadding) * sizeof(__half),
        c10::cuda::getCurrentCUDAStream()>>>(
        live, prepared_query.data_ptr<float>(), live_scores.data_ptr<float>(),
        query_start_loc.data_ptr<int>(), seq_lens.data_ptr<int>(),
        partition_counts.data_ptr<int>(),
        reinterpret_cast<AttentionWork*>(work_list.data_ptr<int>()),
        seq_lens.numel(), work_capacity, found->second / (2 * Heads)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    LoopTable::Kernels[loop_index]<<<
        Heads * work_capacity, Threads, 0,
        c10::cuda::getCurrentCUDAStream()>>>(
        live, reinterpret_cast<const unsigned char*>(kv_cache.data_ptr()),
        partition_results.data_ptr<float>(), block_table.data_ptr<int>(),
        reinterpret_cast<const AttentionWork*>(work_list.data_ptr<int>()),
        prepared_query.data_ptr<float>(), live_scores.data_ptr<float>(),
        block_table.stride(0), kv_cache.stride(0)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    attention_merge_kernel<Heads, HeadDim>
        <<<dim3(query_token_count, Heads), 128, 0,
           c10::cuda::getCurrentCUDAStream()>>>(
            live, partition_results.data_ptr<float>(),
            live_scores.data_ptr<float>(),
            query_start_loc.data_ptr<int>() + seq_lens.numel(),
            partition_counts.data_ptr<int>()
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
