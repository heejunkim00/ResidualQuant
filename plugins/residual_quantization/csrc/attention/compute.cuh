#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "common/cache.cuh"
#include "async_copy.cuh"
#include "output.cuh"
#include "types.h"

// Four contiguous channels, loaded once and decoded in the consuming thread.
template <int Bits>
struct PackedCodeBlock {
    static_assert(Bits == 2 || Bits == 4);
    PackedCodeStorage<Bits> codes;

    __device__ __forceinline__ __nv_bfloat16 element(int index) const {
        constexpr unsigned Mask = (1U << Bits) - 1;
        const unsigned code = (unsigned(codes) >> (index * Bits)) & Mask;
        return __ushort2bfloat16_rn(static_cast<unsigned short>(code));
    }
};

// Load Base/Residual codes and metadata once per four-channel chunk.
// Both BF16 pairs reuse those values and the Residual coefficient.
// Shared: page (read). values refers to a thread-local output array;
// it is not shared/global storage (register placement is compiler-dependent).
template <int HeadDim, typename Options, int LoopIndex>
__device__ __forceinline__ void restore_anchor_values(
    const unsigned char* page, int token, int channel,
    __nv_bfloat162 (&values)[2]
) {
    static_assert(Options::ReferenceMode == ReferenceMode::Anchor);
    using Layout = PackedLayout<HeadDim, Options>;
    constexpr QuantizationRecord Base =
        Layout::template Record<Options::AnchorIndex>();
    using BaseStorage = PackedCodeStorage<Base.bits>;
    const PackedCodeBlock<Base.bits> base_codes{
        reinterpret_cast<const BaseStorage*>(
            page + token * (HeadDim * Base.bits / 8)
        )[channel / 4]};
    const __nv_fp8_e4m3* base_parameters =
        reinterpret_cast<const __nv_fp8_e4m3*>(
            page + Options::PageTokens * (HeadDim * Base.bits / 8)
        );
    const int base_metadata = token * (HeadDim / Base.quantization_group_size) +
                              channel / Base.quantization_group_size;
    const __nv_bfloat16 base_scale =
        static_cast<__nv_bfloat16>(base_parameters[base_metadata]);
    const __nv_bfloat16 base_offset = static_cast<__nv_bfloat16>(
        base_parameters
            [Options::PageTokens * (HeadDim / Base.quantization_group_size) +
             base_metadata]
    );
#pragma unroll
    for (int pair = 0; pair < 2; ++pair)
        values[pair] = __hfma2(
            __halves2bfloat162(
                base_codes.element(2 * pair), base_codes.element(2 * pair + 1)
            ),
            __bfloat162bfloat162(base_scale), __bfloat162bfloat162(base_offset)
        );
    if constexpr (LoopIndex != Options::AnchorIndex) {
        constexpr QuantizationRecord Residual =
            Layout::template Record<LoopIndex>();
        const unsigned char* record =
            page + StagedAnchorBytes<HeadDim, Options>();
        using ResidualStorage = PackedCodeStorage<Residual.bits>;
        const PackedCodeBlock<Residual.bits> residual_codes{
            reinterpret_cast<const ResidualStorage*>(
                record + token * (HeadDim * Residual.bits / 8)
            )[channel / 4]};
        const __nv_fp8_e4m3* residual_parameters =
            reinterpret_cast<const __nv_fp8_e4m3*>(
                record + Options::PageTokens * (HeadDim * Residual.bits / 8)
            );
        const int residual_metadata =
            token * (HeadDim / Residual.quantization_group_size) +
            channel / Residual.quantization_group_size;
        const __nv_bfloat16 residual_scale =
            static_cast<__nv_bfloat16>(residual_parameters[residual_metadata]);
        const __nv_bfloat16 residual_offset = static_cast<__nv_bfloat16>(
            residual_parameters
                [Options::PageTokens *
                     (HeadDim / Residual.quantization_group_size) +
                 residual_metadata]
        );
        constexpr int CodeBytes = HeadDim * Residual.bits / 8;
        constexpr int Groups = HeadDim / Residual.quantization_group_size;
        const __nv_bfloat16 alpha = *reinterpret_cast<const __nv_bfloat16*>(
            record + Options::PageTokens * (CodeBytes + 2 * Groups) + token * 2
        );
#pragma unroll
        for (int pair = 0; pair < 2; ++pair)
            values[pair] = __hfma2(
                __bfloat162bfloat162(alpha), values[pair],
                __hfma2(
                    __halves2bfloat162(
                        residual_codes.element(2 * pair),
                        residual_codes.element(2 * pair + 1)
                    ),
                    __bfloat162bfloat162(residual_scale),
                    __bfloat162bfloat162(residual_offset)
                )
            );
    }
}

// Sum eight lanes in the fixed BF16 shuffle order 1,4,2.
__device__ __forceinline__ __nv_bfloat16 sum_eight_lanes(__nv_bfloat16 value) {
    value = __hadd(value, __shfl_xor_sync(0xffffffff, value, 1));
    value = __hadd(value, __shfl_xor_sync(0xffffffff, value, 4));
    return __hadd(value, __shfl_xor_sync(0xffffffff, value, 2));
}

// Adjacent restoration includes earlier K means. The current loop's mean
// is omitted here and subtracted from the live score by the prepare kernel.
// Shared: page and key_mean (read). key_mean may be null for V or
// when mean correction is disabled.
template <
    int HeadDim, typename Options, typename Value, int LoopIndex,
    int RecordIndex = LoopIndex>
__device__ __forceinline__ Value restore_adjacent(
    const unsigned char* page, const __nv_bfloat16* key_mean, int token,
    int dimension
) {
    static_assert(Options::ReferenceMode == ReferenceMode::Adjacent);
    using Layout = PackedLayout<HeadDim, Options>;
    constexpr QuantizationRecord Record =
        Layout::template Record<RecordIndex>();
    constexpr int CodeBytes = HeadDim * Record.bits / 8;
    constexpr int Groups = HeadDim / Record.quantization_group_size;
    const unsigned char* record =
        page +
        Layout::template StorageOffset<RecordIndex>() * Options::PageTokens;
    using RecordStorage = PackedCodeStorage<Record.bits>;
    const PackedCodeBlock<Record.bits> record_codes{
        reinterpret_cast<const RecordStorage*>(
            record + token * (HeadDim * Record.bits / 8)
        )[dimension / 4]};
    const __nv_fp8_e4m3* record_parameters =
        reinterpret_cast<const __nv_fp8_e4m3*>(
            record + Options::PageTokens * (HeadDim * Record.bits / 8)
        );
    const int record_metadata =
        token * (HeadDim / Record.quantization_group_size) +
        dimension / Record.quantization_group_size;
    const __nv_bfloat16 record_scale =
        static_cast<__nv_bfloat16>(record_parameters[record_metadata]);
    const __nv_bfloat16 record_offset = static_cast<__nv_bfloat16>(
        record_parameters
            [Options::PageTokens * (HeadDim / Record.quantization_group_size) +
             record_metadata]
    );
    Value value;
    if constexpr (std::is_same_v<Value, __nv_bfloat162>)
        value = __hfma2(
            __halves2bfloat162(
                record_codes.element(dimension % 4),
                record_codes.element(dimension % 4 + 1)
            ),
            __bfloat162bfloat162(record_scale),
            __bfloat162bfloat162(record_offset)
        );
    else
        value = __hfma(
            record_codes.element(dimension % 4), record_scale, record_offset
        );
    if constexpr (Options::HasMean && RecordIndex != LoopIndex) {
        if (key_mean) {
            if constexpr (std::is_same_v<Value, __nv_bfloat162>)
                value = __hadd2(
                    value, *reinterpret_cast<const __nv_bfloat162*>(
                               key_mean + RecordIndex * HeadDim + dimension
                           )
                );
            else
                value =
                    __hadd(value, key_mean[RecordIndex * HeadDim + dimension]);
        }
    }
    if constexpr (RecordIndex == Options::AnchorIndex) {
        return value;
    } else {
        const Value previous = restore_adjacent<
            HeadDim, Options, Value, LoopIndex, RecordIndex - 1>(
            page, key_mean, token, dimension
        );
        __nv_bfloat16 alpha = *reinterpret_cast<const __nv_bfloat16*>(
            record + Options::PageTokens * (CodeBytes + 2 * Groups) + token * 2
        );
        if constexpr (std::is_same_v<Value, __nv_bfloat162>)
            return __hfma2(previous, __bfloat162bfloat162(alpha), value);
        else
            return __hfma(previous, alpha, value);
    }
}

// Compute the history exponential once, keeping multiply-then-divide order.
// A separate V inverse rotation always finalizes in merge, even for one part.
template <int HeadDim, int Threads>
__device__ __forceinline__ void write_single_partition(
    LiveAttention live, float live_score, int row, int head,
    float history_maximum, float history_denominator, float accumulated_value
) {
    __shared__ float history_scale, denominator, live_weight;
    if (threadIdx.x == 0) {
        const float maximum = fmaxf(history_maximum, live_score);
        const float current = expf(live_score - maximum);
        history_scale =
            history_denominator > 0.f ? expf(history_maximum - maximum) : 0.f;
        denominator = history_denominator * history_scale + current;
        live_weight = current / denominator;
    }
    __syncthreads();
    const float history = history_denominator > 0.f
                              ? accumulated_value * history_scale / denominator
                              : 0.f;
    write_attention_output<HeadDim, Threads>(
        live.value, live.output, nullptr, row, head,
        live.value_stride, live.output_stride, history, live_weight, nullptr
    );
}

// CTA size follows page size; no occupancy launch bound is imposed.
template <int Heads, int HeadDim, typename Options, int LoopIndex>
__global__ void attention_kernel(
    LiveAttention live, const unsigned char* kv_cache, float* partition_results,
    const int* block_table, const AttentionWork* work_list,
    const float* prepared_query, const float* live_scores,
    int block_table_stride, int64_t value_offset
) {
    // One block computes one work entry and one head. Preparation fills unused
    // grid slots with row=-1; no query-count or partition-table lookup is
    // needed.
    constexpr int Threads = Options::PageTokens * 8;
    using Layout = PackedLayout<HeadDim, Options>;
    const int head = blockIdx.x % Heads;
    const AttentionWork work = work_list[blockIdx.x / Heads];
    const int row = work.row;
    if (row < 0) return;
    const int request = work.request;
    const int part = work.part;
    const int splits = work.partitions;
    const int begin = work.begin;
    const int stop = work.stop;
    const int* request_pages = block_table + request * block_table_stride;
    const int first_page =
        begin < stop ? request_pages[begin / Options::PageTokens] : 0;
    int current_page = first_page;
    int next_page = begin + Options::PageTokens < stop
                        ? request_pages[begin / Options::PageTokens + 1]
                        : 0;
    constexpr int Bytes = StagedPageBytes<HeadDim, Options, LoopIndex>();
    // One K plane and one V plane; each is reused after its final CTA read.
    // Sizes are compile-time constants for this kernel specialization.
    __shared__ __align__(32) unsigned char packed[2 * Bytes];
    __shared__ __align__(8) std::uint64_t bulk_barriers[2];
    AsyncCopy async_copy;
    __shared__ float local_query[HeadDim];
    async_copy.initialize(bulk_barriers);
    if (begin < stop) {
        const int64_t offset = int64_t(first_page * Heads + head) *
                               (Options::PageTokens * Layout::PageWidth());
        async_copy.template stage<HeadDim, Options, LoopIndex>(
            packed, kv_cache + offset,
            bulk_barriers
        );
        async_copy.template stage<HeadDim, Options, LoopIndex>(
            packed + Bytes, kv_cache + value_offset + offset,
            bulk_barriers + 1
        );
    }
    if (threadIdx.x < HeadDim)
        local_query[threadIdx.x] =
            prepared_query[int64_t(row * Heads + head) * HeadDim + threadIdx.x];
    __syncthreads();

    static_assert(
        HeadDim == 96 || HeadDim == 128,
        "Supported storage dimensions are 96 and 128"
    );

    const int tid = threadIdx.x;
    const int64_t result_offset =
        int64_t(row * Heads + head) * (MaxAttentionPartitions * (HeadDim + 2));
    // An Anchor Residual needs the Base mean; the Base itself needs none here.
    // Adjacent needs the means of all preceding loops.
    constexpr int MeanCount = !Options::HasMean ? 0
                              : Options::ReferenceMode == ReferenceMode::Anchor
                                  ? int(LoopIndex != Options::AnchorIndex)
                                  : LoopIndex;
    __shared__ __nv_bfloat16 cached_mean[MeanCount ? MeanCount * HeadDim : 1];
    __shared__ float score[Options::PageTokens], prob[Options::PageTokens],
        factor, newm, newl;
    constexpr int ChannelsPerLane = 4;
    constexpr int ChunksPerLane = HeadDim / (8 * ChannelsPerLane);
    constexpr int QueryPairCount = HeadDim / 16;
    // Each lane keeps local query sums and mean products for its own chunks.
    __nv_bfloat162 query_corrections
        [(Options::ReferenceMode == ReferenceMode::Anchor) ? ChunksPerLane : 1];
    float accum = 0.f, m = -INFINITY, l = 0.f;
    __nv_bfloat162 query_pairs[QueryPairCount];
    __nv_bfloat162 stream_anchor
        [(Options::ReferenceMode == ReferenceMode::Anchor) ? QueryPairCount
                                                           : 1] = {};
    __shared__ float warp_anchor
        [(Options::ReferenceMode == ReferenceMode::Anchor)
             ? (Threads / 32) * HeadDim
             : 1];

    // Anchor lanes own four consecutive channels per 32-channel tile. Adjacent
    // lanes own pairs spaced by 16 channels. Both paths retain BF16 pairs.
#pragma unroll
    for (int pair = 0; pair < QueryPairCount; ++pair) {
        int channel;
        if constexpr (Options::ReferenceMode == ReferenceMode::Anchor)
            channel =
                (pair / (ChannelsPerLane / 2) * 8 + tid % 8) * ChannelsPerLane +
                2 * (pair % (ChannelsPerLane / 2));
        else
            channel = 2 * (tid % 8) + 16 * pair;
        query_pairs[pair] = __floats2bfloat162_rn(
            local_query[channel], local_query[channel + 1]
        );
    }
    if constexpr (Options::HasMean) {
        if constexpr (Options::ReferenceMode == ReferenceMode::Anchor) {
            if constexpr (LoopIndex != Options::AnchorIndex) {
                if (tid < HeadDim)
                    cached_mean[tid] = __float2bfloat16_rn(
                        live.key_mean
                            [(Options::AnchorIndex * Heads + head) * HeadDim +
                             tid]
                    );
            }
        } else {
#pragma unroll
            for (int loop = Options::AnchorIndex; loop < LoopIndex; ++loop) {
                if (tid < HeadDim)
                    cached_mean[loop * HeadDim + tid] = __float2bfloat16_rn(
                        live.key_mean[(loop * Heads + head) * HeadDim + tid]
                    );
            }
        }
    }
    if constexpr (Options::ReferenceMode == ReferenceMode::Anchor) {
        __syncthreads();
#pragma unroll
        for (int chunk = 0; chunk < ChunksPerLane; ++chunk) {
            const int dimension = (chunk * 8 + tid % 8) * ChannelsPerLane;
            __nv_bfloat162 sums = __float2bfloat162_rn(0.f);
            __nv_bfloat162 products = __float2bfloat162_rn(0.f);
#pragma unroll
            for (int pair = 0; pair < ChannelsPerLane / 2; ++pair) {
                const __nv_bfloat162 query =
                    query_pairs[chunk * (ChannelsPerLane / 2) + pair];
                sums = __hadd2(sums, query);
                if constexpr (Options::HasMean &&
                              (LoopIndex != Layout::AnchorIndex)) {
                    const __nv_bfloat162 mean =
                        *reinterpret_cast<const __nv_bfloat162*>(
                            cached_mean + dimension + 2 * pair
                        );
                    products = __hfma2(query, mean, products);
                }
            }
            query_corrections[chunk] = __halves2bfloat162(
                __hadd(__low2bfloat16(sums), __high2bfloat16(sums)),
                __hadd(__low2bfloat16(products), __high2bfloat16(products))
            );
        }
    }
#pragma unroll 2
    for (int start = begin; start < stop; start += Options::PageTokens) {
        const bool next_pending = start + Options::PageTokens < stop;
        const int following_page = next_page;
        const int phase = ((start - begin) / Options::PageTokens) & 1;
        // Current K was prefetched during the preceding PV. The final CTA
        // barrier of that iteration has made the V plane safe to overwrite.
        if (start != begin) {
            const int64_t offset = int64_t(current_page * Heads + head) *
                                   (Options::PageTokens * Layout::PageWidth());
            async_copy.template stage<HeadDim, Options, LoopIndex>(
                packed + Bytes, kv_cache + value_offset + offset,
                bulk_barriers + 1
            );
        }
        if (start + 2 * Options::PageTokens < stop)
            next_page = request_pages[start / Options::PageTokens + 2];
        // Wait only for current K; current V may remain in flight during QK.
        async_copy.wait(bulk_barriers, phase, true);
        unsigned char* current = packed;
        // Anchor keeps [probability, history rescale] in a BF16 pair after
        // its FP32 softmax update; Adjacent uses page-wide shared statistics.
        __nv_bfloat162 softmax_weights = __floats2bfloat162_rn(0.f, 1.f);
        if constexpr (Options::ReferenceMode == ReferenceMode::Anchor) {
            int token = tid / 8, channel = tid % 8;
            __nv_bfloat16 reconstructed_score = __float2bfloat16_rn(0.f);
            auto add_key_record = [&](auto record_tag) {
                constexpr int RecordIndex = decltype(record_tag)::value;
                constexpr int Bits =
                    Layout::template Record<RecordIndex>().bits;
                constexpr int QuantizationGroupSize =
                    Layout::template Record<RecordIndex>()
                        .quantization_group_size;
                constexpr int RecordOffset =
                    RecordIndex == Options::AnchorIndex
                        ? 0
                        : StagedAnchorBytes<HeadDim, Options>();
                __nv_bfloat16 record_score = __float2bfloat16_rn(0.f);
#pragma unroll
                for (int chunk = 0; chunk < ChunksPerLane; ++chunk) {
                    const int dimension =
                        (chunk * 8 + channel) * ChannelsPerLane;

                    const unsigned char* record = current + RecordOffset;
                    using RecordStorage = PackedCodeStorage<Bits>;
                    const PackedCodeBlock<Bits> record_codes{
                        reinterpret_cast<const RecordStorage*>(
                            record + token * (HeadDim * Bits / 8)
                        )[dimension / 4]};
                    const __nv_fp8_e4m3* record_parameters =
                        reinterpret_cast<const __nv_fp8_e4m3*>(
                            record + Options::PageTokens * (HeadDim * Bits / 8)
                        );
                    const int record_metadata =
                        token * (HeadDim / QuantizationGroupSize) +
                        dimension / QuantizationGroupSize;
                    const __nv_bfloat16 record_scale =
                        static_cast<__nv_bfloat16>(
                            record_parameters[record_metadata]
                        );
                    const __nv_bfloat16 record_offset =
                        static_cast<__nv_bfloat16>(
                            record_parameters
                                [Options::PageTokens *
                                     (HeadDim / QuantizationGroupSize) +
                                 record_metadata]
                        );
                    __nv_bfloat162 code_accumulator = __float2bfloat162_rn(0.f);
#pragma unroll
                    for (int pair = 0; pair < ChannelsPerLane / 2; ++pair)
                        code_accumulator = __hfma2(
                            query_pairs[chunk * (ChannelsPerLane / 2) + pair],
                            __halves2bfloat162(
                                record_codes.element(2 * pair),
                                record_codes.element(2 * pair + 1)
                            ),
                            code_accumulator
                        );
                    const __nv_bfloat16 code_dot = __hadd(
                        __low2bfloat16(code_accumulator),
                        __high2bfloat16(code_accumulator)
                    );
                    const __nv_bfloat162 correction = query_corrections[chunk];
                    record_score = __hfma(record_scale, code_dot, record_score);
                    record_score = __hadd(
                        record_score,
                        __hfma(
                            record_offset, __low2bfloat16(correction),
                            RecordIndex == Options::AnchorIndex
                                ? __high2bfloat16(correction)
                                : __float2bfloat16_rn(0.f)
                        )
                    );
                }
                if constexpr (RecordIndex == Options::AnchorIndex &&
                              (LoopIndex != Layout::AnchorIndex)) {
                    constexpr int ResidualBits =
                        Layout::template Record<LoopIndex>().bits;
                    constexpr int ResidualGroups =
                        HeadDim / Layout::template Record<LoopIndex>()
                                      .quantization_group_size;
                    constexpr int ResidualOffset =
                        StagedAnchorBytes<HeadDim, Options>();
                    __nv_bfloat16 alpha =
                        *reinterpret_cast<const __nv_bfloat16*>(
                            current + ResidualOffset +
                            Options::PageTokens * (HeadDim * ResidualBits / 8 +
                                                   2 * ResidualGroups) +
                            token * 2
                        );
                    record_score = __hmul(record_score, alpha);
                }
                reconstructed_score = __hadd(reconstructed_score, record_score);
            };
            add_key_record(std::integral_constant<int, Options::AnchorIndex>{});
            if constexpr (LoopIndex != Layout::AnchorIndex)
                add_key_record(std::integral_constant<int, LoopIndex>{});
            float token_score =
                __bfloat162float(sum_eight_lanes(reconstructed_score));
            if (channel == 0 && start + token < stop) {
                float maximum = fmaxf(m, token_score);
                float stream_factor =
                    exp2f((m - maximum) * 1.4426950408889634f);
                float stream_probability =
                    exp2f((token_score - maximum) * 1.4426950408889634f);
                l = __fmaf_rn(l, stream_factor, stream_probability);
                softmax_weights =
                    __floats2bfloat162_rn(stream_probability, stream_factor);
                m = maximum;
            }
            softmax_weights = __shfl_sync(0xffffffff, softmax_weights, 0, 8);

        } else {
            // Eight lanes reduce one token's QK score in the fixed 1,4,2
            // shuffle order.
            int token = tid / 8, channel = tid % 8;
            __nv_bfloat162 key_accumulator = __float2bfloat162_rn(0.f);
#pragma unroll
            for (int pair = 0; pair < HeadDim / 16; ++pair) {
                int dimension = 2 * channel + 16 * pair;
                const __nv_bfloat162 restored = restore_adjacent<
                    HeadDim, Options, __nv_bfloat162, LoopIndex>(
                    current, cached_mean, token, dimension
                );
                key_accumulator =
                    __hfma2(query_pairs[pair], restored, key_accumulator);
            }
            __nv_bfloat16 key_partial = __hadd(
                __low2bfloat16(key_accumulator),
                __high2bfloat16(key_accumulator)
            );
            float sum = __bfloat162float(sum_eight_lanes(key_partial));
            if (channel == 0)
                score[token] = start + token < stop ? sum : -INFINITY;
        }
        if constexpr (Options::ReferenceMode != ReferenceMode::Anchor)
            __syncthreads();

        if constexpr (Options::ReferenceMode != ReferenceMode::Anchor) {
            // Thread zero computes page-wide softmax statistics and rescales
            // previous sums. exp2 uses log2(e) to evaluate natural
            // exponentials.
            if (tid == 0) {
                float nm = m;
#pragma unroll
                for (int t = 0; t < Options::PageTokens; ++t)
                    nm = fmaxf(nm, score[t]);
                float f = exp2f((m - nm) * 1.4426950408889634f);
#pragma unroll
                for (int t = 0; t < Options::PageTokens; ++t)
                    prob[t] = exp2f((score[t] - nm) * 1.4426950408889634f);
                float psum = 0.f;
                for (int token = 0; token < Options::PageTokens; ++token)
                    psum += prob[token];
                factor = f;
                newm = nm;
                newl = __fmaf_rn(l, f, psum);
            }
            __syncthreads();
        }
        // Every warp has finished reading K before next K overwrites it.
        if constexpr (Options::ReferenceMode == ReferenceMode::Anchor)
            __syncthreads();
        if (next_pending) {
            const int64_t offset = int64_t(following_page * Heads + head) *
                                   (Options::PageTokens * Layout::PageWidth());
            async_copy.template stage<HeadDim, Options, LoopIndex>(
                packed, kv_cache + offset, bulk_barriers
            );
        }
        // Next K overlaps current PV; the final V must drain all writes.
        async_copy.wait(bulk_barriers + 1, phase, next_pending);
        if constexpr (Options::ReferenceMode == ReferenceMode::Anchor) {
            int token = tid / 8, lane = tid % 8;
#pragma unroll
            for (int chunk = 0; chunk < ChunksPerLane; ++chunk) {
                const int dimension = (chunk * 8 + lane) * ChannelsPerLane;
                __nv_bfloat162 restored[2] = {};
                if (start + token < stop)
                    restore_anchor_values<HeadDim, Options, LoopIndex>(
                        current + Bytes, token, dimension, restored
                    );
#pragma unroll
                for (int pair = 0; pair < 2; ++pair) {
                    const int index = chunk * 2 + pair;
                    stream_anchor[index] = __hfma2(
                        __low2bfloat162(softmax_weights), restored[pair],
                        __hmul2(
                            stream_anchor[index],
                            __high2bfloat162(softmax_weights)
                        )
                    );
                }
            }
        } else {
            // Page-wide ownership: even/odd tokens accumulate independently in
            // BF16, then combine with the rescaled BF16 history once per page.
            __nv_bfloat162 page_accumulator = __float2bfloat162_rn(0.f);
#pragma unroll
            for (int token = 0; token < Options::PageTokens; token += 2) {
                __nv_bfloat16 value0 = __float2bfloat16_rn(0.f);
                __nv_bfloat16 value1 = __float2bfloat16_rn(0.f);
                if (tid < HeadDim) {
                    value0 = restore_adjacent<
                        HeadDim, Options, __nv_bfloat16, LoopIndex>(
                        current + Bytes, nullptr, token, tid
                    );
                    value1 = restore_adjacent<
                        HeadDim, Options, __nv_bfloat16, LoopIndex>(
                        current + Bytes, nullptr, token + 1, tid
                    );
                }
                __nv_bfloat162 probability = __floats2bfloat162_rn(
                    start + token < stop ? prob[token] : 0.f,
                    start + token + 1 < stop ? prob[token + 1] : 0.f
                );
                page_accumulator = __hfma2(
                    probability, __halves2bfloat162(value0, value1),
                    page_accumulator
                );
            }
            __nv_bfloat16 page_sum = __hadd(
                __low2bfloat16(page_accumulator),
                __high2bfloat16(page_accumulator)
            );
            accum = __bfloat162float(__hadd(
                __hmul(__float2bfloat16_rn(accum), __float2bfloat16_rn(factor)),
                page_sum
            ));
        }
        if constexpr (Options::ReferenceMode != ReferenceMode::Anchor) {
            m = newm;
            l = newl;
        }
        __syncthreads();
        current_page = following_page;
    }
    if constexpr (Options::ReferenceMode == ReferenceMode::Anchor) {
        int token = tid / 8, lane = tid % 8;
        if (lane == 0) {
            score[token] = m;
            prob[token] = l;
        }
        __syncthreads();
        if (tid < 32) {
            float maximum = tid < Options::PageTokens ? score[tid] : -INFINITY;
#pragma unroll
            for (int offset = 16; offset > 0; offset /= 2)
                maximum = fmaxf(
                    maximum, __shfl_xor_sync(0xffffffff, maximum, offset)
                );
            const float denominator =
                tid < Options::PageTokens ? prob[tid] : 0.f;
            const float scale =
                denominator > 0.f
                    ? exp2f((score[tid] - maximum) * 1.4426950408889634f)
                    : 0.f;
            if (tid < Options::PageTokens) {
                score[tid] = denominator;
                prob[tid] = scale;
            }
            __syncwarp();
            if (tid == 0) {
                newm = maximum;
                float sum = 0.f;
                for (int token = 0; token < Options::PageTokens; ++token)
                    sum = __fmaf_rn(prob[token], score[token], sum);
                newl = sum;
            }
        }
        __syncthreads();
        float scale = prob[token];
#pragma unroll
        for (int j = 0; j < QueryPairCount * 2; ++j) {
            float2 partial = __bfloat1622float2(stream_anchor[j / 2]);
            float anchor = __fmul_rn(scale, j % 2 ? partial.y : partial.x);
            // Four token streams per warp; corresponding channel lanes merge.
            anchor = __fadd_rn(anchor, __shfl_xor_sync(0xffffffff, anchor, 8));
            anchor = __fadd_rn(anchor, __shfl_xor_sync(0xffffffff, anchor, 16));
            const int dimension =
                (j / ChannelsPerLane * 8 + lane) * ChannelsPerLane +
                j % ChannelsPerLane;
            if (tid % 32 < 8)
                warp_anchor[(tid / 32) * HeadDim + dimension] = anchor;
        }
        __syncthreads();
        if (tid < HeadDim) {
            accum = 0.f;
            for (int warp = 0; warp < Threads / 32; ++warp)
                accum += warp_anchor[warp * HeadDim + tid];
        }
        m = newm;
        l = newl;
    }
    float live_score = -INFINITY;
    if (part == 0) {
        live_score = live_scores[row * Heads + head];
    }
    if (splits == 1 && !live.value_rotation) {
        write_single_partition<HeadDim, Threads>(
            live, live_score, row, head, m, l, accum
        );
        return;
    }
    // With absorbed V rotation, live V shares the history coordinate system.
    // Include it once in partition zero; merging then handles only partitions.
    // Separate inverse rotation must finish before live V is added instead.
    if (part == 0 && !live.value_rotation) {
        __shared__ float joined[4];
        if (tid == 0) {
            joined[0] = fmaxf(m, live_score);
            joined[1] = l > 0.f ? expf(m - joined[0]) : 0.f;
            joined[2] = expf(live_score - joined[0]);
            joined[3] = l * joined[1] + joined[2];
        }
        __syncthreads();
        if (tid < HeadDim) {
            const float current = __bfloat162float(
                live.value[row * live.value_stride + head * HeadDim + tid]
            );
            accum = __fmaf_rn(current, joined[2], accum * joined[1]);
        }
        m = joined[0];
        l = joined[3];
    }
    if (tid == 0) {
        partition_results[result_offset + part] = m;
        partition_results[result_offset + MaxAttentionPartitions + part] = l;
    }
    if (tid < HeadDim)
        partition_results
            [result_offset + 2 * MaxAttentionPartitions + part * HeadDim +
             tid] = accum;
}
