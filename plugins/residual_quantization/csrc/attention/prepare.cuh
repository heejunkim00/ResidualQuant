#pragma once

#include <cuda_pipeline.h>

#include "rotation.cuh"
#include "types.h"

inline constexpr int PrepareQueryRows = 8;
inline constexpr int PrepareMatrixPadding = 16;

// Global: input and live_key (read). Each thread returns its four-channel
// partial product; the caller performs the 16-lane row reduction.
template <int HeadDim>
__device__ __forceinline__ float query_key_product(
    const __nv_bfloat16* input, const __nv_bfloat16* live_key,
    int64_t input_stride, int64_t key_stride, int valid_rows
) {
    float live_product = 0.f;

    const int live_row = threadIdx.x / 16, live_lane = threadIdx.x % 16;
    if (live_row < valid_rows) {
#pragma unroll
        for (int channel = live_lane * 4; channel < HeadDim; channel += 64) {
            const uint2 q = *reinterpret_cast<const uint2*>(
                input + live_row * input_stride + channel
            );
            const uint2 k = *reinterpret_cast<const uint2*>(
                live_key + live_row * key_stride + channel
            );
            const __nv_bfloat162 q0 = __hmul2(
                reinterpret_cast<const __nv_bfloat162*>(&q)[0],
                __float2bfloat162_rn(AttentionScale<HeadDim>)
            );
            const __nv_bfloat162 q1 = __hmul2(
                reinterpret_cast<const __nv_bfloat162*>(&q)[1],
                __float2bfloat162_rn(AttentionScale<HeadDim>)
            );
            const float2 qlo = __bfloat1622float2(q0),
                         qhi = __bfloat1622float2(q1);
            const float2 klo = __bfloat1622float2(
                reinterpret_cast<const __nv_bfloat162*>(&k)[0]
            );
            const float2 khi = __bfloat1622float2(
                reinterpret_cast<const __nv_bfloat162*>(&k)[1]
            );
            live_product += qlo.x * klo.x;
            live_product += qlo.y * klo.y;
            live_product += qhi.x * khi.x;
            live_product += qhi.y * khi.y;
        }
    }

    return live_product;
}

// M16 holds output channels and N8 holds queries. One warp produces sixteen
// channels for up to eight queries; all warps share the same query tile.
// Shared: two FP16 matrix buffers, each holding 32 reduction rows. Matrix
// padding changes only shared stride; global storage and output order stay fixed.
template <int Heads, int HeadDim>
__device__ __forceinline__ void rotate_queries(
    const __nv_bfloat16* input, const __half* matrix, float* output,
    int valid_rows, int64_t input_stride, const __nv_bfloat16* live_key,
    int64_t key_stride, float& live_product
) {
    static_assert(HeadDim % 32 == 0);
    constexpr int Threads = PrepareThreads;
    constexpr int OutputStride = Heads * HeadDim;
    constexpr int Width = HeadDim + PrepareMatrixPadding;
    constexpr int Chunk = 32;
    extern __shared__ __align__(16) unsigned char rotation_storage[];
    __half* shared_matrix = reinterpret_cast<__half*>(rotation_storage);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int row = lane / 4, inner = lane % 4;
    const int column = warp * 16;
    float accumulated[4] = {};
    auto copy_chunk = [&](int start, int buffer) {
        for (int index = threadIdx.x * 8; index < Chunk * HeadDim;
             index += Threads * 8) {
            __half* destination = shared_matrix +
                (buffer * Chunk + index / HeadDim) * Width + index % HeadDim;
            const __half* source = matrix + start * HeadDim + index;
            __pipeline_memcpy_async(destination, source, 16);
        }
        __pipeline_commit();
    };
    copy_chunk(0, 0);
    if constexpr (Chunk < HeadDim) copy_chunk(Chunk, 1);
    live_product = query_key_product<HeadDim>(
        input, live_key, input_stride, key_stride, valid_rows
    );
#pragma unroll
    for (int start = 0; start < HeadDim; start += Chunk) {
        const int buffer = (start / Chunk) % 2;
        if (start + Chunk < HeadDim)
            __pipeline_wait_prior(1);
        else
            __pipeline_wait_prior(0);
        __syncthreads();
        if (column < HeadDim) {
#pragma unroll
            for (int part = 0; part < Chunk; part += 8) {
                float a[4];
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    a[i] = __half2float(shared_matrix[
                        (buffer * Chunk + part + inner + (i / 2) * 4) * Width +
                        column + row + (i % 2) * 8
                    ]);
                float b[2] = {};
                if (row < valid_rows) {
#pragma unroll
                    for (int i = 0; i < 2; ++i)
                        b[i] = __bfloat162float(input[
                            row * input_stride + start + part + inner + i * 4
                        ]);
                }
                asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(accumulated[0]), "+f"(accumulated[1]),
                      "+f"(accumulated[2]), "+f"(accumulated[3])
                    : "r"(__float_as_uint(a[0])), "r"(__float_as_uint(a[1])),
                      "r"(__float_as_uint(a[2])), "r"(__float_as_uint(a[3])),
                      "r"(__float_as_uint(b[0])), "r"(__float_as_uint(b[1]))
                );
            }
        }
        if (start + 2 * Chunk < HeadDim) {
            __syncthreads();
            copy_chunk(start + 2 * Chunk, buffer);
        }
    }
    if (column < HeadDim) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int token = inner * 2 + i % 2;
            const int channel = column + row + (i / 2) * 8;
            if (token < valid_rows)
                output[token * OutputStride + channel] =
                    accumulated[i] * AttentionScale<HeadDim>;
        }
    }
    __syncthreads();
}

// Global: query/key/key_rotation/key_mean (read; rotation and mean nullable),
// prepared_query/live_scores (write, prepared Q also read). Pointers already
// select this block's head and first row; valid_rows is at most eight.
template <int Heads, int HeadDim, typename Options>
__device__ __forceinline__ void prepare_queries_and_live_scores(
    const __nv_bfloat16* query, const __nv_bfloat16* key,
    const __half* key_rotation, const float* key_mean, float* prepared_query,
    float* live_scores, int64_t query_stride, int64_t key_stride, int valid_rows
) {
    float product = 0.f;
    if (key_rotation) {
        rotate_queries<Heads, HeadDim>(
            query, key_rotation, prepared_query, valid_rows, query_stride, key,
            key_stride, product
        );
    } else {
        product = query_key_product<HeadDim>(
            query, key, query_stride, key_stride, valid_rows
        );
    }

    // Two 16-lane reductions per warp; valid_rows bounds active queries.
    const int row = threadIdx.x / 16;
    const int lane = threadIdx.x % 16;
    if (row >= valid_rows) return;
    const __nv_bfloat16* row_query = query + row * query_stride;
    float correction = 0.f;

#pragma unroll
    for (int channel = lane * 4; channel < HeadDim; channel += 64) {
        __nv_bfloat162 query_pairs[2];
        const uint2 packed_query =
            *reinterpret_cast<const uint2*>(row_query + channel);

        query_pairs[0] =
            reinterpret_cast<const __nv_bfloat162*>(&packed_query)[0];
        query_pairs[1] =
            reinterpret_cast<const __nv_bfloat162*>(&packed_query)[1];

        query_pairs[0] = __hmul2(
            query_pairs[0], __float2bfloat162_rn(AttentionScale<HeadDim>)
        );
        query_pairs[1] = __hmul2(
            query_pairs[1], __float2bfloat162_rn(AttentionScale<HeadDim>)
        );

        const float2 query_low = __bfloat1622float2(query_pairs[0]);
        const float2 query_high = __bfloat1622float2(query_pairs[1]);
        const float4 query_values =
            make_float4(query_low.x, query_low.y, query_high.x, query_high.y);

        if (!key_rotation) {
            *reinterpret_cast<float4*>(
                prepared_query + row * (Heads * HeadDim) + channel
            ) = query_values;
        }

        if constexpr (Options::HasMean) {
            const float4 mean_values =
                *reinterpret_cast<const float4*>(key_mean + channel);

            float4 rotated_query_values = query_values;
            if (key_rotation) {
                rotated_query_values = *reinterpret_cast<const float4*>(
                    prepared_query + row * (Heads * HeadDim) + channel
                );
            }

            correction += rotated_query_values.x * mean_values.x;
            correction += rotated_query_values.y * mean_values.y;
            correction += rotated_query_values.z * mean_values.z;
            correction += rotated_query_values.w * mean_values.w;
        }
    }

#pragma unroll
    for (int offset = 8; offset > 0; offset /= 2) {
        product += __shfl_down_sync(0xffffffff, product, offset, 16);
        if constexpr (Options::HasMean)
            correction += __shfl_down_sync(0xffffffff, correction, offset, 16);
    }

    if (lane == 0) live_scores[row * Heads] = product - correction;
}

// Global: request metadata (read), partition_counts and head-shared work_list
// (write). Shared: warp sums/prefixes and page-weighted work allocation. The
// host reserves one work slot per captured query plus a GPU-sized spare budget.
// GPU lengths choose partitions without exceeding that fixed capacity.
template <int PageTokens>
__device__ __forceinline__ void prepare_partition_schedule(
    const int* query_start_loc, const int* seq_lens, int* partition_counts,
    AttentionWork* work_list, int request_count, int work_capacity,
    int minimum_work
) {
    __shared__ uint64_t page_work[8];
    __shared__ int warp_prefix[8], short_warp_prefix[8];
    __shared__ uint64_t target_pages, total_page_work, extra_work;
    constexpr int PriorityTokens = 256;
    constexpr int PagesPerExtraWork = 8;
    const int thread = threadIdx.x;
    const int lane = thread % 32;
    const int warp = thread / 32;

    for (int index = thread; index < work_capacity; index += PrepareThreads)
        work_list[index] = {-1, 0, 0, 0, 0, 0, {0, 0}};

    uint64_t total_pages = 0;
    for (int request = thread; request < request_count;
         request += PrepareThreads) {
        const int rows =
            query_start_loc[request + 1] - query_start_loc[request];
        if (rows > 0) {
            const int history = max(0, seq_lens[request] - 1);
            const int pages = (uint64_t(history) + PageTokens - 1) / PageTokens;
            total_pages += uint64_t(pages) * rows;
        }
    }
#pragma unroll
    for (int offset = 16; offset; offset /= 2)
        total_pages += __shfl_down_sync(0xffffffff, total_pages, offset);
    if (lane == 0) page_work[warp] = total_pages;
    __syncthreads();
    if (warp == 0) {
        total_pages = lane < 8 ? page_work[lane] : 0;
#pragma unroll
        for (int offset = 16; offset; offset /= 2)
            total_pages += __shfl_down_sync(0xffffffff, total_pages, offset);
        if (lane == 0) {
            const int query_count = query_start_loc[request_count];
            const int spare = work_capacity - query_count;
            // Fill up to half the resident work capacity when the batch is
            // small, then permit one extra work slot per eight history pages.
            // Keep all work inside the captured spare-slot budget.
            extra_work =
                min(uint64_t(spare),
                    uint64_t(max(0, minimum_work - query_count)) +
                        total_pages / PagesPerExtraWork);
            total_page_work = max(uint64_t{1}, total_pages);
            constexpr int MinimumPages = (64 + PageTokens - 1) / PageTokens;
            target_pages = spare > 0 ? max(uint64_t(MinimumPages),
                                           (total_pages + spare - 1) / spare)
                                     : 0;
        }
    }
    __syncthreads();

    // Each tile scans its work counts, then appends directly to the common
    // list. The running offset is identical in every thread; no atomic needed.
    int work_offset = 0;
    for (int first = 0; first < request_count; first += PrepareThreads) {
        const int request = first + thread;
        int row_begin = 0, row_end = 0, history = 0, pages = 0, partitions = 1;
        if (request < request_count) {
            row_begin = query_start_loc[request];
            row_end = query_start_loc[request + 1];
            history = max(0, seq_lens[request] - 1);
            pages = (uint64_t(history) + PageTokens - 1) / PageTokens;
            partitions = int(min(
                uint64_t(min(MaxAttentionPartitions, max(1, pages))),
                uint64_t{1} + uint64_t(pages) * extra_work / total_page_work
            ));
        }
        const int work_count = (row_end - row_begin) * partitions;
        // Within each 256-request tile, issue chunks longer than the target
        // first, so small requests cannot delay long capped requests. Both
        // buckets share the existing metadata reads, scan and barriers for all
        // lengths.
        const uint64_t priority_pages =
            max(target_pages,
                uint64_t((PriorityTokens + PageTokens - 1) / PageTokens));
        const bool longer =
            (pages + partitions - 1) / partitions > priority_pages;
        int prefix = longer ? work_count : 0;
        int short_prefix = longer ? 0 : work_count;
#pragma unroll
        for (int offset = 1; offset < 32; offset *= 2) {
            const int preceding = __shfl_up_sync(0xffffffff, prefix, offset);
            const int preceding_short =
                __shfl_up_sync(0xffffffff, short_prefix, offset);
            if (lane >= offset) {
                prefix += preceding;
                short_prefix += preceding_short;
            }
        }
        if (lane == 31) {
            warp_prefix[warp] = prefix;
            short_warp_prefix[warp] = short_prefix;
        }
        __syncthreads();
        if (warp == 0) {
            int sum = lane < 8 ? warp_prefix[lane] : 0;
            int short_sum = lane < 8 ? short_warp_prefix[lane] : 0;
#pragma unroll
            for (int offset = 1; offset < 8; offset *= 2) {
                const int preceding = __shfl_up_sync(0xffffffff, sum, offset);
                const int preceding_short =
                    __shfl_up_sync(0xffffffff, short_sum, offset);
                if (lane >= offset) {
                    sum += preceding;
                    short_sum += preceding_short;
                }
            }
            if (lane < 8) {
                warp_prefix[lane] = sum;
                short_warp_prefix[lane] = short_sum;
            }
        }
        __syncthreads();
        int destination;
        if (longer) {
            destination = work_offset + prefix - work_count;
            if (warp > 0) destination += warp_prefix[warp - 1];
        } else {
            destination =
                work_offset + warp_prefix[7] + short_prefix - work_count;
            if (warp > 0) destination += short_warp_prefix[warp - 1];
        }
        for (int row = row_begin; row < row_end; ++row) {
            partition_counts[row] = partitions;
            for (int part = 0; part < partitions; ++part) {
                const int first_page = uint64_t(part) * pages / partitions;
                const int last_page = uint64_t(part + 1) * pages / partitions;
                work_list[destination++] = {
                    row,
                    request,
                    part,
                    partitions,
                    first_page * PageTokens,
                    int(
                        min(uint64_t(history), uint64_t(last_page) * PageTokens)
                    ),
                    {0, 0}};
            }
        }
        work_offset += warp_prefix[7] + short_warp_prefix[7];
        // No thread may overwrite the prefix scratch before its last reader.
        __syncthreads();
    }
}

// Block zero prepares partition metadata; the remaining blocks prepare Q and
// current-token scores. The next kernel waits for both through stream ordering.
template <int Heads, int HeadDim, typename Options, int LoopIndex>
__global__ void attention_prepare_kernel(
    LiveAttention live, float* prepared_query, float* live_scores,
    const int* query_start_loc, const int* seq_lens, int* partition_counts,
    AttentionWork* work_list, int request_count, int work_capacity,
    int minimum_work
) {
    if (blockIdx.x == 0) {
        prepare_partition_schedule<Options::PageTokens>(
            query_start_loc, seq_lens, partition_counts, work_list,
            request_count, work_capacity, minimum_work
        );
    } else {
        const int first_query_row = (blockIdx.x - 1) / Heads * PrepareQueryRows;
        const int head = (blockIdx.x - 1) % Heads;
        const int query_token_count = query_start_loc[request_count];
        if (first_query_row >= query_token_count) return;

        const __nv_bfloat16* query =
            live.query + first_query_row * live.query_stride + head * HeadDim;
        const __nv_bfloat16* key =
            live.key + first_query_row * live.key_stride + head * HeadDim;

        const __half* key_rotation =
            live.key_rotation ? live.key_rotation + head * (HeadDim * HeadDim)
                              : nullptr;
        const float* key_mean = nullptr;
        if constexpr (Options::HasMean)
            key_mean = live.key_mean + (LoopIndex * Heads + head) * HeadDim;

        float* destination =
            prepared_query + int64_t(first_query_row * Heads + head) * HeadDim;
        float* scores = live_scores + first_query_row * Heads + head;

        prepare_queries_and_live_scores<Heads, HeadDim, Options>(
            query, key, key_rotation, key_mean, destination, scores,
            live.query_stride, live.key_stride,
            min(PrepareQueryRows, query_token_count - first_query_row)
        );
    }
}
