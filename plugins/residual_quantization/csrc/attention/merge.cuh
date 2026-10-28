#pragma once

#include "output.cuh"
#include "types.h"

// Warp zero prepares the softmax statistics and one scale per partition.
// Accumulate weighted V sums in partition order, then normalize.
template <int Heads, int HeadDim>
__global__ void attention_merge_kernel(
    LiveAttention live, const float* partition_results,
    const float* live_scores, const int* query_count,
    const int* partition_counts
) {
    static_assert(MaxAttentionPartitions == 16);
    const int row = blockIdx.x, head = blockIdx.y, channel = threadIdx.x;
    if (row >= *query_count) return;
    const int partitions = partition_counts[row];
    // Explicit V inverse rotation is owned by merge, including a single part.
    if (partitions == 1 && !live.value_rotation) return;

    __shared__ float scales[MaxAttentionPartitions];
    __shared__ float history[HeadDim];
    __shared__ float denominator, live_weight;

    const float* record =
        partition_results +
        int64_t(row * Heads + head) * (MaxAttentionPartitions * (HeadDim + 2));
    if (channel < 32) {
        float current = -INFINITY;
        float maximum = -INFINITY;
        if (channel < partitions) {
            if (live.value_rotation) current = live_scores[row * Heads + head];
            maximum = fmaxf(record[channel], current);
        }

#pragma unroll
        for (int offset = MaxAttentionPartitions / 2; offset > 0; offset /= 2)
            maximum =
                fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));

        float scale = 0.f, contribution = 0.f;
        if (channel < partitions) {
            const float partition_denominator =
                record[MaxAttentionPartitions + channel];
            scale = expf(record[channel] - maximum);
            contribution = partition_denominator * scale;
            scales[channel] = scale;
        }

#pragma unroll
        for (int offset = MaxAttentionPartitions / 2; offset > 0; offset /= 2)
            contribution += __shfl_down_sync(0xffffffff, contribution, offset);

        if (channel == 0) {
            denominator = contribution;
            live_weight = 0.f;
            if (live.value_rotation) {
                const float current_weight = expf(current - maximum);
                denominator += current_weight;
                live_weight = current_weight / denominator;
            }
        }
    }
    __syncthreads();

    float accumulated = 0.f;
    if (channel < HeadDim) {
        const float* value_sums = record + 2 * MaxAttentionPartitions;
        for (int part = 0; part < partitions; ++part)
            accumulated += value_sums[part * HeadDim + channel] * scales[part];
        accumulated /= denominator;
    }

    if (live.value_rotation) {
        write_attention_output<HeadDim, 128>(
            live.value, live.output, live.value_rotation, row, head,
            live.value_stride, live.output_stride, accumulated, live_weight,
            history
        );
    } else if (channel < HeadDim) {
        live.output[row * live.output_stride + head * HeadDim + channel] =
            __float2bfloat16_rn(accumulated);
    }
}
