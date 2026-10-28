#pragma once

#include "types.h"

// Compute Y^T = R^T X^T: m16 owns output channels, n8 owns tokens.
// All records share each loaded R fragment. No full matrix preload is needed.
namespace quantization {

__device__ __forceinline__ void rotation_mma(
    float (&c)[4], const float (&a)[4], const float (&b)[2]
) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(__float_as_uint(a[0])), "r"(__float_as_uint(a[1])),
          "r"(__float_as_uint(a[2])), "r"(__float_as_uint(a[3])),
          "r"(__float_as_uint(b[0])), "r"(__float_as_uint(b[1]))
    );
}

// T=8 keeps one n8 fragment per record.
template <int HeadDim, typename Layout, int Plane, int Tokens>
__device__ __forceinline__ void rotate_full_eight(
    const QuantizeInputs<Layout::RecordCount>& inputs,
    const int64_t* slots, const __half* matrix, int first, int tokens,
    float* output
) {
    const int row = threadIdx.x % 32 / 4, inner = threadIdx.x % 4;
    const int warp = threadIdx.x / 32, head = blockIdx.y;
    const int token = first + row;
    const bool valid = token < tokens && slots[token] >= 0;
#pragma unroll
    for (int tile = 0; tile < (HeadDim + 63) / 64; ++tile) {
        const int column = warp * 16 + tile * 64;
        if (column >= HeadDim) continue;
        float accumulated[Layout::RecordCount][4] = {};
#pragma unroll
        for (int offset = 0; offset < HeadDim; offset += 8) {
            float a[4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
                a[i] = __half2float(matrix[
                    (offset + inner + (i / 2) * 4) * HeadDim +
                    column + row + (i % 2) * 8
                ]);
#pragma unroll
            for (int loop = 0; loop < Layout::RecordCount; ++loop) {
                float b[2];
#pragma unroll
                for (int i = 0; i < 2; ++i)
                    b[i] = valid ? __bfloat162float(inputs.pointers[Plane][loop][
                        token * inputs.strides[Plane][loop] + head * HeadDim +
                        offset + inner + i * 4
                    ]) : 0.f;
                rotation_mma(accumulated[loop], a, b);
            }
        }
#pragma unroll
        for (int loop = 0; loop < Layout::RecordCount; ++loop) {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                output[loop * Tokens * RotationStride +
                       (inner * 2 + i % 2) * RotationStride +
                       column + row + (i / 2) * 8] = accumulated[loop][i];
        }
    }
    __syncthreads();
}

// For T=1/2/4, concatenate records virtually into n8; preserve source strides.
// For a partial final CTA, Rows is the number of remaining rows. Padding slots
// are tested on the GPU and never read from K/V or written to the packed cache.
template <int HeadDim, typename Layout, int Plane, int Tokens, bool FullTile>
__device__ __forceinline__ void rotate_compact(
    const QuantizeInputs<Layout::RecordCount>& inputs,
    const int64_t* slots, const __half* matrix, int first, int tokens,
    float* output
) {
    constexpr int Groups = FullTile ? (Layout::RecordCount * Tokens + 7) / 8
                                   : Layout::RecordCount;
    const int rows = FullTile ? Tokens : min(Tokens, tokens - first);
    const int row = threadIdx.x % 32 / 4, inner = threadIdx.x % 4;
    const int warp = threadIdx.x / 32, head = blockIdx.y;
    const __nv_bfloat16* pointer[Groups];
    int64_t stride[Groups];
    int token[Groups];
    bool valid[Groups];
#pragma unroll
    for (int group = 0; group < Groups; ++group) {
        const int flat = group * 8 + row;
        const int record = min(Layout::RecordCount - 1, flat / rows);
        pointer[group] = inputs.pointers[Plane][0];
        stride[group] = inputs.strides[Plane][0];
#pragma unroll
        for (int loop = 1; loop < Layout::RecordCount; ++loop) {
            if (record == loop) {
                pointer[group] = inputs.pointers[Plane][loop];
                stride[group] = inputs.strides[Plane][loop];
            }
        }
        token[group] = first + flat % rows;
        valid[group] = flat < Layout::RecordCount * rows &&
                       token[group] < tokens && slots[token[group]] >= 0;
    }
#pragma unroll
    for (int tile = 0; tile < (HeadDim + 63) / 64; ++tile) {
        const int column = warp * 16 + tile * 64;
        if (column >= HeadDim) continue;
        float accumulated[Groups][4] = {};
#pragma unroll
        for (int offset = 0; offset < HeadDim; offset += 8) {
            float a[4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
                a[i] = __half2float(matrix[
                    (offset + inner + (i / 2) * 4) * HeadDim +
                    column + row + (i % 2) * 8
                ]);
#pragma unroll
            for (int group = 0; group < Groups; ++group) {
                if (FullTile || group * 8 < Layout::RecordCount * rows) {
                    float b[2];
#pragma unroll
                    for (int i = 0; i < 2; ++i)
                        b[i] = valid[group] ? __bfloat162float(pointer[group][
                            token[group] * stride[group] + head * HeadDim +
                            offset + inner + i * 4
                        ]) : 0.f;
                    rotation_mma(accumulated[group], a, b);
                }
            }
        }
#pragma unroll
        for (int group = 0; group < Groups; ++group) {
            if (FullTile || group * 8 < Layout::RecordCount * rows) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int flat = group * 8 + inner * 2 + i % 2;
                    if (flat < Layout::RecordCount * rows)
                        output[(flat / rows) * Tokens * RotationStride +
                               (flat % rows) * RotationStride + column + row +
                               (i / 2) * 8] = accumulated[group][i];
                }
            }
        }
    }
    __syncthreads();
}
}  // namespace quantization
