#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdint>

// Inverse-rotate one FP32 vector; input and output may share storage.
template <int HeadDim, int Threads>
__device__ __forceinline__ void inverse_rotate_vector(
    const float* input, const __half* matrix, float* output
) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int row = lane / 4;
    const int inner = lane % 4;
    constexpr int Tiles = (HeadDim + Threads / 2 - 1) / (Threads / 2);
    float accumulated[Tiles][2][4] = {};

#pragma unroll
    for (int tile = 0; tile < Tiles; ++tile) {
        const int column = warp * 16 + tile * (Threads / 2);
        if (column < HeadDim) {
#pragma unroll
            for (int offset = 0; offset < HeadDim; offset += 8) {
                float a[4] = {};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int input_row = row + (i % 2) * 8;
                    if (input_row == 0) {
                        const int index = offset + inner + (i / 2) * 4;
                        a[i] = nvcuda::wmma::__float_to_tf32(input[index]);
                    }
                }
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int matrix_column = column + half * 8 + row;
                    const int k = offset + inner;
                    const float b0 = __half2float(
                        matrix[matrix_column * HeadDim + k]
                    );
                    const float b1 = __half2float(
                        matrix[matrix_column * HeadDim + k + 4]
                    );
                    asm volatile(
                        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                        "{%0, %1, %2, %3};"
                        : "+f"(accumulated[tile][half][0]),
                          "+f"(accumulated[tile][half][1]),
                          "+f"(accumulated[tile][half][2]),
                          "+f"(accumulated[tile][half][3])
                        : "r"(__float_as_uint(a[0])),
                          "r"(__float_as_uint(a[1])),
                          "r"(__float_as_uint(a[2])),
                          "r"(__float_as_uint(a[3])), "r"(__float_as_uint(b0)),
                          "r"(__float_as_uint(b1))
                    );
                }
            }
        }
    }
    // Finish all reads before overwriting shared input.
    __syncthreads();
#pragma unroll
    for (int tile = 0; tile < Tiles; ++tile) {
        const int column = warp * 16 + tile * (Threads / 2);
        if (column < HeadDim) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int output_row = row + (i / 2) * 8;
                    const int channel = column + half * 8 + inner * 2 + i % 2;
                    if (output_row == 0)
                        output[channel] = accumulated[tile][half][i];
                }
            }
        }
    }
    __syncthreads();
}
