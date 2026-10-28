#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math_constants.h>
#include <mma.h>

#include <cstdint>

#include "common/cache.cuh"
#include "types.h"

namespace quantization {
// Sum the 32 lane-local partials of one token row.
__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int distance = 16; distance; distance >>= 1)
        value += __shfl_xor_sync(0xffffffff, value, distance);
    return value;
}

template <int RecordCount>
struct QuantizeInputs {
    const __nv_bfloat16* pointers[2][RecordCount];
    int64_t strides[2][RecordCount];
};

// A warp owns one token; each lane owns four consecutive channels and packs
// their codes directly. D96 lanes 24-31 contribute zero to coefficient sums
// and neutral bounds, and never read or write a channel.
// Shared: input (read; nullable).
// Registers: channel_mean (read, reused across this thread's token rows).
template <
    int Heads, int HeadDim, typename Layout, CoefficientMethod Coefficient,
    int LoopIndex>
__device__ __forceinline__ void quantize_token(
    const QuantizeInputs<Layout::RecordCount>& inputs, unsigned char* page,
    const float (&channel_mean)[4], int token, int head, int row, int plane,
    int input_token, float* input, __nv_bfloat162 (&reference)[2]
) {
    int lane = threadIdx.x % 32;
    float x[4] = {};
    float previous[4] = {};
    float numerator = 0.f;
    float denominator = 0.f;

    if constexpr (LoopIndex != Layout::AnchorIndex) {
        if (lane * 4 < HeadDim) {
#pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
                const float2 restored = __bfloat1622float2(reference[pair]);
                previous[2 * pair] = restored.x;
                previous[2 * pair + 1] = restored.y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int channel = lane * 4 + i;
        // Rotation produces FP32 rows. Without rotation, read the original BF16
        // directly in its consuming lane; no FP32 shared-memory staging is
        // needed.
        if (channel < HeadDim) {
            if (input != nullptr) {
                x[i] = input[row * RotationStride + channel];
            } else {
                const __nv_bfloat16* token_input =
                    inputs.pointers[plane][LoopIndex] +
                    input_token * inputs.strides[plane][LoopIndex] +
                    head * HeadDim;
                x[i] = __bfloat162float(token_input[channel]);
            }
        }

        if constexpr (LoopIndex != Layout::AnchorIndex &&
                      Coefficient != CoefficientMethod::Unit) {
            if constexpr (Coefficient == CoefficientMethod::Norm) {
                numerator += x[i] * x[i];
            } else {
                numerator += x[i] * previous[i];
            }
            denominator += previous[i] * previous[i];
        }
    }

    float alpha = 1.f;
    if constexpr (LoopIndex != Layout::AnchorIndex &&
                  Coefficient != CoefficientMethod::Unit) {
        numerator = warp_sum(numerator);
        denominator = warp_sum(denominator);
        if constexpr (Coefficient == CoefficientMethod::Norm) {
            alpha = denominator > 0
                        ? sqrtf(numerator) / fmaxf(sqrtf(denominator), 1e-6f)
                        : 0.f;
        } else {
            alpha = numerator / fmaxf(denominator, 1e-12f);
        }
    }
    const __nv_bfloat16 stored_alpha = __float2bfloat16_rn(alpha);
    alpha = __bfloat162float(stored_alpha);

    constexpr int Bits = Layout::template Record<LoopIndex>().bits,
                  QuantizationGroupSize = Layout::template Record<LoopIndex>()
                                              .quantization_group_size;
    constexpr int CodeBytes = HeadDim * Bits / 8,
                  Groups = HeadDim / QuantizationGroupSize;
    unsigned char* record =
        page + Layout::PageTokens * Layout::template StorageOffset<LoopIndex>();

    float low = CUDART_INF_F, high = -CUDART_INF_F;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int channel = lane * 4 + i;
        if (channel < HeadDim) {
            if constexpr (LoopIndex != Layout::AnchorIndex)
                x[i] = __fmaf_rn(-previous[i], alpha, x[i]);
            if constexpr (Layout::HasMean) {
                if (plane == 0) x[i] -= channel_mean[i];
            }
            low = fminf(low, x[i]);
            high = fmaxf(high, x[i]);
        }
    }
    // Each group covers 4 lanes for G16 or 8 lanes for G32.
#pragma unroll
    for (int distance = QuantizationGroupSize / 8; distance; distance >>= 1) {
        low = fminf(low, __shfl_xor_sync(0xffffffff, low, distance));
        high = fmaxf(high, __shfl_xor_sync(0xffffffff, high, distance));
    }

    if (lane * 4 < HeadDim) {
        const float center = (low + high) * 0.5f;
        float radius;
        if constexpr (Bits == 2) {
            radius = (high - low) * 0.9f * 0.5f;
        } else {
            radius = (high - low) * 0.5f;
        }

        const __nv_fp8_e4m3 offset_code(center - radius);
        const __nv_fp8_e4m3 scale_code(2.f * radius / ((1 << Bits) - 1));
        const float scale = float(scale_code), offset = float(offset_code);

        unsigned packed = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            unsigned code = 0;
            if (scale > 0.f) {
                const float normalized = (x[i] - offset) / scale;
                const float floor = floorf(normalized);
                code = unsigned(fminf(
                    fmaxf(floor + float(normalized - floor > 0.5f), 0.f),
                    float((1 << Bits) - 1)
                ));
            }
            packed |= code << (i * Bits);
        }

        // Restore two adjacent channels at once using the stored parameters.
        if constexpr (LoopIndex == Layout::AnchorIndex ||
                      Layout::ReferenceMode == ReferenceMode::Adjacent) {
            const __nv_bfloat162 scale_pair =
                __bfloat162bfloat162(static_cast<__nv_bfloat16>(scale_code));
            const __nv_bfloat162 offset_pair =
                __bfloat162bfloat162(static_cast<__nv_bfloat16>(offset_code));

#pragma unroll
            for (int i = 0; i < 4; i += 2) {
                constexpr unsigned Mask = (1U << Bits) - 1;
                const __nv_bfloat162 code_pair = __halves2bfloat162(
                    __ushort2bfloat16_rn(
                        static_cast<unsigned short>(
                            (packed >> (i * Bits)) & Mask
                        )
                    ),
                    __ushort2bfloat16_rn(
                        static_cast<unsigned short>(
                            (packed >> ((i + 1) * Bits)) & Mask
                        )
                    )
                );

                __nv_bfloat162 restored =
                    __hfma2(code_pair, scale_pair, offset_pair);
                if constexpr (Layout::HasMean) {
                    if (plane == 0)
                        restored = __hadd2(
                            restored, __floats2bfloat162_rn(
                                          channel_mean[i], channel_mean[i + 1]
                                      )
                        );
                }

                if constexpr (LoopIndex != Layout::AnchorIndex) {
                    restored = __hfma2(
                        reference[i / 2], __bfloat162bfloat162(stored_alpha),
                        restored
                    );
                }
                reference[i / 2] = restored;
            }
        }

        // Record and token starts are aligned for INT4's two-byte store.
        // The four codes occupy one byte for INT2 or two bytes for INT4.
        using Storage = PackedCodeStorage<Bits>;
        reinterpret_cast<Storage*>(record + token * CodeBytes)[lane] =
            static_cast<Storage>(packed);

        if (lane % (QuantizationGroupSize / 4) == 0) {
            __nv_fp8_e4m3* parameters = reinterpret_cast<__nv_fp8_e4m3*>(
                record + Layout::PageTokens * CodeBytes
            );
            const int metadata =
                token * Groups + lane * 4 / QuantizationGroupSize;
            parameters[metadata] = scale_code;
            parameters[Layout::PageTokens * Groups + metadata] = offset_code;
        }
    }

    if (lane == 0) {
        __nv_bfloat16* target = reinterpret_cast<__nv_bfloat16*>(
            record + Layout::PageTokens * (CodeBytes + 2 * Groups) + token * 2
        );
        *target = stored_alpha;
    }
}

// Rotation scratch is shared by all records; commit the anchor before each
// residual so LS references remain in registers and packed bytes stay unchanged.
}  // namespace quantization
#include "rotation.cuh"
namespace quantization {

template <
    int Heads, int HeadDim, typename Layout, CoefficientMethod Coefficient,
    int Plane, int Tokens, int LoopIndex>
__device__ __forceinline__ void commit_record(
    const QuantizeInputs<Layout::RecordCount>& inputs, unsigned char* cache,
    const int64_t* slots, const float* mean, int64_t value_offset, int first,
    int tokens, bool rotated, float* output,
    __nv_bfloat162 (&reference)[(Tokens + 3) / 4][2]
) {
    const int head = blockIdx.y, lane = threadIdx.x % 32;
    float centering[4] = {};
    if constexpr (Plane == 0 && Layout::HasMean) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int channel = lane * 4 + i;
            if (channel < HeadDim)
                centering[i] = mean[(LoopIndex * Heads + head) * HeadDim + channel];
        }
    }
#pragma unroll
    for (int iteration = 0; iteration < (Tokens + 3) / 4; ++iteration) {
        const int row = threadIdx.x / 32 + iteration * 4;
        const int token = first + row;
        if (row < Tokens && token < tokens && slots[token] >= 0) {
            const int32_t slot = static_cast<int32_t>(slots[token]);
            unsigned char* page = cache + Plane * value_offset +
                int64_t(slot / Layout::PageTokens * Heads + head) *
                (Layout::PageTokens * Layout::PageWidth());
            quantize_token<Heads, HeadDim, Layout, Coefficient, LoopIndex>(
                inputs, page, centering, slot % Layout::PageTokens, head, row,
                Plane, token,
                rotated ? output + LoopIndex * Tokens * RotationStride : nullptr,
                reference[iteration]
            );
        }
    }
    __syncthreads();
}

template <
    int Heads, int HeadDim, typename Layout, CoefficientMethod Coefficient,
    int Plane, int Tokens, int LoopIndex = 0>
__device__ __forceinline__ void commit_residuals(
    const QuantizeInputs<Layout::RecordCount>& inputs, unsigned char* cache,
    const int64_t* slots, const float* mean, int64_t value_offset, int first,
    int tokens, bool rotated, float* output,
    __nv_bfloat162 (&reference)[(Tokens + 3) / 4][2]
) {
    if constexpr (LoopIndex != Layout::AnchorIndex)
        commit_record<Heads, HeadDim, Layout, Coefficient, Plane, Tokens, LoopIndex>(
            inputs, cache, slots, mean, value_offset, first, tokens, rotated,
            output, reference
        );
    if constexpr (LoopIndex + 1 < Layout::RecordCount)
        commit_residuals<Heads, HeadDim, Layout, Coefficient, Plane, Tokens, LoopIndex + 1>(
            inputs, cache, slots, mean, value_offset, first, tokens, rotated,
            output, reference
        );
}

template <
    int Heads, int HeadDim, typename Layout, CoefficientMethod Coefficient,
    int Plane, int Tokens>
__device__ __forceinline__ void quantize_plane(
    QuantizeInputs<Layout::RecordCount> inputs, unsigned char* cache,
    const int64_t* slots, const __half* key_rotation,
    const __half* value_rotation, const float* key_mean, int64_t value_offset,
    int tokens, float* output
) {
    const int first = blockIdx.x * Tokens;
    const __half* rotation = Plane == 0 ? key_rotation : value_rotation;
    if (rotation) {
        const __half* matrix = rotation + blockIdx.y * (HeadDim * HeadDim);
        if (tokens - first >= Tokens) {
            if constexpr (Tokens == 8)
                rotate_full_eight<HeadDim, Layout, Plane, Tokens>(
                    inputs, slots, matrix, first, tokens, output);
            else
                rotate_compact<HeadDim, Layout, Plane, Tokens, true>(
                    inputs, slots, matrix, first, tokens, output);
        } else {
            rotate_compact<HeadDim, Layout, Plane, Tokens, false>(
                inputs, slots, matrix, first, tokens, output);
        }
    }
    __nv_bfloat162 reference[(Tokens + 3) / 4][2];
    commit_record<Heads, HeadDim, Layout, Coefficient, Plane, Tokens, Layout::AnchorIndex>(
        inputs, cache, slots, key_mean, value_offset, first, tokens,
        rotation != nullptr, output, reference
    );
    commit_residuals<Heads, HeadDim, Layout, Coefficient, Plane, Tokens>(
        inputs, cache, slots, key_mean, value_offset, first, tokens,
        rotation != nullptr, output, reference
    );
}

template <
    int Heads, int HeadDim, typename Layout, CoefficientMethod Coefficient,
    int Tokens>
__global__ void quantize_kernel(
    QuantizeInputs<Layout::RecordCount> inputs, unsigned char* kv_cache,
    const int64_t* slot_mapping, const __half* key_rotation,
    const __half* value_rotation, const float* key_mean, int64_t value_offset,
    int tokens
) {
    static_assert(Tokens == 1 || Tokens == 2 || Tokens == 4 || Tokens == 8);
    static_assert(HeadDim <= 128 && HeadDim % 16 == 0);
    extern __shared__ __align__(32) float input[];
    // A single launch with separate K/V CTAs and fused anchor/residual records.
    if (blockIdx.z == 0)
        quantize_plane<Heads, HeadDim, Layout, Coefficient, 0, Tokens>(
            inputs, kv_cache, slot_mapping, key_rotation, value_rotation,
            key_mean, value_offset, tokens, input
        );
    else
        quantize_plane<Heads, HeadDim, Layout, Coefficient, 1, Tokens>(
            inputs, kv_cache, slot_mapping, key_rotation, value_rotation,
            key_mean, value_offset, tokens, input
        );
}
}  // namespace quantization
