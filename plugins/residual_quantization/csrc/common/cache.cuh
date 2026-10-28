#pragma once

#include <cuda_runtime.h>

#include <array>
#include <type_traits>

#include "types.h"

struct QuantizationRecord {
    int bits;
    int quantization_group_size;

    template <int HeadDim>
    __host__ __device__ constexpr int Bytes() const {
        return HeadDim * bits / 8 + 2 * HeadDim / quantization_group_size + 2;
    }
};

template <int HeadDim, typename Records, int LoopIndex>
__host__ __device__ constexpr int RecordOffset() {
    static_assert(LoopIndex >= 0 && LoopIndex <= Records::Count);
    if constexpr (LoopIndex == 0)
        return 0;
    else
        return RecordOffset<HeadDim, Records, LoopIndex - 1>() +
               Records::template Record<LoopIndex - 1>()
                   .template Bytes<HeadDim>();
}

// Four INT2 channels occupy one byte; four INT4 channels occupy two bytes.
template <int Bits>
using PackedCodeStorage =
    std::conditional_t<Bits == 2, unsigned char, unsigned short>;

// RecordSchedule<2, 16, 4, 32> stores (bits, group size): (2, 16), (4, 32).
template <int... RecordFields>
struct RecordSchedule {
    static_assert(
        sizeof...(RecordFields) > 0 && sizeof...(RecordFields) % 2 == 0
    );
    static constexpr int Count = sizeof...(RecordFields) / 2;
    static constexpr std::array<QuantizationRecord, Count> Records = [] {
        constexpr int Fields[] = {RecordFields...};
        std::array<QuantizationRecord, Count> result{};
        for (int loop = 0; loop < Count; ++loop)
            result[loop] = {Fields[2 * loop], Fields[2 * loop + 1]};
        return result;
    }();

    template <int LoopIndex>
    __host__ __device__ static constexpr QuantizationRecord Record() {
        static_assert(LoopIndex >= 0 && LoopIndex < Count);
        return Records[LoopIndex];
    }
};

template <
    typename RecordTypes, ReferenceMode Reference, int Anchor,
    bool HasMeanValue, int PageTokensValue>
struct KernelOptions {
    using Records = RecordTypes;
    static constexpr ::ReferenceMode ReferenceMode = Reference;
    static constexpr int AnchorIndex = Anchor;
    static constexpr bool HasMean = HasMeanValue;
    static constexpr int PageTokens = PageTokensValue;
};

// Records follow loop order: packed codes, FP8 scale/offset, BF16 alpha.
// Base records use alpha=1; channels have no padding.
template <int HeadDim, typename Options>
struct PackedLayout : Options {
    static constexpr int RecordCount = Options::Records::Count;
    static_assert(
        Options::ReferenceMode != ::ReferenceMode::Adjacent ||
        Options::AnchorIndex == 0
    );
    __host__ __device__ static constexpr int PageWidth() {
        return RecordOffset<HeadDim, typename Options::Records, RecordCount>();
    }

    static_assert(
        HeadDim > 0 && HeadDim % 32 == 0,
        "Head dimension must be a positive multiple of 32"
    );
    static_assert(
        [] {
            for (int loop = 0; loop < Options::Records::Count; ++loop) {
                const int group =
                    Options::Records::Records[loop].quantization_group_size;
                if (group <= 0 || HeadDim % group != 0) return false;
            }
            return true;
        }(),
        "Head dimension must be divisible by every quantization group size"
    );

    template <int LoopIndex>
    __host__ __device__ static constexpr QuantizationRecord Record() {
        return Options::Records::template Record<LoopIndex>();
    }
    template <int LoopIndex>
    __host__ __device__ static constexpr int StorageOffset() {
        return RecordOffset<HeadDim, typename Options::Records, LoopIndex>();
    }
};
