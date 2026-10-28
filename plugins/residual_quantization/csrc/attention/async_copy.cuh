#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cuda/pipeline>
#include <cuda/ptx>

#include "common/cache.cuh"

// Anchor restoration never uses the Base's alpha=1 region. Compact staging
// only; persistent cache offsets and the stored Base alpha remain unchanged.
template <int HeadDim, typename Options>
__host__ __device__ constexpr int StagedAnchorBytes() {
    using Layout = PackedLayout<HeadDim, Options>;
    return Options::PageTokens *
           (Layout::template Record<Options::AnchorIndex>()
                .template Bytes<HeadDim>() - 2);
}

// Bytes staged for one K or V page: Base plus current Residual for Anchor,
// or the prefix through the current loop for Adjacent.
template <int HeadDim, typename Options, int LoopIndex>
__host__ __device__ constexpr int StagedPageBytes() {
    using Layout = PackedLayout<HeadDim, Options>;
    if constexpr (Options::ReferenceMode == ReferenceMode::Adjacent) {
        return Options::PageTokens *
               Layout::template StorageOffset<LoopIndex + 1>();
    } else {
        constexpr int BaseBytes =
            Layout::template Record<Options::AnchorIndex>()
                .template Bytes<HeadDim>() - 2;
        if constexpr (LoopIndex == Options::AnchorIndex) {
            return Options::PageTokens * BaseBytes;
        } else {
            return Options::PageTokens *
                   (BaseBytes + Layout::template Record<LoopIndex>()
                                    .template Bytes<HeadDim>());
        }
    }
}

// Ampere/Ada distribute collective copies across the block. Hopper uses bulk
// transfers with one completion barrier per K/V plane.
template <bool Bulk>
struct AsyncCopyImpl;

template <>
struct AsyncCopyImpl<false> {
    // Each thread issues its share of the collective copy. After waiting, the
    // block barrier makes every thread's completed writes visible to its peers.
    cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

    // The unnamed pointer addresses shared barriers; unused on this path.
    __device__ void initialize(std::uint64_t*) {}

    // Commit the entire plane as one group.
    template <int HeadDim, typename Options, int LoopIndex>
    __device__ __forceinline__ void stage(
        unsigned char* dst, const unsigned char* source, std::uint64_t*
    ) {
        constexpr int Bytes = StagedPageBytes<HeadDim, Options, LoopIndex>();
        using Layout = PackedLayout<HeadDim, Options>;
        const cooperative_groups::thread_block block =
            cooperative_groups::this_thread_block();
        pipeline.producer_acquire();
        if constexpr (Options::ReferenceMode == ReferenceMode::Anchor) {
            constexpr int BaseBytes =
                StagedAnchorBytes<HeadDim, Options>();
            constexpr int BaseOffset =
                Options::PageTokens *
                Layout::template StorageOffset<Options::AnchorIndex>();
            cuda::memcpy_async(
                block, dst, source + BaseOffset,
                cuda::aligned_size_t<16>(BaseBytes), pipeline
            );
            if constexpr (LoopIndex != Options::AnchorIndex) {
                constexpr int ResidualOffset =
                    Options::PageTokens *
                    Layout::template StorageOffset<LoopIndex>();
                cuda::memcpy_async(
                    block, dst + BaseBytes, source + ResidualOffset,
                    cuda::aligned_size_t<16>(Bytes - BaseBytes), pipeline
                );
            }
        } else {
            cuda::memcpy_async(
                block, dst, source, cuda::aligned_size_t<16>(Bytes), pipeline
            );
        }
        pipeline.producer_commit();
    }

    // The unnamed pointer addresses shared barriers; unused on this path.
    __device__ __forceinline__ void wait(
        std::uint64_t*, int, bool newer_plane_pending
    ) {
        // Wait for current K while V is in flight, or for current V while
        // next K is in flight. The final V drains all outstanding writes.
        if (newer_plane_pending)
            cuda::pipeline_consumer_wait_prior<1>(pipeline);
        else
            cuda::pipeline_consumer_wait_prior<0>(pipeline);
        __syncthreads();
    }
};

template <>
struct AsyncCopyImpl<true> {
    __device__ static __forceinline__ void initialize(std::uint64_t* barriers) {
        if (threadIdx.x == 0) {
            for (int i = 0; i < 2; ++i)
                cuda::ptx::mbarrier_init(barriers + i, 1);
            cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
        }
        __syncthreads();
    }

    template <int HeadDim, typename Options, int LoopIndex>
    __device__ static __forceinline__ void stage(
        unsigned char* dst, const unsigned char* source,
        std::uint64_t* barrier
    ) {
        constexpr int Bytes = StagedPageBytes<HeadDim, Options, LoopIndex>();
        if (threadIdx.x == 0) {
            using Layout = PackedLayout<HeadDim, Options>;
            cuda::ptx::mbarrier_arrive_expect_tx(
                cuda::ptx::sem_release, cuda::ptx::scope_cta,
                cuda::ptx::space_shared, barrier, Bytes
            );
            if constexpr (Options::ReferenceMode == ReferenceMode::Anchor) {
                constexpr int BaseBytes =
                    StagedAnchorBytes<HeadDim, Options>();
                constexpr int BaseOffset =
                    Options::PageTokens *
                    Layout::template StorageOffset<Options::AnchorIndex>();
                cuda::ptx::cp_async_bulk(
                    cuda::ptx::space_cluster, cuda::ptx::space_global, dst,
                    source + BaseOffset, BaseBytes, barrier
                );
                if constexpr (LoopIndex != Options::AnchorIndex) {
                    constexpr int ResidualOffset =
                        Options::PageTokens *
                        Layout::template StorageOffset<LoopIndex>();
                    cuda::ptx::cp_async_bulk(
                        cuda::ptx::space_cluster, cuda::ptx::space_global,
                        dst + BaseBytes, source + ResidualOffset,
                        Bytes - BaseBytes, barrier
                    );
                }
            } else {
                cuda::ptx::cp_async_bulk(
                    cuda::ptx::space_cluster, cuda::ptx::space_global, dst,
                    source, Bytes, barrier
                );
            }
        }
    }

    __device__ static __forceinline__ void wait(
        std::uint64_t* barrier, int phase, bool
    ) {
        while (!cuda::ptx::mbarrier_try_wait_parity(barrier, phase)) {
        }
        __syncthreads();
    }
};

// Select the copy implementation for each device target in a fat binary.
// Hopper and newer use bulk transfers; Ampere/Ada issue 16-byte copies.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
using AsyncCopy = AsyncCopyImpl<true>;
#else
using AsyncCopy = AsyncCopyImpl<false>;
#endif
