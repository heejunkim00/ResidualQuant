#pragma once

#include <tuple>

#include "attention/launch.cuh"
#include "kernel_registry.h"
#include "quantize/launch.cuh"

template <
    int Heads, int HeadDim, ReferenceMode Reference, int Anchor,
    typename... Schedules>
void register_schedules() {
    auto register_schedule = [](auto schedule) {
        using Records = decltype(schedule);

        auto register_page = [](auto page) {
            constexpr int PageTokens = decltype(page)::value;

            using Plain =
                KernelOptions<Records, Reference, Anchor, false, PageTokens>;
            using Centered =
                KernelOptions<Records, Reference, Anchor, true, PageTokens>;

            // A single-record group has only its Base, regardless of reference.
            using QuantizePlain = PackedLayout<
                HeadDim,
                KernelOptions<
                    Records,
                    Records::Count == 1 ? ReferenceMode::Anchor : Reference,
                    Anchor, false, PageTokens>>;
            using QuantizeCentered = PackedLayout<
                HeadDim,
                KernelOptions<
                    Records,
                    Records::Count == 1 ? ReferenceMode::Anchor : Reference,
                    Anchor, true, PageTokens>>;

            const std::vector<std::vector<int>> records = std::apply(
                [](auto... record) {
                    return std::vector<std::vector<int>>{
                        {record.bits, record.quantization_group_size}...};
                },
                Records::Records
            );

            register_kernels(
                {Heads, HeadDim, records, Anchor, Reference, false, PageTokens},
                {&launch_attention<Heads, HeadDim, Plain>,
                 {&launch_quantize<
                      Heads, HeadDim, QuantizePlain, CoefficientMethod::Unit>,
                  &launch_quantize<
                      Heads, HeadDim, QuantizePlain, CoefficientMethod::Norm>,
                  &launch_quantize<
                      Heads, HeadDim, QuantizePlain,
                      CoefficientMethod::LeastSquares>}}
            );
            register_kernels(
                {Heads, HeadDim, records, Anchor, Reference, true, PageTokens},
                {&launch_attention<Heads, HeadDim, Centered>,
                 {&launch_quantize<
                      Heads, HeadDim, QuantizeCentered,
                      CoefficientMethod::Unit>,
                  &launch_quantize<
                      Heads, HeadDim, QuantizeCentered,
                      CoefficientMethod::Norm>,
                  &launch_quantize<
                      Heads, HeadDim, QuantizeCentered,
                      CoefficientMethod::LeastSquares>}}
            );
        };

        register_page(std::integral_constant<int, 16>{});
        register_page(std::integral_constant<int, 32>{});
    };

    (register_schedule(Schedules{}), ...);
}
