#pragma once

// Residual coefficient used only when producing quantized records.
enum class CoefficientMethod { Unit, Norm, LeastSquares };

namespace quantization {
inline constexpr int RotationStride = 132;
}
