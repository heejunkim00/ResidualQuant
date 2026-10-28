"""Register optional residual quantization with vLLM."""

from residual_quantization.plugin import register_residual_quantization as register

__all__ = ("register",)
