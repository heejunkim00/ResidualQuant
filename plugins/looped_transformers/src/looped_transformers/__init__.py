"""Register the Ouro and Huginn model adapters with vLLM."""

from vllm import ModelRegistry

__all__ = ("register",)


def register() -> None:
    ModelRegistry.register_model(
        "OuroForCausalLM", "looped_transformers.ouro:OuroForVLLM"
    )
    ModelRegistry.register_model(
        "RavenForCausalLM", "looped_transformers.huginn:HuginnForVLLM"
    )
