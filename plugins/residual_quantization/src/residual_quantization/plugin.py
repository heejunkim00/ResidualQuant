"""Register optional packed-KV integration without importing model adapters."""

import hashlib
import inspect
from copy import copy
from functools import wraps
from pathlib import Path

import torch
from vllm.attention.layer import Attention
from vllm.model_executor.model_loader import (
    DefaultModelLoader,
    register_model_loader,
)

from residual_quantization.install_cache import install_vllm_cache_hooks
from residual_quantization.weights import (
    absorb_model_value_rotations,
    prepare_value_rotation,
)

__all__ = ("register_residual_quantization",)


def _load_artifact(config, model_config):
    """Validate group-major rotations and recurrence-major K means."""
    spec = config.get("rotation_artifact")
    if spec is None:
        return None
    path = Path(spec["artifact"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError("Rotation artifact hash mismatch")
    data = torch.load(path, map_location="cpu", weights_only=True)
    schedules = config["packed_groups"]
    records = [record for schedule in schedules for record in schedule]
    assert data["schedule"] == [bit for bit, _ in records]
    assert data["groups"] == [group for _, group in records]
    h = model_config.num_attention_heads
    d = model_config.hidden_size // h
    layers = model_config.n_layers_in_recurrent_block

    assert data["format_version"] == 2
    assert data["model"] == model_config.loop_model
    if sum(map(len, schedules)) != model_config.mean_recurrence:
        raise ValueError(
            "Packed groups must describe every model recurrence exactly once"
        )
    group_count = len(schedules)
    matrix_shape = (layers, h, d, d)
    for name in ("key_rotations", "value_rotations"):
        rotation = data[name]
        if rotation.ndim not in (4, 5):
            raise ValueError(f"{name} must have four or five dimensions")
        expected = (group_count, *matrix_shape) if rotation.ndim == 5 else matrix_shape
        if rotation.shape != expected:
            raise ValueError(
                f"{name} shape {tuple(rotation.shape)} != model shape {expected}"
            )
        assert rotation.dtype == torch.float16
        assert torch.isfinite(rotation).all()
    assert data["key_means"].shape == (model_config.mean_recurrence, layers, h, d)
    assert data["key_means"].dtype == torch.float32
    assert torch.isfinite(data["key_means"]).all()
    return data


def _configure_loaded_model(model, vllm_config, config, attention_settings):
    """Attach packed attention after weight loading and before KV allocation."""

    from residual_quantization.install_model import install_model_attention

    model_config = vllm_config.model_config.hf_config
    heads = model_config.num_attention_heads
    if heads <= 0 or model_config.hidden_size % heads:
        raise ValueError("Hidden size must be divisible by the attention head count")
    if model_config.num_key_value_heads != heads:
        raise ValueError("Packed attention requires matching Q/K/V head counts")
    if (
        model_config.mean_recurrence <= 0
        or model_config.n_layers_in_recurrent_block <= 0
    ):
        raise ValueError("Positive physical layers and recurrence count required")
    if model_config.n_layers_in_prelude < 0 or model_config.n_layers_in_coda < 0:
        raise ValueError("Prelude/coda layer counts must be nonnegative")

    artifact, value_rotations = prepare_value_rotation(
        _load_artifact(config, model_config)
    )
    install_model_attention(config, artifact, vllm_config, attention_settings)
    if value_rotations is not None:
        absorb_model_value_rotations(model, value_rotations)


class ResidualQuantizationModelLoader(DefaultModelLoader):
    """Load normal auto-format weights, then attach packed attention."""

    def __init__(self, load_config):
        weight_load_config = copy(load_config)
        weight_load_config.load_format = "auto"
        super().__init__(weight_load_config)

    def load_model(self, vllm_config, model_config):
        config = vllm_config.additional_config.get("residual_quantization")
        if not config or not config["packed_groups"]:
            return super().load_model(vllm_config, model_config)

        attention_settings = {}
        original_attention_init = Attention.__init__
        signature = inspect.signature(original_attention_init)

        @wraps(original_attention_init)
        def initialize_attention(layer, *args, **kwargs):
            parameters = signature.bind(layer, *args, **kwargs)
            parameters.apply_defaults()
            original_attention_init(layer, *args, **kwargs)
            attention_settings[layer.layer_name] = parameters.arguments

        Attention.__init__ = initialize_attention
        try:
            model = super().load_model(vllm_config, model_config)
        finally:
            Attention.__init__ = original_attention_init

        _configure_loaded_model(model, vllm_config, config, attention_settings)
        return model


def register_residual_quantization():
    """Register the packed loader and process-local vLLM cache hooks."""
    install_vllm_cache_hooks()
    register_model_loader("residual_quantization")(ResidualQuantizationModelLoader)
