"""One full-attention block table with independent native/packed layer storage."""

from dataclasses import dataclass

import torch
from vllm.config import get_layers_from_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backends.utils import create_fast_prefill_custom_backend
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.kv_cache_utils import check_enough_kv_cache_memory, get_num_blocks
from vllm.v1.engine import core
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.utils import AttentionGroup

__all__ = ("install_vllm_cache_hooks",)

_installed = False


@dataclass(frozen=True)
class LoopCacheSpec(FullAttentionSpec):
    """Identify packed layer storage when selecting the mixed allocator."""


@dataclass
class LoopCacheConfig(KVCacheConfig):
    """Preserve real owner specs through vLLM config deepcopy/serialization."""

    layer_specs: dict[str, FullAttentionSpec]


def mixed_cache_config(vllm_config, specs, available_memory):
    """Share logical page IDs across owners while allocating their real formats."""
    if vllm_config.parallel_config.decode_context_parallel_size != 1:
        raise ValueError("Packed mixed storage requires DCP1")
    check_enough_kv_cache_memory(vllm_config, specs, available_memory)

    for spec in specs.values():
        if (
            type(spec) not in (LoopCacheSpec, FullAttentionSpec)
            or spec.sliding_window is not None
            or spec.attention_chunk_size is not None
        ):
            raise ValueError("Only native/packed full-attention caches are supported")

    block_sizes = {spec.block_size for spec in specs.values()}
    if len(block_sizes) != 1:
        raise ValueError("Group token block sizes differ")
    block_size = block_sizes.pop()

    bytes_per_block = sum(spec.page_size_bytes for spec in specs.values())
    # bytes_per_block already includes all layers.
    num_blocks = get_num_blocks(vllm_config, 1, available_memory, bytes_per_block)
    required_blocks = (
        vllm_config.model_config.max_model_len + block_size - 1
    ) // block_size
    if num_blocks <= required_blocks:
        raise ValueError("Layer buffers cannot hold max_model_len plus the null block")

    # vLLM selects its coordinator by exact spec type.
    coordinator_spec = FullAttentionSpec(**vars(next(iter(specs.values()))))

    return LoopCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(size=num_blocks * spec.page_size_bytes, shared_by=[name])
            for name, spec in specs.items()
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=list(specs), kv_cache_spec=coordinator_spec)
        ],
        layer_specs=dict(specs),
    )


def install_vllm_cache_hooks():
    """Opt-in hooks; importing this module never changes another backend."""
    global _installed

    if _installed:
        return

    original_profile = GPUModelRunner.profile_run

    def profile(runner, *args, **kwargs):
        result = original_profile(runner, *args, **kwargs)
        _warmup_packed_kernels(runner)
        return result

    GPUModelRunner.profile_run = profile

    original_specs = GPUModelRunner.get_kv_cache_spec

    def specs(runner):
        result = original_specs(runner)
        layers = get_layers_from_vllm_config(
            runner.vllm_config, AttentionLayerBase, list(result)
        )

        for name in result:
            layer = layers.get(name)
            if layer is None or not hasattr(layer, "packed_group"):
                continue
            model_config = runner.vllm_config.model_config.hf_config
            heads = model_config.num_attention_heads
            head_dim = model_config.hidden_size // heads
            result[name] = LoopCacheSpec(
                block_size=layer.packed_group.page_tokens,
                num_kv_heads=heads,
                head_size=layer.packed_group.page_width(head_dim),
                dtype=torch.uint8,
                use_mla=False,
            )

        return result

    GPUModelRunner.get_kv_cache_spec = specs

    original_initialize = GPUModelRunner.initialize_attn_backend
    original_iterator = GPUModelRunner._kv_cache_spec_attn_group_iterator

    def initialize(runner, config):
        if not isinstance(config, LoopCacheConfig):
            return original_initialize(runner, config)
        assert not runner.attn_groups, "Attention backends are already initialized"

        layer_names_by_spec = {}
        for group in config.kv_cache_groups:
            for name in group.layer_names:
                owner = runner.shared_kv_cache_layers.get(name, name)
                spec = config.layer_specs[owner]
                layer_names_by_spec.setdefault(spec, []).append(name)

        attention_groups = []
        for spec, names in layer_names_by_spec.items():
            layers = get_layers_from_vllm_config(
                runner.vllm_config, AttentionLayerBase, names
            )
            backends, layer_names_by_backend = {}, {}
            for name in names:
                backend = layers[name].get_attn_backend()
                if name in runner.kv_sharing_fast_prefill_eligible_layers:
                    backend = create_fast_prefill_custom_backend("FastPrefill", backend)

                key = backend.full_cls_name()
                backends[key] = backend
                layer_names_by_backend.setdefault(key, []).append(name)

            for key, layer_names in layer_names_by_backend.items():
                backend = backends[key]
                builder = backend.get_builder_cls()(
                    spec, layer_names, runner.vllm_config, runner.device
                )
                attention_groups.append(AttentionGroup(backend, builder, layer_names))

        runner.attn_groups = [attention_groups]
        runner.calculate_reorder_batch_threshold()

    def iterator(runner):
        if not isinstance(runner.kv_cache_config, LoopCacheConfig):
            yield from original_iterator(runner)
            return
        for group in runner._attn_group_iterator():
            yield group.metadata_builder.kv_cache_spec, group

    GPUModelRunner.initialize_attn_backend = initialize
    GPUModelRunner._kv_cache_spec_attn_group_iterator = iterator

    original_config = kv_cache_utils.get_kv_cache_config

    def cache_config(vllm_config, kv_cache_spec, available_memory):
        if any(type(s) is LoopCacheSpec for s in kv_cache_spec.values()):
            return mixed_cache_config(vllm_config, kv_cache_spec, available_memory)
        return original_config(vllm_config, kv_cache_spec, available_memory)

    kv_cache_utils.get_kv_cache_config = cache_config
    core.get_kv_cache_config = cache_config
    _installed = True


@torch.inference_mode()
def _warmup_packed_kernels(runner):
    """Account for lazy-loaded CUDA modules before vLLM sizes the KV pool."""
    seen = set()
    layers = get_layers_from_vllm_config(runner.vllm_config, AttentionLayerBase)
    for layer in layers.values():
        if not hasattr(layer, "packed_group"):
            continue
        from residual_quantization._cuda import attention, quantize

        group = layer.packed_group
        heads, dim = layer.num_kv_heads, layer.head_size
        signature = (
            heads,
            dim,
            group.records,
            group.page_tokens,
            group.reference_mode,
            group.anchor_index,
            group.coefficient_method,
            layer.key_mean is not None,
            layer.key_rotation is not None,
            layer.value_rotation is not None,
        )
        if signature in seen:
            continue
        seen.add(signature)
        x = torch.zeros((1, heads, dim), dtype=torch.bfloat16, device=runner.device)
        output = torch.empty_like(x)
        cache = torch.zeros(
            (2, 1, heads, group.page_tokens, group.page_width(dim)),
            dtype=torch.uint8,
            device=runner.device,
        )
        slots = torch.zeros(1, dtype=torch.int64, device=runner.device)
        starts = torch.tensor([0, 1], dtype=torch.int32, device=runner.device)
        lengths = torch.ones(1, dtype=torch.int32, device=runner.device)
        table = torch.zeros((1, 1), dtype=torch.int32, device=runner.device)
        quantize(
            [(x, x)] * len(group.records),
            cache,
            slots,
            layer.key_rotation,
            layer.value_rotation,
            layer.key_mean,
            group.records,
            group.reference_mode,
            group.anchor_index,
            group.coefficient_method,
        )
        for loop in range(len(group.records)):
            attention(
                x,
                x,
                x,
                cache,
                output,
                starts,
                lengths,
                table,
                layer.key_rotation,
                layer.value_rotation,
                layer.key_mean,
                group.records,
                group.reference_mode,
                group.anchor_index,
                loop,
            )
    if seen:
        torch.cuda.synchronize(runner.device)
