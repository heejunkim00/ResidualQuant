"""Bind recurrent attention slots to their group state, rotations and means."""

import inspect

from vllm.attention.layer import Attention
from vllm.attention.selector import backend_name_to_enum
from vllm.attention.utils.kv_sharing_utils import validate_kv_sharing_target
from vllm.config import get_layers_from_vllm_config
from vllm.model_executor.models.utils import extract_layer_index

from residual_quantization.packed_backend import PackedBackend
from residual_quantization.prefill_offload import PrefillCPUOffload

__all__ = ("install_model_attention",)


class DeferredKVGroup:
    """Own the packed format and keep raw KV until every group loop completes.

    Incomplete groups copy KV to GPU or optional prefill host storage. The
    completing call borrows its input, so a one-loop group needs no raw-KV copy.
    """

    def __init__(self, config, records, page_tokens, prefill_offload=None):
        self.pending = []
        self.records = records
        self.page_tokens = page_tokens
        self.reference_mode = config["reference_mode"]
        self.anchor_index = config["anchor_index"]
        self.coefficient_method = config["coefficient_method"]
        self.prefill_offload = prefill_offload
        self.host_buffers = [None] * (len(records) - 1)

    def page_width(self, head_dim):
        code_bytes = head_dim * sum(bit for bit, _ in self.records) // 8
        metadata_bytes = 2 * sum(head_dim // group for _, group in self.records)
        alpha_bytes = 2 * len(self.records)
        return code_bytes + metadata_bytes + alpha_bytes

    def stage(self, key, value, offload=False):
        if len(self.pending) + 1 == len(self.records):
            records = self.pending + [(key, value)]
            self.pending = []
            return records

        if offload:
            index = len(self.pending)
            buffer = self.prefill_offload.store(key, value, self.host_buffers[index])
            self.host_buffers[index] = buffer
            self.pending.append((buffer[0, : key.size(0)], buffer[1, : value.size(0)]))
        else:
            self.pending.append((key.clone(), value.clone()))


def _create_prefill_offload(config, vllm_config):
    model_config = vllm_config.model_config.hf_config
    schedules = config["packed_groups"]
    offload_enabled = config.get("prefill_cpu_offload", False)
    chunk_tokens = config.get("prefill_offload_chunk_tokens", 512)
    lookahead = config.get("prefill_offload_lookahead", True)
    if not isinstance(offload_enabled, bool):
        raise ValueError("prefill_cpu_offload must be a boolean")
    if type(chunk_tokens) is not int or chunk_tokens <= 0:
        raise ValueError("prefill_offload_chunk_tokens must be a positive integer")
    if not isinstance(lookahead, bool):
        raise ValueError("prefill_offload_lookahead must be a boolean")
    prefill_offload = None
    pending_records = max(map(len, schedules)) - 1
    if offload_enabled and pending_records:
        from vllm.config import CUDAGraphMode

        graph_mode = vllm_config.compilation_config.cudagraph_mode
        if graph_mode.mixed_mode() == CUDAGraphMode.FULL:
            raise ValueError(
                "Prefill CPU offload requires eager prefill attention; "
                "use FULL_DECODE_ONLY, PIECEWISE or NONE CUDA graphs"
            )
        prefill_offload = PrefillCPUOffload(
            chunk_tokens,
            pending_records,
            model_config.num_attention_heads,
            model_config.hidden_size // model_config.num_attention_heads,
        )
        prefill_offload.lookahead_enabled = lookahead
        prefill_offload.reserve(vllm_config.scheduler_config.max_num_batched_tokens)

    return prefill_offload, lookahead


def _install_attention_backend(
    layer, owner, settings, context, implementation, implementation_parameters
):
    if owner is not None:
        validate_kv_sharing_target(layer.layer_name, owner, context)
    parameters = dict(settings)
    extra_impl_args = parameters.pop("extra_impl_args")
    parameters.update(
        num_heads=layer.num_heads,
        head_size=layer.head_size,
        num_kv_heads=layer.num_kv_heads,
        sliding_window=layer.sliding_window,
        kv_cache_dtype=layer.kv_cache_dtype,
        attn_type=layer.attn_type,
        kv_sharing_target_layer_name=owner,
    )
    layer.impl = implementation(
        **{
            name: parameters[name]
            for name in implementation_parameters
            if name in parameters
        },
        **extra_impl_args,
    )
    layer.attn_backend = PackedBackend
    layer.backend = backend_name_to_enum(PackedBackend.get_name())
    layer.use_output = PackedBackend.accept_output_buffer
    layer.kv_sharing_target_layer_name = owner


def install_model_attention(config, data, vllm_config, attention_settings):
    """Configure existing recurrent Attention objects before KV allocation."""
    model_config = vllm_config.model_config.hf_config
    page_tokens = vllm_config.cache_config.block_size
    recurrent_layer_count = model_config.n_layers_in_recurrent_block
    recurrent_slots_start = (
        model_config.n_layers_in_prelude + model_config.n_layers_in_coda
    )

    reference_mode = config["reference_mode"]
    coefficient_method = config["coefficient_method"]
    schedules = config["packed_groups"]
    if reference_mode not in ("adjacent", "anchor"):
        raise ValueError("Unsupported reference")
    if reference_mode == "adjacent" and config["anchor_index"] != 0:
        raise ValueError("Adjacent reference currently requires anchor_index=0")
    if coefficient_method not in ("unit", "norm", "ls"):
        raise ValueError("Unsupported coefficient method")
    if not schedules or any(not records for records in schedules):
        raise ValueError("Every packed group must contain at least one record")
    if sum(map(len, schedules)) != model_config.mean_recurrence:
        raise ValueError(
            "Packed groups must describe every model recurrence exactly once"
        )

    prefill_offload, lookahead = _create_prefill_offload(config, vllm_config)

    context = vllm_config.compilation_config.static_forward_context
    slots = {
        extract_layer_index(layer.layer_name): layer
        for layer in get_layers_from_vllm_config(vllm_config, Attention).values()
    }
    implementation = PackedBackend.get_impl_cls()
    implementation_parameters = inspect.signature(implementation).parameters
    shared_key_rotations = None
    if data is not None and data["key_rotations"].ndim == 4:
        shared_key_rotations = data["key_rotations"].contiguous().cuda()

    group_start = 0
    for group_index, schedule in enumerate(schedules):
        group_size = len(schedule)
        records = tuple(tuple(record) for record in schedule)

        group_key_rotations = shared_key_rotations
        group_value_rotations = group_key_means = None
        if data is not None:
            group_key_means = data["key_means"][group_start : group_start + group_size]
            if data["key_rotations"].ndim == 5:
                group_key_rotations = (
                    data["key_rotations"][group_index].contiguous().cuda()
                )
            if data["value_rotations"] is not None:
                group_value_rotations = (
                    data["value_rotations"][group_index].contiguous().cuda()
                )

        offload_chain = []
        for recurrent_layer_index in range(recurrent_layer_count):
            group = DeferredKVGroup(config, records, page_tokens, prefill_offload)
            if prefill_offload is not None and lookahead:
                group.offload_chain = offload_chain
                group.offload_index = recurrent_layer_index
                offload_chain.append(group)
            owner_index = (
                recurrent_slots_start
                + group_start * recurrent_layer_count
                + recurrent_layer_index
            )
            cache_owner_name = slots[owner_index].layer_name
            key_rotation = value_rotation = key_mean = None

            if group_key_means is not None:
                key_mean = group_key_means[:, recurrent_layer_index].contiguous().cuda()
            if group_key_rotations is not None:
                key_rotation = group_key_rotations[recurrent_layer_index]
            if group_value_rotations is not None:
                value_rotation = group_value_rotations[recurrent_layer_index]

            for group_loop_index in range(group_size):
                recurrence = group_start + group_loop_index
                index = (
                    recurrent_slots_start
                    + recurrence * recurrent_layer_count
                    + recurrent_layer_index
                )
                layer = slots[index]
                owner = cache_owner_name if group_loop_index > 0 else None
                _install_attention_backend(
                    layer,
                    owner,
                    attention_settings[layer.layer_name],
                    context,
                    implementation,
                    implementation_parameters,
                )
                layer.packed_group = group
                layer.group_loop_index = group_loop_index
                layer.key_rotation = key_rotation
                layer.value_rotation = value_rotation
                layer.key_mean = key_mean

        group_start += group_size
