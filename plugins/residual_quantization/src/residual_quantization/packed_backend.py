"""vLLM packed attention execution, metadata building and backend contracts."""

from dataclasses import replace

from vllm.attention.utils.fa_utils import flash_attn_varlen_func
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
    UbatchSlice,
    split_attn_metadata,
)

from residual_quantization._cuda import (
    attention,
    quantize,
)

__all__ = ("PackedBackend",)


class PackedImpl(FlashAttentionImpl):
    """Common packed-backend implementation with native full-prompt prefill."""

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        if output is None or output_scale is not None or output_block_scale is not None:
            raise ValueError("BF16 output required")
        if attn_metadata is None:
            return output

        actual_token_count = attn_metadata.slot_mapping.numel()
        group = layer.packed_group
        offload = group.prefill_offload is not None and attn_metadata.max_query_len > 1
        lookahead = (
            offload
            and len(group.records) > 1
            and len(group.pending) + 1 == len(group.records)
            and group.prefill_offload.lookahead_enabled
        )
        if lookahead:
            group.prefill_offload.prepare(group, actual_token_count)

        for begin, end, request_metadata in attn_metadata.request_metadata:
            request_query = query[begin:end]
            request_key = key[begin:end]
            request_value = value[begin:end]
            request_output = output[begin:end]

            if request_metadata.max_query_len > 1:
                # Varlen boundaries keep the batched full prompts independent.
                flash_attn_varlen_func(
                    q=request_query,
                    k=request_key,
                    v=request_value,
                    out=request_output,
                    cu_seqlens_q=request_metadata.query_start_loc,
                    cu_seqlens_k=request_metadata.query_start_loc,
                    max_seqlen_q=request_metadata.max_query_len,
                    max_seqlen_k=request_metadata.max_query_len,
                    softmax_scale=query.shape[-1] ** -0.5,
                    causal=True,
                    fa_version=self.vllm_flash_attn_version,
                )
            else:
                attention(
                    request_query,
                    request_key,
                    request_value,
                    kv_cache,
                    request_output,
                    request_metadata.query_start_loc,
                    request_metadata.seq_lens,
                    request_metadata.block_table,
                    layer.key_rotation,
                    layer.value_rotation,
                    layer.key_mean,
                    group.records,
                    group.reference_mode,
                    group.anchor_index,
                    layer.group_loop_index,
                )

        self._store_kv(layer, key, value, kv_cache, attn_metadata, offload, lookahead)

        return output

    def _store_kv(self, layer, key, value, kv_cache, attn_metadata, offload, lookahead):
        group = layer.packed_group
        actual_token_count = attn_metadata.slot_mapping.numel()
        records = group.stage(
            key[:actual_token_count], value[:actual_token_count], offload=offload
        )
        if (
            offload
            and records is None
            and len(group.pending) + 1 == len(group.records)
            and group.prefill_offload.lookahead_enabled
        ):
            group.prefill_offload.preload(group, actual_token_count)
        if records is not None:
            if lookahead:
                chunks = group.prefill_offload.chunks(
                    records, attn_metadata.slot_mapping, group=group
                )
            elif offload and len(records) > 1:
                chunks = group.prefill_offload.chunks(
                    records, attn_metadata.slot_mapping
                )
            else:
                chunks = ((records, attn_metadata.slot_mapping),)
            for chunk_records, chunk_slots in chunks:
                quantize(
                    chunk_records,
                    kv_cache,
                    chunk_slots,
                    layer.key_rotation,
                    layer.value_rotation,
                    layer.key_mean,
                    group.records,
                    group.reference_mode,
                    group.anchor_index,
                    group.coefficient_method,
                )


class PackedBuilder(FlashAttentionMetadataBuilder):
    # vLLM puts one-token requests first before preparing model inputs.
    reorder_batch_threshold = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aot_schedule = False

    def use_cascade_attention(self, *args, **kwargs):
        return False

    def build(
        self,
        common_prefix_len,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build=False,
    ):
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)

        if common_attn_metadata.max_query_len == 1:
            # Decode reuses graph buffers without reading stale CPU offsets.
            metadata.request_metadata = [
                (0, metadata.num_actual_tokens, replace(metadata))
            ]
            return metadata

        starts = common_attn_metadata.query_start_loc_cpu.tolist()
        slices = []
        for request, (begin, end) in enumerate(zip(starts, starts[1:])):
            query_length = end - begin
            if query_length > 1:
                if query_length != int(common_attn_metadata.seq_lens_cpu[request]):
                    raise ValueError(
                        "Chunked/recomputed partial prefill is unsupported"
                    )

                # Split the decode prefix from the full-prompt suffix.
                if not slices:
                    if request > 0:
                        slices.append(UbatchSlice(slice(0, request), slice(0, begin)))
                    slices.append(
                        UbatchSlice(
                            slice(request, common_attn_metadata.num_reqs),
                            slice(begin, common_attn_metadata.num_actual_tokens),
                        )
                    )

        metadata.request_metadata = []
        for part, request_metadata in zip(
            slices, split_attn_metadata(slices, common_attn_metadata)
        ):
            metadata.request_metadata.append(
                (
                    part.token_slice.start,
                    part.token_slice.stop,
                    super().build(0, request_metadata, fast_build),
                )
            )

        return metadata


class PackedBackend(FlashAttentionBackend):
    """Packed recurrent attention with its own cache and metadata contracts."""

    @staticmethod
    def get_name():
        return "LOOP_PACKED_ACCELERATED"

    @staticmethod
    def get_impl_cls():
        return PackedImpl

    @staticmethod
    def get_builder_cls():
        return PackedBuilder

    @staticmethod
    def get_kv_cache_stride_order():
        return (0, 1, 2, 3, 4)

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
        return (2, num_blocks, num_kv_heads, block_size, head_size)
