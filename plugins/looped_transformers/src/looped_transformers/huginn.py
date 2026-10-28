"""Fixed 32-loop Huginn with depth-specific attention and shared weights."""

from types import MethodType

import raven_vllm
import torch
from torch import nn
from vllm.attention.layer import Attention

from looped_transformers.model_setup import prepare_model_config


def _forward_recurrent_layer(layer, freqs_cis, hidden_states, attention):
    """Run shared Huginn projections and route reconstructed K/V by depth."""
    residual = hidden_states
    normalized = layer.norm_1(hidden_states)
    qkv, _ = layer.self_attn.qkv_proj(normalized)
    q, k, v = qkv.split(
        [layer.self_attn.q_size, layer.self_attn.kv_size, layer.self_attn.kv_size],
        dim=-1,
    )
    total_tokens = hidden_states.shape[0]
    q = q.view(total_tokens, layer.self_attn.num_heads, layer.self_attn.head_dim)
    k = k.view(total_tokens, layer.self_attn.num_kv_heads, layer.self_attn.head_dim)
    q_bias, k_bias = layer.self_attn.qk_bias.split(1, dim=0)
    q = (q + q_bias).to(q.dtype)
    k = (k + k_bias).to(q.dtype)
    q, k = layer.self_attn._apply_rotary_emb_complex_like(q, k, freqs_cis)
    attended = attention(q.reshape(total_tokens, -1), k.reshape(total_tokens, -1), v)
    attention_output, _ = layer.self_attn.o_proj(attended)
    hidden_states = layer.norm_2(attention_output + residual)
    mlp_input = layer.norm_3(hidden_states)
    mlp_output = layer.mlp(mlp_input)
    return layer.norm_4(mlp_output + hidden_states)


class HuginnAttentionBank:
    def __init__(self, model, vllm_config):
        recurrent_layer_count = model.config.n_layers_in_recurrent_block
        recurrent_slots_start = (
            model.config.n_layers_in_prelude + model.config.n_layers_in_coda
        )
        physical = [
            model.layers[model.config.n_layers_in_prelude + i].self_attn.attn
            for i in range(recurrent_layer_count)
        ]
        slots = list(physical)
        for recurrence in range(1, model.config.mean_recurrence):
            for layer, source in enumerate(physical):
                slots.append(
                    Attention(
                        source.num_heads,
                        source.head_size,
                        source.head_size**-0.5,
                        num_kv_heads=source.num_kv_heads,
                        quant_config=vllm_config.quant_config,
                        cache_config=vllm_config.cache_config,
                        prefix=(
                            "model.recurrent_virtual_kv."
                            f"{recurrent_slots_start + recurrence * recurrent_layer_count + layer}"
                        ),
                        attn_backend=None,
                        kv_sharing_target_layer_name=None,
                    )
                )
        self.slots = tuple(slots)
        self.recurrent_layer_count = recurrent_layer_count
        model.recurrent_virtual_attentions = nn.ModuleList(
            slots[recurrent_layer_count:]
        )

    def slot(self, recurrence, layer):
        return self.slots[recurrence * self.recurrent_layer_count + layer]


def _fixed_forward(
    model, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
):
    if intermediate_tensors is not None:
        raise NotImplementedError("pipeline intermediate tensors are unsupported")
    input_embeds = (
        model.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
    )
    if model.embed_scale != 1.0:
        input_embeds *= model.embed_scale
    freqs_cis = model.freqs_cis.index_select(0, positions)
    for layer_idx in range(model.config.n_layers_in_prelude):
        input_embeds = model.layers[layer_idx](input_embeds, freqs_cis)
    hidden_states = model.initialize_state(
        input_embeds, scale=float(getattr(model.config, "vllm_init_scale", 1.0))
    )
    for recurrence in range(model.config.mean_recurrence):
        adapter_input = torch.cat([hidden_states, input_embeds], dim=-1)
        hidden_states, _ = model.adapter(adapter_input)
        for layer_idx in range(model.config.n_layers_in_recurrent_block):
            layer = model.layers[model.config.n_layers_in_prelude + layer_idx]
            attention = model.recurrent_attention_bank.slot(recurrence, layer_idx)
            hidden_states = _forward_recurrent_layer(
                layer, freqs_cis, hidden_states, attention
            )
    hidden_states = model.ln_f(hidden_states)
    coda_layer_start = (
        model.config.n_layers_in_prelude + model.config.n_layers_in_recurrent_block
    )
    for layer_idx in range(model.config.n_layers_in_coda):
        hidden_states = model.layers[coda_layer_start + layer_idx](
            hidden_states, freqs_cis
        )
    return model.ln_f(hidden_states)


class HuginnForVLLM(raven_vllm.RavenForvLLM):
    def __init__(self, vllm_config, prefix=""):
        config = prepare_model_config(vllm_config.model_config.hf_config)
        original_attention = raven_vllm.Attention

        def create_physical_attention(*args, **kwargs):
            # Rename attention slots only. Physical modules and checkpoint
            # names retain Raven's prelude/recurrent/coda weight ordering.
            layer_prefix, layer_suffix = kwargs["prefix"].rsplit(".layers.", 1)
            index, separator, suffix = layer_suffix.partition(".")
            index = int(index)
            coda_start = config.n_layers_in_prelude + config.n_layers_in_recurrent_block
            if config.n_layers_in_prelude <= index < coda_start:
                index += config.n_layers_in_coda
            elif index >= coda_start:
                index -= config.n_layers_in_recurrent_block
            kwargs["prefix"] = f"{layer_prefix}.layers.{index}{separator}{suffix}"
            return Attention(*args, **kwargs)

        raven_vllm.Attention = create_physical_attention
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            raven_vllm.Attention = original_attention
        if self.config.mean_recurrence != 32:
            raise ValueError("Huginn requires 32 depth-specific recurrences")
        self.model.recurrent_attention_bank = HuginnAttentionBank(
            self.model, vllm_config
        )
        self.model.forward = MethodType(_fixed_forward, self.model)

    @staticmethod
    def _map_fused_weight(name, value):
        """Map HF ModelOpt fused-linear auxiliaries to vLLM parameters."""
        mappings = (
            ("attn.Wqkv.", "self_attn.qkv_proj.", 3),
            ("attn.proj.", "self_attn.o_proj.", 1),
            ("mlp.fc.", "mlp.gate_up_proj.", 2),
            ("mlp.proj.", "mlp.down_proj.", 1),
        )
        for source, target, shards in mappings:
            if source not in name:
                continue
            name = name.replace(source, target)
            if (
                shards > 1
                and value.ndim == 0
                and name.endswith((".input_scale", ".weight_scale_2"))
            ):
                value = value.repeat(shards)
            break
        return (name, value)

    def load_weights(self, weights):
        return super().load_weights(
            self._map_fused_weight(name, value) for name, value in weights
        )

    def compute_logits(self, hidden_states, sampling_metadata=None):
        return super().compute_logits(hidden_states)
