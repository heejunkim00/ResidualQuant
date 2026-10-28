"""Inference-only Ouro implementation for vLLM 0.10.2.

Trainable decoder parameters are stored once. Each `(UT step, physical layer)`
gets a separate, weight-free vLLM Attention object so its KV cache matches
Ouro's `current_ut * num_hidden_layers + layer_idx` cache indexing.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn
from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen2 import Qwen2MLP
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from looped_transformers.model_setup import prepare_model_config


class OuroAttentionBank(nn.Module):
    def __init__(self, layers, loops, prefix, factory):
        super().__init__()
        self.layers = layers
        self.slots = nn.ModuleList(
            factory(f"{prefix}.virtual_attention.{i}") for i in range(layers * loops)
        )

    def virtual_index(self, loop, layer):
        return loop * self.layers + layer

    def slot(self, loop, layer):
        return self.slots[self.virtual_index(loop, layer)]


class OuroSharedAttention(nn.Module):
    """Own QKV/O weights once and receive a virtual cache slot per call."""

    def __init__(
        self, config, quant_config: Optional[QuantizationConfig], prefix: str
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        if config.num_attention_heads % tp_size != 0:
            raise ValueError(
                "num_attention_heads must be divisible by tensor parallel size"
            )
        if config.num_key_value_heads % tp_size != 0:
            raise ValueError(
                "num_key_value_heads must be divisible by tensor parallel size"
            )
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=getattr(config, "rope_scaling", None),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_slot: Attention,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        query, key, value = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        query, key = self.rotary_emb(positions, query, key)
        attention_output = attention_slot(query, key, value)
        output, _ = self.o_proj(attention_output)
        return output


class OuroSharedDecoderLayer(nn.Module):
    """One physical Ouro layer, reused at every UT step."""

    def __init__(
        self, config, quant_config: Optional[QuantizationConfig], prefix: str
    ) -> None:
        super().__init__()
        self.self_attn = OuroSharedAttention(
            config, quant_config, f"{prefix}.self_attn"
        )
        self.mlp = Qwen2MLP(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            quant_config,
            f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm_2 = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_slot: Attention,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, attention_slot)
        hidden_states = self.input_layernorm_2(hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_attention_layernorm_2(hidden_states)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class OuroModel(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "model") -> None:
        super().__init__()
        config = prepare_model_config(vllm_config.model_config.hf_config)
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise NotImplementedError("Ouro requires pipeline_parallel_size=1")
        if "sliding_attention" in getattr(config, "layer_types", []):
            raise NotImplementedError("Ouro supports full attention only")
        self.config = config
        self.quant_config = quant_config
        self.total_ut_steps = config.mean_recurrence
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.layers = nn.ModuleList(
            OuroSharedDecoderLayer(
                config, quant_config, prefix=f"{prefix}.layers.{layer_idx}"
            )
            for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.early_exit_gate = nn.Linear(config.hidden_size, 1, bias=True)
        scaling = (config.hidden_size // config.num_attention_heads) ** (-0.5)
        tp_size = get_tensor_model_parallel_world_size()

        def create_attention_slot(slot_prefix: str) -> Attention:
            attention_kwargs = {
                "num_kv_heads": config.num_key_value_heads // tp_size,
                "cache_config": cache_config,
                "quant_config": quant_config,
                "prefix": slot_prefix,
            }
            return Attention(
                config.num_attention_heads // tp_size,
                config.hidden_size // config.num_attention_heads,
                scaling,
                **attention_kwargs,
            )

        self.virtual_attention = OuroAttentionBank(
            config.num_hidden_layers, self.total_ut_steps, prefix, create_attention_slot
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            raise NotImplementedError(
                "pipeline parallel intermediate tensors are unsupported"
            )
        hidden_states = (
            inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        )
        for loop_idx in range(self.total_ut_steps):
            deferred_residual = None
            for layer_idx, layer in enumerate(self.layers):
                attention_slot = self.virtual_attention.slot(loop_idx, layer_idx)
                hidden_states, deferred_residual = layer(
                    positions,
                    hidden_states,
                    attention_slot,
                    residual=deferred_residual,
                )
            hidden_states, _ = self.norm(hidden_states, deferred_residual)
        return hidden_states


class OuroForVLLM(nn.Module):
    """Ouro causal LM backend with shared weights across recurrent steps."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = prepare_model_config(vllm_config.model_config.hf_config)
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        model_prefix = f"{prefix}.model" if prefix else "model"
        self.model = OuroModel(
            vllm_config=vllm_config,
            prefix=model_prefix,
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.lm_head" if prefix else "lm_head",
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids,
            positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata
    ) -> Optional[torch.Tensor]:
        return self.logits_processor(self.lm_head, hidden_states, sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = (
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        )
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for original_name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in original_name:
                continue
            name = original_name
            for packed_name, checkpoint_name, shard_id in stacked_params_mapping:
                if checkpoint_name not in name:
                    continue
                name = name.replace(checkpoint_name, packed_name)
                if name not in params_dict:
                    raise KeyError(
                        f"checkpoint parameter {original_name!r} mapped to missing {name!r}"
                    )
                parameter = params_dict[name]
                weight_loader = getattr(
                    parameter, "weight_loader", default_weight_loader
                )
                weight_loader(parameter, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if name not in params_dict:
                    raise KeyError(f"unexpected Ouro checkpoint parameter: {name}")
                parameter = params_dict[name]
                weight_loader = getattr(
                    parameter, "weight_loader", default_weight_loader
                )
                weight_loader(parameter, loaded_weight)
                loaded_params.add(name)
        return loaded_params


__all__ = [
    "OuroForVLLM",
    "OuroModel",
    "OuroSharedDecoderLayer",
]
