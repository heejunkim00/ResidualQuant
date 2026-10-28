"""Shared model weight transformations applied after checkpoint loading."""

import torch

__all__ = ("prepare_value_rotation", "absorb_model_value_rotations")


@torch.no_grad()
def absorb_value_rotation(attention, rotation):
    """Fold row-vector V @ R and output @ R.T into BF16 projections."""
    heads = attention.num_kv_heads
    dimension = attention.head_dim
    if rotation.shape != (heads, dimension, dimension):
        raise ValueError("Value rotation does not match attention geometry")
    qkv = attention.qkv_proj.weight
    output = attention.o_proj.weight
    if qkv.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
        raise ValueError("Value rotation absorption requires BF16 weights")
    value = qkv[attention.q_size + attention.kv_size :]
    hidden = value.shape[1]
    if value.shape[0] != heads * dimension or output.shape[1] != heads * dimension:
        raise ValueError("Value rotation absorption requires tensor parallel size 1")
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        rotation = rotation.float()
        transformed = torch.bmm(
            rotation.transpose(1, 2), value.float().reshape(heads, dimension, hidden)
        )
        value.copy_(transformed.reshape_as(value))
        transformed = torch.bmm(
            output.float().reshape(-1, heads, dimension).permute(1, 0, 2),
            rotation,
        )
        output.copy_(transformed.permute(1, 0, 2).reshape_as(output))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32


def prepare_value_rotation(artifact):
    """Select projection folding from the artifact matrix dimensions."""
    if artifact is None:
        return None, None
    rotation = artifact["value_rotations"]
    if rotation.ndim == 5:
        return artifact, None
    if rotation.ndim != 4:
        raise ValueError("Value rotations must have four or five dimensions")
    return dict(artifact, value_rotations=None), rotation


@torch.no_grad()
def absorb_model_value_rotations(model, rotations):
    """Transform each physical layer once, after checkpoint loading."""
    start = model.config.n_layers_in_prelude
    recurrent_layer_count = model.config.n_layers_in_recurrent_block
    if rotations.shape[0] != recurrent_layer_count:
        raise ValueError("Value rotations do not match the physical layer count")
    for layer, rotation in zip(
        model.model.layers[start : start + recurrent_layer_count], rotations
    ):
        absorb_value_rotation(layer.self_attn, rotation.cuda())
