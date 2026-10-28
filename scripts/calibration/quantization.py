"""FP32 OptR quantization and attention-output objectives."""

from functools import partial

import torch
from calibration.rotation_math import (
    rotate,
    rounded,
    probabilities as default_probabilities,
    output_error as default_output_error,
)

CLIP_BY_BITS = {2: 0.9, 3: 0.95, 4: 1.0}


def predict(x, reference):
    if reference is None:
        return (torch.zeros_like(x), None)
    coefficient = (
        (x.float() * reference.float()).sum(-1, keepdim=True)
        / reference.float().square().sum(-1, keepdim=True).clamp_min(1e-12)
    ).bfloat16()
    return ((reference.float() * coefficient.float()).bfloat16(), coefficient)


def quantization_target(x, reference):
    prediction, coefficient = predict(x, reference)
    return ((x.float() - prediction.float()).bfloat16(), prediction, coefficient)


def affine_qdq(residual, rotation, bits, group_size, mean=None, ste=False):
    if bits not in (2, 3, 4):
        raise ValueError("Only INT2/3/4")
    centered = residual.float() if mean is None else residual.float() - mean.float()
    rotated = rotate(centered, rotation)
    if group_size not in (16, 32) or residual.shape[-1] % group_size:
        raise ValueError(
            "Group size must be 16 or 32 and divide the head dimension"
        )
    grouped = rotated.reshape(
        *rotated.shape[:-1], residual.shape[-1] // group_size, group_size
    )
    minimum, maximum = (grouped.amin(-1, keepdim=True), grouped.amax(-1, keepdim=True))
    center = (maximum + minimum) * 0.5
    radius = (maximum - minimum) * 0.5 * CLIP_BY_BITS[bits]
    lower, upper = (center - radius, center + radius)
    offset = lower.clamp(-448, 448).to(torch.float8_e4m3fn).float()
    scale = (2 * radius / (2**bits - 1)).clamp_max(448).to(torch.float8_e4m3fn).float()
    normalized = (grouped - offset) / torch.where(
        scale > 0, scale, torch.ones_like(scale)
    )
    floor = normalized.floor()
    code = (floor + (normalized - floor > 0.5)).clamp(0, 2**bits - 1)
    decoded = code * scale + offset
    if ste:
        surrogate = torch.clamp(grouped, min=lower.detach(), max=upper.detach())
        decoded = surrogate + (decoded - surrogate).detach()
    restored = rotate(decoded.reshape_as(rotated), rotation.transpose(-1, -2))
    if mean is not None:
        restored = restored + mean.float()
    return rounded(restored, torch.bfloat16, ste)


def reconstruct(x, reference, rotation, bits, group_size, mean=None, ste=False):
    residual, prediction, _ = quantization_target(x, reference)
    return rounded(
        prediction.float()
        + affine_qdq(residual, rotation, bits, group_size, mean, ste),
        torch.bfloat16,
        ste,
    )


def output_loss(
    trace,
    target,
    reference,
    rotation,
    bits,
    group_size,
    gram,
    mean=None,
    fixed_k=None,
    ste=False,
    query_window=64,
    lam=1.0,
    probability_fn=None,
    output_error_fn=None,
):
    """OptR per-head K KL+post-WO / V post-WO objective on tail queries."""
    probabilities = probability_fn or partial(
        default_probabilities, head_dim=trace["q"].shape[-1]
    )
    output_error = output_error_fn or partial(
        default_output_error, hidden=trace["q"].shape[1] * trace["q"].shape[2]
    )
    q = trace["q"][-query_window:]
    indices = trace["query_indices"][-query_window:]
    actual = reconstruct(
        trace[target], reference, rotation, bits, group_size, mean, ste
    )
    with torch.no_grad():
        p = probabilities(q, trace["k"], indices)
    if target == "k":
        pq = probabilities(q, actual, indices)
        delta = (pq - p) @ trace["v"].float().transpose(0, 1)
        kl = (
            (p * (p.clamp_min(1e-12).log() - pq.clamp_min(1e-12).log()))
            .sum(-1)
            .mean(-1)
        )
        return kl + lam * output_error(delta, gram)
    if fixed_k is None:
        raise ValueError("V calibration requires selected reconstructed keys")
    with torch.no_grad():
        pq = probabilities(q, fixed_k, indices)
    delta = pq @ (actual - trace["v"]).float().transpose(0, 1)
    return output_error(delta, gram)
