"""Shared FP32 rotation and attention-loss math for calibration."""

import torch


def hadamard(device="cpu", dimension=128):
    if dimension == 96:
        residues = {i * i % 11 for i in range(1, 11)}
        matrix = torch.ones(12, 12, dtype=torch.float32, device=device)
        for i in range(11):
            for j in range(11):
                matrix[i + 1, j + 1] = (
                    -1 if i == j else 1 if (i - j) % 11 in residues else -1
                )
        sylvester = torch.ones(1, 1, dtype=torch.float32, device=device)
        while sylvester.shape[0] < 8:
            sylvester = torch.cat(
                (
                    torch.cat((sylvester, sylvester), 1),
                    torch.cat((sylvester, -sylvester), 1),
                ),
                0,
            )
        return torch.kron(matrix, sylvester) / (96**0.5)
    if dimension <= 0 or dimension & (dimension - 1):
        raise ValueError("Hadamard dimension must be a power of two or 96")
    matrix = torch.ones(1, 1, device=device, dtype=torch.float32)
    while matrix.shape[0] < dimension:
        matrix = torch.cat(
            [torch.cat([matrix, matrix], 1), torch.cat([matrix, -matrix], 1)], 0
        )
    return matrix / (dimension**0.5)


def rotate(x, rotation):
    return torch.bmm(x.float().transpose(0, 1), rotation.float()).transpose(0, 1)


def rounded(x, dtype, ste=False):
    y = x.to(dtype).float()
    return x + (y - x).detach() if ste else y


def probabilities(q, k, indices, head_dim):
    logits = torch.bmm(q.float().transpose(0, 1), k.float().permute(1, 2, 0)) / (
        head_dim**0.5
    )
    mask = torch.arange(k.shape[0], device=k.device)[None, :] > indices[:, None]
    return logits.masked_fill(mask[None], -float("inf")).softmax(-1)


def output_error(delta, gram, hidden):
    # Per-head post-WO error, exact without materializing the hidden dimension.
    return (torch.bmm(delta.float(), gram.float()) * delta.float()).sum(-1).mean(
        -1
    ) / hidden
