"""FP32 OptR calibration with packed-code reconstruction."""

import json

import torch

from calibration.quantization import output_loss, quantization_target, reconstruct


class FP32Chain:
    """OptR objective parameterized by precision schedule and reference policy.

    LS is computed before rotation in original coordinates. References are
    detached BF16 reconstructed quantized values; rotations/optimizer are FP32.
    """

    def __init__(
        self,
        bits,
        mode,
        groups,
        probability_fn=None,
        output_error_fn=None,
        anchor_index=None,
        independent=False,
    ):
        self.bits = tuple(bits)
        self.mode = mode
        self.groups = groups
        self.independent = independent
        self.mapping = [0, 1, 2, 3] if independent else [0] * 4
        self.anchor_index = anchor_index
        self.order = (
            [anchor_index] + [i for i in range(4) if i != anchor_index]
            if mode == "anchor" and not independent
            else list(range(4))
        )
        self.probability_fn = probability_fn
        self.output_error_fn = output_error_fn

    def reference_depth(self, depth):
        if self.independent or depth == self.order[0]:
            return None
        return depth - 1 if self.mode == "adjacent" else self.anchor_index

    def reconstruct(self, sample, target, rotations, means=None):

        outputs = [None] * 4
        for depth in self.order:
            t = sample[depth]
            rd = self.reference_depth(depth)
            ref = None if rd is None else outputs[rd].detach()
            outputs[depth] = reconstruct(
                t[target],
                ref,
                rotations[self.mapping[depth]],
                self.bits[depth],
                self.groups[depth],
                None if means is None else means[depth],
                ste=torch.is_grad_enabled(),
            )
        return outputs

    def loss(self, sample, target, rotations, gram, means=None, fixed_keys=None):

        outputs = [None] * 4
        losses = [None] * 4
        for depth in self.order:
            t = sample[depth]
            rd = self.reference_depth(depth)
            ref = None if rd is None else outputs[rd]
            r = rotations[self.mapping[depth]]
            mean = None if means is None else means[depth]
            losses[depth] = output_loss(
                t,
                target,
                ref,
                r,
                self.bits[depth],
                self.groups[depth],
                gram,
                mean,
                None if fixed_keys is None else fixed_keys[depth],
                ste=torch.is_grad_enabled(),
                probability_fn=self.probability_fn,
                output_error_fn=self.output_error_fn,
            )
            with torch.no_grad():
                outputs[depth] = reconstruct(
                    t[target], ref, r, self.bits[depth], self.groups[depth], mean
                )
        return torch.stack(losses).mean(0)


def select_best_rotations(candidates, values):
    """Choose each head's lowest holdout-loss candidate, including the seed."""
    selected = []
    metrics = []
    for unit, choices in enumerate(candidates):
        scores = [torch.stack(losses).mean(0) for losses in values[unit]]
        best = scores[0].clone()
        best_r = choices[0].clone()
        for rotation, current in zip(choices[1:], scores[1:]):
            better = current < best
            best_r[:, better] = rotation[:, better]
            best = torch.minimum(best, current)
        selected.append(best_r)
        metrics.append({"initial": scores[0].tolist(), "selected": best.tolist()})
    return selected, metrics


def streaming_key_means(sample, unit_count, head_count, fit_count, initial):
    """Accumulate Direct fit-token K means in their original token order."""
    dimension = initial.shape[-1]
    totals = torch.zeros((unit_count, 4, head_count, dimension), device=initial.device)
    tokens = 0
    with torch.no_grad():
        for ordinal in range(fit_count):
            batch = sample(ordinal)
            for unit in range(unit_count):
                for depth in range(4):
                    totals[unit, depth] += (
                        quantization_target(batch[unit][depth]["k"], None)[0]
                        .float()
                        .sum(0)
                    )
            tokens += len(batch[0][0]["k"])
            del batch
        means = totals / tokens
    return means


def train_direct_rotations(
    sample, chain, initial, means, grams, target, steps, fit_count, chosen_k=None
):
    """Fit one optimizer and one objective per physical layer/recurrence."""
    if not chain.independent:
        raise ValueError("Independent fitting requires Direct")
    base = initial[0]
    records = [(unit, loop) for unit in range(len(means)) for loop in range(4)]
    parameters = [torch.zeros_like(base, requires_grad=True) for _ in records]
    optimizers = [torch.optim.Adam([parameter], lr=0.02) for parameter in parameters]
    candidates = [[base.unsqueeze(0).clone()] for _ in records]
    for step in range(steps):
        for optimizer in optimizers:
            optimizer.zero_grad()
        # Generate each fit prompt once; every record has its own loss/gradient.
        for batch_index in range(2):
            batch = sample((2 * step + batch_index) % fit_count)
            for index, (unit, loop) in enumerate(records):
                trace = batch[unit][loop]
                rotation = base @ torch.matrix_exp(
                    parameters[index] - parameters[index].transpose(-1, -2)
                )
                keys = None
                if target == "v":
                    with torch.no_grad():
                        keys = reconstruct(
                            trace["k"],
                            None,
                            chosen_k[unit][loop],
                            chain.bits[loop],
                            chain.groups[loop],
                            means[unit, loop],
                        )
                loss = (
                    output_loss(
                        trace,
                        target,
                        None,
                        rotation,
                        chain.bits[loop],
                        chain.groups[loop],
                        grams[unit % len(grams)],
                        means[unit, loop] if target == "k" else None,
                        keys,
                        ste=True,
                        probability_fn=chain.probability_fn,
                        output_error_fn=chain.output_error_fn,
                    ).mean()
                    / 2
                )
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite Direct calibration loss")
                loss.backward()
                del rotation, loss, keys
            del batch, trace
        for index, optimizer in enumerate(optimizers):
            if not torch.isfinite(parameters[index].grad).all():
                raise ValueError("Non-finite Direct calibration gradient")
            optimizer.step()
            if step % 20 == 0 or step == steps - 1:
                with torch.no_grad():
                    rotation = base @ torch.matrix_exp(
                        parameters[index] - parameters[index].transpose(-1, -2)
                    )
                    candidates[index].append(rotation.unsqueeze(0).clone())
        if step % 20 == 0 or step == steps - 1:
            print(
                json.dumps(
                    dict(
                        target=target,
                        step=step,
                        independent_records=len(records),
                        allocated_gib=torch.cuda.memory_allocated() / 2**30,
                    )
                ),
                flush=True,
            )
    return candidates


def select_direct_rotations(
    sample, chain, candidates, means, grams, target, hold_count, chosen_k=None
):
    """Choose each recurrence/head's checkpoint from its own holdout losses."""
    if not chain.independent:
        raise ValueError("Independent selection requires Direct")
    records = [(unit, loop) for unit in range(len(means)) for loop in range(4)]
    values = [[[] for _ in choices] for choices in candidates]
    with torch.no_grad():
        for ordinal in range(128, 128 + hold_count):
            batch = sample(ordinal)
            for index, (unit, loop) in enumerate(records):
                trace = batch[unit][loop]
                keys = (
                    None
                    if target == "k"
                    else reconstruct(
                        trace["k"],
                        None,
                        chosen_k[unit][loop],
                        chain.bits[loop],
                        chain.groups[loop],
                        means[unit, loop],
                    )
                )
                for checkpoint, rotation in enumerate(candidates[index]):
                    values[index][checkpoint].append(
                        output_loss(
                            trace,
                            target,
                            None,
                            rotation[0],
                            chain.bits[loop],
                            chain.groups[loop],
                            grams[unit % len(grams)],
                            means[unit, loop] if target == "k" else None,
                            keys,
                            probability_fn=chain.probability_fn,
                            output_error_fn=chain.output_error_fn,
                        )
                    )
            del batch, trace, keys
    selected, metrics = select_best_rotations(candidates, values)
    return (
        [torch.cat(selected[unit * 4 : unit * 4 + 4]) for unit in range(len(means))],
        [metrics[unit * 4 : unit * 4 + 4] for unit in range(len(means))],
    )


def fit_direct(sample, chain, initial, grams, unit_count, head_count, stage):
    means = stage(
        "fit_means",
        lambda: streaming_key_means(sample, unit_count, head_count, 128, initial),
    )
    selected, metrics = {}, {}
    for target in ("k", "v"):
        candidates = stage(
            f"train_{target}",
            lambda: train_direct_rotations(
                sample, chain, initial, means, grams, target, 80, 128, selected.get("k")
            ),
        )
        selected[target], metrics[target] = stage(
            f"holdout_{target}",
            lambda candidates=candidates: select_direct_rotations(
                sample, chain, candidates, means, grams, target, 128, selected.get("k")
            ),
        )
        del candidates
    return means, selected["k"], selected["v"], metrics["k"], metrics["v"]
