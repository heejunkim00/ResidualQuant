"""Fit shared OptR rotations from saved QKV traces, one physical layer at a time."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


from reproduction import digest, save_tensor


def fixed_means(samples, chain, initial):
    import torch
    from calibration.quantization import quantization_target, reconstruct

    history = [[None] * 4 for _ in samples]
    means = [None] * 4
    with torch.no_grad():
        for depth in chain.order:
            reference_depth = chain.reference_depth(depth)
            refs = [
                None if reference_depth is None else h[reference_depth] for h in history
            ]
            targets = [
                quantization_target(sample[depth]["k"].to(initial.device), ref)[0]
                for sample, ref in zip(samples, refs)
            ]
            means[depth] = torch.cat([z.float() for z in targets], dim=0).mean(0)
            del targets
            for sample, previous, ref in zip(samples, history, refs):
                previous[depth] = reconstruct(
                    sample[depth]["k"].to(initial.device),
                    ref,
                    initial[chain.mapping[depth]],
                    chain.bits[depth],
                    chain.groups[depth],
                    means[depth],
                )
    return torch.stack(means).contiguous()


def fit_layer(fit, held, chains, initial, gram, steps=80):
    import torch

    means = [
        fixed_means([sample[group] for sample in fit], chain, initial)
        for group, chain in enumerate(chains)
    ]
    selected, metrics = {}, {}
    for target in ("k", "v"):

        def score(samples, rotation, subset=None):
            scores = []
            for index in range(len(samples)) if subset is None else subset:
                losses = []
                for group, chain in enumerate(chains):
                    sample = [
                        {
                            name: tensor.to(initial.device)
                            for name, tensor in trace.items()
                        }
                        for trace in samples[index][group]
                    ]
                    with torch.no_grad():
                        keys = (
                            None
                            if target == "k"
                            else chain.reconstruct(
                                sample, "k", selected["k"], means[group]
                            )
                        )
                    losses.append(
                        chain.loss(
                            sample,
                            target,
                            rotation,
                            gram,
                            means[group] if target == "k" else None,
                            keys,
                        )
                    )
                scores.append(torch.stack(losses).mean(0))
            return torch.stack(scores).mean(0)

        with torch.no_grad():
            before = score(held, initial)
            best, best_rotation = before.clone(), initial.clone()
        params = torch.zeros_like(initial, requires_grad=True)
        optimizer = torch.optim.Adam([params], lr=0.02)
        for step in range(steps):
            optimizer.zero_grad()
            rotation = initial @ torch.matrix_exp(params - params.transpose(-1, -2))
            indexes = list(
                dict.fromkeys([(2 * step) % len(fit), (2 * step + 1) % len(fit)])
            )
            loss = score(fit, rotation, indexes).mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite calibration loss")
            loss.backward()
            if not torch.isfinite(params.grad).all():
                raise ValueError("Non-finite calibration gradient")
            optimizer.step()
            if step % 20 != 0 and step != steps - 1:
                continue
            with torch.no_grad():
                rotation = initial @ torch.matrix_exp(params - params.transpose(-1, -2))
                current = score(held, rotation)
                better = current < best
                best_rotation[:, better] = rotation[:, better]
                best = torch.minimum(best, current)
        selected[target] = best_rotation.detach()
        metrics[target] = {"initial": before.tolist(), "selected": best.tolist()}
    rotations = torch.stack([selected[name][0] for name in ("k", "v")]).half().cpu()
    key_means = torch.cat(
        [
            torch.einsum(
                "lhd,lhde->lhe",
                mean.cpu(),
                rotations[0].unsqueeze(0).expand(4, -1, -1, -1).float(),
            )
            for mean in means
        ]
    )
    return dict(
        key_rotations=rotations[0],
        value_rotations=rotations[1],
        key_means=key_means,
        metrics=metrics,
    )


def validate_inputs(config, manifest):
    protocol = json.loads(
        (ROOT / "experiments/KV_main_experiments/prompt_protocol.json").read_text()
    )
    models = [
        m
        for m in protocol["models"].values()
        if m["name"] == manifest["model"] and m["revision"] == manifest["revision"]
    ]
    if (
        not models
        or manifest["precision"] != "bf16"
        or manifest["loops"] != models[0]["loops"]
    ):
        raise ValueError(
            "Trace model, revision or precision does not match the protocol"
        )
    if manifest.get("synthetic", False):
        raise ValueError("Synthetic traces are not supported")
    if manifest["query_count"] < manifest["max_length"]:
        raise ValueError("Calibration requires full-query traces")
    dataset = manifest.get("dataset", "")
    if "gsm8k" in dataset:
        domain = "gsm8k"
    elif "mbpp" in dataset:
        domain = "code" if "huginn" in manifest["model"].lower() else "mbpp"
    else:
        raise ValueError("Expected a GSM8K or MBPP training trace dataset")
    partitions = manifest["partitions"]
    if (
        len(partitions["fit"]) != 128
        or len(partitions["holdout"]) != 128
        or set(partitions["fit"]) & set(partitions["holdout"])
    ):
        raise ValueError("Expected disjoint fit128 and holdout128 partitions")
    schedules = config["packed_groups"]
    if (
        config.get("coefficient_method") != "ls"
        or not config.get("use_rotation")
        or config["reference_mode"] != "anchor"
        or config["anchor_index"] not in (0, 3)
        or not schedules
        or any(len(group) != 4 for group in schedules)
        or any(group != schedules[0] for group in schedules)
        or sum(map(len, schedules)) != manifest["loops"]
    ):
        raise ValueError(
            "Expected identical four-loop LS anchor groups with shared rotations"
        )
    return domain


def load_samples(paths, loop_count, heads, dimension):
    import torch

    samples = []
    for path in paths:
        loops = torch.load(path, map_location="cpu", weights_only=True, mmap=True)[
            "loops"
        ]
        if len(loops) != loop_count:
            raise ValueError("Trace loop count mismatch")
        for trace in loops:
            for name in ("q", "k", "v"):
                tensor = trace[name]
                if (
                    tensor.ndim != 3
                    or tensor.shape[1:] != (heads, dimension)
                    or tensor.dtype != torch.bfloat16
                    or not torch.isfinite(tensor).all()
                ):
                    raise ValueError("Trace geometry or precision mismatch")
        for trace in loops:
            indices = trace["query_indices"]
            if (
                trace["q"].shape != trace["k"].shape
                or trace["k"].shape != trace["v"].shape
                or indices.dtype != torch.int64
                or not torch.equal(indices, torch.arange(len(trace["q"])))
            ):
                raise ValueError("Expected full causal Q/K/V traces")
        samples.append(
            [
                [
                    {k: t[k] for k in ("q", "k", "v", "query_indices")}
                    for t in loops[start : start + 4]
                ]
                for start in range(0, len(loops), 4)
            ]
        )
    return samples


def fit_layers(
    traces, output, selected_layers, loop_count, chains, initial, grams, contract
):
    import fcntl
    import torch

    heads, dimension = grams.shape[1:3]
    for layer in selected_layers:
        with (output / f"layer_{layer:02d}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            paths = [
                traces / f"sample_{i:03d}" / f"layer_{layer:02d}.pt"
                for i in contract["fit"] + contract["holdout"]
            ]
            trace_hashes = {str(p.relative_to(traces)): digest(p) for p in paths}
            checkpoint = output / f"layer_{layer:02d}.pt"
            if checkpoint.exists():
                part = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if part["trace_sha256"] != trace_hashes or part["contract"] != contract:
                    raise ValueError("Layer checkpoint inputs changed")
                continue
            samples = load_samples(paths, loop_count, heads, dimension)
            part = fit_layer(
                samples[: len(contract["fit"])],
                samples[len(contract["fit"]) :],
                chains,
                initial,
                grams[layer].cuda(),
                contract["steps"],
            )
            part.update(layer=layer, trace_sha256=trace_hashes, contract=contract)
            save_tensor(checkpoint, part)
            del samples, part
            torch.cuda.empty_cache()
            print(json.dumps(dict(layer=layer, completed=True)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", help="Comma-separated physical layer indices")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import fcntl
    from functools import partial
    import torch
    from calibration.rotation_math import hadamard, probabilities, output_error
    from calibration.rotation_training import FP32Chain
    from reproduction import atomic

    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = json.loads(args.config.read_text())
    manifest = json.loads((args.traces / "manifest.json").read_text())
    domain = validate_inputs(config, manifest)
    schedules = config["packed_groups"]
    grams = torch.load(
        args.traces / "wo_grams.pt", map_location="cpu", weights_only=True
    )
    if (
        grams.ndim != 4
        or grams.shape[-1] != grams.shape[-2]
        or not torch.isfinite(grams).all()
    ):
        raise ValueError("Invalid output-projection Gram matrices")
    layers, heads, dimension, _ = grams.shape
    selected_layers = (
        list(range(layers))
        if args.layers is None
        else list(map(int, args.layers.split(",")))
    )
    if (
        not selected_layers
        or len(set(selected_layers)) != len(selected_layers)
        or not set(selected_layers) <= set(range(layers))
    ):
        raise ValueError("Invalid physical layer selection")
    initial = (
        hadamard("cuda", dimension=dimension)
        .expand(1, heads, dimension, dimension)
        .clone()
    )
    chains = [
        FP32Chain(
            [bit for bit, _ in group],
            "anchor",
            [size for _, size in group],
            partial(probabilities, head_dim=dimension),
            partial(output_error, hidden=heads * dimension),
            anchor_index=config["anchor_index"],
        )
        for group in schedules
    ]
    fit_ids = list(range(128))
    held_ids = list(range(128, 256))
    steps = 80
    sources = [
        Path(__file__),
        *[
            ROOT / "scripts/calibration" / name
            for name in ("quantization.py", "rotation_math.py", "rotation_training.py")
        ],
    ]
    contract = dict(
        model=manifest["model"],
        revision=manifest["revision"],
        dataset=manifest["dataset"],
        calibration_domain=domain,
        fit=fit_ids,
        holdout=held_ids,
        steps=steps,
        seed=0,
        lr=0.02,
        query_window=64,
        physical_layers=layers,
        heads=heads,
        head_dim=dimension,
        rotation_sharing="all_loops_per_physical_layer",
        trace_manifest_sha256=digest(args.traces / "manifest.json"),
        gram_sha256=digest(args.traces / "wo_grams.pt"),
        config_sha256=digest(args.config),
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
    )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = args.output / "contract.json"
        if path.exists() and json.loads(path.read_text()) != contract:
            raise ValueError(
                "Calibration contract changed; use a fresh output directory"
            )
        atomic(path, contract)
    fit_layers(
        args.traces,
        args.output,
        selected_layers,
        manifest["loops"],
        chains,
        initial,
        grams,
        contract,
    )
    assemble(args.output, config, contract)


def assemble(output, config, contract):
    import fcntl
    import torch
    from reproduction import atomic

    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        paths = [
            output / f"layer_{layer:02d}.pt"
            for layer in range(contract["physical_layers"])
        ]
        if not all(path.exists() for path in paths):
            return
        parts = [
            torch.load(path, map_location="cpu", weights_only=True) for path in paths
        ]
        if any(part["contract"] != contract for part in parts):
            raise ValueError("Layer checkpoint contract mismatch")
        records = [record for group in config["packed_groups"] for record in group]
        artifact = dict(
            model="huginn" if "huginn" in contract["model"].lower() else "ouro",
            format_version=2,
            key_rotations=torch.stack([p["key_rotations"] for p in parts]),
            value_rotations=torch.stack([p["value_rotations"] for p in parts]),
            key_means=torch.stack([p["key_means"] for p in parts], dim=1),
            has_mean=True,
            schedule=[b for b, _ in records],
            groups=[g for _, g in records],
            mode="anchor",
            coefficient_method="ls",
            use_rotation=True,
            coordinate_system="grouped_rotated",
            calibration_contract=contract,
            metrics=[p["metrics"] for p in parts],
        )
        save_tensor(output / "rotations.pt", artifact)
        ready = dict(
            config,
            calibration_domain=contract["calibration_domain"],
            rotation_artifact=dict(
                artifact=str((output / "rotations.pt").resolve()),
                sha256=digest(output / "rotations.pt"),
            ),
        )
        ready.pop("use_rotation", None)
        atomic(output / "config.json", ready)


if __name__ == "__main__":
    main()
