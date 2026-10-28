"""Packed-KV theory and fixed-memory decode capacity experiment (one explicit GPU)."""

import argparse, json, math, subprocess, sys
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from reproduction import atomic, digest, snapshot


@lru_cache(maxsize=1)
def model_geometry():
    from transformers import AutoConfig

    protocol = json.loads(
        (ROOT / "experiments/KV_main_experiments/prompt_protocol.json").read_text()
    )
    model = protocol["models"]["ouro"]
    config = AutoConfig.from_pretrained(
        model["name"], revision=model["revision"], trust_remote_code=True
    )
    return dict(
        layers=config.num_hidden_layers,
        loops=model["loops"],
        heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
    )


def footprint(config, geometry=None):
    geometry = model_geometry() if geometry is None else geometry
    layers, loops, heads, dimension = (
        geometry[key] for key in ("layers", "loops", "heads", "head_dim")
    )
    if "bf16_variant" in config:
        raise ValueError("Legacy BF16 configurations are unsupported")
    if not config["packed_groups"]:
        return 2 * layers * loops * heads * dimension * 2
    records = [record for group in config["packed_groups"] for record in group]
    if len(records) != loops or any(
        bit not in (2, 3, 4) or group not in (16, 32) or dimension % group
        for bit, group in records
    ):
        raise ValueError("Invalid precision schedule or quantization group size")
    stride = sum(
        dimension * bit // 8 + 2 * (dimension // group) + 2 for bit, group in records
    )
    return 2 * layers * heads * stride


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--context", type=int, choices=[1024, 2048, 4096, 8192], default=8192
    )
    p.add_argument("--gpu", help="Required with --run")
    p.add_argument("--run", action="store_true")
    p.add_argument(
        "--sweep",
        action="store_true",
        help="After capacity validation, also measure smaller batches",
    )
    p.add_argument("--page-tokens", type=int, choices=(16, 32), default=32)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--trials", type=int, default=3)
    a = p.parse_args()
    if not 2 <= a.tokens <= 128 or a.trials < 1:
        p.error("tokens=2..128 and trials>=1 required")
    if a.run and a.gpu is None:
        p.error("--run requires --gpu")
    cfg = json.loads(a.config.read_text())
    a.output.mkdir(parents=True, exist_ok=True)
    geometry = model_geometry()
    size = footprint(cfg, geometry)
    source_hashes = snapshot(ROOT)
    policy = (
        "staged_full_prompt_barrier"
        if cfg["packed_groups"]
        else "staged_chunked_barrier"
    )
    artifact = cfg.get("rotation_artifact")
    if artifact:
        artifact_path = Path(artifact["artifact"])
        if not artifact_path.is_absolute():
            artifact_path = a.config.resolve().parent / artifact_path
        if digest(artifact_path) != artifact["sha256"]:
            raise ValueError("Rotation artifact hash mismatch")
    length = math.ceil((a.context + a.tokens) / a.page_tokens) * a.page_tokens
    report = dict(
        config=cfg,
        geometry=geometry,
        page_tokens=a.page_tokens,
        context=a.context,
        output_tokens=a.tokens,
        trials=a.trials,
        warmups=1,
        bytes_per_token=size,
        allocated_tokens_per_request=length,
        kv_bytes_per_request=size * length,
        bf16_compression_ratio=footprint({"packed_groups": []}, geometry) / size,
        formula="Bmax=floor(actual available KV bytes / block-rounded KV bytes per request)",
        excludes="weights, rotation matrices, means, activations, graph/workspace, allocator overhead",
        implementation_batch_limit=32,
        prefill=policy,
        gpu_memory_utilization=0.90,
    )

    def save():
        atomic(a.output / "capacity.json", report)

    save()
    if not a.run:
        print(json.dumps(report, indent=2))
        return

    def measure(batch, probe=False, quick=False):
        dest = a.output / (
            "probe_b32" if probe else (f"quick_b{batch}" if quick else f"b{batch}")
        )
        dest.mkdir(exist_ok=True)
        command = [
            sys.executable,
            str(HERE / "measure.py"),
            "--config",
            str(a.config.resolve()),
            "--output",
            str(dest.resolve()),
            "--gpu",
            a.gpu,
            "--page-tokens",
            str(a.page_tokens),
            "--batch",
            str(batch),
            "--contexts",
            str(a.context),
            "--staged-prefill",
            "--tokens",
            str(a.tokens),
            "--trials",
            str(a.trials),
        ]
        if quick:
            command[command.index("--tokens") + 1] = "2"
            command[command.index("--trials") + 1] = "1"
            command += ["--warmups", "0", "--capacity-tokens", str(a.tokens)]
        if probe:
            command.append("--probe-only")
        if (dest / "completed.json").exists():
            if (
                not (dest / "command.json").exists()
                or json.loads((dest / "command.json").read_text()) != command
            ):
                raise ValueError(
                    f"Measurement command changed; use a fresh output: {dest}"
                )
        else:
            atomic(dest / "command.json", command)
        if not (dest / "completed.json").exists():
            with (dest / "run.log").open("w") as log:
                r = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if r.returncode:
                raise RuntimeError(f"Measurement failed: {dest}/run.log")

        result = json.loads((dest / "result.json").read_text())
        expected = dict(
            config=cfg,
            geometry=geometry,
            contexts=[a.context],
            batch=batch,
            physical_gpu=a.gpu,
            page_tokens=a.page_tokens,
            prefill_policy=policy,
            output_tokens=2 if quick else a.tokens,
            trials=1 if quick else a.trials,
            warmups=0 if quick else 1,
            capacity_output_tokens=a.tokens,
            source_sha256=source_hashes,
        )
        contract = result["contract"]
        for key, value in expected.items():
            if contract.get(key) != value:
                raise ValueError(
                    f"Measurement {key} changed; use a fresh output: {dest}"
                )
        if bool(contract.get("probe_only")) != probe and not (probe and result["rows"]):
            raise ValueError(f"Measurement probe mode changed: {dest}")
        atomic(dest / "completed.json", {"returncode": 0})
        return result

    probe = measure(32, True)
    if probe["rows"]:
        report["status"] = "probe_initialization_oom"
        report["probe"] = probe
        save()
        return
    cap = probe["contract"]["kv_capacity"]
    per = sum(math.ceil((a.context + a.tokens) / b) for b in cap["block_sizes"])
    predicted = cap["free_blocks"] // per
    report.update(
        probe=cap,
        predicted_batch_from_probe=predicted,
        probe_note="B32 graph/profile; actual per-batch capacity is rechecked on each run",
        measurements={},
        feasibility={},
    )
    save()

    def sample(batch):
        key = str(batch)
        if key not in report["measurements"]:
            result = measure(batch)
            row = result["rows"][0]
            report["measurements"][key] = row
            save()
        return report["measurements"][key]["status"] == "ok"

    def feasible(batch):
        full = a.output / f"b{batch}"
        if (full / "completed.json").exists():
            result = measure(batch)
            row = result["rows"][0]
        else:
            result = measure(batch, quick=True)
            row = result["rows"][0]
        report["feasibility"][str(batch)] = row
        save()
        return row["status"] == "ok"

    boundary = max(1, min(32, predicted))
    while not feasible(boundary):
        if boundary == 1:
            report["status"] = "no_supported_batch"
            save()
            return
        boundary -= 1
    while boundary < 32 and feasible(boundary + 1):
        boundary += 1
    report.update(
        capacity_boundary_verified_batch=boundary,
        capacity_validation="Actual 2-token generation without warmup; reserve full output KV budget. Long-run performance validation pending.",
    )
    save()
    if not sample(boundary):
        report["status"] = "long_run_capacity_failed"
        save()
        return
    if a.sweep:
        for b in [1, 2, 4, 8, 16, 32]:
            if b <= boundary:
                sample(b)
    ok = [(int(b), r) for b, r in report["measurements"].items() if r["status"] == "ok"]
    best = max(ok, key=lambda item: item[1]["output_tokens_per_second"])
    report.update(
        status="complete",
        max_verified_batch=boundary,
        capacity_censored_at_implementation_limit=boundary == 32,
        best_sampled_batch=best[0],
        best_sampled_output_tokens_per_second=best[1]["output_tokens_per_second"],
        best_note="Best among measured batches, not a proof of a global optimum",
    )
    save()
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("config", "measurements")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
