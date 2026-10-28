"""Collect and fit shared rotations one physical layer at a time."""

import argparse
import fcntl
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-key", choices=("ouro", "ouro26", "huginn"), required=True
    )
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=("gsm8k", "mbpp"), default="gsm8k")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers")
    parser.add_argument("--trace-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not config["packed_groups"] or any(
        len(group) != 4 for group in config["packed_groups"]
    ):
        parser.error("This command fits shared four-loop anchor rotations")
    import torch
    from transformers import AutoConfig
    from calibration.layerwise import assemble
    from reproduction import atomic, digest

    protocol = json.loads(
        (ROOT / "experiments/KV_main_experiments/prompt_protocol.json").read_text()
    )
    spec = protocol["models"][args.model_key]
    model_config = AutoConfig.from_pretrained(
        spec["name"], revision=spec["revision"], trust_remote_code=True
    )
    layer_count = (
        model_config.n_layers_in_recurrent_block
        if args.model_key == "huginn"
        else model_config.num_hidden_layers
    )
    layers = (
        list(range(layer_count))
        if args.layers is None
        else list(map(int, args.layers.split(",")))
    )
    if (
        not layers
        or len(set(layers)) != len(layers)
        or not set(layers) <= set(range(layer_count))
    ):
        parser.error("Invalid physical layer selection")
    args.output.mkdir(parents=True, exist_ok=True)
    source_paths = sorted((ROOT / "scripts/calibration").glob("*.py"))
    source_paths += [
        ROOT / "scripts/ouro_prompt_protocol.py",
        ROOT / "scripts/code_prompt_protocol.py",
    ]
    run_contract = dict(
        model=spec,
        dataset=args.dataset,
        max_length=1024,
        config_sha256=digest(args.config),
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in source_paths},
    )
    run_contract_path = args.output / "run_contract.json"
    layer_contract_path = args.output / "contract.json"
    serving_config_path = args.output / "config.json"
    with (args.output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (
            run_contract_path.exists()
            and json.loads(run_contract_path.read_text()) != run_contract
        ):
            raise ValueError("Calibration inputs changed; use a fresh output directory")
        atomic(run_contract_path, run_contract)
        if serving_config_path.exists():
            ready = json.loads(serving_config_path.read_text())
            artifact = ready["rotation_artifact"]
            if digest(artifact["artifact"]) != artifact["sha256"]:
                raise ValueError("Completed rotation artifact changed")
            return
    trace_parent = args.trace_dir or args.output.parent
    trace_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="calibration-traces-", dir=trace_parent
    ) as trace_folder:
        trace_directory = Path(trace_folder)
        for layer in layers:
            checkpoint = args.output / f"layer_{layer:02d}.pt"
            if checkpoint.exists():
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                contract = json.loads(layer_contract_path.read_text())
                if saved["contract"] != contract:
                    raise ValueError("Layer checkpoint contract mismatch")
                continue
            collect_command = [
                sys.executable,
                str(ROOT / "scripts/calibration/collect.py"),
                "--model-key",
                args.model_key,
                "--gpu",
                args.gpu,
                "--dataset",
                args.dataset,
                "--output",
                str(trace_directory),
                "--layers",
                str(layer),
            ]
            subprocess.run(collect_command, check=True)
            fit_command = [
                sys.executable,
                str(ROOT / "scripts/calibration/layerwise.py"),
                "--gpu",
                args.gpu,
                "--config",
                str(args.config),
                "--traces",
                str(trace_directory),
                "--output",
                str(args.output),
                "--layers",
                str(layer),
            ]
            subprocess.run(fit_command, check=True)
            for trace_path in trace_directory.glob(f"sample_*/layer_{layer:02d}.pt"):
                trace_path.unlink()

    if layer_contract_path.exists():
        assemble(args.output, config, json.loads(layer_contract_path.read_text()))


if __name__ == "__main__":
    main()
