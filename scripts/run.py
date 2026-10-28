"""Download models, prepare and run experiments, and summarize results."""

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from reproduction import atomic, load, snapshot, run, ouro_data


def validate_plan(queue):
    ids = [j["id"] for j in queue["jobs"]]
    assert len(ids) == len(set(ids))
    known = set(ids)
    for job in queue["jobs"]:
        assert set(job["depends_on"]) <= known, job["id"]
        for argument in job["command"]:
            if argument.endswith(".py"):
                source = Path(argument).resolve()
                assert source.is_relative_to(ROOT), argument
                assert source.is_file(), argument
    done = set()
    while True:
        ready = {j["id"] for j in queue["jobs"] if set(j["depends_on"]) <= done}
        if ready <= done:
            break
        done |= ready
    assert done == known, "Cyclic dependencies"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("download", "prepare", "run", "status", "summarize")
    )
    parser.add_argument("--model", choices=("ouro", "ouro26", "huginn"), required=True)
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU for sequential reproduction"
    )
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--huginn-last-anchor",
        action="store_true",
        help=(
            "Huginn shared D96 rotation, [2,2,2,4] x8; "
            "separate calibration/evaluation plan"
        ),
    )
    args = parser.parse_args()
    data = (
        ouro_data("uniform_residual_quantization", args.model)
        if args.model != "huginn"
        else ROOT / "experiments/KV_main_experiments/data/huginn_full_fp32"
    )
    if args.huginn_last_anchor:
        if args.model != "huginn":
            parser.error("--huginn-last-anchor requires --model huginn")
        data = ROOT / "experiments/KV_main_experiments/data/huginn_last_anchor_fp32"
    queue_path = data / "plan.json"
    huginn = (
        load("huginn_runner", ROOT / "experiments/KV_main_experiments/huginn/run.py")
        if args.model == "huginn"
        else None
    )
    if args.action == "download":
        from huggingface_hub import snapshot_download

        protocol = json.loads(
            (ROOT / "experiments/KV_main_experiments/prompt_protocol.json").read_text()
        )
        model = protocol["models"][args.model]
        print(
            snapshot_download(
                model["name"],
                revision=model["revision"],
                allow_patterns=["*.json", "*.py", "*.safetensors", "*.txt", "*.model"],
            )
        )
        return
    if args.action == "prepare":
        if queue_path.exists():
            raise FileExistsError(
                f"Preserve existing queue: {queue_path}; use status/run"
            )
        if args.model != "huginn":
            ouro = load(
                "ouro_runner", ROOT / "experiments/KV_main_experiments/ouro/run.py"
            )
            queue = ouro.plan(args.model)
        else:
            queue = huginn.plan(data, (args.gpu,), reverse_only=args.huginn_last_anchor)
        queue["gpus"] = [args.gpu]
        validate_plan(queue)
        queue["input_sha256"] = snapshot(ROOT)
        atomic(queue_path, queue)
    elif args.action == "run":
        runner = huginn or load(
            "ouro_runner", ROOT / "experiments/KV_main_experiments/ouro/run.py"
        )
        run(
            queue_path,
            ROOT,
            validator=runner.validated_output,
            retry_failed=args.retry_failed,
            gpu=args.gpu,
        )
    elif args.action == "summarize":
        paths = (
            ["KV_main_experiments/huginn/summarize.py"]
            if huginn
            else [
                f"{x}/ouro/summarize.py"
                if x == "KV_main_experiments"
                else f"{x}/summarize.py"
                for x in (
                    "uniform_residual_quantization",
                    "bit_schedules",
                    "KV_main_experiments",
                )
            ]
        )
        for path in paths:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "experiments" / path),
                    *(["--data", str(data)] if huginn else ["--model-key", args.model]),
                ],
                check=True,
            )
    if queue_path.exists():
        queue = json.loads(queue_path.read_text())
        print(
            json.dumps(
                dict(
                    queue=str(queue_path),
                    counts=dict(Counter(j["status"] for j in queue["jobs"])),
                    stages=dict(Counter(j["stage"] for j in queue["jobs"])),
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
