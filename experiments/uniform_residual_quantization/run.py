"""Run resumable uniform-quantization evaluations."""

import json, sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


from reproduction import ouro_data


def commands(stage, dry_run=False, model_key="ouro", *, directory=HERE, expected=19):
    if stage not in ("calibrate", "evaluate"):
        raise ValueError(f"Unsupported experiment stage: {stage}")
    directory = Path(directory)
    data = ouro_data(directory.name, model_key)
    configs = sorted((directory / "configs").glob("*.json"))
    if len(configs) != expected:
        raise ValueError("Run make_configs.py first")
    jobs = []
    for path in configs:
        config = json.loads(path.read_text())
        name = config["id"]
        rotated = bool(config.get("rotation_artifact") or config.get("use_rotation"))
        if stage == "calibrate":
            if rotated:
                command = [
                    sys.executable,
                    str(
                        ROOT
                        / (
                            "scripts/calibration/calibrate.py"
                            if any(len(group) > 1 for group in config["packed_groups"])
                            else "scripts/calibration/direct_ouro.py"
                        )
                    ),
                    "--model-key",
                    model_key,
                    "--gpu",
                    "{gpu}",
                    "--config",
                    str(path),
                    "--output",
                    str(data / "calibration" / name),
                ]
                jobs.append((name, command))
            continue
        ready = data / "calibration" / name / "config.json" if rotated else path
        if not dry_run and not ready.exists():
            raise FileNotFoundError(f"Calibration missing: {ready}")
        output = data / "evaluations" / name / "result.json"
        command = [
            sys.executable,
            str(ROOT / "scripts/evaluation/ouro.py"),
            "--model-key",
            model_key,
            "--benchmark",
            "gsm8k",
            "--gpu",
            "{gpu}",
            "--config",
            str(ready),
            "--output",
            str(output),
            "--gpu-memory-utilization",
            "0.93",
            "--blas-library",
            "default",
        ]
        command += ["--limit", "1319"]
        jobs.append((name, command))
    return jobs
