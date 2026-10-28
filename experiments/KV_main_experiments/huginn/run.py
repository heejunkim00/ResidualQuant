"""Build the Huginn calibration and benchmark execution plan."""

import hashlib, json, sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "experiments/KV_main_experiments/data/huginn_full_fp32"
SELF = Path(__file__).resolve()
BENCHMARKS = ("gsm8k", "math500", "humaneval", "mbpp")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2) + "\n"
    if path.exists():
        assert path.read_text() == text, f"Preserve existing artifact: {path}"
    else:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text)
        tmp.replace(path)


def configurations(reverse_only=False):
    result = {
        "bf16": dict(
            packed_groups=[],
        )
    }
    if reverse_only:
        for group in (16, 32):
            result[f"2224_g{group}_ls_optr"] = dict(
                coefficient_method="ls",
                use_rotation=True,
                reference_mode="anchor",
                anchor_index=3,
                packed_groups=[[[2, group]] * 3 + [[4, 32]] for _ in range(8)],
            )
        for group in (16, 32):
            result[f"2224_g{group}_ls"] = dict(
                coefficient_method="ls",
                reference_mode="anchor",
                anchor_index=3,
                packed_groups=[[[2, group]] * 3 + [[4, 32]] for _ in range(8)],
            )
        return result
    for bits, groups in (
        ([2] * 4, (16, 32)),
        ([4, 2, 2, 2], (16, 32)),
        ([4] * 4, (32,)),
    ):
        for group in groups:
            for label, method, rotation in (
                ("direct", "unit", False),
                ("optr", "ls", True),
            ):
                name = "".join(map(str, bits)) + f"_g{group}_{label}"
                result[name] = dict(
                    coefficient_method=method,
                    reference_mode="anchor",
                    anchor_index=0,
                    packed_groups=[
                        [[b, group if b == 2 else 32]] for _ in range(8) for b in bits
                    ],
                )
                if rotation:
                    result[name]["use_rotation"] = True
    return result


def plan(data=DATA, gpus=(0, 1), reverse_only=False):
    data = Path(data).resolve()
    configs = configurations(reverse_only)
    jobs = []
    rotation = [
        n
        for n, c in configs.items()
        if bool(c.get("rotation_artifact") or c.get("use_rotation"))
    ]
    python = sys.executable

    def job(key, stage, command, deps=(), output=None):
        jobs.append(
            dict(
                id=key,
                stage=stage,
                status="pending",
                depends_on=list(deps),
                command=[str(x) for x in command],
                output=str(output) if output else None,
            )
        )

    for name, cfg in configs.items():
        write(data / "configs" / f"{name}.json", cfg)
    assemblies = {}
    for domain in ("gsm8k", "code"):
        for name in rotation:
            key = f"{domain}:calibrate:{name}"
            output = data / "calibrated" / domain / name
            cfg = configs[name]
            command = [
                python,
                ROOT
                / (
                    "scripts/calibration/calibrate.py"
                    if any(len(group) > 1 for group in cfg["packed_groups"])
                    else "scripts/calibration/direct_huginn.py"
                ),
                *(
                    ["--model-key", "huginn"]
                    if any(len(group) > 1 for group in cfg["packed_groups"])
                    else []
                ),
                "--gpu",
                "{gpu}",
                "--config",
                data / "configs" / f"{name}.json",
                "--dataset",
                "mbpp" if domain == "code" else "gsm8k",
                "--output",
                output,
            ]
            job(key, "calibrate", command, output=output / "config.json")
            assemblies[domain, name] = key
    for name in configs:
        for bench in BENCHMARKS:
            domain = "code" if bench in ("humaneval", "mbpp") else "gsm8k"
            cfg = (
                data / "calibrated" / domain / name / "config.json"
                if name in rotation
                else data / "configs" / f"{name}.json"
            )
            deps = [assemblies[domain, name]] if name in rotation else []
            output = data / "evaluate" / name / bench
            cmd = [
                python,
                ROOT / "scripts/evaluation/huginn.py",
                "--gpu",
                "{gpu}",
                "--config",
                cfg,
                "--benchmark",
                bench,
                "--output",
                output,
            ]
            if reverse_only:
                cmd += [
                    "--gpu-memory-utilization",
                    "0.55",
                    "--max-num-seqs",
                    "8",
                    "--max-num-batched-tokens",
                    "16384",
                ]
            job(
                f"evaluate:{name}:{bench}",
                "evaluate",
                cmd,
                deps,
                output / "result.json",
            )
    job(
        "summarize",
        "summarize",
        [python, SELF.parent / "summarize.py", "--data", data, "--require-complete"],
        [item["id"] for item in jobs if item["stage"] == "evaluate"],
        data / "summary/completed.json",
    )
    for item in jobs:
        item["id"] = "huginn:" + item["id"]
        item["depends_on"] = ["huginn:" + key for key in item["depends_on"]]
        item["experiment"] = "KV_main_experiments"
    return dict(
        model="huginn",
        loops=32,
        block_schedule=(
            "[Residual, Residual, Residual, Anchor] x 8"
            if reverse_only
            else "[Original, Residual, Residual, Residual] x 8"
        ),
        configuration_count=len(configs),
        rotation_configuration_count=len(rotation),
        gpus=list(gpus),
        jobs=jobs,
    )


def validated_output(job):
    """Validate completed calibration and benchmark artifacts."""
    import torch

    sys.path.insert(0, str(SELF.parent))
    from summarize import validate

    output = Path(job["output"])
    stage = job["stage"]
    if stage == "summarize":
        value = json.loads(output.read_text())
        assert value["completed"] and value["cells"] == len(job["depends_on"])
        assert (
            value["results_sha256"]
            == hashlib.sha256(output.with_name("results.json").read_bytes()).hexdigest()
        )
    elif stage == "evaluate":
        result = validate(output.parent)
        command = job["command"]
        cfg = json.loads(Path(command[command.index("--config") + 1]).read_text())
        assert result["config"] == cfg
    elif stage == "calibrate":
        ready = json.loads(output.read_text())
        spec = ready["rotation_artifact"]
        artifact = Path(spec["artifact"])
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == spec["sha256"]
        value = torch.load(artifact, weights_only=True)
        contract = value["calibration_contract"]
        command = job["command"]
        cfg = json.loads(Path(command[command.index("--config") + 1]).read_text())
        assert value["model"] == "huginn"
        schedules = cfg["packed_groups"]
        records = [record for schedule in schedules for record in schedule]
        assert value["mode"] == cfg["reference_mode"]
        assert value["schedule"] == [bit for bit, _ in records]
        assert value["groups"] == [group for _, group in records]
        assert contract["steps"] == 80
        assert len(contract["fit"]) == len(contract["holdout"]) == 128
        assert set(contract["fit"]).isdisjoint(contract["holdout"])
        assert ready["calibration_domain"] in ("gsm8k", "code")
        assert value["format_version"] == 2
        expected = (
            (32, 4, 55, 96, 96)
            if all(len(group) == 1 for group in schedules)
            else (4, 55, 96, 96)
        )
        for name in ("key_rotations", "value_rotations"):
            assert value[name].shape == expected
            assert value[name].dtype == torch.float16
            assert torch.isfinite(value[name]).all()
        assert value["key_means"].shape == (32, 4, 55, 96)
        assert torch.isfinite(value["key_means"]).all()

    else:
        raise ValueError(stage)
    return True
