"""Build Ouro calibration and benchmark execution plans."""

import csv, hashlib, json, subprocess, sys
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
BENCHMARKS = {"math500": 500, "humaneval": 164, "mbpp": 378}


from reproduction import atomic, load, ouro_data


def cells(model_key="ouro"):
    result = []
    for exp in ("uniform_residual_quantization", "bit_schedules"):
        home = HERE.parent / exp
        data = ouro_data(exp, model_key)
        for p in sorted((home / "configs").glob("*.json")):
            cfg = json.loads(p.read_text())
            method = cfg["coefficient_method"] if cfg["packed_groups"] else None
            mode = cfg.get("reference_mode")
            if method not in ("unit", "ls", None):
                continue
            if method == "unit" and any(
                len(group) > 1 for group in cfg["packed_groups"]
            ):
                continue
            if method == "ls" and mode != "anchor":
                continue
            if exp == "bit_schedules" and [
                record[0] for schedule in cfg["packed_groups"] for record in schedule
            ] != [4, 2, 2, 2]:
                continue
            name = cfg["id"]
            rotated = bool(cfg.get("rotation_artifact") or cfg.get("use_rotation"))
            result.append(
                dict(
                    id=name,
                    experiment=exp,
                    config=str(p),
                    ready=str(data / "calibration" / name / "config.json")
                    if rotated
                    else str(p),
                    gsm8k=str(data / "evaluations" / name / "result.json"),
                )
            )
    assert len(result) == 21 and len({c["id"] for c in result}) == 21
    return result


def command(cell, benchmark, model_key="ouro"):
    data = ouro_data(HERE.name, model_key)
    rotated = json.loads(Path(cell["config"]).read_text()).get("use_rotation", False)
    code = benchmark in ("humaneval", "mbpp")
    config = (
        str(data / "code_calibration/calibration" / cell["id"] / "config.json")
        if code and rotated
        else cell["ready"]
    )
    out = data / "evaluations" / benchmark / cell["id"] / "result.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts/evaluation/ouro.py"),
        "--model-key",
        model_key,
        "--gpu",
        "{gpu}",
        "--benchmark",
        benchmark,
        "--config",
        config,
        "--protocol",
        str(HERE / "prompt_protocol.json"),
        "--output",
        str(out),
    ]
    cmd += ["--limit", str(BENCHMARKS[benchmark])]
    return cmd


def plan(model_key="ouro"):
    for experiment in ("uniform_residual_quantization", "bit_schedules"):
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments" / experiment / "make_configs.py"),
            ],
            check=True,
        )
    uniform = load(
        "uniform_runner", ROOT / "experiments/uniform_residual_quantization/run.py"
    )
    selected_cells = cells(model_key)
    main_data = ouro_data("KV_main_experiments", model_key)
    atomic(main_data / "cells.json", selected_cells)
    jobs = []
    for experiment, expected in [
        ("uniform_residual_quantization", 19),
        ("bit_schedules", 16),
    ]:
        for stage in ("calibrate", "evaluate"):
            for name, job_command in uniform.commands(
                stage,
                dry_run=True,
                model_key=model_key,
                directory=ROOT / "experiments" / experiment,
                expected=expected,
            ):
                jobs.append(
                    dict(
                        id=f"{experiment}:{stage}:{name}",
                        stage=stage,
                        experiment=experiment,
                        name=name,
                        command=job_command,
                        depends_on=[],
                        status="pending",
                    )
                )
    for cell in selected_cells:
        config = json.loads(Path(cell["config"]).read_text())
        if not bool(config.get("rotation_artifact") or config.get("use_rotation")):
            continue
        calibration_command = [
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
            cell["config"],
            "--dataset",
            "mbpp",
            "--output",
            str(main_data / "code_calibration/calibration" / cell["id"]),
        ]
        jobs.append(
            dict(
                id=f"code:calibrate:{cell['id']}",
                stage="calibrate",
                experiment="KV_main_experiments",
                name=cell["id"],
                command=calibration_command,
                depends_on=[],
                status="pending",
            )
        )
    calibrations = [j["id"] for j in jobs if j["stage"] == "calibrate"]
    for cell in selected_cells:
        for benchmark in BENCHMARKS:
            jobs.append(
                dict(
                    id=f"main:evaluate:{benchmark}:{cell['id']}",
                    stage="evaluate",
                    experiment="KV_main_experiments",
                    name=cell["id"],
                    command=command(cell, benchmark, model_key=model_key)
                    + ["--blas-library", "default"],
                    depends_on=[],
                    status="pending",
                )
            )
    for job in jobs:
        if job["stage"] == "evaluate":
            job["depends_on"] = calibrations
    for experiment in (
        "uniform_residual_quantization",
        "bit_schedules",
        "KV_main_experiments",
    ):
        jobs.append(
            dict(
                id=f"{experiment}:summarize",
                stage="summarize",
                experiment=experiment,
                name="summary",
                status="pending",
                command=[
                    sys.executable,
                    str(
                        ROOT
                        / "experiments"
                        / experiment
                        / (
                            "ouro/summarize.py"
                            if experiment == "KV_main_experiments"
                            else "summarize.py"
                        )
                    ),
                    "--model-key",
                    model_key,
                ],
                depends_on=[j["id"] for j in jobs if j["stage"] == "evaluate"],
            )
        )
    return dict(
        jobs=jobs,
        protocol=dict(
            model=model_key,
            fit=128,
            holdout=128,
            seed=0,
            gsm8k_test=1319,
            order=["uniform", "schedules", "main"],
            calibration="bf16_qkv_layerwise_shared_online_direct_fp32_fit",
        ),
    )


def validated_output(job):
    command_args = job["command"]
    stage = job["stage"]
    if stage == "summarize":
        model_key = command_args[command_args.index("--model-key") + 1]
        experiment = job["experiment"]
        filename, expected_count = {
            "uniform_residual_quantization": ("results", 19),
            "bit_schedules": ("results", 28),
            "KV_main_experiments": ("ouro_results", 84),
        }[experiment]
        directory = ouro_data(experiment, model_key) / "summary"
        rows = json.loads((directory / f"{filename}.json").read_text())
        with (directory / f"{filename}.csv").open(newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        assert len(rows) == len(csv_rows) == expected_count
        identities = {(row["id"], row.get("benchmark")) for row in rows}
        assert len(identities) == expected_count
        assert identities == {(row["id"], row.get("benchmark")) for row in csv_rows}
        assert all(
            row.get("status", "complete") == "complete"
            and row.get("accuracy_percent") is not None
            and row.get("model", model_key) == model_key
            for row in rows
        )
        return True
    if stage not in ("calibrate", "evaluate"):
        raise ValueError(f"Unknown Ouro stage: {stage}")
    output = Path(command_args[command_args.index("--output") + 1])
    if stage == "calibrate":
        config = json.loads((output / "config.json").read_text())
        artifact = config["rotation_artifact"]
        return (
            hashlib.sha256(Path(artifact["artifact"]).read_bytes()).hexdigest()
            == artifact["sha256"]
        )
    result = json.loads(output.read_text())
    assert result.get("completed")
    records = result["records"]
    benchmark = result["contract"].get("benchmark", "gsm8k")
    count, tokens = {
        "gsm8k": (1319, 1024),
        "math500": (500, 2048),
        "humaneval": (164, 1024),
        "mbpp": (378, 2048),
    }[benchmark]
    assert len(records) == count and result["contract"]["max_new_tokens"] == tokens
    assert {r["document_index"] for r in records} == set(range(count))
    if benchmark != "gsm8k":
        assert "scores" in result
    return True
