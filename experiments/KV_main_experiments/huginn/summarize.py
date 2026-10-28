"""Validate complete Huginn benchmark records before reporting accuracy."""

import hashlib, json, math
from pathlib import Path


COUNTS = {"gsm8k": 1319, "math500": 500, "humaneval": 164, "mbpp": 378}
MODEL = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"


def validate(output):
    output = Path(output)
    canonical = json.loads((output / "result.json").read_text())
    contract = canonical["contract"]
    assert canonical["completed"]
    assert (
        contract["model"] == MODEL
        and contract["model_key"] == "huginn"
        and contract["blas_library"] == "default"
    )
    bench = contract["benchmark"]
    count = contract["requested"]
    config = contract["config"]
    assert bench in COUNTS
    assert count == COUNTS[bench]
    assert contract["max_new_tokens"] == (
        1024 if bench in ("gsm8k", "humaneval") else 2048
    )
    settings = contract["settings"]
    assert contract["revision"] == REVISION and contract["loops"] == 32
    assert 1 <= settings["max_num_seqs"] <= 32
    if (config.get("reference_mode") == "anchor" and config["anchor_index"] == 3) or (
        config["packed_groups"] and settings.get("enable_chunked_prefill") is False
    ):
        assert (
            settings["max_num_batched_tokens"] >= 4096
            and not settings["enable_prefix_caching"]
            and not settings["enable_chunked_prefill"]
        )
    else:
        assert (
            settings["max_num_batched_tokens"] >= 1024
            and settings["enable_prefix_caching"]
        )
    assert settings["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
    if bool(config.get("rotation_artifact") or config.get("use_rotation")):
        assert config["calibration_domain"] == (
            "code" if bench in ("humaneval", "mbpp") else "gsm8k"
        )
        artifact = config["rotation_artifact"]
        assert (
            hashlib.sha256(Path(artifact["artifact"]).read_bytes()).hexdigest()
            == artifact["sha256"]
        )
    matches = list((output / "raw").rglob("results_*.json"))
    assert len(matches) == 1
    result_path = matches[0]
    result = json.loads(result_path.read_text())
    if bench in ("humaneval", "mbpp"):
        from evalplus.data import get_human_eval_plus, get_mbpp_plus

        expected = set(
            (get_human_eval_plus() if bench == "humaneval" else get_mbpp_plus()).keys()
        )
        assert len(expected) == COUNTS[bench]
        samples_path = output / "raw" / f"{bench}_samples.jsonl"
        records = [
            json.loads(line) for line in samples_path.read_text().splitlines() if line
        ]
        ids = [r["task_id"] for r in records]
        assert len(ids) == len(set(ids)) == count
        assert set(ids) == expected
        assert (
            result["model"] == MODEL
            and result["revision"] == REVISION
            and result["prompt"] == "evalplus_chat"
        )
        scores = result["results"][bench]
        accuracy = float(scores["pass@1,base"])
        plus = float(scores["pass@1,plus"])
        assert result["n-samples"][bench]["effective"] == count
        assert math.isclose(accuracy, scores["base_passed"] / count, abs_tol=1e-12)
        assert math.isclose(plus, scores["plus_passed"] / count, abs_tol=1e-12)
        assert 0 <= plus <= accuracy <= 1
    else:
        task = "gsm8k_cot" if bench == "gsm8k" else "huginn_math500_boxed"
        paths = list((output / "raw").rglob(f"samples_{task}_*.jsonl"))
        assert len(paths) == 1
        samples_path = paths[0]
        records = [
            json.loads(line) for line in samples_path.read_text().splitlines() if line
        ]
        filters = (
            ("strict-match", "flexible-extract") if bench == "gsm8k" else ("none",)
        )
        assert {r["filter"] for r in records} == set(filters)
        for filter_name in filters:
            ids = [r["doc_id"] for r in records if r["filter"] == filter_name]
            assert len(ids) == len(set(ids)) == count and set(ids) == set(range(count))
        metric = (
            "exact_match,flexible-extract" if bench == "gsm8k" else "math_verify,none"
        )
        accuracy = float(result["results"][task][metric])
        plus = None
        assert 0 <= accuracy <= 1 and result["n-samples"][task]["effective"] == count
        selected = [
            r
            for r in records
            if r["filter"] == ("flexible-extract" if bench == "gsm8k" else "none")
        ]
        field = "exact_match" if bench == "gsm8k" else "math_verify"
        assert math.isclose(
            accuracy, sum(float(r[field]) for r in selected) / count, abs_tol=1e-12
        )
    assert len(canonical["records"]) == count
    assert all(
        "document_index" in row and "prompt" in row and "generated_text" in row
        for row in canonical["records"]
    )
    assert Path(canonical["raw_result"]).resolve() == result_path.resolve()
    assert math.isclose(
        canonical["scores"]["accuracy_percent"], 100 * accuracy, abs_tol=1e-9
    )
    seconds = float(canonical["runner_seconds"])
    assert math.isfinite(seconds) and seconds > 0
    files = [output / "result.json", result_path, samples_path]
    return dict(
        benchmark=bench,
        config=config,
        documents=count,
        accuracy_percent=100 * accuracy,
        plus_accuracy_percent=None if plus is None else 100 * plus,
        runner_seconds=seconds,
        metric_seconds=result.get("total_evaluation_time_seconds"),
        evidence_sha256={
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
    )


def collect(data):
    from run import configurations, BENCHMARKS

    rows = []
    data = Path(data)
    configs = configurations(
        reverse_only=(data / "configs/2224_g16_ls_optr.json").exists()
    )
    for name, cfg in configs.items():
        for benchmark in BENCHMARKS:
            output = data / "evaluate" / name / benchmark
            value = validate(output) if (output / "result.json").exists() else None
            if value:
                assert (
                    value["config"].get("coefficient_method"),
                    bool(
                        value["config"].get("rotation_artifact")
                        or value["config"].get("use_rotation")
                    ),
                ) == (
                    cfg.get("coefficient_method"),
                    bool(cfg.get("rotation_artifact") or cfg.get("use_rotation")),
                )
                for key in (
                    "reference_mode",
                    "packed_groups",
                ):
                    assert value["config"].get(key) == cfg.get(key)
            rows.append(
                dict(
                    id=name,
                    benchmark=benchmark,
                    config=cfg,
                    status="complete" if value else "pending",
                    accuracy_percent=value["accuracy_percent"] if value else None,
                    result=value,
                )
            )
    assert len(rows) == len(configs) * len(BENCHMARKS)
    return rows


def main():
    import argparse, csv
    from run import DATA

    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--require-complete", action="store_true")
    a = p.parse_args()
    rows = collect(a.data)
    summary = a.data / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    (summary / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (summary / "results.csv").open("w") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "benchmark", "status", "accuracy_percent"]
        )
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
    complete = sum(r["status"] == "complete" for r in rows)
    print(f"{complete}/{len(rows)} Huginn cells")
    if a.require_complete:
        assert complete == len(rows), "All planned full results required"
    if complete == len(rows):
        (summary / "completed.json").write_text(
            json.dumps(
                dict(
                    completed=True,
                    cells=len(rows),
                    results_sha256=hashlib.sha256(
                        (summary / "results.json").read_bytes()
                    ).hexdigest(),
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
