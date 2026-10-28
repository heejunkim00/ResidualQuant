"""Fill only Ouro main-benchmark cells from completed matching full evaluations."""

import argparse, csv, hashlib, json
from pathlib import Path
from run import cells, HERE, ROOT
from reproduction import atomic, ouro_data


COUNTS = {"gsm8k": 1319, "math500": 500, "humaneval": 164, "mbpp": 378}
TOKENS = {"gsm8k": 1024, "math500": 2048, "humaneval": 1024, "mbpp": 2048}


def read_result(path, cfg, benchmark, model_key="ouro"):
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    contract = data["contract"]
    assert contract["model_key"] == model_key, path
    records = data["records"]
    actual = contract["config"]
    assert data["completed"], path
    assert len(records) == COUNTS[benchmark] and {
        r["document_index"] for r in records
    } == set(range(COUNTS[benchmark])), path
    assert contract["max_new_tokens"] == TOKENS[benchmark], path
    assert (
        actual.get("coefficient_method"),
        bool(actual.get("rotation_artifact") or actual.get("use_rotation")),
    ) == (
        cfg.get("coefficient_method"),
        bool(cfg.get("rotation_artifact") or cfg.get("use_rotation")),
    )
    for key in ("id", "reference_mode", "packed_groups"):
        assert actual.get(key) == cfg.get(key), (path, key)
    assert actual.get("source_sha256", {}) == cfg.get("source_sha256", {}), path
    assert contract["blas_library"] == "default", (path, "BLAS mismatch")
    if benchmark in ("humaneval", "mbpp") and bool(
        cfg.get("rotation_artifact") or cfg.get("use_rotation")
    ):
        expected = json.loads(
            (
                ouro_data(HERE.name, model_key)
                / "code_calibration/calibration"
                / cfg["id"]
                / "config.json"
            ).read_text()
        )
        assert actual["rotation_artifact"] == expected["rotation_artifact"], (
            "Code evaluation requires MBPP-calibrated rotations"
        )
    if benchmark == "gsm8k":
        accuracy = 100 * sum(r["flexible_correct"] for r in records) / len(records)
        plus = None
    else:
        assert contract["benchmark"] == benchmark
        assert (
            contract["protocol_sha256"]
            == hashlib.sha256((HERE / "prompt_protocol.json").read_bytes()).hexdigest()
        )
        assert contract["apply_chat_template"] is False and contract["chatml"] is False
        assert (
            contract["source_sha256"]["scripts/evaluation/ouro.py"]
            == hashlib.sha256(
                (ROOT / "scripts/evaluation/ouro.py").read_bytes()
            ).hexdigest()
        )
        accuracy = data["scores"]["accuracy_percent"]
        plus = data["scores"].get("plus_accuracy_percent")
    return dict(
        accuracy_percent=accuracy,
        plus_accuracy_percent=plus,
        generation_seconds=sum(b["elapsed_seconds"] for b in data["batches"]),
        scoring_seconds=data.get("scoring_seconds"),
        result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def collect(model_key="ouro"):
    data = ouro_data(HERE.name, model_key)
    rows = []
    for cell in cells(model_key):
        cfg = json.loads(Path(cell["config"]).read_text())
        method = cfg["coefficient_method"] if cfg["packed_groups"] else "bf16"
        rotated = bool(cfg.get("rotation_artifact") or cfg.get("use_rotation"))
        schedules = cfg["packed_groups"]
        records = [record for schedule in schedules for record in schedule]
        bits = [bit for bit, _ in records]
        groups = [group for _, group in records]
        cost = (
            16
            if method == "bf16"
            else sum(b + 16 / g for b, g in zip(bits, groups)) / 4
            + (
                0.09375
                if any(len(records) > 1 for records in cfg["packed_groups"])
                else 0
            )
        )
        for benchmark in COUNTS:
            path = (
                Path(cell["gsm8k"])
                if benchmark == "gsm8k"
                else data / "evaluations" / benchmark / cell["id"] / "result.json"
            )
            value = read_result(path, cfg, benchmark, model_key)
            row = dict(
                model=model_key,
                id=cell["id"],
                benchmark=benchmark,
                method=method,
                use_rotation=rotated,
                column=2 * int(rotated)
                + int(any(len(records) > 1 for records in cfg["packed_groups"])),
                schedule=bits,
                groups=groups,
                effective_bits=cost,
                status="complete" if value else "pending",
                accuracy_percent=None,
                plus_accuracy_percent=None,
                generation_seconds=None,
                scoring_seconds=None,
                result=str(path),
                result_sha256=None,
            )
            row.update(value or {})
            rows.append(row)
    assert len(rows) == 84
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-key", choices=("ouro", "ouro26"), default="ouro")
    a = p.parse_args()
    rows = collect(a.model_key)
    out = ouro_data(HERE.name, a.model_key) / "summary"
    out.mkdir(parents=True, exist_ok=True)
    atomic(out / "ouro_results.json", rows)
    with (out / "ouro_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{sum(r['status'] == 'complete' for r in rows)}/84 {a.model_key} cells")


if __name__ == "__main__":
    main()
