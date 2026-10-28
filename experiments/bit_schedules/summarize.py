"""Aggregate validated uniform and mixed-schedule evaluations."""

import argparse, csv, json, sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from reproduction import atomic, digest, ouro_data


def full_result(path, config, model_key="ouro"):
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    records = data.get("records", [])
    contract = data["contract"]
    assert contract["model_key"] == model_key, path
    configuration = contract["config"]
    assert data.get("completed") and not data.get("smoke"), path
    assert len(records) == 1319 and {r["document_index"] for r in records} == set(
        range(1319)
    ), path
    assert (
        contract["max_new_tokens"] == 1024
        and contract["fewshot"] == 3
        and contract["loops"] == 4
    ), path
    assert (
        configuration.get("coefficient_method"),
        bool(
            configuration.get("rotation_artifact") or configuration.get("use_rotation")
        ),
    ) == (
        config.get("coefficient_method"),
        bool(config.get("rotation_artifact") or config.get("use_rotation")),
    )
    for key in ("id", "reference_mode", "packed_groups"):
        assert configuration[key] == config[key], f"{path}: mismatched {key}"
    assert configuration.get("source_sha256", {}) == config.get("source_sha256", {}), (
        path,
        "serving source mismatch",
    )
    assert contract["blas_library"] == "default", (path, "BLAS mismatch")
    return dict(
        accuracy_percent=100 * sum(bool(r["flexible_correct"]) for r in records) / 1319,
        evaluation_seconds=sum(b["elapsed_seconds"] for b in data["batches"]),
        result_sha256=digest(path),
    )


def rows(model_key="ouro"):
    data = ouro_data(HERE.name, model_key)
    cells = json.loads((HERE / "reuse.json").read_text())
    for cell in cells:
        cell["result"] = str(
            ouro_data("uniform_residual_quantization", model_key)
            / "evaluations"
            / cell["id"]
            / "result.json"
        )
    for path in sorted((HERE / "configs").glob("*.json")):
        cells.append(
            dict(
                id=path.stem,
                config=str(path),
                result=str(data / "evaluations" / path.stem / "result.json"),
            )
        )
    result = []
    for cell in cells:
        configuration = json.loads(Path(cell["config"]).read_text())
        schedules = configuration["packed_groups"]
        records = [record for schedule in schedules for record in schedule]
        bits = [bit for bit, _ in records]
        groups = [group for _, group in records]
        uses_reference = any(
            len(records) > 1 for records in configuration["packed_groups"]
        )
        method = (
            configuration["coefficient_method"]
            if configuration["packed_groups"]
            else "bf16"
        )
        rotated = bool(
            configuration.get("rotation_artifact") or configuration.get("use_rotation")
        )
        cost = sum(b + 16 / g for b, g in zip(bits, groups)) / 4 + (
            3 * 16 / (4 * 128) if uses_reference else 0
        )
        value = full_result(Path(cell["result"]), configuration, model_key)
        row = dict(
            id=cell["id"],
            schedule=bits,
            groups=groups,
            method=method,
            use_rotation=rotated,
            column=2 * int(rotated) + int(uses_reference),
            reference=configuration["reference_mode"],
            effective_bits=cost,
            status="complete" if value else "pending",
            accuracy_percent=None,
            evaluation_seconds=None,
            result=cell["result"],
        )
        row.update(value or {})
        result.append(row)
    assert len(result) == 28
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-key", choices=("ouro", "ouro26"), default="ouro")
    a = p.parse_args()
    values = rows(a.model_key)
    out = ouro_data(HERE.name, a.model_key) / "summary"
    out.mkdir(parents=True, exist_ok=True)
    atomic(out / "results.json", values)
    with (out / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(dict.fromkeys(k for r in values for k in r))
        )
        writer.writeheader()
        writer.writerows(values)
    print(
        f"{sum(r['status'] == 'complete' for r in values)}/28 full evaluations; missing cells remain blank"
    )


if __name__ == "__main__":
    main()
