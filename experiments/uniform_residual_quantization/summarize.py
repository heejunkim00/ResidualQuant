"""Aggregate completed full evaluations."""

import argparse, csv, json, sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
from reproduction import atomic, ouro_data


FIELDS = (
    "id",
    "method",
    "use_rotation",
    "reference",
    "bits",
    "group",
    "accuracy_percent",
    "evaluation_seconds",
    "result",
)


def read_results(directory, model_key):
    rows = []
    for path in sorted((directory / "evaluations").glob("*/result.json")):
        data = json.loads(path.read_text())
        contract = data["contract"]
        if contract.get("model_key") != model_key:
            raise ValueError(f"Result model does not match {model_key}: {path}")
        records = data.get("records", [])
        if not data.get("completed") or data.get("smoke"):
            continue
        ids = [record["document_index"] for record in records]
        if len(records) != 1319 or set(ids) != set(range(1319)):
            continue
        if contract.get("max_new_tokens") != 1024:
            continue
        config = contract["config"]
        if path.parent.name == "bf16":
            if data.get("backend") != "vLLM BF16":
                continue
        correct = sum(bool(r["flexible_correct"]) for r in records)
        seconds = sum(b.get("elapsed_seconds", 0) for b in data.get("batches", []))
        rows.append(
            dict(
                id=path.parent.name,
                method=config["coefficient_method"]
                if config["packed_groups"]
                else "bf16",
                use_rotation=bool(
                    config.get("rotation_artifact") or config.get("use_rotation")
                ),
                reference=config.get("reference_mode", "none"),
                bits=next(
                    (
                        record[0]
                        for group in config["packed_groups"]
                        for record in group
                    ),
                    16,
                ),
                group=next(
                    (
                        record[1]
                        for group in config["packed_groups"]
                        for record in group
                    ),
                    None,
                ),
                accuracy_percent=100 * correct / 1319,
                evaluation_seconds=seconds,
                result=str(path.relative_to(HERE)),
            )
        )
    return rows


def write_results(directory, rows):
    output = directory / "summary"
    output.mkdir(parents=True, exist_ok=True)
    atomic(output / "results.json", rows)
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=("ouro", "ouro26"), default="ouro")
    args = parser.parse_args()
    directory = ouro_data(HERE.name, args.model_key)
    rows = read_results(directory, args.model_key)
    write_results(directory, rows)
    print(f"{len(rows)} completed full evaluations")


if __name__ == "__main__":
    main()
