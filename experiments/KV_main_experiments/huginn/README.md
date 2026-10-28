# Huginn benchmarks

Run commands from the repository root after completing the [installation](../../../README.md).

```bash
source scripts/environment.sh
python scripts/run.py download --model huginn
```

## Direct and OptR

```bash
python scripts/run.py prepare --model huginn
python scripts/run.py run --model huginn
python scripts/run.py status --model huginn
python scripts/run.py summarize --model huginn
```

This plan evaluates BF16, Direct, and independent OptR on GSM8K, MATH500, HumanEval, and MBPP. Outputs are stored in `experiments/KV_main_experiments/data/huginn_full_fp32/`.

## Last-anchor ResidualQuant

```bash
python scripts/run.py prepare --model huginn --huginn-last-anchor
python scripts/run.py run --model huginn --huginn-last-anchor
python scripts/run.py status --model huginn --huginn-last-anchor
python scripts/run.py summarize --model huginn --huginn-last-anchor
```

This plan evaluates BF16 and last-anchor LS/LS+OptR with G16 and G32. Outputs are stored in `experiments/KV_main_experiments/data/huginn_last_anchor_fp32/`.

Both plans include calibration, evaluation, and result aggregation. Each output directory contains `plan.json`, `configs/`, `calibrated/`, `evaluate/`, and `summary/`. Rerun `run` to continue an existing plan; add `--retry-failed` to retry failed jobs.
