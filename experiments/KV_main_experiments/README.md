# Math and code benchmarks

Evaluate GSM8K, MATH500, HumanEval, and MBPP from the repository root:

```bash
source scripts/environment.sh
python scripts/run.py download --model ouro
python scripts/run.py prepare --model ouro
python scripts/run.py run --model ouro
python scripts/run.py status --model ouro
python scripts/run.py summarize --model ouro
```

Replace `ouro` with `ouro26` for Ouro-2.6B. The plan includes calibration for methods that require learned rotations. See the [Huginn instructions](huginn/README.md) for its default and last-anchor plans.

Ouro execution plans are stored in `experiments/uniform_residual_quantization/data/plan.json`, or `data/ouro26/plan.json` for Ouro-2.6B. Math and code benchmark results are stored in this directory's `data/`, or `data/ouro26/`, with aggregated results under `summary/`.

To run an individual benchmark with a calibrated configuration:

```bash
python scripts/evaluation/ouro.py --model-key ouro --benchmark math500 \
  --gpu 0 --config path/to/config.json --output path/to/result.json
```

Use `gsm8k`, `math500`, `humaneval`, or `mbpp` for `--benchmark`. Omit `--config` to evaluate the BF16 baseline. Raw evaluator outputs are saved beside the result in `raw/`; use a separate output directory for each run.
