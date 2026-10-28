# Uniform quantization

Compare uniform INT2 G16/G32 and INT4 G32 configurations on GSM8K using the shared Ouro experiment plan. Run from the repository root after [installation](../../README.md):

```bash
source scripts/environment.sh
python scripts/run.py download --model ouro
python scripts/run.py prepare --model ouro
python scripts/run.py run --model ouro
python scripts/run.py status --model ouro
python scripts/run.py summarize --model ouro
```

Use `--model ouro26` for Ouro-2.6B. The shared plan also runs the mixed-schedule and math/code experiments; calibration is included for methods that require learned rotations.

The plan is saved to `experiments/uniform_residual_quantization/data/plan.json`. Uniform evaluation results are under `data/evaluations/`, and JSON/CSV summaries are under `data/summary/`. Ouro-2.6B uses `data/ouro26/` instead of `data/`.

Rerun `run` to continue an existing plan. Add `--retry-failed` to retry failed jobs.
