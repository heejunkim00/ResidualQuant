# Mixed precision schedules

Compare loop-wise KV precision schedules on GSM8K using the [shared Ouro experiment plan](../uniform_residual_quantization/README.md):

```bash
source scripts/environment.sh
python scripts/run.py prepare --model ouro
python scripts/run.py run --model ouro
python scripts/run.py status --model ouro
python scripts/run.py summarize --model ouro
```

Use `--model ouro26` for Ouro-2.6B. Run these commands as part of the shared plan; a second `prepare` is not needed if the plan already exists.

Matching uniform configurations reuse results from `uniform_residual_quantization`. Mixed-schedule results are saved under `experiments/bit_schedules/data/`, with aggregated JSON/CSV results under `data/summary/`. Ouro-2.6B uses `data/ouro26/` instead of `data/`.
