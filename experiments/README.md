# Experiments

Run commands from the repository root after completing the [installation](../README.md).

```bash
source scripts/environment.sh
python scripts/run.py download --model ouro
python scripts/run.py prepare --model ouro
python scripts/run.py run --model ouro
python scripts/run.py status --model ouro
python scripts/run.py summarize --model ouro
```

Use `ouro` for Ouro-1.4B, `ouro26` for Ouro-2.6B, or `huginn` for Huginn-0125. `prepare` creates the execution plan; `run` executes it and skips completed jobs. Add `--retry-failed` to retry failed jobs.

| Experiment | Usage |
| --- | --- |
| Uniform quantization | [Instructions](uniform_residual_quantization/README.md) |
| Mixed precision schedules | [Instructions](bit_schedules/README.md) |
| Math and code benchmarks | [Instructions](KV_main_experiments/README.md) |
| Last-anchor ResidualQuant | [Instructions](reverse_anchor/README.md) |
| KV memory and latency | [Instructions](kv_memory_latency/README.md) |

Calibration artifacts, results, and summaries are stored under each experiment's `data/` directory. Ouro-2.6B uses the `data/ouro26/` subdirectory.
