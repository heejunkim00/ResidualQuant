# KV memory and decode latency

Run from the repository root after [installation](../../README.md). These tools currently measure Ouro-1.4B.

## Decode latency

Use a prepared inference configuration. For a method with learned rotations, pass the `config.json` produced by calibration.

```bash
source scripts/environment.sh
python experiments/kv_memory_latency/measure.py --gpu 0 \
  --config path/to/config.json \
  --output experiments/kv_memory_latency/data/run
```

The result is saved as `result.json` in the output directory. Use `--contexts` to select input lengths, `--batch` for batch size, and `--tokens`, `--trials`, and `--warmups` for measurement length and repetitions. Pass one context length per invocation when batch size exceeds one. Decode timing excludes prefill and the first output token.

## KV footprint

```bash
python experiments/kv_memory_latency/capacity.py \
  --config path/to/config.json --context 8192 \
  --output experiments/kv_memory_latency/data/capacity
```

This writes the calculated KV footprint to `capacity.json`.
