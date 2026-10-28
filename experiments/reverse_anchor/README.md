# Last-anchor ResidualQuant

Run from the repository root after [installation](../../README.md):

```bash
source scripts/environment.sh
python scripts/run.py download --model ouro
```

## LS

```bash
python scripts/evaluation/ouro.py --model-key ouro --benchmark gsm8k --gpu 0 \
  --config experiments/reverse_anchor/ls_g32.json \
  --output experiments/reverse_anchor/data/ls_g32/result.json
```

## LS+OptR

Calibrate the rotations, then evaluate the generated configuration:

```bash
python scripts/calibration/calibrate.py --model-key ouro --gpu 0 \
  --config experiments/reverse_anchor/calibration_g32.json \
  --output experiments/reverse_anchor/data/g32
python scripts/evaluation/ouro.py --model-key ouro --benchmark gsm8k --gpu 0 \
  --config experiments/reverse_anchor/data/g32/config.json \
  --output experiments/reverse_anchor/data/g32/result.json
```

Replace `g32` with `g16` in the configuration and output paths to use G16. For Ouro-2.6B, download with `--model ouro26`, use `--model-key ouro26` in both commands, and choose separate output directories.

For MATH500, change `--benchmark gsm8k` to `--benchmark math500` and use a separate evaluation output directory. For HumanEval or MBPP, calibrate with `--dataset mbpp` and evaluate with `--benchmark humaneval` or `--benchmark mbpp`.

Calibration writes `rotations.pt` and `config.json` to its output directory. Rerunning the same calibration resumes completed layers. Evaluation writes `result.json` and retains raw outputs in the adjacent `raw/` directory.

For Huginn, use the [last-anchor plan](../KV_main_experiments/huginn/README.md).
