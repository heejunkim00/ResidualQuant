# ResidualQuant

Low-bit KV cache quantization for looped Transformers. ResidualQuant represents earlier loops with INT2 residuals relative to an INT4 Last-loop anchor, combining least-squares (LS) scaling with shared OptR-H rotations. Packed CUDA attention kernels reconstruct KV tiles during attention, integrated with vLLM's paged cache and CUDA Graph decoding.

## Installation

Linux, an NVIDIA GPU, CUDA 12.8, and [uv](https://docs.astral.sh/uv/) are required. Dependencies are pinned to Python 3.11, PyTorch 2.8.0, vLLM 0.10.2, and Transformers 4.56.2.

```bash
uv sync --locked --python 3.11
source scripts/environment.sh
```

Run commands from the repository root. Installation builds the CUDA extension; set `CUDA_HOME` if the toolkit is outside the default search path.

## Models

| Model | Model key | Loops |
| --- | --- | --- |
| Ouro-1.4B | `ouro` | 4 |
| Ouro-2.6B | `ouro26` | 4 |
| Huginn-3.5B | `huginn` | 32 |

Evaluation supports GSM8K, MATH500, HumanEval, and MBPP.

The final loop in each group is the anchor. INT2 residuals use group size 16 or 32; INT4 anchors use group size 32. LS without rotation is an ablation of ResidualQuant.

## Ouro

### BF16 evaluation

Select Ouro-1.4B with `--model-key ouro` (default), or Ouro-2.6B with `--model-key ouro26`:

```bash
python scripts/evaluation/ouro.py \
  --model-key ouro26 \
  --benchmark gsm8k \
  --gpu 0 \
  --max-num-seqs 32 --num-concurrent 32 \
  --no-prefix-caching \
  --output experiments/ouro26_baseline/data/gsm8k/result.json
```

Select `gsm8k`, `math500`, `humaneval`, or `mbpp` with `--benchmark` and use a separate output path for each evaluation. `--num-concurrent` and `--no-prefix-caching` are GSM8K options; omit them for the other benchmarks. `--max-num-seqs` sets the server concurrency limit for all four benchmarks.

### Calibration

The commands below use Ouro-1.4B. For Ouro-2.6B, select `--model-key ouro26` for both calibration and evaluation and use a separate output directory.

Download the model and calibrate shared rotations for ResidualQuant with group size 32:

```bash
python scripts/run.py download --model ouro

python scripts/calibration/calibrate.py --model-key ouro \
  --gpu 0 \
  --config experiments/reverse_anchor/calibration_g32.json \
  --output experiments/reverse_anchor/data/g32
```

Calibration uses 128 fitting and 128 holdout examples from GSM8K training data with seed 0. The exported `config.json` selects the fitted rotations for evaluation.

For group size 16, use `calibration_g16.json` and a separate output directory such as `experiments/reverse_anchor/data/g16`.

### Quantized evaluation

Evaluate GSM8K:

```bash
python scripts/evaluation/ouro.py --benchmark gsm8k \
  --gpu 0 \
  --config experiments/reverse_anchor/data/g32/config.json \
  --output experiments/reverse_anchor/data/g32/gsm8k/result.json \
  --no-prefix-caching
```

To evaluate the BF16 baseline, omit `--config`.

Evaluate MATH500 with the same calibrated rotations:

```bash
python scripts/evaluation/ouro.py \
  --gpu 0 \
  --benchmark math500 \
  --config experiments/reverse_anchor/data/g32/config.json \
  --output experiments/reverse_anchor/data/g32/math500/result.json
```

Calibrate and evaluate Ouro-2.6B with the same G32 configuration:

```bash
python scripts/calibration/calibrate.py --model-key ouro26 \
  --gpu 0 \
  --config experiments/reverse_anchor/calibration_g32.json \
  --output experiments/reverse_anchor/data/ouro26_g32
python scripts/evaluation/ouro.py --model-key ouro26 \
  --benchmark math500 --gpu 0 --max-num-seqs 32 \
  --config experiments/reverse_anchor/data/ouro26_g32/config.json \
  --output experiments/reverse_anchor/data/ouro26_g32/math500/result.json
```

For the LS-only ablation, use `experiments/reverse_anchor/ls_g16.json` or `ls_g32.json` directly; these configurations require no calibration.

HumanEval and MBPP use rotations calibrated on MBPP training data. Repeat calibration with `--dataset mbpp` and a separate output directory, then evaluate with `--benchmark humaneval` or `--benchmark mbpp` using that directory's `config.json`.

### Decode performance

Measure fixed-batch decoding at 2k, 4k, and 8k context:

```bash
python experiments/kv_memory_latency/measure.py \
  --gpu 0 \
  --config experiments/reverse_anchor/data/g32/config.json \
  --contexts 2048 4096 8192 \
  --batch 1 --tokens 128 --warmups 1 --trials 3 \
  --output experiments/kv_memory_latency/data/residualquant_g32
```

Timing excludes prefill and the first output token. For larger batches, select one context length per invocation. See the [measurement protocol](experiments/kv_memory_latency/README.md) for KV footprint calculations.

## Huginn-3.5B

Run the Last-loop comparison on GSM8K, MATH500, HumanEval, and MBPP:

```bash
python scripts/run.py download --model huginn
python scripts/run.py prepare --model huginn --huginn-last-anchor --gpu 0
python scripts/run.py run --model huginn --huginn-last-anchor --gpu 0
python scripts/run.py status --model huginn --huginn-last-anchor

python experiments/KV_main_experiments/huginn/summarize.py \
  --data experiments/KV_main_experiments/data/huginn_last_anchor_fp32 \
  --require-complete
```

The plan includes BF16, LS without rotation, and ResidualQuant at INT2 group sizes 16 and 32. See the [Huginn protocol](experiments/KV_main_experiments/huginn/README.md) for the default and last-anchor experiment plans.

## Experiments

The comparison and ablation plans accept `--model ouro` or `--model ouro26`:

```bash
python scripts/run.py prepare --model ouro26
python scripts/run.py run --model ouro26 --gpu 0
python scripts/run.py summarize --model ouro26
```

Ouro-2.6B plans, calibration artifacts, evaluations, and summaries are stored under each experiment's `data/ouro26/` directory. Independent OptR uses `direct_ouro.py`; shared ResidualQuant rotations use `calibrate.py`.

| Experiment | Configuration and commands |
| --- | --- |
| Last-loop residual quantization | [Ouro](experiments/reverse_anchor/README.md), [Huginn](experiments/KV_main_experiments/huginn/README.md) |
| Uniform precision and residual/rotation comparisons | [Uniform quantization](experiments/uniform_residual_quantization/README.md) |
| Loop-wise precision allocation | [Mixed precision](experiments/bit_schedules/README.md) |
| Math and code benchmarks | [Benchmark protocols](experiments/KV_main_experiments/README.md) |
| Decode latency and KV memory | [Instructions](experiments/kv_memory_latency/README.md) |

## Prefill CPU offload

To offload retained prefill KV states, add the following fields to the serving configuration:

```json
{
  "prefill_cpu_offload": true,
  "prefill_offload_chunk_tokens": 512
}
```

Offloading overlaps host transfers with recurrent computation to reduce temporary GPU memory. It is disabled by default. Packed attention requires full-prompt prefill; prefix caching and chunked prefill are disabled by the evaluation runners.

## Results

Each benchmark writes `result.json` with evaluation settings, generated responses, scores, and completion status. Raw evaluator outputs are stored under `raw/`.

Math benchmarks report answer accuracy; code benchmarks report base and plus pass@1. See the [benchmark protocols](experiments/KV_main_experiments/README.md) for prompts and scoring rules.
