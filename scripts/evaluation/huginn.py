"""Run pinned Huginn benchmark prompts with the packed KV backend."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluation.server_client import lm_eval_model_args, serving
from reproduction import atomic

COUNTS = {"gsm8k": 1319, "math500": 500, "humaneval": 164, "mbpp": 378}


def model_spec():
    protocol = ROOT / "experiments/KV_main_experiments/prompt_protocol.json"
    return json.loads(protocol.read_text())["models"]["huginn"]


def invocation(
    benchmark,
    output,
    limit,
    packed=False,
    gpu_memory_utilization=0.9,
    max_num_seqs=32,
    max_num_batched_tokens=None,
):
    model = model_spec()
    maximum = 1024 if benchmark in ("gsm8k", "humaneval") else 2048
    settings = dict(
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=False,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=1024,
        enable_prefix_caching=True,
        compilation_config={
            "level": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": list(range(1, max_num_seqs + 1)),
        },
    )
    if packed:
        settings.update(
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            max_num_batched_tokens=4096,
        )
    if max_num_batched_tokens is not None:
        settings["max_num_batched_tokens"] = max_num_batched_tokens
    if benchmark in ("humaneval", "mbpp"):
        script = ROOT / "scripts/run_evalplus_code.py"
        args = [
            "--benchmark",
            benchmark,
            "--output-root",
            str(output),
            "--model",
            model["name"],
            "--revision",
            model["revision"],
            "--chat-template",
        ]
    else:
        task = "gsm8k_cot" if benchmark == "gsm8k" else "huginn_math500_boxed"
        script = ROOT / "scripts/run_lm_eval_timed.py"
        args = [
            "run",
            "--model",
            "local-completions",
            "--model_args",
            lm_eval_model_args(model["name"], model["revision"], "{base_url}"),
            "--tasks",
            task,
            "--num_fewshot",
            "8" if benchmark == "gsm8k" else "4",
            "--batch_size",
            "auto",
            "--gen_kwargs",
            f"do_sample=False,temperature=0.0,max_gen_toks={maximum}",
            "--apply_chat_template",
            "--fewshot_as_multiturn",
            "--output_path",
            str(output),
            "--log_samples",
            "--seed",
            "0,0,0,0",
        ]
        if benchmark == "gsm8k":
            args += [
                "--system_instruction",
                (
                    "You are a helpful assistant that can assist users "
                    "with mathematical reasoning."
                ),
            ]
        else:
            args += ["--include_path", str(ROOT / "configs/eval/tasks")]
    if limit is not None:
        args += ["--limit", str(limit)]
    return script, args, maximum, settings


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--benchmark", choices=COUNTS, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--max-num-batched-tokens", type=int)
    a = p.parse_args()
    if not 0 < a.gpu_memory_utilization < 1:
        p.error("GPU memory utilization must be between 0 and 1")
    if not 1 <= a.max_num_seqs <= 32:
        p.error("max-num-seqs must be in [1, 32]")
    if a.max_num_batched_tokens is not None and a.max_num_batched_tokens < 4096:
        p.error("max-num-batched-tokens must be at least 4096")
    return a


def read_output(output, benchmark, limit):
    result_paths = list((output / "raw").rglob("results_*.json"))
    if len(result_paths) != 1:
        raise RuntimeError("Expected one lm-eval or EvalPlus result file")
    raw_result = json.loads(result_paths[0].read_text())
    task = (
        benchmark
        if benchmark in ("humaneval", "mbpp")
        else "gsm8k_cot"
        if benchmark == "gsm8k"
        else "huginn_math500_boxed"
    )
    if benchmark in ("humaneval", "mbpp"):
        samples_path = output / "raw" / f"{benchmark}_samples.jsonl"
        rows = [json.loads(line) for line in samples_path.read_text().splitlines()]
        records = [
            dict(
                document_index=index,
                task_id=row["task_id"],
                prompt=row["prompt"],
                generated_text=row["raw_output"],
                generated_tokens=row["output_tokens"],
                stop_reason=row["finish_reason"],
                solution=row["solution"],
            )
            for index, row in enumerate(rows)
        ]
        metric = "pass@1,base"
        accuracy = raw_result["results"][task][metric]
        plus = raw_result["results"][task]["pass@1,plus"]
    else:
        sample_paths = list((output / "raw").rglob(f"samples_{task}_*.jsonl"))
        if len(sample_paths) != 1:
            raise RuntimeError("Expected one lm-eval samples file")
        rows = [json.loads(line) for line in sample_paths[0].read_text().splitlines()]
        selected_filter = "flexible-extract" if benchmark == "gsm8k" else "none"
        rows = [row for row in rows if row["filter"] == selected_filter]
        accuracy_field = "exact_match" if benchmark == "gsm8k" else "math_verify"
        records = [
            dict(
                document_index=row["doc_id"],
                task_id=str(row["doc_id"]),
                prompt=row["arguments"]["gen_args_0"]["arg_0"],
                generated_text=row["resps"][0][0],
                correct=bool(row[accuracy_field]),
            )
            for row in rows
        ]
        records.sort(key=lambda row: row["document_index"])
        metric = (
            "exact_match,flexible-extract"
            if benchmark == "gsm8k"
            else "math_verify,none"
        )
        accuracy = raw_result["results"][task][metric]
        plus = None
    expected_count = limit if limit is not None else COUNTS[benchmark]
    if len(records) != expected_count:
        raise RuntimeError("Evaluator produced an incomplete result")
    return records, metric, accuracy, plus, result_paths[0]


def main():
    a = parse_args()
    model = model_spec()
    cfg = json.loads(a.config.read_text())
    method, rotated = (
        cfg.get("coefficient_method"),
        bool(cfg.get("rotation_artifact") or cfg.get("use_rotation")),
    )
    if cfg["packed_groups"] and method not in ("unit", "norm", "ls"):
        raise ValueError("Unsupported Huginn coefficient method")
    if a.limit not in (None, COUNTS[a.benchmark]):
        raise ValueError("Full evaluation requires every benchmark document")
    if rotated:
        assert cfg["calibration_domain"] == (
            "code" if a.benchmark in ("humaneval", "mbpp") else "gsm8k"
        )
    script, args, maximum, settings = invocation(
        a.benchmark,
        a.output / "raw",
        a.limit,
        packed=bool(cfg["packed_groups"]),
        gpu_memory_utilization=a.gpu_memory_utilization,
        max_num_seqs=a.max_num_seqs,
        max_num_batched_tokens=a.max_num_batched_tokens,
    )
    if a.dry_run:
        print(
            json.dumps(
                dict(
                    script=str(script), args=args, settings=settings, max_tokens=maximum
                ),
                indent=2,
            )
        )
        return
    a.output.mkdir(parents=True, exist_ok=True)
    gpu_name, capability = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                a.gpu,
                "--query-gpu=name,compute_cap",
                "--format=csv,noheader",
            ],
            text=True,
        )
        .strip()
        .split(", ")
    )
    contract = dict(
        model=model["name"],
        revision=model["revision"],
        loops=model["loops"],
        model_key="huginn",
        blas_library="default",
        gpu_name=gpu_name,
        compute_capability=[int(x) for x in capability.split(".")],
        benchmark=a.benchmark,
        config=cfg,
        requested=a.limit or COUNTS[a.benchmark],
        physical_gpu=a.gpu,
        max_new_tokens=maximum,
        settings=settings,
        prompt_protocol_sha256=hashlib.sha256(
            (ROOT / "experiments/KV_main_experiments/prompt_protocol.json").read_bytes()
        ).hexdigest(),
        source_sha256={
            str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [
                Path(__file__),
                script,
                ROOT / "scripts/evaluation/server.py",
                ROOT / "scripts/evaluation/server_client.py",
                ROOT / "scripts/code_prompt_protocol.py",
                *[
                    f
                    for directory in (
                        "plugins/residual_quantization/src/residual_quantization",
                        "plugins/residual_quantization/csrc",
                        "plugins/looped_transformers/src/looped_transformers",
                    )
                    for f in sorted((ROOT / directory).rglob("*"))
                    if f.suffix in (".py", ".cu", ".cuh", ".cpp", ".h")
                ],
            ]
        },
    )
    path = a.output / "result.json"
    if path.exists():
        raise ValueError("Use a fresh evaluation output; do not mix partial runs")
    started = time.perf_counter()
    with serving("huginn", a.config, a.gpu, a.output, settings) as base_url:
        initialization_seconds = time.perf_counter() - started
        args = [arg.replace("{base_url}", base_url) for arg in args]
        if a.benchmark in ("humaneval", "mbpp"):
            args += ["--base-url", base_url]
        client_env = os.environ.copy()
        client_env["RESIDUALQUANT_TIMING_FILE"] = str(a.output / "timing.json")
        subprocess.run([sys.executable, str(script), *args], check=True, env=client_env)
    records, metric, accuracy, plus, result_path = read_output(
        a.output, a.benchmark, a.limit
    )
    timing = json.loads((a.output / "timing.json").read_text())
    result = dict(
        contract=contract,
        records=records,
        scores=dict(
            metric=metric,
            accuracy_percent=100 * accuracy,
            plus_accuracy_percent=None if plus is None else 100 * plus,
        ),
        completed=True,
        batches=[dict(elapsed_seconds=timing["generate_until_seconds"])],
        initialization_seconds=initialization_seconds,
        scoring_seconds=timing.get("scoring_seconds"),
        physical_gpu=a.gpu,
        gpu_count=1,
        runner_seconds=time.perf_counter() - started,
        raw_result=str(result_path),
    )
    atomic(a.output / "result.json", result)


if __name__ == "__main__":
    main()
