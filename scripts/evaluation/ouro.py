"""Evaluate pinned Ouro math/code benchmarks through a local vLLM server."""

import argparse
import hashlib
import json
import os
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluation.server_client import lm_eval_model_args, serving
from reproduction import atomic
from ouro_prompt_protocol import FEWSHOT_PREFIX, last_number

COUNTS = {"math500": 500, "humaneval": 164, "mbpp": 378}


def invocation(
    benchmark,
    output,
    base_url,
    model,
    revision,
    math_task="ouro_math500_rotation",
):
    if benchmark in ("humaneval", "mbpp"):
        command = [
            sys.executable,
            str(ROOT / "scripts/run_evalplus_code.py"),
            "--benchmark",
            benchmark,
            "--output-root",
            str(output),
            "--base-url",
            base_url,
            "--model",
            model,
            "--revision",
            revision,
        ]
    else:
        command = [
            sys.executable,
            str(ROOT / "scripts/run_lm_eval_timed.py"),
            "run",
            "--model",
            "local-completions",
            "--model_args",
            lm_eval_model_args(model, revision, base_url),
            "--tasks",
            math_task,
            "--num_fewshot",
            "0",
            "--batch_size",
            "auto",
            "--gen_kwargs",
            "do_sample=False,temperature=0.0,max_gen_toks=2048",
            "--include_path",
            str(ROOT / "configs/eval/tasks"),
            "--output_path",
            str(output),
            "--log_samples",
            "--seed",
            "0,0,0,0",
        ]
    return command


def read_output(benchmark, raw, limit, math_task="ouro_math500_rotation"):
    results = list(raw.rglob("results_*.json"))
    if len(results) != 1:
        raise RuntimeError("Expected exactly one evaluator result")
    result = json.loads(results[0].read_text())
    if benchmark == "math500":
        samples = list(raw.rglob(f"samples_{math_task}_*.jsonl"))
        if len(samples) != 1:
            raise RuntimeError("Expected exactly one MATH500 sample file")
        rows = [json.loads(line) for line in samples[0].read_text().splitlines()]
        rows = [row for row in rows if row["filter"] == "none"]
        records = [
            dict(
                document_index=row["doc_id"],
                task_id=str(row["doc_id"]),
                prompt=row["arguments"]["gen_args_0"]["arg_0"],
                generated_text=row["resps"][0][0],
                correct=bool(row["math_verify"]),
            )
            for row in rows
        ]
        task = math_task
        score = dict(
            metric="math_verify",
            accuracy_percent=100 * result["results"][task]["math_verify,none"],
            plus_accuracy_percent=None,
        )
    else:
        samples = raw / f"{benchmark}_samples.jsonl"
        rows = [json.loads(line) for line in samples.read_text().splitlines()]
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
        values = result["results"][benchmark]
        score = dict(
            metric="pass@1,base",
            accuracy_percent=100 * values["pass@1,base"],
            plus_accuracy_percent=100 * values["pass@1,plus"],
        )
    records.sort(key=lambda row: row["document_index"])
    if len(records) != limit:
        raise RuntimeError("Evaluator returned an incomplete benchmark")
    return records, score, result


def evaluate(benchmark, args, config, protocol, output):
    if "bf16_variant" in config:
        raise ValueError(
            "Legacy matched BF16 control is unsupported; regenerate the BF16 config"
        )
    if output.exists():
        raise ValueError("Use a fresh output path for this evaluation")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = output.parent / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rule = next(
        row
        for row in protocol["benchmarks"]
        if row["model"] == args.model_key and row["benchmark"] == benchmark
    )
    model = protocol["models"][args.model_key]
    native_bf16 = not config["packed_groups"]
    reverse = config.get("reference_mode") == "anchor" and config["anchor_index"] == 3
    max_sequences = args.max_num_seqs or (4 if reverse else 32)
    capture_sizes = [n for n in (1, 2, 4, 8, 16, 32) if n < max_sequences] + [
        max_sequences
    ]
    settings = dict(
        gpu_memory_utilization=0.9,
        max_num_seqs=max_sequences,
        max_num_batched_tokens=4096,
        enable_prefix_caching=False,
        enable_chunked_prefill=native_bf16,
        num_gpu_blocks_override=1025 if reverse else None,
        blas_library=args.blas_library,
        compilation_config=dict(
            level=0,
            cudagraph_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=capture_sizes,
        ),
    )
    limit = COUNTS[benchmark]
    started = time.perf_counter()
    with serving(args.model_key, args.config, args.gpu, output.parent, settings) as url:
        initialization_seconds = time.perf_counter() - started
        command = invocation(
            benchmark,
            raw,
            url,
            model["name"],
            model["revision"],
            math_task=rule["task"],
        )
        client_env = os.environ.copy()
        client_env["RESIDUALQUANT_TIMING_FILE"] = str(raw / "timing.json")
        subprocess.run(command, check=True, env=client_env)

    records, scores, evaluator = read_output(
        benchmark, raw, limit, math_task=rule["task"]
    )
    prompts = [row["prompt"] for row in records]
    prompt_hash = hashlib.sha256(
        json.dumps(prompts, ensure_ascii=False).encode()
    ).hexdigest()
    source_files = (
        Path(__file__),
        ROOT / "scripts/evaluation/server.py",
        ROOT / "scripts/evaluation/server_client.py",
        ROOT / "plugins/looped_transformers/src/looped_transformers/model_setup.py",
        (
            ROOT / "scripts/run_lm_eval_timed.py"
            if benchmark == "math500"
            else ROOT / "scripts/run_evalplus_code.py"
        ),
    )
    contract = dict(
        benchmark=benchmark,
        model=model["name"],
        model_key=args.model_key,
        revision=model["revision"],
        loops=model["loops"],
        config=config,
        int2_experiment=config,
        protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        source_sha256={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
        prompt_rule=rule,
        prompt_sha256=prompt_hash,
        max_new_tokens=rule["max_new_tokens"],
        limit=limit,
        fewshot=0,
        apply_chat_template=False,
        chatml=False,
        stop_strings=rule["stop"],
        seed=0,
        decoding="greedy",
        max_model_len=4096,
        max_num_seqs=settings["max_num_seqs"],
        max_num_batched_tokens=4096,
        prefix_caching=False,
        cudagraph="FULL_DECODE_ONLY",
        blas_library=args.blas_library,
    )
    if reverse:
        contract.update(chunked_prefill=False, num_gpu_blocks_override=1025)
    timing = json.loads((raw / "timing.json").read_text())
    elapsed = timing["generate_until_seconds"]
    result = dict(
        contract=contract,
        records=records,
        scores=scores,
        completed=True,
        batches=[dict(elapsed_seconds=elapsed)],
        scoring_seconds=timing.get("scoring_seconds"),
        physical_gpu=args.gpu,
        gpu_count=1,
        initialization_seconds=initialization_seconds,
        runner_seconds=time.perf_counter() - started,
        raw_result=str(next(raw.rglob("results_*.json"))),
    )
    atomic(output, result)
    return dict(benchmark=benchmark, path=str(output), count=limit, completed=True)


def evaluate_benchmarks(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--model-key", choices=("ouro", "ouro26"), default="ouro")
    parser.add_argument("--benchmark", choices=list(COUNTS), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "experiments/KV_main_experiments/prompt_protocol.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--blas-library", choices=["default", "cublaslt"], default="default"
    )
    args = parser.parse_args(argv)
    if args.max_num_seqs is not None and not 1 <= args.max_num_seqs <= 32:
        parser.error("Maximum sequences must be between 1 and 32")
    if args.limit not in (None, COUNTS[args.benchmark]):
        parser.error("Full evaluation requires every benchmark document")

    config = (
        json.loads(args.config.read_text()) if args.config else {"packed_groups": []}
    )
    protocol = json.loads(args.protocol.read_text())
    for name, digest in protocol["source_sha256"].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"Prompt protocol source hash mismatch: {name}")

    evaluate(args.benchmark, args, config, protocol, args.output)


def parse_gsm8k_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--model-key", choices=("ouro", "ouro26"), default="ouro")
    parser.add_argument("--limit", type=int, default=1319)
    parser.add_argument("--num-gpu-blocks-override", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument(
        "--num-concurrent",
        type=int,
        help="Concurrent API requests; defaults to max(32, twice the server slots)",
    )
    parser.add_argument(
        "--auto-kv-cache",
        action="store_true",
        help="Let vLLM allocate KV cache from the GPU memory budget",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--blas-library",
        choices=["auto", "default", "cublas", "cublaslt"],
        default="auto",
    )
    parser.add_argument("--eager", action="store_true")
    parser.add_argument(
        "--prefix-caching", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("GPU memory utilization must be between 0 and 1")
    if args.max_num_seqs is not None and not 1 <= args.max_num_seqs <= 4096:
        parser.error("Maximum sequences must fit the 4096-token scheduling budget")
    if args.num_concurrent is not None and args.num_concurrent < 1:
        parser.error("Client concurrency must be positive")
    if args.auto_kv_cache and args.num_gpu_blocks_override is not None:
        parser.error("Automatic KV allocation conflicts with a block override")
    if args.limit != 1319:
        parser.error("Full GSM8K evaluation requires all 1319 documents")
    if args.output.exists():
        raise ValueError("Use a fresh output path for this evaluation")
    if args.blas_library == "auto":
        args.blas_library = "default"

    return args


def gsm8k_settings(args, native_bf16, reverse):
    max_sequences = args.max_num_seqs or (4 if reverse else 32)
    if args.num_concurrent is None:
        args.num_concurrent = max(32, 2 * max_sequences)

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    needed = 2 * args.num_concurrent + 256
    if soft != resource.RLIM_INFINITY and soft < needed:
        if hard != resource.RLIM_INFINITY and hard < needed:
            raise ValueError("File descriptor limit is too low for client concurrency")
        resource.setrlimit(resource.RLIMIT_NOFILE, (needed, hard))

    capture_sizes = [1]
    while capture_sizes[-1] * 2 <= max_sequences:
        capture_sizes.append(capture_sizes[-1] * 2)
    if capture_sizes[-1] != max_sequences:
        capture_sizes.append(max_sequences)
    blocks = args.num_gpu_blocks_override
    if reverse and blocks is None and not args.auto_kv_cache:
        blocks = 1025
    settings = dict(
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=max_sequences,
        max_num_batched_tokens=4096,
        enable_prefix_caching=args.prefix_caching if native_bf16 else False,
        enable_chunked_prefill=native_bf16,
        num_gpu_blocks_override=blocks,
        enforce_eager=args.eager,
        blas_library=args.blas_library,
        compilation_config=dict(
            level=0,
            cudagraph_mode="NONE" if args.eager else "FULL_DECODE_ONLY",
            cudagraph_capture_sizes=capture_sizes,
        ),
    )
    return settings


def read_gsm8k_output(raw, count):
    result_paths = list(raw.rglob("results_*.json"))
    samples_paths = list(raw.rglob("samples_ouro_gsm8k_rotation_*.jsonl"))
    if len(result_paths) != 1 or len(samples_paths) != 1:
        raise RuntimeError("Expected one lm-eval result and one samples file")
    samples = [json.loads(line) for line in samples_paths[0].read_text().splitlines()]
    samples = [sample for sample in samples if sample["filter"] == "none"]
    records = []
    for sample in samples:
        row = sample["doc"]
        generated = sample["resps"][0][0]
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        strict = re.search(r"The answer is (\-?[0-9\.\,]+).", generated)
        prediction = strict.group(1).replace(",", "").rstrip(".") if strict else None
        flexible = last_number(generated)
        record = dict(
            document_index=sample["doc_id"],
            task_id=str(sample["doc_id"]),
            prompt=sample["arguments"]["gen_args_0"]["arg_0"],
            question=row["question"],
            gold=gold,
            strict_prediction=prediction,
            flexible_prediction=flexible,
            strict_correct=prediction == gold,
            flexible_correct=flexible == gold,
            generated_text=generated,
        )
        if record["strict_correct"] != bool(sample["strict_match"]):
            raise RuntimeError("Strict scorer differs from the original contract")
        if record["flexible_correct"] != bool(sample["flexible_match"]):
            raise RuntimeError("Flexible scorer differs from the original contract")
        records.append(record)
    records.sort(key=lambda row: row["document_index"])
    if len(records) != count:
        raise RuntimeError("lm-eval returned an incomplete GSM8K benchmark")

    return records, result_paths[0]


def evaluate_gsm8k(argv):
    args = parse_gsm8k_args(argv)
    protocol_path = ROOT / "experiments/KV_main_experiments/prompt_protocol.json"
    model = json.loads(protocol_path.read_text())["models"][args.model_key]
    config = json.loads(args.config.read_text()) if args.config else None
    if config is not None and "bf16_variant" in config:
        raise ValueError(
            "Legacy matched BF16 control is unsupported; regenerate the BF16 config"
        )
    native_bf16 = config is None or not config["packed_groups"]
    reverse = config is not None and (
        config.get("reference_mode") == "anchor" and config["anchor_index"] == 3
    )
    count = 1319
    maximum = 1024
    stop = ["Q:", "</s>", "<|im_end|>"]
    settings = gsm8k_settings(args, native_bf16, reverse)
    max_sequences = settings["max_num_seqs"]
    capture_sizes = settings["compilation_config"]["cudagraph_capture_sizes"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = args.output.parent / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with serving(
        args.model_key, args.config, args.gpu, args.output.parent, settings
    ) as url:
        initialization = time.perf_counter() - started
        command = [
            sys.executable,
            str(ROOT / "scripts/run_lm_eval_timed.py"),
            "run",
            "--model",
            "local-completions",
            "--model_args",
            lm_eval_model_args(
                model["name"],
                model["revision"],
                url,
                num_concurrent=args.num_concurrent,
            ),
            "--tasks",
            "ouro_gsm8k_rotation",
            "--num_fewshot",
            "0",
            "--batch_size",
            "auto",
            "--gen_kwargs",
            f"do_sample=False,temperature=0.0,max_gen_toks={maximum}",
            "--include_path",
            str(ROOT / "configs/eval/tasks"),
            "--output_path",
            str(raw),
            "--log_samples",
            "--seed",
            "0,0,0,0",
        ]
        client_env = os.environ.copy()
        client_env["RESIDUALQUANT_TIMING_FILE"] = str(raw / "timing.json")
        subprocess.run(command, check=True, env=client_env)

    records, result_path = read_gsm8k_output(raw, count)

    sources = (
        "plugins/looped_transformers/src/looped_transformers/ouro.py",
        "scripts/evaluation/ouro.py",
        "scripts/ouro_prompt_protocol.py",
        "scripts/run_lm_eval_timed.py",
        "scripts/evaluation/server.py",
        "scripts/evaluation/server_client.py",
        "plugins/looped_transformers/src/looped_transformers/model_setup.py",
        "configs/eval/tasks/ouro_gsm8k_rotation.yaml",
        "plugins/residual_quantization/setup.py",
        *(
            str(path.relative_to(ROOT))
            for path in sorted((ROOT / "plugins/residual_quantization/csrc").rglob("*"))
            if path.suffix in (".cu", ".cuh", ".cpp", ".h")
        ),
    )
    contract = dict(
        benchmark="gsm8k",
        model=model["name"],
        model_key=args.model_key,
        revision=model["revision"],
        protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        config=config,
        mode="bf16" if native_bf16 else "packed",
        kv_policy=(
            "bf16"
            if native_bf16
            else "direct"
            if all(len(group) == 1 for group in config["packed_groups"])
            else "last_anchor"
            if reverse
            else "first_anchor"
            if config["reference_mode"] == "anchor"
            else config["reference_mode"]
        ),
        schedule=(
            [16] * model["loops"]
            if native_bf16
            else [bits for group in config["packed_groups"] for bits, _ in group]
        ),
        blas_library=args.blas_library,
        fewshot=3,
        prompt_sha256=hashlib.sha256(FEWSHOT_PREFIX.encode()).hexdigest(),
        max_new_tokens=maximum,
        decoding="greedy",
        loops=model["loops"],
        limit=count,
        batch_size=max_sequences,
        client_batch_size="auto",
        client_concurrency=args.num_concurrent,
        seed=0,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        prefix_caching=settings["enable_prefix_caching"],
        scheduling=f"api_concurrent_requests_{args.num_concurrent}",
        cudagraph=settings["compilation_config"]["cudagraph_mode"],
        cudagraph_capture_sizes=capture_sizes,
        compilation_level=0,
        stop_strings=stop,
        gpu_memory_utilization=args.gpu_memory_utilization,
        source_sha256={
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                *sources,
                *(
                    str(path.relative_to(ROOT))
                    for path in sorted(
                        (
                            ROOT
                            / "plugins/residual_quantization/src/residual_quantization"
                        ).rglob("*")
                    )
                    if path.suffix in (".py", ".cu", ".cuh")
                ),
            )
        },
    )
    if args.num_gpu_blocks_override is not None or reverse:
        contract["num_gpu_blocks_override"] = settings["num_gpu_blocks_override"]
    if config:
        contract["int2_experiment"] = config
        contract["int2_experiment_sha256"] = hashlib.sha256(
            args.config.read_bytes()
        ).hexdigest()
    if reverse:
        contract["chunked_prefill"] = False

    timing = json.loads((raw / "timing.json").read_text())
    strict_correct = sum(record["strict_correct"] for record in records)
    flexible_correct = sum(record["flexible_correct"] for record in records)
    gpu_name = subprocess.check_output(
        ["nvidia-smi", "-i", args.gpu, "--query-gpu=name", "--format=csv,noheader"],
        text=True,
    ).strip()
    result = dict(
        contract=contract,
        backend="vLLM BF16" if native_bf16 else "vLLM packed KV",
        physical_gpu=args.gpu,
        gpu_count=1,
        gpu=gpu_name,
        records=records,
        batches=[dict(elapsed_seconds=timing["generate_until_seconds"])],
        done=len(records),
        completed=True,
        initialization_seconds=initialization,
        session_elapsed_seconds=time.perf_counter() - started,
        runner_seconds=time.perf_counter() - started,
        strict_correct=strict_correct,
        flexible_correct=flexible_correct,
        strict_accuracy=strict_correct / count,
        flexible_accuracy=flexible_correct / count,
        scores=dict(
            metric="flexible_match",
            accuracy_percent=100 * flexible_correct / count,
            plus_accuracy_percent=None,
        ),
        raw_result=str(result_path),
    )
    atomic(args.output, result)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Ouro-1.4B or Ouro-2.6B; select a benchmark before --help for its options."
    )
    parser.add_argument("--benchmark", choices=["gsm8k", *COUNTS], required=True)
    if "--benchmark" in sys.argv[1:]:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--benchmark", choices=["gsm8k", *COUNTS], required=True)
    args, remaining = parser.parse_known_args()
    if args.benchmark == "gsm8k":
        evaluate_gsm8k(remaining)
    else:
        evaluate_benchmarks(sys.argv[1:])


if __name__ == "__main__":
    main()
