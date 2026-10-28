"""Pinned EvalPlus plain/chat prompts, generation and benchmark scoring."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from code_prompt_protocol import (
    STOP_STRINGS,
    MAX_TOKENS,
    build_prompt,
    build_chat_prompt,
)
from evaluation.server_client import completions
from reproduction import atomic


def generate(args) -> tuple[list[dict], dict]:
    """Run the model over the split and return one record per problem."""
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    data = get_human_eval_plus() if args.benchmark == "humaneval" else get_mbpp_plus()
    if args.limit is not None and args.limit > len(data):
        raise ValueError("limit exceeds the benchmark size")
    items = list(data.items())
    if args.limit is not None:
        items = items[: args.limit]

    if args.chat_template:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model, revision=args.revision or None, trust_remote_code=True
        )
        if tokenizer.chat_template is None:
            raise SystemExit(
                f"--chat-template asked for, but {args.model} ships no chat "
                "template; stock EvalPlus would use the raw prompt here"
            )
        prompts = [build_chat_prompt(task["prompt"], tokenizer) for _, task in items]
    else:
        prompts = [build_prompt(task["prompt"]) for _, task in items]
    generate_started = time.perf_counter()
    outputs = completions(
        args.base_url + "/completions",
        args.model,
        prompts,
        args.max_tokens or MAX_TOKENS[args.benchmark],
        STOP_STRINGS,
    )
    generate_seconds = time.perf_counter() - generate_started

    from evalplus.sanitize import sanitize

    records = []
    generated_tokens = 0
    for (task_id, task), prompt, output in zip(items, prompts, outputs):
        text = output["text"]
        generated_tokens += len(output["token_ids"])
        records.append(
            {
                "task_id": task_id,
                "solution": sanitize(text, entrypoint=task["entry_point"]),
                "prompt": prompt,
                "raw_output": text,
                "finish_reason": output["finish_reason"],
                "output_tokens": len(output["token_ids"]),
            }
        )

    timing = {
        "model_init_seconds": 0.0,
        "generate_until_seconds": generate_seconds,
        "generated_samples": len(records),
        "generated_tokens": generated_tokens,
        "generation_calls": 1,
        "completed": True,
        "base_url": args.base_url,
    }
    return records, timing


def score(samples_path: Path, benchmark: str) -> dict:
    """Run EvalPlus and read base/plus pass@1 scores."""
    from evalplus.eval import PASS
    from evalplus.evaluate import evaluate

    results_path = samples_path.with_name(samples_path.stem + "_eval_results.json")
    if results_path.exists():
        results_path.unlink()

    evaluate(
        dataset=benchmark,
        samples=str(samples_path),
        base_only=False,
        i_just_wanna_run=False,
    )

    payload = json.loads(results_path.read_text())
    evaluations = payload["eval"]

    total = len(evaluations)
    base_passed = 0
    plus_passed = 0
    for attempts in evaluations.values():
        attempt = attempts[0]
        base_status = attempt["base_status"]
        plus_status = attempt["plus_status"]
        base_passed += int(base_status == PASS)
        plus_passed += int(base_status == PASS and plus_status == PASS)

    return {
        "total": total,
        "base_passed": base_passed,
        "plus_passed": plus_passed,
        "pass@1,base": base_passed / total if total else None,
        "pass@1,plus": plus_passed / total if total else None,
        "eval_results_path": str(results_path),
    }


def write_lm_eval_shaped_result(
    output_root: Path,
    benchmark: str,
    scores: dict,
    seconds: float,
    model: str,
    revision: str,
    prompt: str,
) -> Path:
    """Write EvalPlus scores in the lm-eval result format."""
    directory = output_root / benchmark / model.replace("/", "__")
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%f")
    path = directory / f"results_{stamp}.json"
    atomic(
        path,
        {
            "results": {
                benchmark: {
                    "alias": benchmark,
                    "pass@1,base": scores["pass@1,base"],
                    "pass@1,plus": scores["pass@1,plus"],
                    "base_passed": scores["base_passed"],
                    "plus_passed": scores["plus_passed"],
                }
            },
            "n-samples": {
                benchmark: {"original": scores["total"], "effective": scores["total"]}
            },
            "total_evaluation_time_seconds": seconds,
            "scorer": "evalplus",
            "prompt": prompt,
            "model": model,
            "revision": revision,
        },
    )
    return path


def main() -> None:
    protocol_path = (
        Path(__file__).resolve().parents[1]
        / "experiments/KV_main_experiments/prompt_protocol.json"
    )
    default_model = json.loads(protocol_path.read_text())["models"]["ouro"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("humaneval", "mbpp"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--model", default=default_model["name"])
    parser.add_argument("--revision", default=default_model["revision"])
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help=(
            "wrap the EvalPlus instruction in the model's own chat template, "
            "which is what stock EvalPlus does for any tokenizer that has one"
        ),
    )
    args = parser.parse_args()
    if args.max_tokens is not None and args.max_tokens < 1:
        parser.error("max-tokens must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    records, timing = generate(args)

    samples_path = output_root / f"{args.benchmark}_samples.jsonl"
    with samples_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    scoring_started = time.perf_counter()
    scores = score(samples_path, args.benchmark)
    timing["scoring_seconds"] = time.perf_counter() - scoring_started
    seconds = time.perf_counter() - started

    result_path = write_lm_eval_shaped_result(
        output_root,
        args.benchmark,
        scores,
        seconds,
        args.model,
        args.revision,
        "evalplus_chat" if args.chat_template else "evalplus_plain",
    )

    timing_path = Path(
        os.environ.get("RESIDUALQUANT_TIMING_FILE", output_root / "timing.json")
    )
    atomic(timing_path, timing)

    print(
        json.dumps(
            {
                "benchmark": args.benchmark,
                "scored": scores["total"],
                "pass@1_base": scores["pass@1,base"],
                "pass@1_plus": scores["pass@1,plus"],
                "result": str(result_path),
                "samples": str(samples_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
