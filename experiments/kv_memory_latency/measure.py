"""Ouro fixed-batch decode latency; source scripts/environment.sh before running."""

import argparse, hashlib, json, os, time, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluation.server_client import serving_config
from reproduction import atomic, snapshot
from capacity import model_geometry


def measure_trial(engine, scheduler, params, prompts, context, trial, a):
    import torch

    if a.staged_prefill:
        assert not scheduler.running and not scheduler.waiting
        ids = [f"{context}-{trial}-{i}" for i in range(a.batch)]
        parked = []
        for i, rid in enumerate(ids):
            engine.add_request(
                request_id=rid,
                prompt={"prompt_token_ids": prompts[i][:context]},
                params=params,
            )
            while True:
                output = engine.step()
                assert not any(r.finished for r in output)
                if not output:
                    continue
                assert len(output) == 1 and output[0].request_id == rid
                assert len(output[0].outputs[0].token_ids) == 1
                assert len(scheduler.running) == 1
                request = scheduler.running.pop()
                assert (
                    request.request_id == rid and request.num_computed_tokens == context
                )
                parked.append(request)
                break
        scheduler.running.extend(parked)
        print(
            dict(event="prefill_barrier_ready", trial=trial, batch=a.batch),
            flush=True,
        )
        torch.cuda.synchronize()
        start = time.perf_counter()
        decode_calls = 0
        for count in range(2, a.tokens + 1):
            assert all(r.num_computed_tokens == context + count - 2 for r in parked)
            output = engine.step()
            decode_calls += 1
            assert len(output) == a.batch and {r.request_id for r in output} == set(ids)
            assert all(
                len(r.outputs) == 1 and len(r.outputs[0].token_ids) == count
                for r in output
            )
            assert all(r.finished == (count == a.tokens) for r in output)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        assert not engine.has_unfinished_requests() and not scheduler.running
    elif a.batch == 1:
        engine.add_request(
            request_id=f"{context}-{trial}",
            prompt={"prompt_token_ids": prompts[0][:context]},
            params=params,
        )
        last = 0
        start = None
        decode_calls = 0
        while engine.has_unfinished_requests():
            output = engine.step()
            if start is not None:
                decode_calls += 1
            if not output:
                assert start is None
                continue
            assert len(output) == 1 and len(output[0].outputs) == 1
            count = len(output[0].outputs[0].token_ids)
            assert count == last + 1, (last, count)
            last = count
            if count == 1:
                torch.cuda.synchronize()
                start = time.perf_counter()
            if output[0].finished:
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                assert count == a.tokens and decode_calls == a.tokens - 1, (
                    count,
                    decode_calls,
                )
    else:
        ids = [f"{context}-{trial}-{i}" for i in range(a.batch)]
        for i, rid in enumerate(ids):
            engine.add_request(
                request_id=rid,
                prompt={"prompt_token_ids": prompts[i][:context]},
                params=params,
            )
        last = 0
        start = None
        decode_calls = 0
        while engine.has_unfinished_requests():
            output = engine.step()
            if start is not None:
                decode_calls += 1
            if not output:
                assert start is None
                continue
            assert len(output) == a.batch and {r.request_id for r in output} == set(
                ids
            ), "Active batch changed; do not report latency"
            counts = [len(r.outputs[0].token_ids) for r in output]
            assert (
                all(len(r.outputs) == 1 for r in output)
                and counts == [last + 1] * a.batch
            ), "Requests entered decode at different steps"
            last = counts[0]
            if last == 1:
                torch.cuda.synchronize()
                start = time.perf_counter()
            if any(r.finished for r in output):
                assert all(r.finished for r in output)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                assert last == a.tokens and decode_calls == a.tokens - 1
    return elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--contexts", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    p.add_argument("--page-tokens", type=int, choices=(16, 32), default=32)
    p.add_argument("--batch", type=int, choices=range(1, 33), default=1)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument(
        "--capacity-tokens",
        type=int,
        help="Reserve KV budget for this output length during a short feasibility run",
    )
    p.add_argument(
        "--staged-prefill",
        action="store_true",
        help="Prepare each request to its first output before a synchronized fixed-batch decode",
    )
    p.add_argument(
        "--probe-only",
        action="store_true",
        help="Initialize and report KV capacity without generating",
    )
    a = p.parse_args()
    if a.warmups < 0:
        p.error("warmups must be nonnegative")
    capacity_tokens = a.tokens if a.capacity_tokens is None else a.capacity_tokens
    if not a.tokens <= capacity_tokens <= 128:
        p.error("capacity-tokens must be between tokens and 128")
    if not 2 <= a.tokens <= 128 or a.trials < 1:
        p.error("tokens=2..128 and trials>=1 required")
    if any(c <= 0 or c > 8192 for c in a.contexts):
        p.error("contexts must be in 1..8192")
    if a.batch > 1 and len(a.contexts) != 1:
        p.error(
            "Use one context per B4/B16 process to profile its actual prefill memory"
        )
    max_batched = (
        4096 if a.batch == 1 or a.staged_prefill else max(8320, a.batch * a.contexts[0])
    )
    a.output.mkdir(parents=True, exist_ok=True)

    protocol = json.loads(
        (ROOT / "experiments/KV_main_experiments/prompt_protocol.json").read_text()
    )
    model = protocol["models"]["ouro"]
    geometry = model_geometry()
    cfg = json.loads(a.config.read_text())
    chunked_prefill = not cfg["packed_groups"] and (a.batch == 1 or a.staged_prefill)
    if not chunked_prefill:
        max_batched = max(max_batched, 8320)
    if "bf16_variant" in cfg:
        raise ValueError(
            "Legacy matched BF16 control is unsupported; regenerate the BF16 config"
        )
    os.environ.update(
        CUDA_VISIBLE_DEVICES=a.gpu,
        VLLM_PLUGINS="register_looped_transformers,register_residual_quantization",
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
    )

    import torch
    from vllm import LLM, SamplingParams

    contract = dict(
        gpu=torch.cuda.get_device_name(),
        physical_gpu=a.gpu,
        batch=a.batch,
        contexts=a.contexts,
        output_tokens=a.tokens,
        page_tokens=a.page_tokens,
        decode_steps=a.tokens - 1,
        warmups=a.warmups,
        capacity_output_tokens=capacity_tokens,
        trials=a.trials,
        seed=0,
        blas="default",
        prefix_caching=False,
        graph="FULL_DECODE_ONLY",
        max_model_len=8320,
        max_num_batched_tokens=max_batched,
        prefill_policy=(
            ("staged_chunked_barrier" if a.staged_prefill else "chunked_single_request")
            if chunked_prefill
            else ("staged_full_prompt_barrier" if a.staged_prefill else "full_prompt")
        ),
        scope="wall time from first output completion to last output completion; prefill and first token excluded; no per-step synchronization",
        model=model["name"],
        revision=model["revision"],
        geometry=geometry,
        source_sha256=snapshot(ROOT),
        config=cfg,
        torch=torch.__version__,
    )
    rows = []

    def write():
        atomic(a.output / "result.json", dict(contract=contract, rows=rows))

    try:
        llm = LLM(
            model=model["name"],
            revision=model["revision"],
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=1,
            block_size=a.page_tokens,
            max_model_len=8320,
            max_num_seqs=a.batch,
            max_num_batched_tokens=max_batched,
            enable_chunked_prefill=chunked_prefill,
            gpu_memory_utilization=0.90,
            enable_prefix_caching=False,
            seed=0,
            additional_config=serving_config(cfg, a.config),
            load_format="residual_quantization" if cfg["packed_groups"] else "auto",
            compilation_config={
                "level": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [a.batch],
            },
        )
        params = SamplingParams(
            temperature=0,
            min_tokens=a.tokens,
            max_tokens=a.tokens,
            ignore_eos=True,
            stop=[],
        )
        engine = llm.llm_engine
        scheduler = engine.engine_core.engine_core.scheduler
        manager = scheduler.kv_cache_manager
        specs = manager.kv_cache_config.kv_cache_groups
        if a.staged_prefill:
            original_free = manager.free

            def free_finished_only(request):
                assert request.is_finished(), (
                    "KV eviction/preemption invalidates the fixed-batch experiment"
                )
                return original_free(request)

            manager.free = free_finished_only
        contract["kv_capacity"] = dict(
            free_blocks=manager.block_pool.get_num_free_blocks(),
            block_sizes=[g.kv_cache_spec.block_size for g in specs],
            gpu_memory_utilization=0.90,
        )
        if a.probe_only:
            contract["probe_only"] = True
            write()
            raise SystemExit(0)
        import random, statistics

        rng = random.Random(0)
        tokens = [rng.randrange(100, 10000) for _ in range(8192)]
        contract["prompt_sha256"] = hashlib.sha256(
            json.dumps(tokens).encode()
        ).hexdigest()
        prompts = [tokens] + [
            [rng.randrange(100, 10000) for _ in range(8192)] for _ in range(a.batch - 1)
        ]
        contract["batch_prompt_sha256"] = hashlib.sha256(
            json.dumps(prompts).encode()
        ).hexdigest()
        for context in contract["contexts"]:
            samples = []
            if a.batch > 1:
                manager = engine.engine_core.engine_core.scheduler.kv_cache_manager
                specs = manager.kv_cache_config.kv_cache_groups
                blocks_needed = a.batch * sum(
                    (context + capacity_tokens + g.kv_cache_spec.block_size - 1)
                    // g.kv_cache_spec.block_size
                    for g in specs
                )
                free = manager.block_pool.get_num_free_blocks()
                if blocks_needed > free:
                    rows.append(
                        dict(
                            context=context,
                            status="oom",
                            stage="kv_capacity",
                            required_blocks=blocks_needed,
                            free_blocks=free,
                            active_batch=a.batch,
                        )
                    )
                    write()
                    continue
            try:
                for trial in range(a.trials + a.warmups):
                    print(
                        dict(
                            event="trial_start",
                            trial=trial,
                            warmup=trial < a.warmups,
                            batch=a.batch,
                            context=context,
                        ),
                        flush=True,
                    )
                    elapsed = measure_trial(
                        engine, scheduler, params, prompts, context, trial, a
                    )
                    print(
                        dict(
                            event="trial_done",
                            trial=trial,
                            ms_per_token=elapsed * 1000 / (a.tokens - 1),
                        ),
                        flush=True,
                    )
                    if trial >= a.warmups:
                        samples.append(elapsed * 1000 / (a.tokens - 1))
                rows.append(
                    dict(
                        context=context,
                        status="ok",
                        ms_per_token=statistics.median(samples),
                        min_ms=min(samples),
                        max_ms=max(samples),
                        runs_ms=samples,
                        decode_steps=a.tokens - 1,
                        active_batch=a.batch,
                        output_tokens_per_second=a.batch
                        * 1000
                        / statistics.median(samples),
                    )
                )
                print(rows[-1], flush=True)
                write()
            except torch.OutOfMemoryError as e:
                rows.append(dict(context=context, status="oom", error=str(e)))
                write()
                break
    except Exception as e:
        message = str(e).lower()
        if isinstance(e, torch.OutOfMemoryError) or any(
            t in message
            for t in (
                "cuda out of memory",
                "no available memory for the cache blocks",
                "larger than the available kv cache memory",
                "maximum number of tokens that can be stored in kv cache",
            )
        ):
            for context in a.contexts:
                if not any(r["context"] == context for r in rows):
                    rows.append(
                        dict(
                            context=context,
                            status="oom",
                            stage="initialization",
                            error=str(e),
                        )
                    )
            write()
            raise SystemExit(0)
        atomic(a.output / "error.json", dict(type=type(e).__name__, message=str(e)))
        raise
    write()


if __name__ == "__main__":
    main()
