"""Calibrate depth-specific Direct rotations for Ouro."""

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--model-key", choices=("ouro", "ouro26"), default="ouro")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument("--dataset", choices=("gsm8k", "mbpp"), default="gsm8k")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    schedules = config["packed_groups"]
    if not schedules or any((not records for records in schedules)):
        raise ValueError("Every calibration group must contain records")
    if sum(map(len, schedules)) != 4:
        raise ValueError("Calibration groups must describe all 4 model recurrences")
    if config["reference_mode"] != "anchor":
        raise ValueError("Rotation calibration requires Anchor-relative groups")
    if any(len(group) != 1 for group in schedules):
        raise ValueError("Use calibrate.py for shared anchor rotations")
    if config["anchor_index"] != 0:
        raise ValueError("A single-record group requires anchor index 0")
    all_records = [record for group in schedules for record in group]
    bits = [bit for bit, _ in all_records]
    groups = [group for _, group in all_records]
    assert (config.get("coefficient_method"), config.get("use_rotation", False)) == (
        "ls",
        True,
    )
    mode = "anchor"
    method = "ls"
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "config.json").exists():
        raise FileExistsError("Calibration output is complete; choose a fresh output")
    os.environ.update(
        CUDA_VISIBLE_DEVICES=args.gpu,
        VLLM_PLUGINS="register_looped_transformers",
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
        TOKENIZERS_PARALLELISM="false",
    )
    import torch
    from vllm import LLM, SamplingParams
    from calibration.data import build_dataset
    from reproduction import atomic, digest, save_tensor
    from functools import partial
    from transformers import AutoConfig
    from calibration.rotation_training import (
        FP32Chain,
        fit_direct,
    )
    from calibration.rotation_math import hadamard, probabilities, output_error

    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    protocol = json.loads(
        (ROOT / "experiments/KV_main_experiments/prompt_protocol.json").read_text()
    )
    spec = protocol["models"][args.model_key]
    model_config = AutoConfig.from_pretrained(
        spec["name"], revision=spec["revision"], trust_remote_code=True
    )
    layer_count = model_config.num_hidden_layers
    head_count = model_config.num_attention_heads
    hidden_size = model_config.hidden_size
    head_dim = hidden_size // head_count
    loop_count = spec["loops"]
    slot_count = layer_count * loop_count
    if model_config.num_key_value_heads != head_count:
        raise ValueError("Direct calibration requires matching Q/K/V head counts")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("GPU memory utilization must be between 0 and 1")
    chain = FP32Chain(
        bits,
        mode,
        groups,
        probability_fn=partial(probabilities, head_dim=head_dim),
        output_error_fn=partial(output_error, hidden=hidden_size),
        independent=True,
    )
    initial = (
        hadamard("cuda", dimension=head_dim)
        .expand(1, head_count, head_dim, head_dim)
        .clone()
    )
    fit_count = 128
    hold_count = 128
    steps = 80
    dataset, tokenizer = build_dataset(args.model_key, args.dataset)
    ids = dataset["partitions"]["fit"] + dataset["partitions"]["holdout"]

    started = time.perf_counter()
    llm = LLM(
        model=spec["name"],
        revision=spec["revision"],
        trust_remote_code=True,
        dtype="bfloat16",
        hf_overrides={"total_ut_steps": loop_count},
        tensor_parallel_size=1,
        max_model_len=1025,
        max_num_seqs=1,
        max_num_batched_tokens=1025,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        enforce_eager=True,
        seed=0,
    )
    init_seconds = time.perf_counter() - started

    def attach(model):
        slots = model.model.virtual_attention.slots
        assert len(slots) == slot_count
        model._online_captures = {}
        model._online_handles = []
        for loop in range(loop_count):
            for layer in range(layer_count):
                index = loop * layer_count + layer

                def capture(module, args, kwargs, index=index):
                    assert index not in model._online_captures
                    values = (
                        args[:3]
                        if len(args) >= 3
                        else tuple((kwargs[k] for k in ("query", "key", "value")))
                    )
                    model._online_captures[index] = tuple(
                        (
                            x.detach().reshape(x.shape[0], head_count, head_dim)
                            for x in values
                        )
                    )

                model._online_handles.append(
                    slots[index].register_forward_pre_hook(capture, with_kwargs=True)
                )
        out = []
        for layer in model.model.layers:
            w = (
                layer.self_attn.o_proj.weight.detach()
                .float()
                .reshape(hidden_size, head_count, head_dim)
                .permute(1, 0, 2)
            )
            out.append(w.transpose(1, 2) @ w)
        return (len(model._online_handles), torch.stack(out))

    [(handle_count, grams)] = llm.apply_model(attach)
    assert handle_count == slot_count and grams.shape == (
        layer_count,
        head_count,
        head_dim,
        head_dim,
    )
    generation_count = 0
    generation_seconds = 0.0

    def sample(ordinal):
        nonlocal generation_count, generation_seconds
        prompt = dataset["samples"][ordinal]["text"]
        tokens = (
            tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            .input_ids[0]
            .tolist()
        )
        llm.apply_model(lambda model: model._online_captures.clear())
        tick = time.perf_counter()
        llm.generate(
            [{"prompt_token_ids": tokens}],
            SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
            use_tqdm=False,
        )
        torch.cuda.synchronize()
        generation_seconds += time.perf_counter() - tick
        generation_count += 1

        def take(model):
            assert len(model._online_captures) == slot_count
            result = []
            for layer in range(layer_count):
                loops = []
                for loop in range(loop_count):
                    q, k, v = model._online_captures[loop * layer_count + layer]
                    assert (
                        q.shape
                        == k.shape
                        == v.shape
                        == (len(tokens), head_count, head_dim)
                    )
                    loops.append(
                        {
                            "q": q,
                            "k": k,
                            "v": v,
                            "query_indices": torch.arange(len(tokens), device=k.device),
                        }
                    )
                result.append(loops)
            model._online_captures.clear()
            return result

        raw = llm.apply_model(take)[0]
        assert not torch.is_inference_mode_enabled()
        result = [
            [{name: tensor.clone() for name, tensor in loop.items()} for loop in layer]
            for layer in raw
        ]
        return result

    timings = {"model_init": init_seconds}

    def stage(name, fn):
        tick = time.perf_counter()
        before = generation_count
        result = fn()
        torch.cuda.synchronize()
        timings[name] = time.perf_counter() - tick
        print(
            json.dumps(
                {
                    "stage": name,
                    "seconds": timings[name],
                    "generated": generation_count - before,
                    "total_generated": generation_count,
                    "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    / 1024**2,
                }
            ),
            flush=True,
        )
        return result

    means, chosen_k, chosen_v, k_metrics, v_metrics = fit_direct(
        sample, chain, initial, grams, layer_count, head_count, stage
    )
    parts = []
    for layer in range(layer_count):
        rotations = (
            torch.stack(
                [r[chain.mapping] for r in (chosen_k[layer], chosen_v[layer])], dim=1
            )
            .half()
            .cpu()
        )
        key_means = torch.einsum(
            "lhd,lhde->lhe", means[layer].cpu(), rotations[:, 0].float()
        ).contiguous()
        part = {
            "key_rotations": rotations[:, 0].contiguous(),
            "value_rotations": rotations[:, 1].contiguous(),
            "key_means": key_means,
        }
        parts.append(part)
    timings["generation_count"] = generation_count
    timings["generation_seconds"] = generation_seconds
    timings["sum_stage_seconds"] = sum(
        (
            v
            for k, v in timings.items()
            if k not in ("generation_count", "generation_seconds", "sum_stage_seconds")
        )
    )
    atomic(args.output / "timings.json", timings)
    contract = {
        "method": "single_gpu_online_replay_streaming_mean",
        "dataset": args.dataset,
        "model": spec["name"],
        "model_key": args.model_key,
        "revision": spec["revision"],
        "physical_layers": layer_count,
        "heads": head_count,
        "head_dim": head_dim,
        "prompt_source_sha256": dataset["prompt_source_sha256"],
        "reference_mode": mode,
        "anchor_index": config["anchor_index"],
        "packed_groups": schedules,
        "group_schedule": list(chain.groups),
        "seed": 0,
        "fit": ids[:fit_count],
        "holdout": ids[128 : 128 + hold_count],
        "steps": steps,
        "trace_precision": "bf16",
        "optimizer_precision": "FP32",
        "mean_reduction": "per-sample FP32 sums divided by total token count",
        "source_sha256": digest(Path(__file__)),
        "config_sha256": digest(args.config),
    }
    artifact = {
        "model": "ouro",
        "format_version": 2,
        "metrics": {"k": k_metrics, "v": v_metrics},
        "key_rotations": torch.stack([p["key_rotations"] for p in parts])
        .transpose(0, 1)
        .contiguous(),
        "value_rotations": torch.stack([p["value_rotations"] for p in parts])
        .transpose(0, 1)
        .contiguous(),
        "key_means": torch.stack([p["key_means"] for p in parts])
        .transpose(0, 1)
        .contiguous(),
        "has_mean": True,
        "schedule": [bit for bit, _ in all_records],
        "groups": [group for _, group in all_records],
        "mode": mode,
        "coefficient_method": method,
        "use_rotation": True,
        "coordinate_system": "grouped_rotated",
        "calibration_contract": contract,
    }
    artifact_path = args.output / "rotations.pt"
    save_tensor(artifact_path, artifact)
    ready = dict(
        config, calibration_domain="code" if args.dataset == "mbpp" else "gsm8k"
    )
    ready.pop("use_rotation", None)
    ready["rotation_artifact"] = {
        "artifact": str(artifact_path.resolve()),
        "sha256": digest(artifact_path),
    }
    atomic(args.output / "config.json", ready)
    print(json.dumps(timings), flush=True)


if __name__ == "__main__":
    main()
