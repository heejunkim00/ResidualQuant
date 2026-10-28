"""Calibrate depth-specific Direct rotations for Huginn."""

import argparse
import json
import os
import resource
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=("gsm8k", "mbpp"), default="gsm8k")

    args = parser.parse_args()
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("GPU memory utilization must be between 0 and 1")
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "config.json").exists():
        raise FileExistsError("Calibration output is complete; choose a fresh output")
    for key in list(os.environ):
        if key.startswith(("HUGINN_", "OURO_", "LOOPQ_")):
            del os.environ[key]
    os.environ.update(
        CUDA_VISIBLE_DEVICES=args.gpu,
        VLLM_PLUGINS="register_looped_transformers",
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
        HUGINN_EXPLICIT_LOOPWISE_FORWARD="1",
        HUGINN_KV_LAYOUT="depth_specific",
        TOKENIZERS_PARALLELISM="false",
    )
    import torch
    from vllm import LLM, SamplingParams
    from vllm.attention.layer import Attention
    from vllm.model_executor.models.utils import extract_layer_index
    from calibration.rotation_training import (
        FP32Chain,
        fit_direct,
    )
    from calibration.rotation_math import hadamard, probabilities, output_error
    from transformers import AutoConfig
    from calibration.data import build_dataset
    from reproduction import atomic, digest, save_tensor

    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    protocol = json.loads(
        (ROOT / "experiments/KV_main_experiments/prompt_protocol.json").read_text()
    )
    spec = protocol["models"]["huginn"]
    model_config = AutoConfig.from_pretrained(
        spec["name"], revision=spec["revision"], trust_remote_code=True
    )
    layer_count = model_config.n_layers_in_recurrent_block
    head_count = model_config.n_heads
    hidden_size = model_config.n_embd
    head_dim = hidden_size // head_count
    loop_count = spec["loops"]
    block_count = loop_count // 4
    slot_count = loop_count * layer_count
    if loop_count % 4 or model_config.num_key_value_heads != head_count:
        raise ValueError(
            "Direct calibration requires four-loop blocks and matching Q/K/V heads"
        )
    dataset, tokenizer = build_dataset("huginn", args.dataset)
    assert (
        dataset["model"] == "huginn"
        and dataset["domain"] in ("gsm8k", "code")
        and (dataset["seed"] == 0)
    )
    assert (
        len(dataset["samples"]) == 256
        and len(dataset["partitions"]["fit"])
        == len(dataset["partitions"]["holdout"])
        == 128
    )
    assert [s["id"] for s in dataset["samples"]] == dataset["partitions"][
        "fit"
    ] + dataset["partitions"]["holdout"]
    assert config.get("use_rotation", False)
    schedules = config["packed_groups"]
    if not schedules or any((not records for records in schedules)):
        raise ValueError("Every calibration group must contain records")
    if sum(map(len, schedules)) != loop_count:
        raise ValueError("Calibration groups must describe every model recurrence")
    if config["reference_mode"] != "anchor":
        raise ValueError("Rotation calibration requires Anchor-relative groups")
    independent = all((len(group) == 1 for group in schedules))
    if not independent:
        raise ValueError("Use calibrate.py for shared anchor rotations")
    if config["anchor_index"] != 0:
        raise ValueError("A single-record group requires anchor index 0")
    all_records = [record for group in schedules for record in group]
    training_records = all_records[:4]
    if any(
        all_records[start : start + 4] != training_records
        for start in range(0, loop_count, 4)
    ):
        raise ValueError(
            "Direct Huginn calibration requires a repeated four-loop precision schedule"
        )
    bits = [bit for bit, _ in training_records]
    groups = [group for _, group in training_records]
    assert (config.get("coefficient_method"), config.get("use_rotation", False)) == (
        "ls",
        True,
    )
    mode = "anchor"
    fit_count = 128
    hold_count = 128
    steps = 80
    chain = FP32Chain(
        bits,
        mode,
        groups,
        partial(probabilities, head_dim=head_dim),
        partial(output_error, hidden=hidden_size),
        independent=independent,
    )
    initial = (
        hadamard("cuda", dimension=head_dim)
        .expand(1, head_count, head_dim, head_dim)
        .clone()
    )
    units = [
        (block, physical)
        for block in range(block_count)
        for physical in range(layer_count)
    ]
    started = time.perf_counter()
    llm = LLM(
        model=spec["name"],
        revision=spec["revision"],
        trust_remote_code=True,
        dtype="bfloat16",
        hf_overrides={"mean_recurrence": loop_count},
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
        recurrent_layer_count = model.config.n_layers_in_recurrent_block
        recurrent_slots_start = (
            model.config.n_layers_in_prelude + model.config.n_layers_in_coda
        )
        slot_indices = {}
        for block, physical in units:
            indices = []
            for depth in range(4):
                recurrence = block * 4 + depth
                if (
                    not 0 <= recurrence < model.config.mean_recurrence
                    or not 0 <= physical < recurrent_layer_count
                ):
                    raise IndexError("Invalid recurrence/layer")
                index = (
                    recurrent_slots_start
                    + recurrence * recurrent_layer_count
                    + physical
                )
                indices.append(index)
            slot_indices[block, physical] = tuple(indices)
        slots = {
            extract_layer_index(m.layer_name): m
            for m in model.modules()
            if isinstance(m, Attention)
        }
        assert (
            len(slots)
            == model.config.n_layers_in_prelude
            + model.config.n_layers_in_coda
            + model.config.mean_recurrence * model.config.n_layers_in_recurrent_block
        )
        model._online_captures = {}
        model._online_handles = []
        for block, physical in units:
            for depth in range(4):
                index = slot_indices[block, physical][depth]

                def capture(module, args, kwargs, index=index):
                    assert index not in model._online_captures
                    vals = [
                        args[i] if len(args) > i else kwargs[name]
                        for i, name in enumerate(("query", "key", "value"))
                    ]
                    assert all((x.dtype == torch.bfloat16 for x in vals))
                    model._online_captures[index] = tuple(
                        (x.detach().reshape(-1, head_count, head_dim) for x in vals)
                    )

                model._online_handles.append(
                    slots[index].register_forward_pre_hook(capture, with_kwargs=True)
                )
        grams = []
        for physical in range(layer_count):
            w = (
                model.model.layers[model.config.n_layers_in_prelude + physical]
                .self_attn.o_proj.weight.detach()
                .float()
            )
            w = w.reshape(hidden_size, head_count, head_dim).permute(1, 0, 2)
            grams.append(w.transpose(1, 2) @ w)
        return (len(model._online_handles), torch.stack(grams), slot_indices)

    [(handle_count, grams, slot_indices)] = llm.apply_model(attach)
    assert handle_count == slot_count and grams.shape == (
        layer_count,
        head_count,
        head_dim,
        head_dim,
    )
    sampler = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
    generation_count = 0
    generation_seconds = 0.0

    def sample(ordinal):
        nonlocal generation_count, generation_seconds
        ids = (
            tokenizer(
                dataset["samples"][ordinal]["text"],
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )
            .input_ids[0]
            .tolist()
        )
        assert 1 <= len(ids) <= 1024
        llm.apply_model(lambda model: model._online_captures.clear())
        tick = time.perf_counter()
        llm.generate([{"prompt_token_ids": ids}], sampler, use_tqdm=False)
        torch.cuda.synchronize()
        generation_seconds += time.perf_counter() - tick
        generation_count += 1

        def take(model):
            assert len(model._online_captures) == slot_count
            out = model._online_captures.copy()
            model._online_captures.clear()
            return out

        raw = llm.apply_model(take)[0]
        query_indices = torch.arange(len(ids), device="cuda")
        result = []
        for block, physical in units:
            loops = []
            for depth in range(4):
                index = slot_indices[block, physical][depth]
                q, k, v = raw.pop(index)
                assert q.shape == k.shape == v.shape == (len(ids), head_count, head_dim)
                loops.append(
                    {
                        "q": q.clone(),
                        "k": k.clone(),
                        "v": v.clone(),
                        "query_indices": query_indices,
                    }
                )
                del q, k, v
            result.append(loops)
        assert not raw
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
                    "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
                    "peak_allocated_gib": round(
                        torch.cuda.max_memory_allocated() / 2**30, 3
                    ),
                    "max_rss_gib": round(
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3
                    ),
                }
            ),
            flush=True,
        )
        return result

    means, chosen_k, chosen_v, k_metrics, v_metrics = fit_direct(
        sample, chain, initial, grams, len(units), head_count, stage
    )
    record_rotations = (
        torch.stack(
            [
                torch.stack(
                    [chosen[unit][chain.mapping] for chosen in (chosen_k, chosen_v)],
                    dim=1,
                )
                for unit, (_, physical) in enumerate(units)
            ]
        )
        .half()
        .cpu()
    )
    key_means = (
        torch.stack(
            [
                torch.einsum(
                    "lhd,lhde->lhe",
                    means[unit].cpu(),
                    record_rotations[unit, :, 0].float(),
                ).contiguous()
                for unit in range(len(units))
            ]
        )
        .reshape(block_count, layer_count, 4, head_count, head_dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(loop_count, layer_count, head_count, head_dim)
        .contiguous()
    )
    rotations = record_rotations.reshape(
        block_count, layer_count, 4, 2, head_count, head_dim, head_dim
    )
    rotations = rotations.permute(0, 2, 1, 3, 4, 5, 6).reshape(
        loop_count, layer_count, 2, head_count, head_dim, head_dim
    )
    key_rotations, value_rotations = rotations.unbind(2)
    key_rotations = key_rotations.contiguous()
    value_rotations = value_rotations.contiguous()
    contract = {
        "method": "gpu_online_streaming_mean",
        "model": spec["name"],
        "revision": spec["revision"],
        "physical_layers": layer_count,
        "heads": head_count,
        "head_dim": head_dim,
        "dataset": args.dataset,
        "anchor_index": config["anchor_index"],
        "reference_mode": mode,
        "packed_groups": schedules,
        "group_schedule": list(chain.groups),
        "prompt_source_sha256": dataset["prompt_source_sha256"],
        "fit": dataset["partitions"]["fit"][:fit_count],
        "holdout": dataset["partitions"]["holdout"][:hold_count],
        "steps": steps,
        "seed": 0,
        "trace_precision": "bf16",
        "optimizer_precision": "FP32",
        "mean_reduction": "per-sample FP32 sums divided by total token count",
        "source_sha256": digest(Path(__file__)),
        "config_sha256": digest(args.config),
        "calibration_protocol": "original_fp32_objective_fixed_hadamard_means_v1",
    }
    artifact = {
        "model": "huginn",
        "format_version": 2,
        "metrics": {"k": k_metrics, "v": v_metrics},
        "key_rotations": key_rotations,
        "value_rotations": value_rotations,
        "key_means": key_means,
        "has_mean": True,
        "schedule": [bit for bit, _ in all_records],
        "groups": [group for _, group in all_records],
        "mode": mode,
        "coefficient_method": config["coefficient_method"],
        "use_rotation": True,
        "calibration_contract": contract,
    }
    path = args.output / "rotations.pt"
    save_tensor(path, artifact)
    ready = dict(
        config,
        calibration_domain=dataset["domain"],
        rotation_artifact={
            "artifact": str(path.resolve()),
            "sha256": digest(path),
        },
    )
    ready.pop("use_rotation", None)
    atomic(args.output / "config.json", ready)
    timings.update(
        generation_count=generation_count,
        generation_seconds=generation_seconds,
        sum_stage_seconds=sum(timings.values()),
    )
    atomic(args.output / "timings.json", timings)
    print(
        json.dumps(
            {
                "complete": True,
                "physical_layers": layer_count,
                "artifact": str(path),
                "config": str(args.output / "config.json"),
                "seconds": timings["sum_stage_seconds"],
                "generation_count": generation_count,
                "sha256": ready["rotation_artifact"]["sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
