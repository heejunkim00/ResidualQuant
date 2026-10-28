"""Collect full-query BF16 QKV traces for layer-wise calibration."""

import argparse
import hashlib
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


from reproduction import atomic, digest, save_tensor

MAX_LENGTH = 1024


def configure_environment(gpu, model_key):
    for key in list(os.environ):
        if key.startswith(("HUGINN_", "OURO_", "LOOPQ_")):
            del os.environ[key]
    os.environ.update(
        CUDA_VISIBLE_DEVICES=gpu,
        VLLM_PLUGINS="register_looped_transformers",
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
        TOKENIZERS_PARALLELISM="false",
    )
    if model_key == "huginn":
        os.environ.update(
            HUGINN_EXPLICIT_LOOPWISE_FORWARD="1", HUGINN_KV_LAYOUT="depth_specific"
        )


def attach(model, model_key, layers, loops):
    import torch
    from vllm.attention.layer import Attention
    from vllm.model_executor.models.utils import extract_layer_index

    config = model.config
    count = config.n_layers_in_recurrent_block
    heads = config.num_attention_heads
    dimension = config.hidden_size // heads
    if config.num_key_value_heads != heads:
        raise ValueError("Calibration currently requires matching Q/K/V head counts")
    selected = (
        list(range(count)) if layers is None else list(map(int, layers.split(",")))
    )
    if (
        not selected
        or len(set(selected)) != len(selected)
        or not set(selected) <= set(range(count))
    ):
        raise ValueError("Invalid physical layer selection")
    if model_key == "huginn":
        start = config.n_layers_in_prelude + config.n_layers_in_coda
        slots = {
            extract_layer_index(m.layer_name): m
            for m in model.modules()
            if isinstance(m, Attention)
        }
    else:
        start = 0
        slots = dict(enumerate(model.model.virtual_attention.slots))
    model._calibration_traces = {}
    model._calibration_handles = []
    for layer in selected:
        for loop in range(loops):
            index = start + loop * count + layer

            def capture(module, positional, keywords, layer=layer, loop=loop):
                if (layer, loop) in model._calibration_traces:
                    raise ValueError("Repeated attention slot in one prefill")
                values = [
                    positional[i] if len(positional) > i else keywords[name]
                    for i, name in enumerate(("query", "key", "value"))
                ]
                if any(x.dtype != torch.bfloat16 for x in values):
                    raise ValueError("BF16 QKV required")
                model._calibration_traces[layer, loop] = tuple(
                    x.detach().reshape(-1, heads, dimension).to("cpu", copy=True)
                    for x in values
                )

            model._calibration_handles.append(
                slots[index].register_forward_pre_hook(capture, with_kwargs=True)
            )
    grams = []
    for layer in model.model.layers[
        config.n_layers_in_prelude : config.n_layers_in_prelude + count
    ]:
        weight = (
            layer.self_attn.o_proj.weight.detach()
            .float()
            .reshape(config.hidden_size, heads, dimension)
            .permute(1, 0, 2)
        )
        grams.append((weight.transpose(1, 2) @ weight).cpu())
    return torch.stack(grams), selected


def collect_samples(llm, tokenizer, samples, output, selected, ordinals, loop_count):
    import fcntl
    import torch
    from vllm import SamplingParams

    sampler = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
    started = time.perf_counter()
    for ordinal in ordinals:
        folder = output / f"sample_{ordinal:03d}"
        folder.mkdir(exist_ok=True)
        with (folder / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            paths = [folder / f"layer_{layer:02d}.pt" for layer in selected]
            if all(p.exists() for p in paths):
                continue
            tokens = tokenizer(
                samples[ordinal]["text"], truncation=True, max_length=MAX_LENGTH
            ).input_ids
            llm.apply_model(lambda model: model._calibration_traces.clear())
            llm.generate([{"prompt_token_ids": tokens}], sampler, use_tqdm=False)

            def take(model):
                captured = model._calibration_traces
                model._calibration_traces = {}
                return captured

            raw = llm.apply_model(take)[0]
            if set(raw) != {
                (layer, loop) for layer in selected for loop in range(loop_count)
            }:
                raise ValueError("Incomplete QKV capture")
            for layer, path in zip(selected, paths):
                if path.exists():
                    continue
                loops = []
                for loop in range(loop_count):
                    q, k, v = raw.pop((layer, loop))
                    if (
                        q.shape != k.shape
                        or k.shape != v.shape
                        or len(q) != len(tokens)
                    ):
                        raise ValueError("Trace token count mismatch")
                    loops.append(
                        dict(q=q, k=k, v=v, query_indices=torch.arange(len(tokens)))
                    )
                save_tensor(
                    path, dict(loops=loops, document_index=samples[ordinal]["id"])
                )
            del raw
            print(
                json.dumps(
                    dict(
                        ordinal=ordinal,
                        layers=selected,
                        tokens=len(tokens),
                        elapsed_seconds=time.perf_counter() - started,
                    )
                ),
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-key", choices=("ouro", "ouro26", "huginn"), required=True
    )
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--dataset", choices=("gsm8k", "mbpp"), default="gsm8k")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", help="Comma-separated physical layer indices")
    parser.add_argument("--ordinals", help="Comma-separated sample indices in 0..255")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("Invalid GPU memory fraction")
    ordinals = (
        list(range(256))
        if args.ordinals is None
        else list(map(int, args.ordinals.split(",")))
    )
    if (
        not ordinals
        or len(set(ordinals)) != len(ordinals)
        or not set(ordinals) <= set(range(256))
    ):
        parser.error("Invalid trace ordinals")
    configure_environment(args.gpu, args.model_key)
    import fcntl
    import torch
    from vllm import LLM
    from calibration.data import build_dataset

    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    protocol = json.loads(
        (ROOT / "experiments/KV_main_experiments/prompt_protocol.json").read_text()
    )
    spec = protocol["models"][args.model_key]
    dataset, tokenizer = build_dataset(args.model_key, args.dataset)
    samples = dataset.pop("samples")
    manifest = dict(
        dataset,
        model=spec["name"],
        revision=spec["revision"],
        loops=spec["loops"],
        precision="bf16",
        max_length=MAX_LENGTH,
        query_count=MAX_LENGTH,
        collection_backend="vllm",
        prompts_sha256=hashlib.sha256(
            json.dumps(samples, ensure_ascii=False).encode()
        ).hexdigest(),
        source_sha256={
            str(p.relative_to(ROOT)): digest(p)
            for p in [
                Path(__file__),
                ROOT / "scripts/calibration/data.py",
                ROOT
                / "plugins/looped_transformers/src/looped_transformers"
                / ("huginn.py" if args.model_key == "huginn" else "ouro.py"),
                ROOT
                / "plugins/looped_transformers/src/looped_transformers/model_setup.py",
            ]
        },
    )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = args.output / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("Trace contract changed; use a fresh output directory")
        atomic(path, manifest)
    key = "mean_recurrence" if args.model_key == "huginn" else "total_ut_steps"
    llm = LLM(
        model=spec["name"],
        revision=spec["revision"],
        trust_remote_code=True,
        dtype="bfloat16",
        hf_overrides={key: spec["loops"]},
        tensor_parallel_size=1,
        max_model_len=MAX_LENGTH + 1,
        max_num_seqs=1,
        max_num_batched_tokens=MAX_LENGTH + 1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        enforce_eager=True,
        seed=0,
    )

    [(grams, selected)] = llm.apply_model(
        partial(
            attach, model_key=args.model_key, layers=args.layers, loops=spec["loops"]
        )
    )
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = args.output / "wo_grams.pt"
        if path.exists():
            if not torch.equal(torch.load(path, weights_only=True), grams):
                raise ValueError("Output-projection Gram matrices changed")
        else:
            save_tensor(path, grams)
    collect_samples(
        llm, tokenizer, samples, args.output, selected, ordinals, spec["loops"]
    )


if __name__ == "__main__":
    main()
