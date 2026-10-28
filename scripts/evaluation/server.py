"""Serve one configured checkout through vLLM's OpenAI-compatible API."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluation.server_client import configure_kv_policy, serving_config
from reproduction import digest


def configure_environment(model_key, config, gpu):
    for key in list(os.environ):
        if key.startswith(("OURO_", "HUGINN_", "LOOPQ_")):
            del os.environ[key]

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["VLLM_PLUGINS"] = "register_looped_transformers"
    if config.get("packed_groups"):
        os.environ["VLLM_PLUGINS"] += ",register_residual_quantization"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if model_key in ("ouro", "ouro26"):
        os.environ["OURO_SHARED_KV_CACHE"] = "0"
        os.environ["OURO_FUSED_ROPE_CACHE"] = "0"
    else:
        os.environ["HUGINN_EXPLICIT_LOOPWISE_FORWARD"] = "1"
        os.environ["HUGINN_KV_LAYOUT"] = "depth_specific"


def build_command(model_key, model, config, config_path, settings, port):
    command = [
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        model["name"],
        "--served-model-name",
        model["name"],
        "--revision",
        model["revision"],
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--seed",
        "0",
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        str(settings["gpu_memory_utilization"]),
        "--max-num-seqs",
        str(settings["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(settings["max_num_batched_tokens"]),
        "--compilation-config",
        json.dumps(settings["compilation_config"]),
        "--additional-config",
        json.dumps(serving_config(config, config_path)),
    ]
    command.append(
        "--enable-prefix-caching"
        if settings["enable_prefix_caching"]
        else "--no-enable-prefix-caching"
    )
    if config.get("packed_groups"):
        command.extend(
            [
                "--block-size",
                str(settings["block_size"]),
                "--load-format",
                "residual_quantization",
                "--scheduler-cls",
                "residual_quantization.scheduler.FullPromptScheduler",
            ]
        )
    if settings.get("enable_chunked_prefill") is False:
        command.append("--no-enable-chunked-prefill")
    if settings.get("num_gpu_blocks_override") is not None:
        command.extend(
            ["--num-gpu-blocks-override", str(settings["num_gpu_blocks_override"])]
        )
    if settings.get("enforce_eager"):
        command.append("--enforce-eager")
    if model_key == "huginn":
        command.extend(
            [
                "--hf-overrides",
                json.dumps({"mean_recurrence": model["loops"]}),
            ]
        )
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-key", choices=("ouro", "ouro26", "huginn"), required=True
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settings", required=True, help="JSON vLLM engine settings")
    args = parser.parse_args()

    config = json.loads(args.config.read_text()) if args.config else {}
    if "bf16_variant" in config:
        raise ValueError(
            "Legacy matched BF16 control is unsupported; regenerate the BF16 config"
        )
    settings = json.loads(args.settings)
    configure_kv_policy(settings, config)
    protocol = json.loads(
        (ROOT / "experiments/KV_main_experiments/prompt_protocol.json").read_text()
    )
    for name, expected in protocol["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Prompt protocol source hash mismatch: {name}")
    model = protocol["models"][args.model_key]

    configure_environment(args.model_key, config, args.gpu)
    args.output.mkdir(parents=True, exist_ok=True)

    import torch

    torch.set_num_threads(4)
    torch.backends.cuda.preferred_blas_library(settings.get("blas_library", "default"))
    major, minor = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

    import uvloop
    from vllm.entrypoints.openai.api_server import run_server
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.utils import FlexibleArgumentParser

    command = build_command(
        args.model_key, model, config, args.config, settings, args.port
    )
    server_args = make_arg_parser(FlexibleArgumentParser()).parse_args(command)
    validate_parsed_serve_args(server_args)
    uvloop.run(run_server(server_args))


if __name__ == "__main__":
    main()
