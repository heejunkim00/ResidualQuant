"""Start one local vLLM server and send OpenAI completion requests to it."""

import json
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from reproduction import digest


def request(url, payload, timeout=86400):
    body = json.dumps(payload).encode()
    req = Request(url, body, {"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as response:
        return json.load(response)


def completions(url, model, prompts, max_tokens, stop, batch_size=None):
    """Keep the evaluator outside the serving process, including tokenization."""
    if batch_size is not None and batch_size < 1:
        raise ValueError("Completion batch size must be positive")
    if not prompts:
        return []
    batch_size = batch_size or len(prompts)
    records = []
    for begin in range(0, len(prompts), batch_size):
        batch = prompts[begin : begin + batch_size]
        result = request(
            url,
            dict(
                model=model,
                prompt=batch,
                max_tokens=max_tokens,
                temperature=0,
                top_p=1,
                stop=stop,
                seed=0,
                return_token_ids=True,
            ),
        )
        choices = sorted(result["choices"], key=lambda item: item["index"])
        if [choice["index"] for choice in choices] != list(range(len(batch))):
            raise RuntimeError(
                "Server returned missing, duplicate or invalid completion indices"
            )
        records.extend(choices)
    return records


def lm_eval_model_args(model, revision, base_url, num_concurrent=32):
    """Use the same remote tokenizer and request policy for both models."""
    return ",".join(
        (
            f"model={model}",
            f"revision={revision}",
            f"base_url={base_url}/completions",
            f"tokenizer={model}",
            "tokenizer_backend=huggingface",
            "tokenized_requests=true",
            "trust_remote_code=true",
            "max_length=4096",
            f"num_concurrent={num_concurrent}",
            "timeout=3600",
        )
    )


def configure_kv_policy(settings, config):
    """Keep original prompt KV until its four-loop group is committed."""
    if config and config["packed_groups"]:
        settings.setdefault("block_size", 32)
        settings.update(
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            max_num_batched_tokens=max(4096, settings["max_num_batched_tokens"]),
        )


def serving_config(config, config_path):
    """Pass only inference settings; resolve artifacts beside their JSON file."""
    if not config.get("packed_groups"):
        return {}
    if config.get("use_rotation"):
        raise ValueError("Use a calibrated config containing rotation_artifact")
    runtime_config = {
        name: config[name]
        for name in (
            "packed_groups",
            "coefficient_method",
            "reference_mode",
            "anchor_index",
            "rotation_artifact",
            "prefill_cpu_offload",
            "prefill_offload_chunk_tokens",
            "prefill_offload_lookahead",
        )
        if name in config
    }
    if "rotation_artifact" in runtime_config:
        artifact = dict(runtime_config["rotation_artifact"])
        path = Path(artifact["artifact"])
        if not path.is_absolute():
            path = Path(config_path).resolve().parent / path
        artifact["artifact"] = str(path.resolve())
        runtime_config["rotation_artifact"] = artifact
    return {"residual_quantization": runtime_config}


@contextmanager
def serving(model_key, config, gpu, output, settings):
    experiment_config = json.loads(Path(config).read_text()) if config else None
    if experiment_config is not None:
        root = Path(__file__).resolve().parents[2]
        for relative, expected in experiment_config.get("source_sha256", {}).items():
            path = root / relative
            actual = digest(path) if path.is_file() else None
            if actual != expected:
                raise ValueError(
                    f"Source mismatch; regenerate serving config: {relative}"
                )
    configure_kv_policy(settings, experiment_config)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).with_name("server.py")),
        "--model-key",
        model_key,
        "--gpu",
        str(gpu),
        "--port",
        str(port),
        "--output",
        str(output),
        "--settings",
        json.dumps(settings),
    ]
    if config is not None:
        command.extend(["--config", str(config)])
    with (output / "server.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        base_url = f"http://127.0.0.1:{port}/v1"
        try:
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"vLLM server exited; inspect {log.name}")
                try:
                    with urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                        break
                except (OSError, URLError):
                    time.sleep(1)
            else:
                raise TimeoutError(
                    f"vLLM server did not become ready; inspect {log.name}"
                )
            yield base_url
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
