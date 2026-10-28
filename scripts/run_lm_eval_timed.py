#!/usr/bin/env python3
"""Run lm-eval against a separate vLLM API server and record client timings."""

import os
import re
import runpy
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen

from reproduction import atomic


def read_engine_work(base_url):
    url = urlsplit(base_url)
    metrics_url = urlunsplit((url.scheme, url.netloc, "/metrics", "", ""))
    with urlopen(metrics_url, timeout=5) as response:
        lines = response.read().decode().splitlines()

    counters = {}
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name = line.split("{", 1)[0].split()[0]
        if name in (
            "vllm:prompt_tokens_total",
            "vllm:generation_tokens_total",
            "vllm:iteration_tokens_total_count",
            "vllm:num_preemptions_total",
        ):
            counters[name] = counters.get(name, 0) + float(line.split()[-1])
        elif name == "vllm:iteration_tokens_total_bucket":
            boundary = re.search(r'\ble="([^"]+)"', line).group(1)
            key = f"scheduler_step_tokens_le_{boundary}"
            counters[key] = counters.get(key, 0) + float(line.split()[-1])

    if "vllm:generation_tokens_total" not in counters:
        raise ValueError("vLLM generation counters are missing")
    return counters


def main():
    from lm_eval.models.openai_completions import LocalCompletionsAPI

    started = time.perf_counter()
    timing = {
        "model_init_seconds": 0.0,
        "generate_until_seconds": 0.0,
        "generated_samples": 0,
        "generation_calls": 0,
        "completed": False,
    }
    original_init = LocalCompletionsAPI.__init__
    original_generate = LocalCompletionsAPI.generate_until

    def save():
        target = Path(os.environ["RESIDUALQUANT_TIMING_FILE"])
        timing["process_seconds"] = time.perf_counter() - started
        atomic(target, timing)

    def measured_init(self, *args, **kwargs):
        tick = time.perf_counter()
        try:
            return original_init(self, *args, **kwargs)
        finally:
            timing["model_init_seconds"] += time.perf_counter() - tick

    def measured_generate(self, *args, **kwargs):
        before = None
        try:
            before = read_engine_work(self.base_url)
        except (OSError, ValueError) as error:
            timing["engine_metrics_error"] = str(error)

        tick = time.perf_counter()
        try:
            result = original_generate(self, *args, **kwargs)
        finally:
            timing["generate_until_seconds"] += time.perf_counter() - tick
            timing["generation_calls"] += 1
        timing["generated_samples"] += len(result)

        if before is not None:
            try:
                after = read_engine_work(self.base_url)
                work = timing.setdefault("engine_work", {})
                for name, value in after.items():
                    work[name] = work.get(name, 0) + value - before.get(name, 0)
                timing["generated_tokens_per_second"] = (
                    work["vllm:generation_tokens_total"]
                    / timing["generate_until_seconds"]
                )
            except (OSError, ValueError) as error:
                timing["engine_metrics_error"] = str(error)
        return result

    LocalCompletionsAPI.__init__ = measured_init
    LocalCompletionsAPI.generate_until = measured_generate
    try:
        runpy.run_module("lm_eval", run_name="__main__")
    except SystemExit as exc:
        timing["completed"] = exc.code in (None, 0)
        raise
    else:
        timing["completed"] = True
    finally:
        LocalCompletionsAPI.__init__ = original_init
        LocalCompletionsAPI.generate_until = original_generate
        save()


if __name__ == "__main__":
    main()
