"""Shared experiment paths, source snapshots, and sequential execution."""

import hashlib, importlib.util, json, subprocess
from pathlib import Path


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ouro_data(experiment, model_key="ouro"):
    if model_key not in ("ouro", "ouro26"):
        raise ValueError(f"Unsupported Ouro model: {model_key}")
    data = Path(__file__).resolve().parents[1] / "experiments" / experiment / "data"
    return data if model_key == "ouro" else data / model_key


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save_tensor(path, value):
    import torch

    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def snapshot(root):
    root = Path(root).resolve()
    paths = {
        root / "pyproject.toml",
        root / "uv.lock",
        root / "plugins/looped_transformers/pyproject.toml",
        root / "plugins/residual_quantization/pyproject.toml",
        root / "plugins/residual_quantization/MANIFEST.in",
    }
    source_suffixes = {".py", ".cu", ".cuh", ".cpp", ".h", ".json", ".yaml", ".sh"}
    for folder in ("scripts", "plugins", "configs"):
        paths.update(
            path
            for path in (root / folder).rglob("*")
            if path.is_file() and path.suffix in source_suffixes
        )
    paths.update(
        path
        for path in (root / "experiments").rglob("*")
        if path.is_file()
        and "data" not in path.relative_to(root).parts
        and path.suffix in {".py", ".json", ".tex"}
    )
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def run(path, root, *, validator, retry_failed=False, gpu=0):
    plan = json.loads(path.read_text())
    if plan.get("input_sha256") != snapshot(root):
        raise ValueError("Sources changed; prepare a fresh reproduction plan")
    jobs = plan["jobs"]
    known = {job["id"] for job in jobs}
    assert len(known) == len(jobs)
    assert all(set(job["depends_on"]) <= known for job in jobs)
    for job in jobs:
        if job["status"] == "running":
            raise ValueError("Prior job unfinished; inspect its output before resuming")
        if job["status"] == "failed" and retry_failed:
            job["status"] = "pending"
    while any(job["status"] != "complete" for job in jobs):
        completed = {job["id"] for job in jobs if job["status"] == "complete"}
        ready = [
            job
            for job in jobs
            if job["status"] == "pending" and set(job["depends_on"]) <= completed
        ]
        if not ready:
            raise RuntimeError(
                "Failed or unresolved dependencies; inspect per-job logs"
            )
        job = ready[0]
        command = [argument.replace("{gpu}", str(gpu)) for argument in job["command"]]
        log = path.parent / "logs" / (job["id"].replace(":", "_") + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        print(job["id"], flush=True)
        job.update(status="running", log=str(log))
        atomic(path, plan)
        try:
            with log.open("a") as stream:
                subprocess.run(
                    command,
                    cwd=root,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            if not validator(job):
                raise RuntimeError("Incomplete or invalid result")
        except BaseException:
            job["status"] = "failed"
            atomic(path, plan)
            raise
        job["status"] = "complete"
        atomic(path, plan)
    plan["completed"] = True
    atomic(path, plan)
