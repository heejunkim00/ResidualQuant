"""Generate the uniform comparison matrix; single-record groups contain only an anchor."""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from reproduction import atomic, digest

METHODS = (
    ("direct", "unit", "anchor", False, True),
    ("residual", "unit", "anchor", False, False),
    ("norm", "norm", "anchor", False, False),
    ("ls", "ls", "anchor", False, False),
    ("optr", "ls", "anchor", True, True),
    ("ls_optr", "ls", "anchor", True, False),
)


def source_hashes():
    directories = [
        "plugins/residual_quantization/src/residual_quantization",
        "plugins/residual_quantization/csrc",
    ]
    files = {
        f
        for directory in directories
        for f in (ROOT / directory).rglob("*")
        if f.suffix in (".py", ".cu", ".cuh", ".cpp", ".h")
    }
    files.add(ROOT / "plugins/residual_quantization/setup.py")
    return {str(f.relative_to(ROOT)): digest(f) for f in sorted(files)}


def make_config(
    name,
    coefficient_method,
    reference_mode,
    bits,
    group,
    use_rotation,
    single_record=False,
    *,
    sources=None,
):
    cfg = dict(
        id=name,
        coefficient_method=coefficient_method,
        reference_mode=reference_mode,
        anchor_index=0,
        packed_groups=[[[bit, group if bit == 2 else 32] for bit in bits]],
    )
    if single_record:
        cfg["packed_groups"] = [[record] for record in cfg["packed_groups"][0]]
    if use_rotation:
        cfg["use_rotation"] = True
    cfg["source_sha256"] = source_hashes() if sources is None else sources
    return cfg


def generate(output=HERE / "configs"):
    output.mkdir(parents=True, exist_ok=True)
    sources = source_hashes()
    paths = []
    for bit, group in [(2, 16), (2, 32), (4, 32)]:
        for label, method, mode, rotation, single in METHODS:
            name = f"int{bit}_g{group}_{label}_{mode}"
            cfg = make_config(
                name,
                method,
                mode,
                [bit] * 4,
                group,
                use_rotation=rotation,
                single_record=single,
                sources=sources,
            )
            path = output / (name + ".json")
            atomic(path, cfg)
            paths.append(path)
    bf16 = dict(
        id="bf16",
        packed_groups=[],
    )
    atomic(output / "bf16.json", bf16)
    assert len(paths) == 18
    return paths


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=HERE / "configs")
    a = p.parse_args()
    generate(a.output)
