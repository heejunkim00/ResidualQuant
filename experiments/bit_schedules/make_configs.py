"""Mixed-schedule cells; static cells reference the uniform experiment."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from reproduction import atomic, load

shared = load(
    "uniform_configs", HERE.parent / "uniform_residual_quantization/make_configs.py"
)
METHODS = tuple(
    cell for cell in shared.METHODS if cell[0] in ("direct", "ls", "optr", "ls_optr")
)


def generate():
    out = HERE / "configs"
    out.mkdir(exist_ok=True)
    sources = shared.source_hashes()
    for bits in ([2, 2, 2, 4], [4, 2, 2, 2]):
        for group in (16, 32):
            for label, method, mode, rotation, single in METHODS:
                name = "".join(map(str, bits)) + f"_g{group}_{label}_{mode}"
                cfg = shared.make_config(
                    name,
                    method,
                    mode,
                    bits,
                    group,
                    use_rotation=rotation,
                    single_record=single,
                    sources=sources,
                )
                atomic(out / (name + ".json"), cfg)
    reused = []
    for bit, group in [(4, 32), (2, 16), (2, 32)]:
        for label, method, mode, rotation, single in METHODS:
            name = f"int{bit}_g{group}_{label}_{mode}"
            reused.append(
                dict(
                    id=name,
                    config=str(
                        HERE.parent
                        / "uniform_residual_quantization/configs"
                        / (name + ".json")
                    ),
                    result=str(
                        HERE.parent
                        / "uniform_residual_quantization/data/evaluations"
                        / name
                        / "result.json"
                    ),
                )
            )
    atomic(HERE / "reuse.json", reused)


if __name__ == "__main__":
    generate()
