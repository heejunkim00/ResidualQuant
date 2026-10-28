#!/usr/bin/env bash
_residualquant_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
source "$_residualquant_root/.venv/bin/activate" || { unset _residualquant_root; return 2; }

export RESIDUALQUANT_DATA_ROOT="${RESIDUALQUANT_DATA_ROOT:-$_residualquant_root/local-data}"
export HF_HOME="${HF_HOME:-$RESIDUALQUANT_DATA_ROOT/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$RESIDUALQUANT_DATA_ROOT/cache/datasets}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
unset _residualquant_root
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"
