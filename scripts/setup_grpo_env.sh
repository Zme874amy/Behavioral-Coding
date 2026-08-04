#!/usr/bin/env bash
# Build the dedicated venv for the GRPO arm.
#
#   bash scripts/setup_grpo_env.sh        # once, on a login node
#   source scripts/setup_grpo_env.sh use  # every session (activates only)
#
# Kept separate from scripts/env.sh on purpose. That one reuses the read-only
# MLeRP DSKS conda env (torch 2.4.0) which every other part of the pipeline
# runs against; vLLM pins its own torch and would break it. Nothing here
# touches DSKS.
#
# The venv lives OUTSIDE the repo; a vLLM install is well over 10GB and does
# not belong near a directory that gets committed.
#
# HF_HOME defaults to the cache that already exists on MLeRP, which already
# holds Qwen2.5-7B-Instruct (~15GB). Pointing it at a fresh directory would
# silently re-download the whole model into a home directory that is under a
# 50GB quota, which is how this script ran out of disk the first time.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
GRPO_ENV="${GRPO_ENV:-$(dirname "$REPO_ROOT")/grpo-env}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# /tmp on the MLeRP login node is a 30GB root filesystem shared by every user
# and typically has ~2GB free, which is not enough to unpack the CUDA wheels
# vLLM depends on (several are 200-500MB each). Keep scratch on the same
# volume as the venv.
export TMPDIR="${TMPDIR_OVERRIDE:-$(dirname "$REPO_ROOT")/tmp}"

_activate() {
  # shellcheck disable=SC1091
  source "$GRPO_ENV/bin/activate"
  export PYTHONPATH="$REPO_ROOT/src"
  export HF_HOME
  export TOKENIZERS_PARALLELISM=false
  # DSKS ships TensorFlow on the PATH; keep transformers from importing it.
  export USE_TF=0
  cd "$REPO_ROOT" || return 1
}

if [[ "${1:-}" == "use" ]]; then
  if [[ ! -x "$GRPO_ENV/bin/python" ]]; then
    echo "No GRPO env at $GRPO_ENV. Build it: bash scripts/setup_grpo_env.sh" >&2
    return 1 2>/dev/null || exit 1
  fi
  _activate
  echo "GRPO env active: $(python --version), HF_HOME=$HF_HOME"
  return 0 2>/dev/null || exit 0
fi

set -e
echo "Building GRPO env at $GRPO_ENV"
echo "HuggingFace cache at $HF_HOME"
echo "Scratch (TMPDIR) at $TMPDIR"

# Fail here rather than 20 minutes into an install. The home directory is under
# a 50GB quota and the resolved vLLM stack (torch plus the nvidia-* CUDA 13
# wheels, flashinfer, triton) is the bulk of it. Override with NEED_GB=n if a
# future pin resolves smaller.
NEED_GB="${NEED_GB:-18}"
avail_gb="$(df -BG --output=avail "$(dirname "$GRPO_ENV")" 2>/dev/null | tail -1 | tr -dc '0-9')"
if [[ -n "$avail_gb" && "$avail_gb" -lt "$NEED_GB" ]]; then
  echo "ERROR: only ${avail_gb}GB free at $(dirname "$GRPO_ENV"), need ~${NEED_GB}GB." >&2
  echo "Reclaim space first. Optimizer state from COMPLETED training runs is" >&2
  echo "the usual culprit and is safe to delete (it is only used to resume an" >&2
  echo "interrupted run; the adapters are what matter):" >&2
  echo "  find outputs data -type f \\( -name optimizer.pt -o -name scheduler.pt \\" >&2
  echo "       -o -name rng_state.pth \\) -delete" >&2
  echo "  rm -rf ~/.cache/pip" >&2
  exit 1
fi

# 3.10+ required by vLLM. Prefer an explicit interpreter over `python3`, which
# on a login node may already be a conda env we do not want to inherit from.
PY=""
for cand in python3.12 python3.11 python3.10 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    ver="$("$cand" -c 'import sys; print("%d%02d" % sys.version_info[:2])')"
    if [[ "$ver" -ge 310 ]]; then PY="$cand"; break; fi
  fi
done
[[ -n "$PY" ]] || { echo "Need python >= 3.10, found none" >&2; exit 1; }
echo "Base interpreter: $PY ($("$PY" --version 2>&1))"

mkdir -p "$HF_HOME" "$TMPDIR"
"$PY" -m venv "$GRPO_ENV"
_activate

python -m pip install --no-cache-dir --upgrade pip wheel
# --no-cache-dir because the wheel cache for this stack runs to ~7GB and lands
# in the quota'd home directory, where it is pure overhead: the env is built
# once and the cache is never reused.
#
# vLLM resolves and installs its own torch build; let it lead so pip does not
# first pull a torch that vLLM then has to replace.
python -m pip install --no-cache-dir -r "$REPO_ROOT/requirements-grpo.txt"

echo
echo "=== resolved versions ==="
python - <<'PY'
import importlib
for m in ("torch", "transformers", "trl", "peft", "accelerate", "vllm", "datasets"):
    try:
        print(f"  {m:14s} {getattr(importlib.import_module(m), '__version__', '?')}")
    except Exception as exc:
        print(f"  {m:14s} FAILED: {type(exc).__name__}: {exc}")

# TRL hard-fails outside its supported vLLM window, and it does so at trainer
# construction, i.e. after a GPU job has already been scheduled and the model
# loaded. Catch it here instead.
try:
    import vllm
    from packaging.version import Version
    v = Version(vllm.__version__)
    if not (Version("0.17.0") <= v <= Version("0.25.1")):
        print(f"\n  WARNING: vllm {v} is outside TRL's supported 0.17.0-0.25.1 window")
    else:
        print(f"\n  vllm {v} is inside TRL's supported window")
except Exception as exc:
    print(f"\n  WARNING: could not check vllm version: {exc}")

import torch
print(f"  torch CUDA build: {torch.version.cuda}, visible devices: {torch.cuda.device_count()}")
PY

echo
echo "Done. Every session:  source scripts/setup_grpo_env.sh use"
