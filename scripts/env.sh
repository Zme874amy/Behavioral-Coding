#!/usr/bin/env bash
# Project environment.
#   bash scripts/env.sh setup   # once: install extras
#   source scripts/env.sh       # every session
#
# MLeRP: reuses the read-only DSKS conda env + `pip --user` for extras.
# Local: uses a repo .venv built from requirements.txt.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

# Newest /apps/mambaforge/envs/dsks_* that has a python (override with MLERP_DSKS).
_dsks() {
  local p
  for p in ${MLERP_DSKS:-} $(ls -d /apps/mambaforge/envs/dsks_* 2>/dev/null | sort -rV); do
    [[ -x "$p/bin/python" ]] && { echo "$p"; return 0; }
  done
  return 1
}

if [[ "${1:-}" == "setup" ]]; then
  set -euo pipefail
  cd "$REPO_ROOT"
  if dsks="$(_dsks)"; then
    echo "MLeRP DSKS: $dsks"
    "$dsks/bin/python" -m pip install --user -r requirements-extras.txt
  else
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
  fi
  echo "Done. Run: source scripts/env.sh"
  exit 0
fi

cd "$REPO_ROOT" || return 1 2>/dev/null || exit 1
if dsks="$(_dsks)"; then
  export PATH="$dsks/bin:$PATH"          # gives `python`, plus DSKS torch/transformers
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  source "$REPO_ROOT/.venv/bin/activate"
else
  echo "No env yet. Run: bash scripts/env.sh setup" >&2
  return 1 2>/dev/null || exit 1
fi
export PYTHONPATH="$REPO_ROOT/src"
# DSKS ships TensorFlow; SFT is PyTorch-only — skip TF imports in transformers.
export USE_TF=0
