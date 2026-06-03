#!/usr/bin/env bash
# MLeRP: dsks_2025.08 + pip --user for hydra/peft.  Mac: .venv from requirements.txt
#
#   bash scripts/env.sh setup    # once
#   source scripts/env.sh        # every session

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# MLeRP DSKS — change version here if needed (dsks_2024.06 also works)
DSKS="/apps/mambaforge/envs/dsks_2025.08"

_is_mlerp() { [[ -x "${DSKS}/bin/python3" ]]; }

_mlerp_on() {
  export PATH="${DSKS}/bin:${PATH}"
  export CONDA_PREFIX="${DSKS}"
  export PYTHONPATH="${REPO_ROOT}/src"
}

_setup_mlerp() {
  _mlerp_on
  python3 -m pip install --user \
    hydra-core omegaconf peft accelerate huggingface_hub datasets
  echo "OK. Run: source ${REPO_ROOT}/scripts/env.sh"
}

_setup_local() {
  cd "$REPO_ROOT"
  python3 -m venv .venv
  # shellcheck source=/dev/null
  source .venv/bin/activate
  pip install -r requirements.txt
  echo "OK. Run: source ${REPO_ROOT}/scripts/env.sh"
}

if [[ "${1:-}" == "setup" ]]; then
  _is_mlerp && _setup_mlerp || _setup_local
  exit 0
fi

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Run: source ${REPO_ROOT}/scripts/env.sh" >&2
  exit 1
fi

cd "$REPO_ROOT"
if _is_mlerp; then
  _mlerp_on
elif [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
  export PYTHONNOUSERSITE=1
  export PYTHONPATH="${REPO_ROOT}/src"
else
  echo "Run: bash ${REPO_ROOT}/scripts/env.sh setup" >&2
  return 1 2>/dev/null || exit 1
fi

# dsks only has python3
python() { python3 "$@"; }
export -f python 2>/dev/null || true
