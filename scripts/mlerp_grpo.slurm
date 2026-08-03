#!/bin/bash
# MLeRP: one job of the single-call GRPO ladder.
#
# Submit from the repo root. STAGE selects what runs:
#
#   STAGE=calibrate sbatch scripts/mlerp_grpo.slurm            # do this FIRST
#   STAGE=sft TARGET=bare sbatch scripts/mlerp_grpo.slurm      # sc_ft_bare
#   STAGE=sft TARGET=rat  sbatch scripts/mlerp_grpo.slurm      # sc_ft_rat
#   STAGE=grpo SEED=0 sbatch scripts/mlerp_grpo.slurm          # headline arm
#   STAGE=grpo SEED=0 WEIGHTING=off sbatch scripts/mlerp_grpo.slurm   # Phase 2
#   STAGE=grpo SEED=0 INIT=cold      sbatch scripts/mlerp_grpo.slurm  # Phase 2
#   STAGE=predict ARM=sc_grpo SEED=0 sbatch scripts/mlerp_grpo.slurm
#
# CTX defaults to 5 (Phase 1); pass CTX=3 for the Phase 3 replication.
#
# Run STAGE=calibrate before committing to the 3-seed ladder. The GPU-hour
# figures in docs/GRPO.md assume a vLLM prefix-cache hit rate that has not been
# measured on this prompt shape, so they carry roughly a factor-of-two
# uncertainty; calibrate replaces them with a measurement from this cluster.

#SBATCH --job-name=grpo
#SBATCH --partition=BigCats
#SBATCH --qos=lion
#SBATCH --gres=gpu:40gb:1     # full A100 40GB; a 20GB MIG slice OOMs on 7B
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/grpo/%x_%j.log

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p outputs/grpo

# The GRPO venv, NOT scripts/env.sh: vLLM pins its own torch and the DSKS conda
# env ships torch 2.4.0 that the rest of the pipeline runs against.
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR}/scripts/setup_grpo_env.sh" use

STAGE="${STAGE:-calibrate}"
CTX="${CTX:-5}"
SEED="${SEED:-0}"
TARGET="${TARGET:-bare}"
ARM="${ARM:-sc_grpo}"
WEIGHTING="${WEIGHTING:-on}"
INIT="${INIT:-ft_bare}"

echo "=== stage=$STAGE ctx=$CTX seed=$SEED target=$TARGET arm=$ARM"
echo "=== weighting=$WEIGHTING init=$INIT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

FLAGS=()
[[ "$WEIGHTING" == "off" ]] && FLAGS+=(--no-rare-weighting)
[[ "$INIT" == "cold" ]] && FLAGS+=(--cold-start)

case "$STAGE" in
  calibrate)
    python -m baseline.grpo calibrate --ctx "$CTX" --limit 64 --steps 5 "${FLAGS[@]}"
    ;;
  sft)
    if [[ "$TARGET" == "rat" && ! -f "data/fine_tuning/rationales/hlqc_rationales_ctx${CTX}.json" ]]; then
      echo "Missing frozen rationales for ctx=${CTX}. Generate them first (needs Azure, not a GPU):" >&2
      echo "  PYTHONPATH=src python -m baseline.rationalize --ctx ${CTX}" >&2
      exit 1
    fi
    python -m baseline.sc_arm train --target "$TARGET" --ctx "$CTX"
    ;;
  grpo)
    python -m baseline.grpo train --seed "$SEED" --ctx "$CTX" "${FLAGS[@]}"
    ;;
  predict)
    PRED=(--arm "$ARM" --ctx "$CTX")
    [[ "$ARM" == "sc_grpo" ]] && PRED+=(--seed "$SEED")
    [[ "$WEIGHTING" == "off" ]] && PRED+=(--unweighted)
    python -m baseline.sc_arm predict "${PRED[@]}"
    ;;
  *)
    echo "Unknown STAGE=$STAGE (calibrate|sft|grpo|predict)" >&2
    exit 1
    ;;
esac

echo "=== done"
