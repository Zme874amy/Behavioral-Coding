#!/bin/bash
# MLeRP: one job of the two-call GRPO ladder (cells A and B; baseline.grpo_tc).
#
# Submit from the repo root. STAGE selects what runs:
#
#   STAGE=calibrate REGIME=pair sbatch scripts/mlerp_grpo_tc.slurm       # do FIRST
#   STAGE=train REGIME=pair CREDIT=dec   SEED=0 sbatch scripts/mlerp_grpo_tc.slurm
#   STAGE=train REGIME=mix  CREDIT=joint SEED=0 sbatch scripts/mlerp_grpo_tc.slurm
#   STAGE=train REGIME=pair CREDIT=dec SEED=0 WEIGHTING=off sbatch ...    # Phase 2
#   STAGE=train REGIME=pair CREDIT=dec SEED=0 INIT=cold      sbatch ...   # Phase 2
#   STAGE=predict  ARM=grpo_pair_dec SEED=0 sbatch scripts/mlerp_grpo_tc.slurm
#   STAGE=recovery ARM=grpo_pair_dec SEED=0 sbatch scripts/mlerp_grpo_tc.slurm
#   STAGE=recovery ARM=ft_bare              sbatch scripts/mlerp_grpo_tc.slurm
#
# `train` and `calibrate` need a full 40GB card: decoupled colocates a vLLM
# engine with the policy, and joint fans each prompt out to G rollouts x 2 turns.
# `predict` and `recovery` are single-sequence HF two-call decoding at ~16GB and
# fit a 20GB MIG slice:  ... sbatch --gres=gpu:3g.20gb:1 scripts/mlerp_grpo_tc.slurm
#
# The four arms warm-start from the SFT two-call arms, so those must be trained
# first (baseline.local_arm), and decoupled T2 needs the frozen oof_t1 store
# (baseline.oof_t1) -- both are preconditions the submitter wires as deps.
#
# CTX defaults to 5 (Phase 1); pass CTX=3 for the Phase 4 replication.

#SBATCH --job-name=grpotc
#SBATCH --partition=BigCats
#SBATCH --qos=lion
#SBATCH --gres=gpu:40gb:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/grpo/%x_%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p outputs/grpo

# The GRPO venv, not scripts/env.sh (vLLM pins its own torch).
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR}/scripts/setup_grpo_env.sh" use

# Per-job rendezvous port so concurrent jobs cannot collide on 29500 (decoupled
# uses vLLM's torch.distributed group; harmless for the joint loop).
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$(( 20000 + SLURM_JOB_ID % 20000 ))}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

STAGE="${STAGE:-calibrate}"
CTX="${CTX:-5}"
SEED="${SEED:-0}"
REGIME="${REGIME:-pair}"
CREDIT="${CREDIT:-dec}"
ARM="${ARM:-grpo_pair_dec}"
VARIANT="${VARIANT:-weighted}"
WEIGHTING="${WEIGHTING:-on}"
INIT="${INIT:-warm}"

echo "=== stage=$STAGE ctx=$CTX seed=$SEED regime=$REGIME credit=$CREDIT arm=$ARM"
echo "=== variant=$VARIANT weighting=$WEIGHTING init=$INIT master_port=$MASTER_PORT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

COLD=()
[[ "$INIT" == "cold" ]] && COLD+=(--cold-start)
FLAGS=("${COLD[@]}")
[[ "$WEIGHTING" == "off" ]] && FLAGS+=(--no-rare-weighting)

case "$STAGE" in
  calibrate)
    # Only times the loop; takes --cold-start (changes what loads) but not
    # --no-rare-weighting, which calibrate does not define.
    python -m baseline.grpo_tc calibrate --regime "$REGIME" --ctx "$CTX" \
      --limit 64 --steps 5 "${COLD[@]}"
    ;;
  train)
    python -m baseline.grpo_tc train --regime "$REGIME" --credit "$CREDIT" \
      --seed "$SEED" --ctx "$CTX" "${FLAGS[@]}"
    ;;
  predict)
    python -m baseline.grpo_tc predict --arm "$ARM" --seed "$SEED" \
      --variant "$VARIANT" --ctx "$CTX"
    ;;
  recovery)
    PROBE=(--arm "$ARM" --ctx "$CTX" --variant "$VARIANT")
    [[ "$ARM" == grpo_* ]] && PROBE+=(--seed "$SEED")
    python -m baseline.recovery "${PROBE[@]}"
    ;;
  *)
    echo "Unknown STAGE=$STAGE (calibrate|train|predict|recovery)" >&2
    exit 1
    ;;
esac

echo "=== done"
