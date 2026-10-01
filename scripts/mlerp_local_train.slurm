#!/bin/bash
# MLeRP: train the LoRA adapter(s) for one arm of the Qwen tier of the
# Model Scale x Adaptation x Rationale Alignment grid.
#
# Two axes. TARGET is what the model is trained to emit; REGIME is how many
# adapters carry that training:
#
#   REGIME=pair        two adapters, one per tier      -> ft_bare    / ft_rat
#   REGIME=mixed       one adapter, shuffled T1+T2     -> ft1mix_*
#   REGIME=sequential  one adapter, T1 then T2         -> ft1seq_*
#
# Submit from repo root, one job per (target, regime):
#   TARGET=bare sbatch scripts/mlerp_local_train.slurm     # FT-Bare (label targets)
#   TARGET=rat  sbatch scripts/mlerp_local_train.slurm     # FT-Rat  (rationale + label)
#   TARGET=rat CTX=3 sbatch scripts/mlerp_local_train.slurm
#   TARGET=bare REGIME=mixed      sbatch scripts/mlerp_local_train.slurm
#   TARGET=bare REGIME=sequential sbatch scripts/mlerp_local_train.slurm
#
# TF_MODE adds the teacher-forcing axis (REGIME=pair TARGET=bare only). It trains
# ONLY the T2 adapter, reusing the base arm's T1, and varies which T1 label
# conditioned T2 during training:
#
#   TF_MODE=full   always the gold T1        -> ft_bare_tffull  (the control)
#   TF_MODE=dyn    decayed gold -> predicted -> ft_bare_tfdyn
#   TF_MODE=zero   always the predicted T1   -> ft_bare_tfzero
#
#   TF_MODE=zero CTX=5 sbatch scripts/mlerp_local_train.slurm
#
# dyn and zero need the frozen out-of-fold predicted-T1 file for this context
# length; see scripts/mlerp_oof_t1.slurm.
#
# All three regimes see the same number of example-passes, so the wall clock
# below covers any of them: `pair` trains two adapters over one tier each, and
# the single-adapter regimes train one adapter over both tiers.
#
# TARGET=rat requires the frozen rationale file for this context length. Generate
# it first from a machine with Azure access (it needs the gpt-4o deployment, not
# a GPU):
#   PYTHONPATH=src python -m baseline.rationalize --ctx <N>
#
# See conf/baseline_ft_local_config.yaml and docs/BASELINE.md.

#SBATCH --job-name=qwen_ft_train
#SBATCH --partition=BigCats
#SBATCH --qos=lion
#SBATCH --gres=gpu:40gb:1        # full A100 40GB; a 20GB MIG slice OOMs on 7B
#SBATCH --mem=48G
#SBATCH --time=06:00:00          # ~3850 example-passes x 3 epochs, any regime
#SBATCH --output=outputs/local_arm/train_%j.log

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source "${SLURM_SUBMIT_DIR}/scripts/env.sh"
export TOKENIZERS_PARALLELISM=false

TARGET="${TARGET:-bare}"
REGIME="${REGIME:-pair}"
CTX="${CTX:-5}"
TF_MODE="${TF_MODE:-}"
# Replicate index for the teacher-forcing arms; blank = the original run.
SEED="${SEED:-}"

bash "${SLURM_SUBMIT_DIR}/scripts/check_data.sh"
mkdir -p outputs/local_arm

if [[ "$TARGET" == "rat" ]]; then
  RAT_FILE="data/fine_tuning/rationales/hlqc_rationales_ctx${CTX}.json"
  # An interrupted generation run leaves an empty `{}` behind, so presence alone
  # is not enough; local_arm additionally checks per-row coverage.
  if [[ ! -f "$RAT_FILE" ]] || ! grep -q '_explanation' "$RAT_FILE" 2>/dev/null; then
    echo "Missing or empty $RAT_FILE. Generate it first (needs Azure, not a GPU):" >&2
    echo "  PYTHONPATH=src python -m baseline.rationalize --ctx ${CTX}" >&2
    exit 1
  fi
fi

if [[ -n "$TF_MODE" && "$TF_MODE" != "full" ]]; then
  OOF_FILE="data/fine_tuning/oof_t1/hlqc_oof_t1_ctx${CTX}.json"
  # An interrupted fold run leaves a valid but partial file, which would let the
  # uncovered rows fall back to gold conditioning; local_arm additionally checks
  # per-row coverage and refuses rather than training a mislabelled arm.
  if [[ ! -f "$OOF_FILE" ]] || ! grep -q 't1_pred' "$OOF_FILE" 2>/dev/null; then
    echo "Missing or empty $OOF_FILE. Generate it first (one job per fold):" >&2
    echo "  FOLD=k STAGE=train   CTX=${CTX} sbatch scripts/mlerp_oof_t1.slurm" >&2
    echo "  FOLD=k STAGE=predict CTX=${CTX} sbatch scripts/mlerp_oof_t1.slurm" >&2
    exit 1
  fi
fi

echo "Training target=${TARGET} regime=${REGIME} ctx=${CTX} tf=${TF_MODE:-none}"
python3 -m baseline.local_arm train --target "$TARGET" --regime "$REGIME" \
  --ctx "$CTX" ${TF_MODE:+--tf "$TF_MODE"} ${SEED:+--seed "$SEED"}
