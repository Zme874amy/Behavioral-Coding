#!/bin/bash
# MLeRP: frozen OUT-OF-FOLD predicted Tier-1 labels for the HLQC training set.
#
# The teacher-forcing arms (TF_MODE=dyn/zero in mlerp_local_train.slurm) train T2
# conditioned on a PREDICTED T1 rather than the gold one. Those predictions have
# to be out of fold: the production T1 adapter trained on all 1,925 HLQC rows, so
# its predictions on those same rows are partly memorised and far cleaner than
# the ~79% it manages on held-out data. Training against them would understate
# the exposure roughly threefold.
#
# So the corpus is split into K conversation-level folds; each fold gets a T1-only
# adapter trained on the other folds, which then predicts the held-out fold. Two
# stages per fold, in order:
#
#   FOLD=0 STAGE=train   CTX=5 sbatch scripts/mlerp_oof_t1.slurm
#   FOLD=0 STAGE=predict CTX=5 sbatch scripts/mlerp_oof_t1.slurm
#
# All folds write into one shared JSON keyed by corp_utt_idx, checkpointed every
# 25 utterances, so predict jobs resume and may run in any order. When every fold
# has landed, check calibration before training anything downstream:
#
#   PYTHONPATH=src python -m baseline.oof_t1 report --ctx 5
#
# Pooled out-of-fold accuracy should sit near the ~0.79 the production T1 adapter
# scores on the evaluation corpus. Far above and the exposure being trained on is
# gentler than the one met at inference, which makes every downstream number a
# lower bound rather than an estimate.
#
# See src/baseline/oof_t1.py and docs/EXPERIMENTS.md.

#SBATCH --job-name=qwen_oof_t1
#SBATCH --partition=BigCats
#SBATCH --qos=lion
#SBATCH --gres=gpu:40gb:1        # full A100 40GB; a 20GB MIG slice OOMs on 7B
#SBATCH --mem=48G
#SBATCH --time=04:00:00          # train: ~1.5k rows x 3 epochs; predict: ~400 rows
#SBATCH --output=outputs/local_arm/oof_t1_%j.log

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source "${SLURM_SUBMIT_DIR}/scripts/env.sh"
export TOKENIZERS_PARALLELISM=false

CTX="${CTX:-5}"
FOLDS="${FOLDS:-5}"
FOLD="${FOLD:?set FOLD to the fold index, 0..$((FOLDS - 1))}"
STAGE="${STAGE:-train}"

bash "${SLURM_SUBMIT_DIR}/scripts/check_data.sh"
mkdir -p outputs/local_arm

case "$STAGE" in
  train|predict) ;;
  *) echo "STAGE must be 'train' or 'predict', got '${STAGE}'" >&2; exit 1 ;;
esac

echo "Out-of-fold T1: stage=${STAGE} fold=${FOLD}/${FOLDS} ctx=${CTX}"
python3 -m baseline.oof_t1 "${STAGE}-fold" --ctx "$CTX" --fold "$FOLD" --folds "$FOLDS"
