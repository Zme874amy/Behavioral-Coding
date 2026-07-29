#!/bin/bash
# MLeRP: train one LoRA adapter pair for the Qwen tier of the
# Model Scale x Adaptation x Rationale Alignment grid.
#
# Submit from repo root, one job per training target:
#   TARGET=bare sbatch scripts/mlerp_local_train.slurm     # FT-Bare (label targets)
#   TARGET=rat  sbatch scripts/mlerp_local_train.slurm     # FT-Rat  (rationale + label)
#   TARGET=rat CTX=3 sbatch scripts/mlerp_local_train.slurm
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
#SBATCH --time=06:00:00          # 2 adapters x ~1925 examples x 3 epochs
#SBATCH --output=outputs/local_arm/train_%j.log

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source "${SLURM_SUBMIT_DIR}/scripts/env.sh"
export TOKENIZERS_PARALLELISM=false

TARGET="${TARGET:-bare}"
CTX="${CTX:-5}"

bash "${SLURM_SUBMIT_DIR}/scripts/check_data.sh"
mkdir -p outputs/local_arm

if [[ "$TARGET" == "rat" ]]; then
  RAT_FILE="data/fine_tuning/rationales/hlqc_rationales_ctx${CTX}.json"
  if [[ ! -f "$RAT_FILE" ]]; then
    echo "Missing $RAT_FILE. Generate it first (needs Azure, not a GPU):" >&2
    echo "  PYTHONPATH=src python -m baseline.rationalize --ctx ${CTX}" >&2
    exit 1
  fi
fi

echo "Training target=${TARGET} ctx=${CTX}"
python3 -m baseline.local_arm train --target "$TARGET" --ctx "$CTX"
