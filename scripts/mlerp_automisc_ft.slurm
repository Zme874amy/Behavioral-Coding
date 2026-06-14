#!/bin/bash
# MLeRP batch run: AutoMISC pipeline + LoRA fine-tuning fair comparison.
# Submit from repo root:
#   sbatch scripts/mlerp_automisc_ft.slurm
#
# Trains two LoRA adapters per CV fold (T1 + T2) and compares the zero-shot
# AutoMISC annotator against the fine-tuned one on MIV6.3A (conversation-level
# 5-fold CV). See conf/automisc_ft_config.yaml and docs/HF_SETUP.md.

#SBATCH --job-name=automisc_ft
#SBATCH --partition=BigCats
#SBATCH --qos=lion
#SBATCH --gres=gpu:40gb:1        # full A100 40GB; a 20GB MIG slice OOMs on 7B
#SBATCH --mem=48G
#SBATCH --time=16:00:00          # 10 LoRA trainings + tiered inference over 821x(ZS+CV)
#SBATCH --output=outputs/automisc_ft/%j.log

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source "${SLURM_SUBMIT_DIR}/scripts/env.sh"
export TOKENIZERS_PARALLELISM=false

RUN_DIR="outputs/automisc_ft/qwen7b_miv63a"

python3 src/automisc_ft/main.py \
  "hydra.run.dir=${RUN_DIR}"
