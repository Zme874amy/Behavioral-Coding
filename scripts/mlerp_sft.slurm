#!/bin/bash
# MLeRP batch SFT. Submit from repo root:
#   export PRESET=qwen7b
#   sbatch scripts/mlerp_sft.slurm
#
# Presets: qwen7b | gemma3n_e4b | gemma2_9b | qwen05b
# Change --qos if needed (lion / cheetah). See docs/HF_SETUP.md.

#SBATCH --job-name=sft_rich
#SBATCH --partition=BigCats
#SBATCH --qos=lion
#SBATCH --gres=gpu:40gb:1        # full A100 40GB; a 20GB MIG slice OOMs on 7B
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/sft_runs/%j.log

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source "${SLURM_SUBMIT_DIR}/scripts/env.sh"
export TOKENIZERS_PARALLELISM=false

PRESET="${PRESET:-qwen7b}"
RUN_DIR="outputs/sft_runs/${PRESET}_rich"

python3 src/sft/main.py \
  "+sft_models=${PRESET}" \
  sft_device=cuda \
  "hydra.run.dir=${RUN_DIR}"
