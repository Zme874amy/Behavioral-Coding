#!/bin/bash
#SBATCH --job-name=sft_rich
#SBATCH --partition=BigCats
#SBATCH --qos=lion
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/sft_runs/%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "${SLURM_SUBMIT_DIR}/scripts/env.sh"
export TOKENIZERS_PARALLELISM=false

PRESET="${PRESET:-qwen7b}"
python3 src/sft/main.py \
  "+sft_models=${PRESET}" \
  sft_device=cuda \
  "hydra.run.dir=outputs/sft_runs/${PRESET}_rich"
