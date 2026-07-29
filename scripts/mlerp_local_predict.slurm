#!/bin/bash
# MLeRP: annotate the evaluation set for ONE cell of the Qwen tier.
#
# One job per cell rather than one job for all eight: the Inf-CoT cells generate
# 100-200 tokens per call and take hours, so a single job would blow past the
# wall clock. Each job here is roughly 0.5-3h, and local_arm checkpoints every 25
# utterances so a requeue resumes instead of restarting.
#
# Submit from repo root:
#   ARM=zs      INF=bare sbatch scripts/mlerp_local_predict.slurm
#   ARM=zs      INF=cot  sbatch scripts/mlerp_local_predict.slurm
#   ARM=fs      INF=bare sbatch scripts/mlerp_local_predict.slurm
#   ARM=fs      INF=cot  sbatch scripts/mlerp_local_predict.slurm
#   ARM=ft_bare INF=bare sbatch scripts/mlerp_local_predict.slurm
#   ARM=ft_bare INF=cot  sbatch scripts/mlerp_local_predict.slurm
#   ARM=ft_rat  INF=bare sbatch scripts/mlerp_local_predict.slurm
#   ARM=ft_rat  INF=cot  sbatch scripts/mlerp_local_predict.slurm
#
# All eight at once (see also scripts/submit_local_grid.sh):
#   CTX=5 bash scripts/submit_local_grid.sh
#
# The ft_* arms need their adapters trained first via
# scripts/mlerp_local_train.slurm.

#SBATCH --job-name=qwen_predict
#SBATCH --partition=BigCats
#SBATCH --qos=lion
#SBATCH --gres=gpu:40gb:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/local_arm/predict_%j.log

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source "${SLURM_SUBMIT_DIR}/scripts/env.sh"
export TOKENIZERS_PARALLELISM=false

ARM="${ARM:?set ARM to one of zs, fs, ft_bare, ft_rat}"
INF="${INF:?set INF to bare or cot}"
CTX="${CTX:-5}"

bash "${SLURM_SUBMIT_DIR}/scripts/check_data.sh"
mkdir -p outputs/local_arm data/annotated/baseline

echo "Predicting arm=${ARM} inf=${INF} ctx=${CTX}"
python3 -m baseline.local_arm predict --arm "$ARM" --inf "$INF" --ctx "$CTX"

# Rescore everything present so the comparison stays current as cells land.
python3 -m baseline.eval
