#!/bin/bash
# MLeRP: P5 student pilot for one candidate (docs/experiments/2026-10-06-student-pilot.md).
# Zero-shot on the HLQC validation fold, then a short LoRA run on the HLQC rows
# outside it. Never reads MIV6.3A. Runs in grpo-env (transformers 5.14.1), which
# supports every candidate; the DSKS env (4.55) does not know olmo3/mistral3.
#
#   MODEL=/mnt/userdata4/jia-wen/models/Olmo-3-7B-Instruct sbatch scripts/mlerp_student_pilot.slurm
#   MODEL=Qwen/Qwen2.5-7B-Instruct STEP=zs sbatch scripts/mlerp_student_pilot.slurm   # cached reference
#
# Env: MODEL (staged path or cached HF id), STEP=zs|lora|both|ftval (both; ftval = predict with the pilot adapter), ROWS (300), EXTRA overrides.

#SBATCH --job-name=student_pilot
#SBATCH --partition=BigCats
#SBATCH --qos=lion
#SBATCH --gres=gpu:40gb:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/student_pilot/%x_%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/setup_grpo_env.sh use
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p outputs/student_pilot
: "${MODEL:?}"; STEP="${STEP:-both}"; ROWS="${ROWS:-300}"
# shellcheck disable=SC2206
OV=(model.bf16=true model.fp16=false ${EXTRA:-})
echo "== pilot $MODEL step $STEP rows $ROWS git $(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l) =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [[ "$STEP" == zs || "$STEP" == both ]]; then
  python -m baseline.student_pilot zs --model "$MODEL" "${OV[@]}"
fi
if [[ "$STEP" == lora || "$STEP" == both ]]; then
  python -m baseline.student_pilot lora --model "$MODEL" --rows "$ROWS" "${OV[@]}"
fi
if [[ "$STEP" == ftval ]]; then
  python -m baseline.student_pilot ftval --model "$MODEL" "${OV[@]}"
fi
