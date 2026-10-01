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
# The 1-adapter arms take the same flags; only the adapter behind them differs:
#   ARM=ft1mix_bare INF=bare sbatch scripts/mlerp_local_predict.slurm
#   ARM=ft1seq_rat  INF=cot  sbatch scripts/mlerp_local_predict.slurm
#
# The whole grid at once (see also scripts/submit_local_grid.sh):
#   CTX=5 bash scripts/submit_local_grid.sh
#
# The ft_* and ft1* arms need their adapters trained first via
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

ARM="${ARM:?set ARM to one of zs, fs, ft_bare, ft_rat, ft1mix_bare, ft1mix_rat, ft1seq_bare, ft1seq_rat, ft_bare_tffull, ft_bare_tfdyn, ft_bare_tfzero}"
INF="${INF:?set INF to bare or cot}"
CTX="${CTX:-5}"
# Replicate index for the teacher-forcing arms; blank = the original run.
SEED="${SEED:-}"
# Cross-dataset evaluation: point at another corpus WITHOUT clobbering the
# MIV6.3A results. Set both together (EVAL_NAME namespaces the output file):
#   EVAL_CSV=data/manual/AnnoMI_eval.csv   EVAL_NAME=annomi   ARM=ft_bare INF=bare ...
#   EVAL_CSV=data/manual/Welivita_eval.csv EVAL_NAME=welivita ARM=ft_bare INF=bare ...
# Build those CSVs first: python3 -m baseline.prep_crossscheme all
EVAL_CSV="${EVAL_CSV:-}"
EVAL_NAME="${EVAL_NAME:-}"

bash "${SLURM_SUBMIT_DIR}/scripts/check_data.sh"
mkdir -p outputs/local_arm data/annotated/baseline

# Forward the eval-corpus overrides only when set, so the default run is
# byte-for-byte unchanged.
OVERRIDES=()
[ -n "$EVAL_CSV" ]  && OVERRIDES+=("dataset.eval_csv=${EVAL_CSV}")
[ -n "$EVAL_NAME" ] && OVERRIDES+=("dataset.eval_name=${EVAL_NAME}")

echo "Predicting arm=${ARM} inf=${INF} ctx=${CTX} ${EVAL_NAME:+ds=${EVAL_NAME}}"
python3 -m baseline.local_arm predict --arm "$ARM" --inf "$INF" --ctx "$CTX" \
  ${SEED:+--seed "$SEED"} "${OVERRIDES[@]}"

# Rescore everything present so the comparison stays current as cells land.
# `baseline.eval` scans all result files and reports each corpus in its own
# section, so one rescore keeps MIV6.3A and the cross-scheme corpora in sync.
python3 -m baseline.eval
