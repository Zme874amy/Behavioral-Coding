#!/bin/bash
# MLeRP: annotate the evaluation set with the smart-retrieval `ag` arm.
#
# Same shape and cost as an `fs` predict cell (two calls per utterance): the
# exemplars are just retrieved per utterance (src/agentic/retriever.py) instead
# of loaded from a frozen file. agentic.arm checkpoints every 25 utterances, so a
# requeue resumes rather than restarts.
#
# Submit from repo root (RETRIEVER=qwen|tfidf, BACKBONE=none|ft_bare|ft1mix_bare):
#   RETRIEVER=qwen  BACKBONE=ft1mix_bare INF=bare sbatch scripts/mlerp_agentic.slurm  # main setting (1 adapter)
#   RETRIEVER=qwen  BACKBONE=ft_bare sbatch scripts/mlerp_agentic.slurm  # 2-adapter headline
#   RETRIEVER=tfidf BACKBONE=ft_bare sbatch scripts/mlerp_agentic.slurm  # tfidf comparison
#   RETRIEVER=qwen  BACKBONE=none    sbatch scripts/mlerp_agentic.slurm  # rep vs v1
#   RETRIEVER=qwen  BACKBONE=ft_bare RARE_RESERVE=0 sbatch scripts/mlerp_agentic.slurm  # ablation
#
# BACKBONE=ft_bare needs the ft_bare adapter pair and BACKBONE=ft1mix_bare the
# shared mixed-regime adapter, both trained via scripts/mlerp_local_train.slurm
# (TARGET=bare; REGIME=mixed for the latter). Each config writes its own CSV
# (qwen_ag_<retriever>_<backbone>_inf_<inf>_ctx<N>.csv).

#SBATCH --job-name=qwen_agentic
#SBATCH --partition=BigCats
#SBATCH --qos=lion
#SBATCH --gres=gpu:40gb:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/local_arm/agentic_%j.log

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source "${SLURM_SUBMIT_DIR}/scripts/env.sh"
export TOKENIZERS_PARALLELISM=false

INF="${INF:-cot}"
CTX="${CTX:-5}"
RETRIEVER="${RETRIEVER:-qwen}"
BACKBONE="${BACKBONE:-none}"

bash "${SLURM_SUBMIT_DIR}/scripts/check_data.sh"
mkdir -p outputs/local_arm outputs/agentic data/annotated/baseline

RESERVE_ARG=""
[ -n "${RARE_RESERVE:-}" ] && RESERVE_ARG="--rare-reserve ${RARE_RESERVE}"

echo "Predicting arm=ag retriever=${RETRIEVER} backbone=${BACKBONE} inf=${INF} ctx=${CTX} ${RESERVE_ARG}"
python3 -m agentic.arm predict --inf "$INF" --ctx "$CTX" \
    --retriever "$RETRIEVER" --backbone "$BACKBONE" $RESERVE_ARG

# Rescore everything present so the comparison stays current as cells land.
python3 -m baseline.eval
