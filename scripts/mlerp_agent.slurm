#!/bin/bash
# MLeRP: annotate the evaluation set with the bounded structured AGENT
# (agentic.agent_arm) -- model-directed tool use + self-critique + revise, not RAG.
#
# The agent runs ~3-6 generations per utterance, so it is several times the cost
# of an `ag`/`fs` cell. Start with the tail-only pilot (TAIL=1, ~130 rows) before
# the full 821. agent_arm checkpoints every 25 rows, so a requeue resumes.
#
# Submit from repo root:
#   TAIL=1                      sbatch scripts/mlerp_agent.slurm   # tail pilot, base model
#   TAIL=1 RETRIEVER=qwen       sbatch scripts/mlerp_agent.slurm   # tail pilot, semantic tool
#   (no TAIL)                   sbatch scripts/mlerp_agent.slurm   # full 821, base model
#   BACKBONE=ft_bare            sbatch scripts/mlerp_agent.slurm   # (not recommended; adapters fight JSON)

#SBATCH --job-name=qwen_agent
#SBATCH --partition=BigCats
#SBATCH --qos=lion
#SBATCH --gres=gpu:40gb:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/local_arm/agent_%j.log

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source "${SLURM_SUBMIT_DIR}/scripts/env.sh"
export TOKENIZERS_PARALLELISM=false

CTX="${CTX:-5}"
RETRIEVER="${RETRIEVER:-tfidf}"
BACKBONE="${BACKBONE:-none}"

bash "${SLURM_SUBMIT_DIR}/scripts/check_data.sh"
mkdir -p outputs/local_arm outputs/agentic data/annotated/baseline

TAIL_ARG=""; [ -n "${TAIL:-}" ] && TAIL_ARG="--tail-only"
LIMIT_ARG=""; [ -n "${LIMIT:-}" ] && LIMIT_ARG="--limit ${LIMIT}"

echo "Agent: retriever=${RETRIEVER} backbone=${BACKBONE} ctx=${CTX} ${TAIL_ARG} ${LIMIT_ARG}"
python3 -m agentic.agent_arm predict --ctx "$CTX" \
    --retriever "$RETRIEVER" --backbone "$BACKBONE" $TAIL_ARG $LIMIT_ARG
