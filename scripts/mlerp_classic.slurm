#!/bin/bash
# MLeRP: the classic (non-LLM) pathway that needs a GPU.
#   MODE=encoder      -> fine-tune roberta-base (T1 + T2), predict MIV6.3A
#   MODE=qwen_sklearn -> sklearn classifier on Qwen-embedding features
# The ngram sklearn variants run CPU-locally on the Mac; only these need the GPU.
#
# roberta-base must be staged into the HF cache first (scp from the Mac); loaded
# with HF_HUB_OFFLINE=1 so no internet is needed at train time.
#
# Submit from repo root:
#   MODE=encoder BALANCED=1        sbatch scripts/mlerp_classic.slurm
#   MODE=encoder BALANCED=1 CONTEXT=1 sbatch scripts/mlerp_classic.slurm
#   MODE=qwen_sklearn CLF=logreg BALANCED=1 sbatch scripts/mlerp_classic.slurm

#SBATCH --job-name=classic
#SBATCH --partition=BigCats
#SBATCH --qos=lion
#SBATCH --gres=gpu:40gb:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/classic/classic_%j.log

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source "${SLURM_SUBMIT_DIR}/scripts/env.sh"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

MODE="${MODE:-encoder}"
CTX="${CTX:-5}"
CLF="${CLF:-logreg}"
MODEL="${MODEL:-roberta-base}"
EPOCHS="${EPOCHS:-3}"

BAL=""; [ -n "${BALANCED:-}" ] && BAL="--balanced"
CTXARG=""; [ -n "${CONTEXT:-}" ] && CTXARG="--context"

bash "${SLURM_SUBMIT_DIR}/scripts/check_data.sh"
mkdir -p outputs/classic data/annotated/classic

if [ "$MODE" = "encoder" ]; then
  echo "Encoder fine-tune model=${MODEL} ctx=${CTX} epochs=${EPOCHS} ${BAL} ${CTXARG}"
  python3 -m classic.encoder --model "$MODEL" --ctx "$CTX" --epochs "$EPOCHS" $BAL $CTXARG
else
  echo "sklearn on qwen features clf=${CLF} ctx=${CTX} ${BAL} ${CTXARG}"
  python3 -m classic.sklearn_run --features qwen --clf "$CLF" --ctx "$CTX" $BAL $CTXARG
fi
