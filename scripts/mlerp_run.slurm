#!/bin/bash
# MLeRP: one re-run cell = train (if fine-tuned) + predict one fold of one arm.
# Generic over manifest/design/fold/seed/arm/student (docs/RERUN_PLAN.md P3).
# Submitted by scripts/submit_stage.py, which expands conf/stages/<stage>.yaml;
# by hand:
#
#   MANIFEST=misc_pooled_cv DESIGN=loso FOLD=0 SEED=42 ARM=ft1mix_bare INF=bare \
#     sbatch scripts/mlerp_run.slurm
#
# Env:
#   MANIFEST DESIGN FOLD SEED ARM INF     the cell (required)
#   STUDENT     staged model path (/mnt/userdata4/jia-wen/models/<name>) or cached HF id
#   CTX         context volleys (5)
#   PROMPT      prompt set (v2)
#   SOURCE      training-file swap, e.g. misc.hlqc.gold.cleaned (ablation)
#   EXTRA       further OmegaConf overrides (e.g. QLoRA settings for Gemma-3-12B)
#   KEEP_ADAPTERS=1   keep the fold adapter after prediction (default: delete it)
#
# Runs must come from a clean, tagged checkout (git pull of a rerun-* tag); the
# job refuses a dirty tree so every .meta.json sha names the exact code.

#SBATCH --job-name=rerun
#SBATCH --partition=BigCats
#SBATCH --qos=lion
#SBATCH --gres=gpu:40gb:1        # full A100 40GB; a 20GB MIG slice OOMs on 7B
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/rerun/%x_%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# grpo-env (transformers 5.14.1): the DSKS env (4.55) cannot load Qwen3.5 / Gemma-4 /
# Ministral-3 / OLMo-3. Compute nodes are offline, so models come from the staged
# copies or the HF cache.
source "${SLURM_SUBMIT_DIR}/scripts/setup_grpo_env.sh" use
export TOKENIZERS_PARALLELISM=false PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p outputs/rerun

: "${MANIFEST:?}" "${DESIGN:?}" "${FOLD:?}" "${SEED:?}" "${ARM:?}" "${INF:?}"
CTX="${CTX:-5}"; PROMPT="${PROMPT:-v2}"
if [[ -n "$(git status --porcelain --untracked-files=no)" && "${ALLOW_DIRTY:-0}" != "1" ]]; then
  echo "Refusing to run from a dirty tree (commit, tag, pull). ALLOW_DIRTY=1 for debugging only." >&2
  exit 1
fi
python -m components.prompts.registry --check

COMMON=(--manifest "$MANIFEST" --design "$DESIGN" --fold "$FOLD" --seed "$SEED"
        --ctx "$CTX" --prompt-version "$PROMPT")
[[ -n "${SOURCE:-}" ]] && COMMON+=(--training-source "$SOURCE")
OV=()
[[ -n "${STUDENT:-}" ]] && OV+=("model.base_model=${STUDENT}")
# shellcheck disable=SC2206
[[ -n "${EXTRA:-}" ]] && OV+=(${EXTRA})

echo "== cell: $MANIFEST/$DESIGN fold $FOLD seed $SEED arm $ARM inf $INF student ${STUDENT:-config} prompt $PROMPT source ${SOURCE:-none} =="
case "$ARM" in
  zs|fs) ;;
  *) python -m baseline.local_arm train --arm "$ARM" "${COMMON[@]}" "${OV[@]}" ;;
esac
DROP=(); [[ "${KEEP_ADAPTERS:-0}" != "1" ]] && DROP=(--drop-adapter)
python -m baseline.local_arm predict --arm "$ARM" --inf "$INF" "${COMMON[@]}" "${DROP[@]}" "${OV[@]}"
