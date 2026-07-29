#!/usr/bin/env bash
# Submit the whole Qwen tier: two training jobs, then the eight evaluation cells
# as dependent jobs so the ft_* arms only start once their adapters exist.
#
# Usage (from repo root, on MLeRP):
#   bash scripts/submit_local_grid.sh              # ctx=5
#   CTX=3 bash scripts/submit_local_grid.sh
#   SKIP_TRAIN=1 bash scripts/submit_local_grid.sh # adapters already trained
#
# FT-Rat additionally needs the frozen rationale file for this context length:
#   PYTHONPATH=src python -m baseline.rationalize --ctx <N>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$REPO_ROOT"

CTX="${CTX:-5}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

RAT_FILE="data/fine_tuning/rationales/hlqc_rationales_ctx${CTX}.json"
if [[ ! -f "$RAT_FILE" ]]; then
  echo "WARNING: $RAT_FILE not found; the ft_rat arm will fail." >&2
  echo "  Generate it first: PYTHONPATH=src python -m baseline.rationalize --ctx ${CTX}" >&2
fi

bare_dep=""
rat_dep=""
if [[ "$SKIP_TRAIN" != "1" ]]; then
  bare_job=$(TARGET=bare CTX="$CTX" sbatch --parsable scripts/mlerp_local_train.slurm)
  rat_job=$(TARGET=rat  CTX="$CTX" sbatch --parsable scripts/mlerp_local_train.slurm)
  echo "train ft_bare adapters: job ${bare_job}"
  echo "train ft_rat  adapters: job ${rat_job}"
  bare_dep="--dependency=afterok:${bare_job}"
  rat_dep="--dependency=afterok:${rat_job}"
fi

submit() {  # submit <arm> <inf> [dependency]
  local arm="$1" inf="$2" dep="${3:-}"
  local job
  # shellcheck disable=SC2086  # dep must word-split into sbatch flags
  job=$(ARM="$arm" INF="$inf" CTX="$CTX" sbatch --parsable $dep scripts/mlerp_local_predict.slurm)
  echo "  qwen_${arm}_inf_${inf}_ctx${CTX}: job ${job}"
}

echo "evaluation cells:"
# In-context arms need no adapters, so they start immediately.
submit zs bare
submit zs cot
submit fs bare
submit fs cot
submit ft_bare bare "$bare_dep"
submit ft_bare cot  "$bare_dep"
submit ft_rat  bare "$rat_dep"
submit ft_rat  cot  "$rat_dep"

echo
echo "Submitted. Watch with: squeue -u \$USER"
echo "Results land in data/annotated/baseline/qwen_<arm>_inf_<inf>_ctx${CTX}.csv"
echo "Scores are rewritten to docs/BASELINE_RESULTS.md as each cell finishes."
