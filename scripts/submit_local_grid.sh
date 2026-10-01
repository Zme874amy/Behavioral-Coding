#!/usr/bin/env bash
# Submit the two-call Qwen grid: the training jobs, then the evaluation cells as
# dependent jobs so each ft arm only starts once its adapters exist.
#
# Three adapter regimes share this grid. They run the SAME two-call inference and
# differ only in how many adapters carry the training, which is what makes the
# adapter-count comparison identifiable:
#
#   pair        2 adapters, one per tier    -> ft_bare     / ft_rat
#   mixed       1 adapter, shuffled T1+T2   -> ft1mix_bare / ft1mix_rat
#   sequential  1 adapter, T1 then T2       -> ft1seq_bare / ft1seq_rat
#
# Usage (from repo root, on MLeRP):
#   bash scripts/submit_local_grid.sh                     # ctx=5, all regimes
#   CTX=3 bash scripts/submit_local_grid.sh
#   REGIMES=pair bash scripts/submit_local_grid.sh        # the original grid only
#   REGIMES="mixed sequential" bash scripts/submit_local_grid.sh
#   SKIP_TRAIN=1 bash scripts/submit_local_grid.sh        # adapters already trained
#   SKIP_INCONTEXT=1 bash scripts/submit_local_grid.sh    # skip zs/fs
#   DRYRUN=1 bash scripts/submit_local_grid.sh            # print, submit nothing
#
# The rat arms additionally need the frozen rationale file for this context
# length. When it is absent they are skipped rather than submitted to fail:
#   PYTHONPATH=src python -m baseline.rationalize --ctx <N>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$REPO_ROOT"

CTX="${CTX:-5}"
# `-` not `:-`, so REGIMES="" means "no adapter arms, just the in-context ones"
# rather than silently falling back to the full default.
REGIMES="${REGIMES-pair mixed sequential}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_INCONTEXT="${SKIP_INCONTEXT:-0}"
DRYRUN="${DRYRUN:-0}"

RAT_FILE="data/fine_tuning/rationales/hlqc_rationales_ctx${CTX}.json"
HAVE_RAT=1
# An interrupted generation run leaves an empty `{}` behind, which a bare -f
# test would happily accept, so require actual content.
if [[ ! -f "$RAT_FILE" ]] || ! grep -q '_explanation' "$RAT_FILE" 2>/dev/null; then
  HAVE_RAT=0
  echo "NOTE: $RAT_FILE not found, so every 'rat' arm is skipped at ctx=${CTX}." >&2
  echo "  Generate it first: PYTHONPATH=src python -m baseline.rationalize --ctx ${CTX}" >&2
  echo >&2
fi

# regime + target -> the arm name that prediction and scoring use.
arm_for() {
  case "$1:$2" in
    pair:bare)       echo ft_bare ;;
    pair:rat)        echo ft_rat ;;
    mixed:bare)      echo ft1mix_bare ;;
    mixed:rat)       echo ft1mix_rat ;;
    sequential:bare) echo ft1seq_bare ;;
    sequential:rat)  echo ft1seq_rat ;;
    *) echo "unknown regime/target $1/$2" >&2; return 1 ;;
  esac
}

submit_train() {  # submit_train <target> <regime>
  local target="$1" regime="$2"
  if [[ "$DRYRUN" == "1" ]]; then
    echo "dry-run"
    return
  fi
  TARGET="$target" REGIME="$regime" CTX="$CTX" \
    sbatch --parsable scripts/mlerp_local_train.slurm
}

submit_predict() {  # submit_predict <arm> <inf> [dependency]
  local arm="$1" inf="$2" dep="${3:-}"
  if [[ "$DRYRUN" == "1" ]]; then
    echo "  [dry] predict ${arm} inf=${inf} ctx=${CTX} ${dep}"
    return
  fi
  local job
  # shellcheck disable=SC2086  # dep must word-split into sbatch flags
  job=$(ARM="$arm" INF="$inf" CTX="$CTX" \
    sbatch --parsable $dep scripts/mlerp_local_predict.slurm)
  echo "  qwen_${arm}_inf_${inf}_ctx${CTX}: job ${job}"
}

# In-context arms need no adapters, so they start immediately and are submitted
# once regardless of which regimes were asked for.
if [[ "$SKIP_INCONTEXT" != "1" ]]; then
  echo "=== in-context arms (no adapters) ==="
  for arm in zs fs; do
    for inf in bare cot; do
      submit_predict "$arm" "$inf"
    done
  done
  echo
fi

for regime in $REGIMES; do
  echo "=== regime: ${regime} ==="
  for target in bare rat; do
    if [[ "$target" == "rat" && "$HAVE_RAT" != "1" ]]; then
      echo "  skipping ${regime}/rat: no rationale file for ctx=${CTX}"
      continue
    fi
    arm="$(arm_for "$regime" "$target")"

    dep=""
    if [[ "$SKIP_TRAIN" != "1" ]]; then
      job="$(submit_train "$target" "$regime")"
      echo "  train ${arm}: job ${job}"
      [[ "$DRYRUN" == "1" ]] || dep="--dependency=afterok:${job}"
    fi

    for inf in bare cot; do
      submit_predict "$arm" "$inf" "$dep"
    done
  done
  echo
done

echo "Submitted. Watch with: squeue -u \$USER"
echo "Results land in data/annotated/baseline/qwen_<arm>_inf_<inf>_ctx${CTX}.csv"
echo "Scores are rewritten to docs/BASELINE_RESULTS.md as each cell finishes."
