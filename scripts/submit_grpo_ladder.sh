#!/bin/bash
# Submit Phases 1-3 of the single-call GRPO ladder as one dependency graph.
#
#   bash scripts/submit_grpo_ladder.sh              # submits calibration too
#   CAL=158198 bash scripts/submit_grpo_ladder.sh   # reuse a queued calibration
#   DRYRUN=1 bash scripts/submit_grpo_ladder.sh     # print the graph, submit nothing
#
# Every training job hangs off `afterok` on calibration, so a configuration
# that cannot even start vLLM costs one short job rather than eight 20-hour
# ones. Each prediction hangs off its own training job and the faithfulness
# probe off its prediction, so the whole ladder runs unattended and nothing
# reads an adapter or a result file before it exists.
#
# Placement follows the memory split in docs/GRPO.md: GRPO colocates a vLLM
# engine with the policy and needs a full 40GB card, while prediction and the
# probe are single-sequence HF decoding at ~16GB and go to 20GB MIG slices,
# which are usually idle while the A100s are contended.
#
# HouseCats reaches only node07's two 40GB cards but is rarely queued for them;
# BigCats has twelve and a deep queue. Splitting across both means the ladder
# draws from whichever frees first instead of serialising behind one partition.
#
# Kept to POSIX-ish bash 3 features (no namerefs, no associative arrays) so it
# can be dry-run on a mac before it spends a day of A100 time.
set -euo pipefail
cd "$(dirname "$0")/.."

SLURM=scripts/mlerp_grpo.slurm
# `submit` is called as $(submit ...), so it runs in a subshell and cannot
# increment a shell variable the parent will see. The tally goes through a file.
TALLY=$(mktemp)
trap 'rm -f "$TALLY"' EXIT

# submit <label> <bigcats|housecats> <train|light> <dep-or-empty> <VAR=val ...>
submit() {
  label="$1"; where="$2"; kind="$3"; dep="$4"; shift 4

  case "$where" in
    housecats) place="--partition=HouseCats --qos=tabby" ;;
    bigcats)   place="--partition=BigCats --qos=lion" ;;
    *) echo "bad partition $where" >&2; exit 1 ;;
  esac
  case "$kind" in
    train) res="--gres=gpu:40gb:1 --mem=64G --time=20:00:00" ;;
    cal)   res="--gres=gpu:40gb:1 --mem=64G --time=01:00:00" ;;
    light) res="--gres=gpu:3g.20gb:1 --mem=48G --time=08:00:00" ;;
    *) echo "bad kind $kind" >&2; exit 1 ;;
  esac
  dep_arg=""
  [ -n "$dep" ] && dep_arg="--dependency=afterok:$dep"

  if [ -n "${DRYRUN:-}" ]; then
    id="dry.$label"
  else
    id=$(env "$@" sbatch --parsable --job-name="$label" \
          $place $res $dep_arg "$SLURM")
  fi
  echo x >> "$TALLY"
  # Progress to stderr, job id to stdout, so `$(submit ...)` captures only the
  # id while the log line survives a caller that discards stdout.
  if [ -n "$dep" ]; then
    printf '  %-14s %-9s %-5s %-10s after %s\n' "$label" "$where" "$kind" "$id" "$dep" >&2
  else
    printf '  %-14s %-9s %-5s %-10s\n' "$label" "$where" "$kind" "$id" >&2
  fi
  echo "$id"
}

echo "=== Phase 0: calibration ===" >&2
if [ -n "${CAL:-}" ]; then
  echo "  reusing queued calibration $CAL" >&2
else
  CAL=$(submit cal housecats cal "" STAGE=calibrate CTX=5)
fi

echo "=== Phase 1: headline arm, 3 seeds ===" >&2
# Seeds 0 and 1 to HouseCats' two cards, seed 2 into the BigCats pool.
G0=$(submit g_s0 housecats train "$CAL" STAGE=grpo SEED=0 CTX=5)
G1=$(submit g_s1 housecats train "$CAL" STAGE=grpo SEED=1 CTX=5)
G2=$(submit g_s2 bigcats   train "$CAL" STAGE=grpo SEED=2 CTX=5)
P0=$(submit p_s0 bigcats light "$G0" STAGE=predict ARM=sc_grpo SEED=0 CTX=5)
submit p_s1 bigcats light "$G1" STAGE=predict ARM=sc_grpo SEED=1 CTX=5 >/dev/null
submit p_s2 bigcats light "$G2" STAGE=predict ARM=sc_grpo SEED=2 CTX=5 >/dev/null

echo "=== Phase 2: ablations ===" >&2
for s in 0 1 2; do
  u=$(submit "unw_s$s" bigcats train "$CAL" STAGE=grpo SEED=$s CTX=5 WEIGHTING=off)
  submit "p_unw_s$s" bigcats light "$u" \
    STAGE=predict ARM=sc_grpo_unw SEED=$s CTX=5 >/dev/null
done
COLD=$(submit cold_s0 bigcats train "$CAL" STAGE=grpo SEED=0 CTX=5 INIT=cold)
submit p_cold_s0 bigcats light "$COLD" \
  STAGE=predict ARM=sc_grpo_cold SEED=0 CTX=5 >/dev/null

echo "=== Phase 3: defence ===" >&2
# The probe sources donor rationales from the arm's own predictions, so it
# follows the prediction rather than the training job.
submit faith_grpo bigcats light "$P0" \
  STAGE=faith ARM=sc_grpo SEED=0 CTX=5 >/dev/null

# ctx3 replication. There is no ctx3 sc_ft_bare adapter yet and GRPO
# initialises from it, so that SFT heads this branch. It needs no calibration:
# it is the supervised recipe already run at ctx5, not the RL loop.
SFT3=$(submit sft3_bare bigcats train "" STAGE=sft TARGET=bare CTX=3)
submit p3_bare bigcats light "$SFT3" STAGE=predict ARM=sc_ft_bare CTX=3 >/dev/null
G3=$(submit g3_s0 bigcats train "$SFT3:$CAL" STAGE=grpo SEED=0 CTX=3)
submit p3_s0 bigcats light "$G3" STAGE=predict ARM=sc_grpo SEED=0 CTX=3 >/dev/null

echo >&2
echo "$(wc -l < "$TALLY" | tr -d ' ') jobs submitted. Watch with:  squeue -u \$USER" >&2
echo "Score everything once they land:  PYTHONPATH=src python -m baseline.eval" >&2
