#!/bin/bash
# Submit Phases 1-3 of the single-call GRPO ladder as one dependency graph.
#
#   bash scripts/submit_grpo_ladder.sh              # submits calibration too
#   CAL=158198 bash scripts/submit_grpo_ladder.sh   # reuse a queued calibration
#   CAL=none   bash scripts/submit_grpo_ladder.sh   # calibration already passed
#   SFT3=done  bash scripts/submit_grpo_ladder.sh   # ctx3 sc_ft_bare already trained
#   DRYRUN=1   bash scripts/submit_grpo_ladder.sh   # print the graph, submit nothing
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

  # Concurrency is capped per QOS, not per user: lion allows 4 running jobs,
  # panther 4, tabby 1. Eight GRPO runs all sit under lion by default and six
  # of them would idle behind the cap for a full 20-hour wave, so the training
  # jobs are spread across all three. BigCats permits lion and panther alike;
  # panther differs only in a 7-day wall, which nothing here needs.
  case "$where" in
    housecats) place="--partition=HouseCats --qos=tabby" ;;
    bigcats)   place="--partition=BigCats --qos=lion" ;;
    panther)   place="--partition=BigCats --qos=panther" ;;
    *) echo "bad partition $where" >&2; exit 1 ;;
  esac
  case "$kind" in
    # 24h is the ceiling on both the lion and tabby QOS. Calibration measured
    # 17.8 s/prompt, so two epochs over the ~1650-prompt training split is
    # ~16.3h before oversampling and the validation callbacks, which lands near
    # 21h. Anything shorter would wall-clock out; the best-checkpoint callback
    # means a job that does hit the wall still leaves a usable adapter.
    train) res="--gres=gpu:40gb:1 --mem=64G --time=24:00:00" ;;
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
if [ "${CAL:-}" = "none" ]; then
  echo "  calibration already passed; training jobs ungated" >&2
  CAL=""
elif [ -n "${CAL:-}" ]; then
  echo "  reusing queued calibration $CAL" >&2
else
  CAL=$(submit cal housecats cal "" STAGE=calibrate CTX=5)
fi

echo "=== Phase 1: headline arm, 3 seeds ===" >&2
# One seed onto tabby's single slot, the other two onto panther, leaving all
# four lion slots for Phase 2 so that every training job can run concurrently.
G0=$(submit g_s0 housecats train "$CAL" STAGE=grpo SEED=0 CTX=5)
G1=$(submit g_s1 panther   train "$CAL" STAGE=grpo SEED=1 CTX=5)
G2=$(submit g_s2 panther   train "$CAL" STAGE=grpo SEED=2 CTX=5)
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

# ctx3 replication. GRPO initialises from a ctx3 sc_ft_bare adapter, so that
# SFT heads this branch. It needs no calibration: it is the supervised recipe
# already run at ctx5, not the RL loop.
if [ "${SFT3:-}" = "done" ]; then
  echo "  ctx3 sc_ft_bare already trained" >&2
  SFT3=""
else
  SFT3=$(submit sft3_bare bigcats train "" STAGE=sft TARGET=bare CTX=3)
fi
submit p3_bare panther light "$SFT3" STAGE=predict ARM=sc_ft_bare CTX=3 >/dev/null
# Join whichever of the two gates actually exist into one afterok list.
G3DEP=$(echo "$SFT3:$CAL" | sed 's/^://; s/:$//')
G3=$(submit g3_s0 panther train "$G3DEP" STAGE=grpo SEED=0 CTX=3)
submit p3_s0 bigcats light "$G3" STAGE=predict ARM=sc_grpo SEED=0 CTX=3 >/dev/null

echo >&2
echo "$(wc -l < "$TALLY" | tr -d ' ') jobs submitted. Watch with:  squeue -u \$USER" >&2
echo "Score everything once they land:  PYTHONPATH=src python -m baseline.eval" >&2
