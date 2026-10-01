#!/usr/bin/env bash
# Submit the whole teacher-forcing campaign for one context length, with the
# dependency graph wired end to end.
#
# The axis: every two-call arm trains T2 conditioned on the GOLD Tier-1 label but
# runs it conditioned on the PREDICTED one, so roughly a fifth of T2 calls meet a
# group spec they never saw in training. Three arms vary that one thing.
#
#   ft_bare_tffull   p_gold = 1 throughout        the control
#   ft_bare_tfdyn    p_gold decays 1 -> 0         scheduled sampling
#   ft_bare_tfzero   p_gold = 0 throughout        always the predicted T1
#
# Four waves, each gated on the last:
#
#   1. K fold adapters      T1-only, each trained on K-1 folds
#   2. K fold predictions   each predicts its held-out fold into one shared JSON
#   3. 3 T2 trainings       staged, one 1-epoch stage per p_gold in the schedule
#   4. 3 prediction cells   Inf-Bare only, then a rescore
#
# Usage:
#   CTX=5 bash scripts/submit_tf_axis.sh
#   CTX=3 bash scripts/submit_tf_axis.sh
#   DRYRUN=1 CTX=5 bash scripts/submit_tf_axis.sh      # print, submit nothing
#   SKIP_OOF=1 CTX=5 bash scripts/submit_tf_axis.sh    # predictions already frozen
#
# Split around the calibration check, which is the recommended way to run it:
#   CTX=5 TF_MODES="" bash scripts/submit_tf_axis.sh     # waves 1-2 only
#   PYTHONPATH=src python -m baseline.oof_t1 report --ctx 5
#   CTX=5 SKIP_OOF=1 bash scripts/submit_tf_axis.sh      # waves 3-4
#
# Run ctx5 to completion before ctx3. The out-of-fold calibration check between
# waves 2 and 3 is what decides whether the campaign's numbers are estimates or
# only lower bounds, and it is cheaper to learn that once.
set -euo pipefail
cd "$(dirname "$0")/.."

CTX="${CTX:-5}"
FOLDS="${FOLDS:-5}"
DRYRUN="${DRYRUN:-}"
SKIP_OOF="${SKIP_OOF:-}"
# `-` not `:-`, deliberately: TF_MODES="" must mean "no T2 arms, out-of-fold
# stages only", which is how the campaign is split around the calibration
# check. Same convention as REGIMES in submit_local_grid.sh.
TF_MODES="${TF_MODES-full dyn zero}"
# Which structure hosts the axis. pair = cell A (2 adapters, ft_bare_tf*),
# mixed = cell B (1 adapter, ft1mix_bare_tf*). `sequential` is not offered: its
# T2 stage overwrites the T1 stage, which would confound the axis with
# catastrophic forgetting.
REGIME="${REGIME:-pair}"
case "$REGIME" in
  pair)  ARM_BASE=ft_bare ;;
  mixed) ARM_BASE=ft1mix_bare ;;
  *) echo "REGIME must be pair or mixed, got '$REGIME'" >&2; exit 1 ;;
esac

# Everything goes to lion, which caps at 4 concurrent jobs.
#
# Do NOT spread these across panther to dodge that cap, however tempting: panther
# is capped at `gres/gpu=0` (`sacctmgr show qos`), so a GPU job submitted there
# never starts -- it sits PENDING with reason QOSMaxGRESPerJob indefinitely
# rather than failing. tabby (HouseCats) allows GPUs but only 1 job at a time and
# is provisioned with 20GB MIG slices, which OOM on a 7B model.
place_for() {
  echo "--partition=BigCats --qos=lion"
}

submit() {  # submit <label> <place> <dep-or-empty> <script> <VAR=val ...>
  local label="$1" place="$2" dep="$3" script="$4"; shift 4
  local dep_arg="" id
  [ -n "$dep" ] && dep_arg="--dependency=afterok:$dep"
  if [ -n "$DRYRUN" ]; then
    id="dry.$label"
  else
    # shellcheck disable=SC2086  # place and dep_arg must word-split
    id=$(env "$@" sbatch --parsable --job-name="$label" $place $dep_arg "$script")
  fi
  printf '  %-24s %-10s %s\n' "$label" "$id" "${dep:+after $dep}" >&2
  echo "$id"
}

echo "=== teacher-forcing axis, ${ARM_BASE} (regime=${REGIME}), ctx=${CTX} ===" >&2

train_dep=""
if [ -z "$SKIP_OOF" ]; then
  echo "--- wave 1: out-of-fold T1 adapters ---" >&2
  fold_train_ids=()
  for k in $(seq 0 $((FOLDS - 1))); do
    fold_train_ids+=("$(submit "ooft1_tr_f${k}_c${CTX}" "$(place_for "$k")" "" \
      scripts/mlerp_oof_t1.slurm CTX="$CTX" FOLDS="$FOLDS" FOLD="$k" STAGE=train)")
  done

  echo "--- wave 2: out-of-fold T1 predictions ---" >&2
  # Each predict waits only on its OWN fold's adapter. They all append to one
  # shared JSON, which is checkpointed and resumes, so ordering does not matter.
  fold_pred_ids=()
  for k in $(seq 0 $((FOLDS - 1))); do
    fold_pred_ids+=("$(submit "ooft1_pr_f${k}_c${CTX}" "$(place_for "$k")" \
      "${fold_train_ids[$k]}" scripts/mlerp_oof_t1.slurm \
      CTX="$CTX" FOLDS="$FOLDS" FOLD="$k" STAGE=predict)")
  done
  train_dep=$(IFS=:; echo "${fold_pred_ids[*]}")
fi

# Waves 3 and 4 interleave per mode: each arm's prediction cell depends only on
# its own training, so there is no reason to make them wait on each other.
# (Kept free of associative arrays so this runs under bash 3.2 as well as 4+.)
echo "--- waves 3-4: staged T2 training, then its prediction cell ---" >&2
echo "    CHECK FIRST once wave 2 lands:" >&2
echo "      PYTHONPATH=src python -m baseline.oof_t1 report --ctx ${CTX}" >&2
i=0
for mode in $TF_MODES; do
  # tf=full never reads the predicted-T1 store, so it does not wait for it.
  dep="$train_dep"
  [ "$mode" = "full" ] && dep=""
  train_id=$(submit "tf_${mode}_train_c${CTX}" "$(place_for "$i")" "$dep" \
    scripts/mlerp_local_train.slurm TARGET=bare REGIME="$REGIME" CTX="$CTX" TF_MODE="$mode")
  submit "tf_${mode}_pred_c${CTX}" "$(place_for "$i")" "$train_id" \
    scripts/mlerp_local_predict.slurm ARM="${ARM_BASE}_tf${mode}" INF=bare CTX="$CTX" >/dev/null
  i=$((i + 1))
done

echo >&2
echo "Submitted. mlerp_local_predict.slurm reruns baseline.eval as each cell" >&2
echo "lands, so docs/BASELINE_RESULTS.md updates itself." >&2
