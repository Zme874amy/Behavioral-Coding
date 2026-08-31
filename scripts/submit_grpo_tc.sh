#!/usr/bin/env bash
# Submit the whole two-call GRPO campaign (cells A and B) for one context length,
# with train -> predict -> recovery -> rescore wired end to end.
#
# The four arms cross regime with credit scheme, each warm-started from the
# matching SFT two-call arm (which must already be trained):
#
#   grpo_pair_dec    2 adapters, decoupled    from ft_bare pair
#   grpo_pair_joint  2 adapters, joint traj   from ft_bare pair
#   grpo_mix_dec     1 adapter,  decoupled    from ft1mix_bare
#   grpo_mix_joint   1 adapter,  joint traj   from ft1mix_bare
#
# Preconditions (NOT submitted here -- train the SFT arms and freeze oof_t1 first):
#   PYTHONPATH=src python -m baseline.local_arm train --target bare --regime pair  --ctx $CTX
#   PYTHONPATH=src python -m baseline.local_arm train --target bare --regime mixed --ctx $CTX
#   bash scripts/submit_tf_axis.sh    # produces the oof_t1 store decoupled T2 needs
#
# Waves, each gated on the last:
#   1. (optional) calibrate pair + mix
#   2. train    4 arms x SEEDS  (+ ablations when ABLATE=1)
#   3. predict  one MIG cell per trained arm         (dep: its train)
#   4. recovery one MIG probe per trained arm + the two SFT baselines
#   5. rescore  baseline.eval                         (dep: all predicts)
#
# Usage:
#   CTX=5 bash scripts/submit_grpo_tc.sh
#   CTX=5 ABLATE=1 bash scripts/submit_grpo_tc.sh          # + weighting-off / cold-start
#   CTX=5 CALIBRATE=1 bash scripts/submit_grpo_tc.sh       # calibrate, submit nothing else
#   DRYRUN=1 CTX=5 bash scripts/submit_grpo_tc.sh          # print, submit nothing
#   CTX=5 SEEDS="0" bash scripts/submit_grpo_tc.sh         # one seed
set -euo pipefail
cd "$(dirname "$0")/.."

CTX="${CTX:-5}"
SEEDS="${SEEDS:-0 1 2}"
DRYRUN="${DRYRUN:-}"
ABLATE="${ABLATE:-}"
CALIBRATE="${CALIBRATE:-}"
SLURM="scripts/mlerp_grpo_tc.slurm"

# arm -> "regime credit". The order is the reading order in the report.
ARMS=("grpo_pair_dec:pair dec" "grpo_pair_joint:pair joint"
      "grpo_mix_dec:mix dec" "grpo_mix_joint:mix joint")

full="--partition=BigCats --qos=lion"                       # 40GB, for train
mig="--partition=BigCats --qos=lion --gres=gpu:3g.20gb:1"   # MIG, for predict/recovery

submit() {  # submit <label> <place> <dep-or-empty> <VAR=val ...>
  local label="$1" place="$2" dep="$3"; shift 3
  local dep_arg="" id
  [ -n "$dep" ] && dep_arg="--dependency=afterok:$dep"
  if [ -n "$DRYRUN" ]; then
    id="dry.$label"
  else
    # shellcheck disable=SC2086
    id=$(env "$@" sbatch --parsable --job-name="$label" $place $dep_arg "$SLURM")
  fi
  printf '  %-32s %-12s %s\n' "$label" "$id" "${dep:+after $dep}" >&2
  echo "$id"
}

submit_after_any() {  # like submit(), but the dependency is afterany (runs even
  local label="$1" place="$2" dep="$3"; shift 3   # if the predecessor timed out)
  local id
  if [ -n "$DRYRUN" ]; then
    id="dry.$label"
  else
    # shellcheck disable=SC2086
    id=$(env "$@" sbatch --parsable --job-name="$label" $place \
         --dependency=afterany:"$dep" "$SLURM")
  fi
  printf '  %-32s %-12s %s\n' "$label" "$id" "afterany $dep" >&2
  echo "$id"
}

echo "=== two-call GRPO campaign, ctx=${CTX}, seeds='${SEEDS}' ===" >&2

if [ -n "$CALIBRATE" ]; then
  echo "--- calibrate ---" >&2
  submit "cal_pair_c${CTX}" "$full" "" STAGE=calibrate REGIME=pair CTX="$CTX" >/dev/null
  submit "cal_mix_c${CTX}"  "$full" "" STAGE=calibrate REGIME=mix  CTX="$CTX" >/dev/null
  echo "Re-cost docs/GRPO_TWOCALL.md against these before the full ladder." >&2
  exit 0
fi

# Optional space-separated list to restrict which arms to submit (e.g. to
# resubmit only the paths that failed without duplicating healthy jobs).
ARMS_FILTER="${ARMS_FILTER:-}"

# Submit the train job(s) for one arm/seed/variant and echo a colon-joined dep of
# their job id(s). Decoupled pair splits into two INDEPENDENT stage jobs (t1, t2),
# each fitting the wall; mix and joint are a single job.
submit_train() {  # <lab> <regime> <credit> <seed> <extra VAR=val ...>
  local lab="$1" regime="$2" credit="$3" seed="$4"; shift 4
  if [ "$regime" = "pair" ] && [ "$credit" = "dec" ]; then
    # Two INDEPENDENT stage jobs (the stages don't depend on each other).
    local a b
    a=$(submit "tr_${lab}_t1" "$full" "" STAGE=train REGIME=pair CREDIT=dec SEED="$seed" TIER=t1 CTX="$CTX" "$@")
    b=$(submit "tr_${lab}_t2" "$full" "" STAGE=train REGIME=pair CREDIT=dec SEED="$seed" TIER=t2 CTX="$CTX" "$@")
    echo "${a}:${b}"
  elif [ "$regime" = "mix" ] && [ "$credit" = "dec" ]; then
    # One shared adapter can't be tier-split, so split it in TIME: MIX_CHAIN
    # chained jobs, each resuming from the last checkpoint after the previous
    # ends (afterany, so a wall-truncated job still hands off). Predict deps the
    # last. A run that finishes early makes the remaining links quick no-ops.
    local prev="" id
    for lnk in $(seq 1 "${MIX_CHAIN:-2}"); do
      if [ -z "$prev" ]; then
        id=$(submit "tr_${lab}_p${lnk}" "$full" "" STAGE=train REGIME=mix CREDIT=dec SEED="$seed" CTX="$CTX" "$@")
      else
        id=$(submit_after_any "tr_${lab}_p${lnk}" "$full" "$prev" STAGE=train REGIME=mix CREDIT=dec SEED="$seed" CTX="$CTX" "$@")
      fi
      prev="$id"
    done
    echo "$prev"
  else
    # Joint is a pure-HF loop (~57s/step at G=8 -> ~24.6h, over the wall). Trim to
    # G=6 and lighter validation so it fits one 24h job; G is exploration breadth,
    # secondary to the decoupled-vs-joint credit-scheme comparison (documented).
    local extra=("$@")
    if [ "$credit" = "joint" ]; then
      extra+=("OVERRIDES=grpo.num_generations=6 grpo.val_every_steps=200 grpo.val_max_examples=64")
    fi
    submit "tr_${lab}" "$full" "" STAGE=train REGIME="$regime" CREDIT="$credit" SEED="$seed" CTX="$CTX" "${extra[@]}"
  fi
}

predict_ids=()

echo "--- train + predict + recovery ---" >&2
for entry in "${ARMS[@]}"; do
  arm="${entry%%:*}"; rc="${entry#*:}"; regime="${rc% *}"; credit="${rc#* }"
  if [ -n "$ARMS_FILTER" ] && ! printf '%s\n' $ARMS_FILTER | grep -qx "$arm"; then
    continue
  fi
  for seed in $SEEDS; do
    # Phase 1: the weighted headline arm.
    tdep=$(submit_train "${arm}_s${seed}_c${CTX}" "$regime" "$credit" "$seed")
    pid=$(submit "pr_${arm}_s${seed}_c${CTX}" "$mig" "$tdep" \
      STAGE=predict ARM="$arm" SEED="$seed" VARIANT=weighted CTX="$CTX")
    predict_ids+=("$pid")
    submit "rc_${arm}_s${seed}_c${CTX}" "$mig" "$tdep" \
      STAGE=recovery ARM="$arm" SEED="$seed" VARIANT=weighted CTX="$CTX" >/dev/null

    if [ -n "$ABLATE" ]; then
      # Phase 2: weighting-off at every seed; cold-start at seed 0 only.
      tdep=$(submit_train "${arm}_unw_s${seed}_c${CTX}" "$regime" "$credit" "$seed" WEIGHTING=off)
      pid=$(submit "pr_${arm}_unw_s${seed}_c${CTX}" "$mig" "$tdep" \
        STAGE=predict ARM="$arm" SEED="$seed" VARIANT=unweighted CTX="$CTX")
      predict_ids+=("$pid")
      if [ "$seed" = "0" ]; then
        tdep=$(submit_train "${arm}_cold_s0_c${CTX}" "$regime" "$credit" 0 INIT=cold)
        pid=$(submit "pr_${arm}_cold_s0_c${CTX}" "$mig" "$tdep" \
          STAGE=predict ARM="$arm" SEED=0 VARIANT=coldstart CTX="$CTX")
        predict_ids+=("$pid")
      fi
    fi
  done
done

if [ -z "$ARMS_FILTER" ]; then
  echo "--- recovery baselines (SFT warm-starts, no training) ---" >&2
  for base in ft_bare ft1mix_bare; do
    submit "rc_${base}_c${CTX}" "$mig" "" STAGE=recovery ARM="$base" CTX="$CTX" >/dev/null
  done
else
  echo "--- ARMS_FILTER set: skipping baseline recovery + auto-rescore (run eval manually) ---" >&2
  echo "=== submitted (filtered). Run 'python -m baseline.eval' after all cells land. ===" >&2
  exit 0
fi

echo "--- rescore (after every prediction lands) ---" >&2
dep=""
[ ${#predict_ids[@]} -gt 0 ] && dep=$(IFS=:; echo "${predict_ids[*]}")
if [ -n "$DRYRUN" ]; then
  echo "  eval  dry.eval  after ${dep:-none}" >&2
elif [ -n "$dep" ]; then
  sbatch --parsable --job-name="eval_grpotc_c${CTX}" $full \
    --dependency=afterok:"$dep" \
    --wrap="source scripts/setup_grpo_env.sh use && PYTHONPATH=src python -m baseline.eval" >&2
fi

echo "=== submitted. Read partial cells as running, not as results. ===" >&2
