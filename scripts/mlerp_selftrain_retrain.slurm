#!/bin/bash
# MLeRP: Direction 1 self-training RETRAIN + PREDICT on an augmented CSV.
#
# Trains a fresh ft_bare adapter pair on the augmented training set (human +
# accepted pseudo-labels) into a SEPARATE adapter tree, then predicts on the 821
# gold eval set into a SEPARATE output dir, so neither the baseline adapters nor
# the baseline predictions are touched. Score the resulting CSV separately (a
# standalone macro-F1 over gold T2), as with the ag arms.
#
#   AUG=data/selftrain/aug/welivita_mi_ft_bare_ctx5.csv \
#   ADAPTER=data/fine_tuning/selftrain_welivita_mi \
#   sbatch scripts/mlerp_selftrain_retrain.slurm

#SBATCH --job-name=selftrain_retrain
#SBATCH --partition=BigCats
#SBATCH --qos=lion
#SBATCH --gres=gpu:40gb:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/selftrain/retrain_%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "${SLURM_SUBMIT_DIR}/scripts/env.sh"
export TOKENIZERS_PARALLELISM=false

CTX="${CTX:-5}"
EPOCHS="${EPOCHS:-3}"
# Backbone cell. ft_bare (pair, 2 adapters) scores T2 macro-F1 0.458; ft1mix_bare
# (mixed, 1 adapter) scores 0.496 -- the stronger point estimate, so augmentation
# should be tested there too, not only on pair. NOTE: adapter_root() reads
# paths.adapter_dir for pair but paths.adapter1_dir for mixed/sequential, so the
# override key must follow the regime or the run silently lands in (and clobbers)
# the shared baseline_local_1a tree.
REGIME="${REGIME:-pair}"             # pair | mixed | sequential
case "$REGIME" in
  pair)       ARM=ft_bare;     ADAPTER_KEY=paths.adapter_dir ;;
  mixed)      ARM=ft1mix_bare; ADAPTER_KEY=paths.adapter1_dir ;;
  sequential) ARM=ft1seq_bare; ADAPTER_KEY=paths.adapter1_dir ;;
  *) echo "REGIME must be pair|mixed|sequential" >&2; exit 1 ;;
esac
AUG="${AUG:?set AUG=data/selftrain/aug/<file>.csv}"
ADAPTER="${ADAPTER:-data/fine_tuning/selftrain_welivita_mi}"
OUT="${OUT:-data/annotated/selftrain}"
# Training seed. The conversation-clustered bootstrap covers eval-set variance
# only; seed variance is a separate source and is what SEED_NOISE_FLOOR=0.08 was
# derived from. Multi-seed runs are the definitive check.
SEED="${SEED:-42}"
# Extra OmegaConf overrides for the train step, space-separated. E.g.
#   EXTRA="training.save_strategy=no"
# skips HF's final checkpoint-<N>/ (a duplicate adapter + ~0.3 GB optimizer
# state per run), which filled the userdata4 quota on 2026-09-28.
EXTRA="${EXTRA:-}"

mkdir -p outputs/selftrain "$OUT"

echo "== train $ARM (regime=$REGIME) on augmented set (epochs=$EPOCHS) =="
# local_arm takes OmegaConf overrides as POSITIONAL dotlist args (not a flag).
python3 -m baseline.local_arm train --target bare --regime "$REGIME" --ctx "$CTX" \
  "dataset.train_csv=${AUG}" "${ADAPTER_KEY}=${ADAPTER}" \
  "training.num_train_epochs=${EPOCHS}" "training.seed=${SEED}" $EXTRA

echo "== predict on 821 gold with the retrained adapters =="
python3 -m baseline.local_arm predict --arm "$ARM" --inf bare --ctx "$CTX" \
  "${ADAPTER_KEY}=${ADAPTER}" "paths.output_dir=${OUT}"

echo "Prediction CSV -> ${OUT}/ (score with selftrain.score against gold t2)"
