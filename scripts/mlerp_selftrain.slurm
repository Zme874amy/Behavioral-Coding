#!/bin/bash
# MLeRP: Direction 1 (self-training) labeling + selection for one pool corpus.
#
# Runs the STUDENT labeler (1a): pseudo-labels a MISC-unlabeled pool with the
# current ft_bare adapters via self-consistency, then selects confident rows and
# writes an augmented training CSV. The retrain step reuses the normal training
# job (see the printed command / scripts/mlerp_local_train.slurm) with
# dataset.train_csv pointed at the augmented CSV.
#
# The stronger-model labeler (1b) is NOT run here: it needs a served teacher, so
# use scripts/mlerp_synth.slurm (STAGE=label) which starts the vLLM endpoint and
# calls `selftrain.label_pool --labeler remote` against it.
#
# Submit from repo root (POOL is a parsed CSV in the standard schema):
#   POOL=data/parsed/AnnoMI_parsed.csv sbatch scripts/mlerp_selftrain.slurm
#   POOL=data/parsed/MIV6.3B_parsed.csv CTX=5 K=5 MIN_CONF=0.8 sbatch scripts/mlerp_selftrain.slurm
#
# For an EXTERNAL corpus, first ingest it into the parsed schema (conv_id,
# speaker, utt_text, corp_utt_idx, ...) and pass its path as POOL.

#SBATCH --job-name=selftrain_label
#SBATCH --partition=BigCats
#SBATCH --qos=lion
#SBATCH --gres=gpu:40gb:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/selftrain/label_%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "${SLURM_SUBMIT_DIR}/scripts/env.sh"
export TOKENIZERS_PARALLELISM=false

POOL="${POOL:?set POOL=data/parsed/<corpus>_parsed.csv}"
ARM="${ARM:-ft_bare}"
CTX="${CTX:-5}"
K="${K:-5}"
TEMP="${TEMP:-0.7}"
MIN_CONF="${MIN_CONF:-0.8}"
PER_CLASS_CAP="${PER_CLASS_CAP:-300}"
TAG="$(basename "$POOL" _parsed.csv)"
PSEUDO="data/selftrain/pseudo/${TAG}_${ARM}_ctx${CTX}.csv"
AUG="data/selftrain/aug/${TAG}_${ARM}_ctx${CTX}.csv"

mkdir -p outputs/selftrain data/selftrain/pseudo data/selftrain/aug

echo "== label pool (self) =="
python3 -m selftrain.label_pool --pool "$POOL" --arm "$ARM" --ctx "$CTX" \
  --k "$K" --temperature "$TEMP" --out "$PSEUDO"

echo "== select =="
python3 -m selftrain.select --pseudo "$PSEUDO" \
  --train-csv data/manual/HLQC_balanced_manual.csv \
  --out "$AUG" --min-confidence "$MIN_CONF" --per-class-cap "$PER_CLASS_CAP"

echo "Augmented train CSV -> $AUG"
echo "Retrain with (separate GPU job):"
echo "  TARGET=bare REGIME=pair CTX=${CTX} sbatch scripts/mlerp_local_train.slurm \\"
echo "    # after adding overrides: dataset.train_csv=${AUG} paths.adapter_dir=data/fine_tuning/selftrain_${TAG}"
