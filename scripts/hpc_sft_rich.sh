#!/usr/bin/env bash
# Run on M3 HPC (CUDA). Requires: huggingface-cli login + Gemma license accepted.
# Usage: bash scripts/hpc_sft_rich.sh [preset] [output_dir]
#   preset: qwen7b | gemma3n_e4b | gemma2_9b | qwen05b  (default: qwen7b)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false

PRESET="${1:-qwen7b}"
OUT_DIR="${2:-outputs/sft_runs/${PRESET}_rich}"

case "${PRESET}" in
  qwen7b) MODEL_ID="Qwen/Qwen2.5-7B-Instruct" ;;
  qwen05b) MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct" ;;
  gemma3n_e4b) MODEL_ID="google/gemma-3n-E4B-it" ;;
  gemma2_9b) MODEL_ID="google/gemma-2-9b-it" ;;
  *) echo "Unknown preset: ${PRESET}"; exit 1 ;;
esac

echo "=== Preflight: ${MODEL_ID} ==="
python src/sft/preflight_hf.py --model "${MODEL_ID}" \
  $( [[ "${PRESET}" == gemma* ]] && echo --trust-remote-code ) || true

echo "=== SFT train + eval: ${PRESET} -> ${OUT_DIR} ==="
python src/sft/main.py \
  "+sft_models=${PRESET}" \
  sft_device=cuda \
  hydra.run.dir="${OUT_DIR}"

echo "Done. Metrics: ${OUT_DIR}/sft_evaluation/metrics.json"
