#!/bin/bash
# MLeRP: screen a candidate teacher straight from the cluster's SHARED Ollama
# store (/apps/ollama/models, ~310 GB of GGUF, not on our quota) -- no staging.
# Same screen as mlerp_teacher_screen.slurm (selftrain.screen_teachers, greedy
# two-call labelling of HLQC), served by Ollama instead of vLLM.
#
# GGUF via Ollama is fine for a greedy screen, but has no prompt logprobs (so no
# likelihood confidence) and cannot be LoRA-fine-tuned: the winner still gets
# staged as safetensors for those steps.
#
#   OMODEL=qwen3.6:27b-q8_0 THINK=false sbatch scripts/mlerp_teacher_screen_ollama.slurm
#   OMODEL=gpt-oss:120b THINK=low MAXTOK=1024 PARALLEL=4 \
#     sbatch --gres=gpu:40gb:2 scripts/mlerp_teacher_screen_ollama.slurm
#
# Models present: ls /apps/ollama/models/manifests/registry.ollama.ai/library/*/

#SBATCH --job-name=teacher_screen_ol
#SBATCH --partition=BigCats
#SBATCH --qos=lion
#SBATCH --gres=gpu:40gb:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/selftrain/screen_ol_%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "${SLURM_SUBMIT_DIR}/scripts/setup_grpo_env.sh" use
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

OMODEL="${OMODEL:?set OMODEL to an Ollama tag, e.g. gpt-oss:120b}"
NAME="${NAME:-ollama_$(echo "$OMODEL" | tr ':/' '__')}"
THINK="${THINK:-}"            # false | low | medium | high | empty (model default)
MAXTOK="${MAXTOK:-48}"
NUMCTX="${NUMCTX:-12288}"
PARALLEL="${PARALLEL:-8}"     # KV cache is allocated for NUMCTX x PARALLEL tokens
OLLAMA_BIN="${OLLAMA_BIN:-/apps/ollama/bin/ollama-0.24.0}"
OUT="${OUT:-data/selftrain/teacher/screen_${NAME}.csv}"
PORT="${PORT:-$(( 11000 + SLURM_JOB_ID % 1000 ))}"

export OLLAMA_MODELS=/apps/ollama/models OLLAMA_NOPRUNE=1   # shared store is read-only
export OLLAMA_HOST="127.0.0.1:${PORT}" OLLAMA_NUM_PARALLEL="$PARALLEL"
export OLLAMA_CONTEXT_LENGTH="$NUMCTX" OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=-1
export OLLAMA_URL="http://127.0.0.1:${PORT}"
mkdir -p outputs/selftrain data/selftrain/teacher

echo "== ollama serve $OMODEL on :$PORT (parallel=$PARALLEL ctx=$NUMCTX think=${THINK:-default}) =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$OLLAMA_BIN" serve > "outputs/selftrain/ollama_${SLURM_JOB_ID}.log" 2>&1 &
OL_PID=$!
trap 'kill $OL_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
  curl -sf "$OLLAMA_URL/api/version" >/dev/null 2>&1 && break
  kill -0 $OL_PID 2>/dev/null || { echo "ollama exited" >&2; tail -30 "outputs/selftrain/ollama_${SLURM_JOB_ID}.log"; exit 1; }
  sleep 5
done
curl -sf "$OLLAMA_URL/api/version"; echo

echo "== load $OMODEL (warm-up) =="
curl -sf "$OLLAMA_URL/api/generate" -d "{\"model\":\"$OMODEL\",\"prompt\":\"hi\",\"stream\":false,\"options\":{\"num_predict\":1,\"num_ctx\":$NUMCTX}}" >/dev/null \
  || { echo "model failed to load" >&2; tail -40 "outputs/selftrain/ollama_${SLURM_JOB_ID}.log"; exit 1; }
"$OLLAMA_BIN" ps || true   # shows the GPU/CPU split: any CPU share means a slow, offloaded run

PYTHONPATH=src python3 -m selftrain.screen_teachers --backend ollama --model "$OMODEL" \
  ${THINK:+--think "$THINK"} --max-tokens "$MAXTOK" --num-ctx "$NUMCTX" --concurrency "$PARALLEL" \
  --pool data/manual/HLQC_balanced_manual.csv --fewshot --exclude-exemplars ${SCREEN_ARGS:-} --out "$OUT"
echo "== done =="
