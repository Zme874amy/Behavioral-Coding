#!/bin/bash
# MLeRP: serve ONE candidate teacher model with vLLM and screen it on labelled
# HLQC (selftrain.screen_teachers): fast batched greedy labelling, then accuracy,
# macro-F1 with clustered CIs, and per-code precision on the tail codes.
#
# The model must already be staged to a local path (compute nodes are offline):
#   bash scripts/stage_model.sh google/gemma-4-31B-it-qat-w4a16-ct
#
#   MODEL=/mnt/userdata4/jia-wen/models/gemma-4-31B-it-qat-w4a16-ct \
#     sbatch scripts/mlerp_teacher_screen.slurm
#   # 2-GPU models (tensor parallel) -- request both cards on the node:
#   MODEL=/mnt/userdata4/jia-wen/models/gpt-oss-120b TP=2 \
#     SCREEN_ARGS="--reasoning low --max-tokens 1024" \
#     sbatch --gres=gpu:40gb:2 scripts/mlerp_teacher_screen.slurm
#
# QUANT is left empty by default: vLLM reads the quantization (FP8, GPTQ,
# compressed-tensors, MXFP4) from the checkpoint's own config.

#SBATCH --job-name=teacher_screen
#SBATCH --partition=BigCats
#SBATCH --qos=lion
#SBATCH --gres=gpu:40gb:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/selftrain/screen_%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "${SLURM_SUBMIT_DIR}/scripts/setup_grpo_env.sh" use
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Same compute-node workaround as mlerp_synth.slurm: flashinfer cannot JIT here.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

MODEL="${MODEL:?set MODEL to a staged local model dir}"
NAME="${NAME:-$(basename "$MODEL")}"
QUANT="${QUANT:-}"
TP="${TP:-1}"
MAXLEN="${MAXLEN:-12288}"
GPUMEM="${GPUMEM:-0.90}"
SERVER_EXTRA="${SERVER_EXTRA:---enable-prefix-caching}"
SCREEN_ARGS="${SCREEN_ARGS:-}"
OUT="${OUT:-data/selftrain/teacher/screen_${NAME}.csv}"
PORT="${PORT:-$(( 8000 + SLURM_JOB_ID % 1000 ))}"
export VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1" VLLM_API_KEY="EMPTY"

[[ -d "$MODEL" ]] || { echo "Model path not found: $MODEL (stage it first)" >&2; exit 1; }
mkdir -p outputs/selftrain data/selftrain/teacher

echo "== serve $NAME (tp=$TP quant=${QUANT:-auto}) on :$PORT =="
python3 -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name "$NAME" \
  --port "$PORT" ${QUANT:+--quantization "$QUANT"} --tensor-parallel-size "$TP" \
  --max-model-len "$MAXLEN" --gpu-memory-utilization "$GPUMEM" --enforce-eager $SERVER_EXTRA &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

echo "== wait for health (up to 30 min) =="
for _ in $(seq 1 180); do
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { echo "server ready"; break; }
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM exited during startup" >&2; exit 1; }
  sleep 10
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || { echo "vLLM never became healthy" >&2; exit 1; }

PYTHONPATH=src python3 -m selftrain.screen_teachers --model "$NAME" \
  --pool data/manual/HLQC_balanced_manual.csv --fewshot --exclude-exemplars \
  $SCREEN_ARGS --out "$OUT"
echo "== done =="
