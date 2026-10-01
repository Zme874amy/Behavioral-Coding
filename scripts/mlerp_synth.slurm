#!/bin/bash
# MLeRP: serve a large open teacher with vLLM, then run Direction 2 (synthesis)
# and/or Direction 1b (remote pseudo-labeling) against it.
#
# The teacher is served from a LOCAL path (pre-staged into the HF-free model dir
# so the compute node needs no internet). Default: Qwen2.5-32B-Instruct-AWQ
# (4-bit, ~19GB VRAM, fits one A100 40GB; same family as the 7B student).
#
#   STAGE=all sbatch scripts/mlerp_synth.slurm                         # gen + verify (Dir 2)
#   STAGE=generate sbatch scripts/mlerp_synth.slurm
#   STAGE=verify   sbatch scripts/mlerp_synth.slurm
#   STAGE=label POOL=data/parsed/AnnoMI_parsed.csv sbatch scripts/mlerp_synth.slurm  # Dir 1b

#SBATCH --job-name=synth_teacher
#SBATCH --partition=BigCats
#SBATCH --qos=lion
#SBATCH --gres=gpu:40gb:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/synth/teacher_%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# vLLM lives in the GRPO venv (it pins its own torch), reused here for serving.
source "${SLURM_SUBMIT_DIR}/scripts/setup_grpo_env.sh" use
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# The compute node cannot JIT-build flashinfer's sampling kernel (ninja fails),
# which crashed engine startup. Disable flashinfer (sampler + attention) and use
# FLASH_ATTN, which loads prebuilt.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

STAGE="${STAGE:-all}"
MODEL="${MODEL:-/mnt/userdata4/jia-wen/models/Qwen2.5-32B-Instruct-AWQ}"   # vLLM --model (local path)
NAME="${NAME:-$(basename "$MODEL")}"                                        # served-model-name / API model id
QUANT="${QUANT:-awq}"
MAXLEN="${MAXLEN:-8192}"
GPUMEM="${GPUMEM:-0.92}"
PORT="${PORT:-$(( 8000 + SLURM_JOB_ID % 1000 ))}"
export VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export VLLM_API_KEY="EMPTY"
CTX="${CTX:-5}"
K="${K:-1}"
VARIANT="${VARIANT:-proto}"          # proto | boundary
WINDOWS="${WINDOWS:-60}"             # dialogue windows to generate
TURNS="${TURNS:-10}"                 # utterances per window
KEEP_HARD="${KEEP_HARD:-0}"          # 1 = keep verifier-disagreed turns (boundary)
VERIFIER="${VERIFIER:-$NAME}"        # override to break generator/verifier circularity
GEN_SEED="${GEN_SEED:-42}"
MIX="${MIX:-train}"                  # train = HLQC topic mix | eval = MIV6.3A mix
RULE="${RULE:-0}"                    # 1 = generate a SocialDial-style rule and condition on it
TAG="$(echo "$NAME" | tr '/:.' '___')"
STEM="${TAG}_${VARIANT}_${MIX}$([[ "$RULE" == "1" ]] && echo _rule || true)"

mkdir -p outputs/synth data/synth/raw data/synth/aug data/selftrain/pseudo

if [[ ! -d "$MODEL" ]]; then
  echo "Teacher model path not found: $MODEL (pre-stage it first)." >&2; exit 1
fi

echo "== start vLLM server: $MODEL (name=$NAME, quant=$QUANT) on :$PORT =="
# Prefix caching: the label stage scores every candidate code as prompt + " CODE",
# so all candidates of a row share one long prefix that is then computed once.
SERVER_EXTRA="${SERVER_EXTRA:---enable-prefix-caching}"
python3 -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name "$NAME" \
  --port "$PORT" --quantization "$QUANT" --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$GPUMEM" --enforce-eager $SERVER_EXTRA &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

echo "== wait for health (up to 20 min for a 32B load) =="
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then echo "server ready"; break; fi
  sleep 10
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || { echo "vLLM never became healthy" >&2; exit 1; }

run_generate() {
  local rf=""; [[ "$RULE" == "1" ]] && rf="--rule"
  python3 -m synth.generate --variant "$VARIANT" --mix "$MIX" $rf \
    --windows "$WINDOWS" --turns "$TURNS" --model "$NAME" --provider vllm_server \
    --seed "$GEN_SEED" --out "data/synth/raw/${STEM}.csv"
}
run_segment() {
  python3 -m synth.segment --raw "data/synth/raw/${STEM}.csv" \
    --model "$NAME" --provider vllm_server \
    --out "data/synth/raw/${STEM}_seg.csv"
}
run_verify() {
  local kh=""; [[ "$KEEP_HARD" == "1" ]] && kh="--keep-hard"
  python3 -m synth.verify --raw "data/synth/raw/${STEM}_seg.csv" \
    --model "$VERIFIER" --provider vllm_server $kh \
    --train-csv data/manual/HLQC_balanced_manual.csv \
    --eval-csv data/manual/MIV6.3A_manual.csv \
    --out "data/synth/aug/${STEM}.csv"
  python3 -m synth.metrics --real data/manual/HLQC_balanced_manual.csv \
    --set old=data/synth/raw/Qwen2_5-32B-Instruct-AWQ.csv \
    --set "${STEM}=data/synth/raw/${STEM}_seg.csv" || true
}
run_judge() {
  # SDF section 3.1.4: evaluate content BEFORE spending downstream compute.
  python3 -m synth.judge --raw "data/synth/raw/${STEM}_seg.csv" \
    --model "$NAME" --provider vllm_server --limit 40 \
    --real-csv data/manual/HLQC_balanced_manual.csv --real-windows 20 \
    --out "outputs/synth/judge_${STEM}.csv"
}

case "$STAGE" in
  all)      run_generate; run_segment; run_verify; run_judge ;;
  generate) run_generate ;;
  segment)  run_segment ;;
  judge)    run_judge ;;
  segverify) run_segment; run_verify ;;   # reuse existing generated windows
  verify)   run_verify ;;
  label)
    POOL="${POOL:?set POOL for STAGE=label}"
    PTAG="$(basename "$POOL" _parsed.csv)"
    # Teacher labelling. LABEL_ARGS passes self-training-v2 flags through, e.g.
    #   LABEL_ARGS="--confidence likelihood --speakers counsellor --sample-convs 800"
    #   LABEL_ARGS="--confidence likelihood --keep-gold --fewshot --exclude-exemplars"
    LABEL_ARGS="${LABEL_ARGS:-}"
    OUT="${OUT:-data/selftrain/pseudo/${PTAG}_distil_${TAG}_ctx${CTX}.csv}"
    python3 -m selftrain.label_pool --pool "$POOL" --labeler remote \
      --labeler-model "$NAME" --labeler-provider vllm_server --ctx "$CTX" --k "$K" \
      --teacher-tokenizer "$MODEL" $LABEL_ARGS --out "$OUT" ;;
  *) echo "Unknown STAGE=$STAGE (all|generate|segment|segverify|verify|judge|label)" >&2; exit 1 ;;
esac
echo "== done (STAGE=$STAGE) =="
