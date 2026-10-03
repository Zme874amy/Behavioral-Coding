#!/bin/bash
# Stage a Hugging Face model into /mnt/userdata4/jia-wen/models/<name> on the
# MLeRP LOGIN node (compute nodes are offline), without leaving a second copy in
# the Hugging Face Xet chunk cache -- on a ~112 GB user quota a 30 GB model must
# not cost 60 GB.
#
#   bash scripts/stage_model.sh google/gemma-4-31B-it-qat-w4a16-ct
#   bash scripts/stage_model.sh openai/gpt-oss-120b   # large: free space first
#
# Run it from a file (as here), never by pattern-killing: see the 2026-10-03 note.
set -euo pipefail
REPO="${1:?usage: stage_model.sh <hf-repo-id> [local-name]}"
NAME="${2:-$(basename "$REPO")}"
DEST="/mnt/userdata4/jia-wen/models/${NAME}"
source "$(dirname "$0")/setup_grpo_env.sh" use
# The grpo env points HF_HOME at /home, where no token lives; gated repos (Gemma)
# need the one saved under userdata4.
TOKEN_FILE=/mnt/userdata4/jia-wen/.cache/huggingface/token
[[ -s "$TOKEN_FILE" ]] && export HF_TOKEN="$(cat "$TOKEN_FILE")"
# Weight duplicates some repos ship (gpt-oss: original/ and metal/ copies; GGUF).
EXCLUDES=("*.gguf" "original/*" "metal/*")

need_gb=$(python3 - "$REPO" "${EXCLUDES[@]}" <<'PY'
import fnmatch, sys
from huggingface_hub import HfApi
repo, excl = sys.argv[1], sys.argv[2:]
info = HfApi().model_info(repo, files_metadata=True)
keep = [s for s in info.siblings if not any(fnmatch.fnmatch(s.rfilename, e) for e in excl)]
print(round(sum((s.size or 0) for s in keep) / 1e9, 1))
PY
)
used_gb=$(du -xs /mnt/userdata4/jia-wen 2>/dev/null | awk '{printf "%.1f", $1/1048576}')
echo "repo $REPO needs ~${need_gb} GB (after excludes); user dir ${used_gb} GB (quota ~112 GB)"
python3 -c "import sys; sys.exit(0 if float('$used_gb') + float('$need_gb') < 108 else 1)" \
  || { echo "Not enough headroom for $REPO -- delete a staged model first." >&2; exit 1; }

# Each pattern needs its own --exclude flag: extra positional values would be read
# as FILENAMES to fetch, silently downloading only metadata.
ex_args=(); for e in "${EXCLUDES[@]}"; do ex_args+=(--exclude "$e"); done
HF_HUB_DISABLE_XET=1 hf download "$REPO" --local-dir "$DEST" "${ex_args[@]}"
rm -rf /mnt/userdata4/jia-wen/.cache/huggingface/xet /home/jia-wen/.cache/huggingface/xet 2>/dev/null || true

n_w=$(find "$DEST" -name "*.safetensors" | wc -l)
if [[ "$n_w" -eq 0 ]]; then
  echo "FAILED: no .safetensors weights in $DEST (gated repo without access? network?)" >&2
  exit 1
fi
echo "staged -> $DEST ($n_w weight files, $(du -sh "$DEST" | cut -f1))"
