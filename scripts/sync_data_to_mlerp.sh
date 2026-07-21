#!/usr/bin/env bash
# Copy human evaluation labels to MLeRP (run from your Mac, repo root).
#
# Usage:
#   bash scripts/sync_data_to_mlerp.sh
#   MLERP_SSH=user@host bash scripts/sync_data_to_mlerp.sh
#
# MLERP_SSH is any ssh destination or ~/.ssh/config alias; the default is the
# Strudel2 alias which carries the right key and login node.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$REPO_ROOT"

MLERP_SSH="${MLERP_SSH:-jia-wen_MLeRP_Monash}"
REMOTE_REPO="${REMOTE_REPO:-/mnt/userdata4/jia-wen/Behavioral-Coding}"

FILES=(
  "data/manual/MIV6.3A_manual.csv"
  "data/manual/HLQC_balanced_manual.csv"
)

for f in "${FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing local file: $f" >&2
    exit 1
  fi
done

echo "Creating remote data dirs on ${MLERP_SSH}:${REMOTE_REPO}"
ssh "${MLERP_SSH}" "mkdir -p ${REMOTE_REPO}/data/manual ${REMOTE_REPO}/data/fewshot"

echo "Copying manual evaluation CSVs..."
scp "${FILES[@]}" "${MLERP_SSH}:${REMOTE_REPO}/data/manual/"

# Baseline reproduction extras: frozen few-shot exemplars + Azure credentials.
shopt -s nullglob
exemplar_files=(data/fewshot/exemplars*.json)
if (( ${#exemplar_files[@]} )); then
  echo "Copying few-shot exemplars (${exemplar_files[*]})..."
  scp "${exemplar_files[@]}" "${MLERP_SSH}:${REMOTE_REPO}/data/fewshot/"
else
  echo "WARN: no data/fewshot/exemplars*.json found (needed for few-shot baseline runs)"
fi
if [[ -f ".env" ]]; then
  echo "Copying .env (Azure credentials)..."
  scp ".env" "${MLERP_SSH}:${REMOTE_REPO}/.env"
else
  echo "WARN: .env not found (needed for Azure OpenAI baseline runs)"
fi

echo "Verifying on MLeRP..."
ssh "${MLERP_SSH}" "wc -l ${REMOTE_REPO}/data/manual/*.csv"

echo "Done. On MLeRP run:  bash scripts/check_data.sh"
