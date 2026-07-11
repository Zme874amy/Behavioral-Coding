#!/usr/bin/env bash
# Copy human evaluation labels to MLeRP (run from your Mac, repo root).
#
# Usage:
#   bash scripts/sync_data_to_mlerp.sh
#   MLERP_USER=you MLERP_HOST=login.example.edu bash scripts/sync_data_to_mlerp.sh
#
# Defaults match a typical MLeRP layout; override MLERP_* if yours differs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$REPO_ROOT"

MLERP_USER="${MLERP_USER:-jia-wen}"
MLERP_HOST="${MLERP_HOST:-mlerp.cloud.edu.au}"
REMOTE_REPO="${REMOTE_REPO:-/mnt/userdata4/${MLERP_USER}/Behavioral-Coding}"

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

echo "Creating remote data dirs on ${MLERP_USER}@${MLERP_HOST}:${REMOTE_REPO}"
ssh "${MLERP_USER}@${MLERP_HOST}" "mkdir -p ${REMOTE_REPO}/data/manual ${REMOTE_REPO}/data/fewshot"

echo "Copying manual evaluation CSVs..."
scp "${FILES[@]}" "${MLERP_USER}@${MLERP_HOST}:${REMOTE_REPO}/data/manual/"

# Baseline reproduction extras: frozen few-shot exemplars + Azure credentials.
if [[ -f "data/fewshot/exemplars.json" ]]; then
  echo "Copying few-shot exemplars..."
  scp "data/fewshot/exemplars.json" "${MLERP_USER}@${MLERP_HOST}:${REMOTE_REPO}/data/fewshot/"
else
  echo "WARN: data/fewshot/exemplars.json not found (needed for few-shot baseline runs)"
fi
if [[ -f ".env" ]]; then
  echo "Copying .env (Azure credentials)..."
  scp ".env" "${MLERP_USER}@${MLERP_HOST}:${REMOTE_REPO}/.env"
else
  echo "WARN: .env not found (needed for Azure OpenAI baseline runs)"
fi

echo "Verifying on MLeRP..."
ssh "${MLERP_USER}@${MLERP_HOST}" "wc -l ${REMOTE_REPO}/data/manual/*.csv"

echo "Done. On MLeRP run:  bash scripts/check_data.sh"
