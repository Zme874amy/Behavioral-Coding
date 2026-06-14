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

echo "Creating remote data/manual/ on ${MLERP_USER}@${MLERP_HOST}:${REMOTE_REPO}"
ssh "${MLERP_USER}@${MLERP_HOST}" "mkdir -p ${REMOTE_REPO}/data/manual"

echo "Copying manual evaluation CSVs..."
scp "${FILES[@]}" "${MLERP_USER}@${MLERP_HOST}:${REMOTE_REPO}/data/manual/"

echo "Verifying on MLeRP..."
ssh "${MLERP_USER}@${MLERP_HOST}" "wc -l ${REMOTE_REPO}/data/manual/*.csv"

echo "Done. On MLeRP run:  bash scripts/check_data.sh"
