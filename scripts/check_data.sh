#!/usr/bin/env bash
# Verify required evaluation CSVs exist before SFT / automisc_ft runs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$REPO_ROOT"

REQUIRED=(
  "data/manual/MIV6.3A_manual.csv"
)
OPTIONAL=(
  "data/manual/HLQC_balanced_manual.csv"
)

missing=0
for f in "${REQUIRED[@]}"; do
  if [[ -f "$f" ]]; then
    n=$(($(wc -l < "$f") - 1))
    echo "OK  $f  ($n data rows)"
  else
    echo "MISSING (required): $f" >&2
    missing=1
  fi
done
for f in "${OPTIONAL[@]}"; do
  if [[ -f "$f" ]]; then
    n=$(($(wc -l < "$f") - 1))
    echo "OK  $f  ($n data rows)"
  else
    echo "WARN (optional for automisc_ft): $f not found"
  fi
done

if (( missing )); then
  echo >&2
  echo "Manual labels are gitignored by default except data/manual/*.csv." >&2
  echo "If you just cloned on MLeRP, run:  git pull" >&2
  echo "Or from your Mac:  bash scripts/sync_data_to_mlerp.sh" >&2
  exit 1
fi

echo "Data check passed."
