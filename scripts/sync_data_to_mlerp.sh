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

# Strudel2 issues short-lived (~28 day) SSH certificates. When one expires the
# only symptom is "Permission denied (publickey)", which looks identical to a key
# that was never registered, so check the expiry up front and say so plainly.
CERT="${MLERP_CERT:-$HOME/.ssh/strudel2/strudel2_ssh_key-cert.pub}"
if [[ -f "$CERT" ]]; then
  valid_to=$(ssh-keygen -L -f "$CERT" 2>/dev/null \
    | awk '/Valid:/ {print $5}')
  if [[ -n "$valid_to" && "$valid_to" != "forever" ]]; then
    # Certificate stamps are UTC "YYYY-MM-DDTHH:MM:SS".
    if [[ "$(date -u +%Y-%m-%dT%H:%M:%S)" > "$valid_to" ]]; then
      echo "ERROR: Strudel2 SSH certificate expired at ${valid_to} (UTC)." >&2
      echo "       $CERT" >&2
      echo "  Renew it by logging in to https://strudel2.cloud.edu.au and" >&2
      echo "  starting an MLeRP session; that reissues the key and certificate." >&2
      echo "  Until then every ssh/scp here fails as 'Permission denied (publickey)'." >&2
      exit 1
    fi
    echo "Strudel2 certificate valid until ${valid_to} (UTC)"
  fi
fi

echo "Creating remote data dirs on ${MLERP_SSH}:${REMOTE_REPO}"
ssh "${MLERP_SSH}" "mkdir -p ${REMOTE_REPO}/data/manual ${REMOTE_REPO}/data/fewshot \
  ${REMOTE_REPO}/data/fine_tuning/rationales"

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

# Distilled rationale targets for the ft_rat arm. These are generated against
# Azure gpt-4o (so they are produced wherever Azure works, not on the GPU node)
# and are frozen per context length; copying them avoids paying for the same
# generation twice and guarantees both tiers train on identical targets.
rationale_files=(data/fine_tuning/rationales/hlqc_rationales_ctx*.json)
if (( ${#rationale_files[@]} )); then
  echo "Copying distilled rationale targets (${rationale_files[*]})..."
  scp "${rationale_files[@]}" "${MLERP_SSH}:${REMOTE_REPO}/data/fine_tuning/rationales/"
else
  echo "WARN: no data/fine_tuning/rationales/hlqc_rationales_ctx*.json found"
  echo "      (needed for the ft_rat arm; build with 'python -m baseline.rationalize --ctx N')"
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
