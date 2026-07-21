#!/bin/bash
# MLeRP batch run for the AutoMISC baseline reproduction (Azure OpenAI gpt-4o).
# Runs all four prompting conditions in parallel (they are API-bound, no GPU
# needed) and then the evaluation. Submit from repo root:
#
#   sbatch scripts/mlerp_baseline.slurm                # ctx=5 (default)
#   CTX=3 sbatch scripts/mlerp_baseline.slurm          # 3 context volleys
#
# Run a single condition instead:
#   CONDITIONS=fewshot_bare CTX=3 sbatch scripts/mlerp_baseline.slurm
#
# Prerequisites on MLeRP (see scripts/sync_data_to_mlerp.sh):
#   - data/manual/*.csv, data/fewshot/exemplars.json
#   - .env with AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY
#   - bash scripts/env.sh setup   (installs requirements-extras.txt)
#
# Each condition checkpoints to data/annotated/baseline/<condition>.csv and
# resumes automatically, so requeued/partial jobs are safe.

#SBATCH --job-name=automisc_baseline
#SBATCH --partition=BigCats
#SBATCH --qos=lion
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/baseline_runs/%j.log

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source "${SLURM_SUBMIT_DIR}/scripts/env.sh"

bash "${SLURM_SUBMIT_DIR}/scripts/check_data.sh"
if [[ ! -f .env ]]; then
  echo "Missing .env with Azure credentials (sync it with scripts/sync_data_to_mlerp.sh)" >&2
  exit 1
fi
mkdir -p outputs/baseline_runs data/annotated/baseline

CONDITIONS="${CONDITIONS:-zeroshot_rationales zeroshot_bare fewshot_rationales fewshot_bare}"
CTX="${CTX:-5}"

pids=()
for cond in $CONDITIONS; do
  echo "Launching condition: $cond (ctx=${CTX})"
  python3 -m baseline.main "condition=${cond}" "num_context_turns=${CTX}" \
    > "outputs/baseline_runs/${SLURM_JOB_ID}_${cond}_ctx${CTX}.log" 2>&1 &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done

if (( fail )); then
  echo "One or more conditions failed; see outputs/baseline_runs/${SLURM_JOB_ID}_*.log" >&2
  exit 1
fi

echo "All conditions complete. Running evaluation..."
python3 -m baseline.eval
