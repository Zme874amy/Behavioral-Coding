# AutoMISC Baseline Reproduction — Runbook (MLeRP)

Reproduces the AutoMISC gpt-4o baseline on Azure OpenAI with a 2x2 prompting
matrix plus a fine-tuned condition, evaluated against human consensus labels.

## Experiment matrix

All conditions: Azure `gpt-4o` deployment, temperature 0, hierarchical T1→T2
prompts, interval context of 5 turns.

| Condition | Prompt | Shots |
|---|---|---|
| `zeroshot_rationales` | original AutoMISC (explanation + label) | 0 |
| `zeroshot_bare` | label-only | 0 |
| `fewshot_rationales` | explanation + label | stratified HLQC exemplars |
| `fewshot_bare` | label-only | stratified HLQC exemplars |
| `finetuned` | label-only, zero-shot | fine-tuned gpt-4o |

- **Eval set:** `data/manual/MIV6.3A_manual.csv` (821 utterances, `t1_label_GT`/`t2_label_GT`).
- **Few-shot exemplars + fine-tuning data:** `data/manual/HLQC_balanced_manual.csv` (held out; no leakage).
- Few-shot exemplars are frozen in `data/fewshot/exemplars.json` (9 T1 + 32 T2,
  one per code; rationales generated once by gpt-4o conditioned on the gold label).
  Rebuild only if needed: `python3 -m baseline.fewshot`.

## One-time setup

From your Mac (repo root) — copies manual CSVs, few-shot exemplars, and `.env`
(Azure credentials) to MLeRP:

```bash
bash scripts/sync_data_to_mlerp.sh
```

On MLeRP:

```bash
cd /mnt/userdata4/$USER/Behavioral-Coding   # or your repo path
git pull
bash scripts/env.sh setup                   # installs requirements-extras.txt
```

## Run the four prompting conditions

```bash
sbatch scripts/mlerp_baseline.slurm
```

This runs all four conditions in parallel (API-bound, no GPU) and then the
evaluation. Each condition writes `data/annotated/baseline/<condition>.csv`
with checkpoint/resume on `corp_utt_idx`, so a requeued job continues where it
stopped. A single condition: `CONDITIONS=fewshot_bare sbatch scripts/mlerp_baseline.slurm`.

Interactive alternative (login/compute node, no sbatch):

```bash
source scripts/env.sh
python3 -m baseline.main condition=zeroshot_rationales   # etc.
```

Expected wall-clock: ~25–45 min per condition (821 utterances x 2 calls at
~1.6–2.6 s/utterance).

## Fine-tuning (Azure-side; already submitted)

The job trains on Azure servers, not locally. Job metadata lives in
`data/fine_tuning/baseline/job_metadata.json` (job `ftjob-d8b97676188d4321aec51d3cefb4234a`,
base `gpt-4o-2024-08-06`, 1 epoch, 3465 train / 385 valid examples, ~4.1M
training tokens ≈ $100 at $25/M).

```bash
python3 -m baseline.finetune status     # poll until status: succeeded
python3 -m baseline.finetune deploy     # tries data-plane deployment API
```

If `deploy` fails with a permission error, deploy the fine-tuned model in the
Azure portal (Deployments -> Create -> select the fine-tuned model), then:

```bash
python3 -m baseline.main condition=finetuned model=<deployment-name>
```

To rebuild/resubmit training data: `python3 -m baseline.finetune build` then
`python3 -m baseline.finetune submit`.

## Evaluation

Runs automatically at the end of the slurm job, or manually:

```bash
python3 -m baseline.eval
```

Outputs:

- `docs/BASELINE_RESULTS.md` — combined comparison table (accuracy, Cohen's
  kappa, macro-F1 for T1 and T2, overall and per speaker).
- `outputs/baseline_eval/comparison.csv` — same table as CSV.
- `outputs/baseline_eval/<condition>_<tier>_report.csv` — per-code P/R/F1.

Conditions whose CSVs are missing are skipped, so you can evaluate as results
come in and re-run after the fine-tuned condition finishes.

## Code map

| File | Role |
|---|---|
| `src/baseline/main.py` | Condition runner (T1→T2 loop, checkpoint/resume, retry) |
| `src/baseline/fewshot.py` | Exemplar builder/loader; frozen rationale generation |
| `src/baseline/finetune.py` | JSONL build, Azure fine-tune submit/status/deploy |
| `src/baseline/eval.py` | Metrics + comparison table |
| `conf/baseline_config.yaml` | Model/provider, conditions, context settings |
| `src/components/prompts/templates/*/t{1,2}_bare.j2` | Label-only prompt variants |
| `src/components/prompts/response_formats.py` | `*_bare` label-only schemas |
| `scripts/mlerp_baseline.slurm` | Batch job for all conditions + eval |
