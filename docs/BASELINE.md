# AutoMISC Baseline Reproduction — Runbook (MLeRP)

Reproduces the AutoMISC gpt-4o baseline on Azure OpenAI with a 2x2 prompting
matrix plus fine-tuned conditions, evaluated against human consensus labels,
across two context lengths (3 vs 5 prior volleys).

## Experiment matrix

All conditions: Azure `gpt-4o` deployment, temperature 0, hierarchical T1→T2
prompts, interval context. Context length (ctx) is a compared dimension:
ctx=3 matches the paper's setting (GPT-4.1, 3 context volleys); ctx=5 was the
repo default used in the first round.

| Condition | Prompt | Shots |
|---|---|---|
| `zeroshot_rationales` | original AutoMISC (explanation + label) | 0 |
| `zeroshot_bare` | label-only | 0 |
| `fewshot_rationales` | explanation + label | stratified HLQC exemplars |
| `fewshot_bare` | label-only | stratified HLQC exemplars |
| `finetuned` | label-only, zero-shot | fine-tuned gpt-4o (matched ctx) |

- **Eval set:** `data/manual/MIV6.3A_manual.csv` (821 utterances, `t1_label_GT`/`t2_label_GT`).
- **Few-shot exemplars + fine-tuning data:** `data/manual/HLQC_balanced_manual.csv` (held out; no leakage).
- Few-shot exemplars are frozen per context length in
  `data/fewshot/exemplars_ctx3.json` and `data/fewshot/exemplars.json` (= ctx5).
  Rebuild only if needed: `python3 -m baseline.fewshot --ctx <N>`.
- Fine-tuning is matched: one model per context length, trained on prompts
  built with the same number of context volleys it is evaluated with.

## One-time setup

From your Mac (repo root) — copies manual CSVs, few-shot exemplar files, and
`.env` (Azure credentials) to MLeRP:

```bash
bash scripts/sync_data_to_mlerp.sh
```

On MLeRP:

```bash
cd /mnt/userdata4/$USER/Behavioral-Coding   # or your repo path
git pull
bash scripts/env.sh setup                   # installs requirements-extras.txt
```

## Run the prompting conditions

```bash
CTX=3 sbatch scripts/mlerp_baseline.slurm    # the 4 conditions at 3 volleys
sbatch scripts/mlerp_baseline.slurm          # same at 5 volleys (already done)
```

Each condition writes `data/annotated/baseline/<condition>_ctx<N>.csv` with
checkpoint/resume on `corp_utt_idx`, so a requeued job continues where it
stopped. Single condition: `CONDITIONS=fewshot_bare CTX=3 sbatch scripts/mlerp_baseline.slurm`.

Interactive alternative:

```bash
source scripts/env.sh
python3 -m baseline.main condition=zeroshot_rationales num_context_turns=3
```

Expected wall-clock: ~25–45 min per condition (821 utterances x 2 calls).

## Fine-tuning (Azure-side)

Two jobs, one per context length; metadata in
`data/fine_tuning/baseline/job_metadata*.json`:

| ctx | Job | Status |
|---|---|---|
| 5 | `ftjob-d8b97676188d4321aec51d3cefb4234a` | succeeded — model `gpt-4o-2024-08-06.ft-d8b97676...-automisc-baseline` |
| 3 | `ftjob-1487dbf17d6b417f85f31cb3e566f007` | submitted (1 epoch, ~3.8M tokens ≈ $96) |

```bash
python3 -m baseline.finetune status --ctx 3      # poll until status: succeeded
python3 -m baseline.finetune deploy --ctx 3      # data-plane deploy attempt
```

The data-plane deployment API returns 404 on this resource, so deployment must
be done in the Azure portal / AI Foundry: Deployments -> Create -> select the
fine-tuned model -> name it exactly `automisc-ft-ctx5` / `automisc-ft-ctx3`.
Then evaluate (matched context):

```bash
python3 -m baseline.main condition=finetuned model=automisc-ft-ctx5 num_context_turns=5
python3 -m baseline.main condition=finetuned model=automisc-ft-ctx3 num_context_turns=3
```

Delete the fine-tuned deployments in the portal after evaluation to stop
hourly hosting charges.

To rebuild/resubmit training data: `python3 -m baseline.finetune build --ctx <N>`
then `submit --ctx <N>`.

## Evaluation

Runs automatically at the end of the slurm job, or manually:

```bash
python3 -m baseline.eval
```

It discovers every `<condition>_ctx<N>.csv` present (legacy un-suffixed files
count as ctx5) and writes:

- `docs/BASELINE_RESULTS.md` — comparison tables grouped by tier and context,
  with the paper's GPT-4.1 reference numbers.
- `outputs/baseline_eval/comparison.csv` — same as CSV.
- `outputs/baseline_eval/<condition>_ctx<N>_<tier>_report.csv` — per-code P/R/F1.

Two macro-F1 columns are reported: `Macro-F1 (gold)` averages only over codes
present in the gold labels (comparable to the paper); `Macro-F1 (all)` also
counts predicted-but-never-gold codes, each contributing 0.

## Code map

| File | Role |
|---|---|
| `src/baseline/main.py` | Condition runner (T1→T2 loop, checkpoint/resume, retry) |
| `src/baseline/fewshot.py` | Per-ctx exemplar builder/loader; frozen rationale generation |
| `src/baseline/finetune.py` | Per-ctx JSONL build, Azure fine-tune submit/status/deploy |
| `src/baseline/eval.py` | Metrics + comparison table across conditions and contexts |
| `conf/baseline_config.yaml` | Model/provider, conditions, context settings |
| `src/components/prompts/templates/*/t{1,2}_bare.j2` | Label-only prompt variants |
| `src/components/prompts/response_formats.py` | `*_bare` label-only schemas |
| `scripts/mlerp_baseline.slurm` | Batch job for all conditions + eval (`CTX` env var) |
