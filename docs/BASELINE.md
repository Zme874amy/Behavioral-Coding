# Model Scale x Adaptation x Rationale Alignment — Runbook

Sixteen conditions: **2 model tiers x 4 adaptation arms x 2 inference styles**.
Each cell reports accuracy, class coverage (macro-F1), and instruction
compliance against human consensus labels.

The grid grew out of the AutoMISC gpt-4o baseline reproduction (a 2x2 prompting
matrix plus a fine-tuned condition). Everything from that round is still here,
just renamed into the tier/arm/style scheme.

> **This grid is the two-call pipeline** (a Tier-1 call, then a Tier-2 call
> conditioned on its answer). The GRPO experiment that follows from these
> results uses a single-call format instead and is documented separately in
> [GRPO.md](GRPO.md). Its arms are prefixed `sc_` and appear as their own block
> in [BASELINE_RESULTS.md](BASELINE_RESULTS.md); the two formats are each
> internally comparable but not comparable to each other. Nothing below is
> affected by it.

## The grid

**Tiers** — model scale.

| Tier | Model | Role |
|---|---|---|
| `gpt4o` | Azure `gpt-4o` | frontier anchor, and the rationale generator |
| `qwen` | `Qwen2.5-7B-Instruct` | edge/local candidate, LoRA-adapted |

**Arms** — how the model is adapted.

| Arm | Adaptation | Training target |
|---|---|---|
| `zs` | none | — |
| `fs` | stratified HLQC exemplars in context | — |
| `ft_bare` | LoRA / Azure fine-tune | label only |
| `ft_rat` | LoRA / Azure fine-tune | distilled rationale + label |

**Inference styles** — how it is prompted at evaluation time.

| Style | Prompt template | Asks for |
|---|---|---|
| `inf_bare` | `t1_bare` / `t2_bare` | label only |
| `inf_cot` | `t1` / `t2` | rationale, then label |

The two fine-tuning arms crossed with the two inference styles are the point of
the design. `ft_bare` under `inf_cot` asks whether label-only training overrides
an explicit instruction to explain; `ft_rat` under `inf_bare` asks whether a
rationale-trained model can go quiet on demand. Either instruction can genuinely
be disobeyed, so those four cells measure instruction control, not just accuracy.

### Data

- **Eval set:** `data/manual/MIV6.3A_manual.csv` — 821 utterances with
  `t1_label_GT` / `t2_label_GT`.
- **Few-shot exemplars and all fine-tuning data:**
  `data/manual/HLQC_balanced_manual.csv` — 1,925 utterances, held out from the
  eval set, so there is no leakage.
- Context length (`ctx`, prior volleys) is a compared dimension: **ctx=3**
  matches the paper (GPT-4.1, 3 volleys), **ctx=5** is what the first round ran.

Three artifacts are frozen **per context length**, because a prompt built with 5
volleys can reference material that is invisible at 3:

| Artifact | Path | Built by |
|---|---|---|
| Few-shot exemplars | `data/fewshot/exemplars_ctx<N>.json` (legacy `exemplars.json` = ctx5) | `baseline.fewshot` |
| Distilled rationales | `data/fine_tuning/rationales/hlqc_rationales_ctx<N>.json` | `baseline.rationalize` |
| Azure FT JSONL | `data/fine_tuning/baseline/hlqc_{train,valid}[_rat]_ctx<N>.jsonl` | `baseline.finetune build` |

## Status of the 16 cells

At **ctx=5**. Nothing is run at ctx=3 yet beyond the gpt-4o in-context cells.

| Cell | State |
|---|---|
| `gpt4o_zs_inf_bare` / `inf_cot` | **done** — see `docs/BASELINE_RESULTS.md` |
| `gpt4o_fs_inf_bare` / `inf_cot` | **done** |
| `qwen_*` (all 8) | **ready to run** — every input exists; needs an MLeRP GPU |
| `gpt4o_ft_bare_inf_bare` / `inf_cot` | **blocked** — model trained, no deployment |
| `gpt4o_ft_rat_inf_bare` / `inf_cot` | **blocked** — training data built, job unsubmitted pending deployment access |

The whole Qwen tier is unblocked: the distilled rationales are generated and the
pipeline is smoke-tested end to end. It only needs GPU time.

The four completed gpt-4o cells were written before this naming existed, as
`zeroshot_rationales`, `zeroshot_bare`, `fewshot_rationales`, `fewshot_bare`.
`baseline.eval` maps them through `LEGACY_STEMS`, so nothing has to be renamed.

### Known asymmetry: compliance is only measurable on the Qwen tier

`src/baseline/main.py` passes a pydantic `response_format` to Azure, so decoding
is schema-constrained: an `explanation` field is either in the schema or not, and
the model *cannot* fail to comply. The Qwen tier generates freely, so its
compliance is genuine behaviour.

Accuracy and macro-F1 are comparable across both tiers. **Instruction-compliance
claims should be scoped to the Qwen tier**, and `BASELINE_RESULTS.md` marks the
gpt-4o rows `constrained` for exactly this reason. Making it symmetric means
re-running the gpt-4o cells with `response_format=None` and free-text parsing.

### Known limitation: the distilled rationales are post-hoc

`baseline.rationalize` shows gpt-4o the gold label and asks it to justify it. The
result reads like reasoning but was generated *after* the answer was fixed, so it
may not be faithful to any process that would independently produce that label.
This is the standard caveat for rationale distillation and belongs in the
write-up alongside any `ft_rat` result.

## One-time setup

From your Mac (repo root) — copies manual CSVs, exemplars, distilled rationales,
and `.env` to MLeRP:

```bash
bash scripts/sync_data_to_mlerp.sh
```

This first checks the Strudel2 SSH certificate. Those are short-lived (~28 days),
and an expired one fails as `Permission denied (publickey)`, which looks exactly
like a key that was never registered. If it reports an expiry, renew by logging in
to <https://strudel2.cloud.edu.au> and starting an MLeRP session.

On MLeRP:

```bash
cd /mnt/userdata4/$USER/Behavioral-Coding
git pull
bash scripts/env.sh setup
bash scripts/check_data.sh
```

## Running the gpt-4o tier

Azure only, no GPU needed.

```bash
CTX=3 sbatch scripts/mlerp_baseline.slurm    # the 4 in-context cells at 3 volleys
sbatch scripts/mlerp_baseline.slurm          # same at 5 volleys (already done)

# one cell, interactively
source scripts/env.sh
python3 -m baseline.main condition=zeroshot_rationales num_context_turns=3
```

Roughly 25-45 min per cell (821 utterances x 2 calls). Checkpointed on
`corp_utt_idx`, so a requeued job resumes.

### Azure fine-tuning

Jobs, one per (target, context); metadata in `data/fine_tuning/baseline/`:

| Target | ctx | Job | Status |
|---|---|---|---|
| bare | 5 | `ftjob-d8b97676188d4321aec51d3cefb4234a` | succeeded — model `gpt-4o-2024-08-06.ft-d8b97676...-automisc-baseline` |
| bare | 3 | `ftjob-1487dbf17d6b417f85f31cb3e566f007` | succeeded |
| rat | 5 | — | **training data built, job not submitted** |

The FT-Rat training data is ready at
`data/fine_tuning/baseline/hlqc_{train,valid}_rat_ctx5.jsonl` (3,850 examples,
~4.5M tokens, **~$112 per epoch**).

It is deliberately not submitted yet. Deployment is blocked (below), so
submitting now would pay for a model that cannot be served. Get deployment
access first, then:

```bash
python3 -m baseline.finetune submit --target rat --ctx 5
python3 -m baseline.finetune status --target rat --ctx 5
python3 -m baseline.finetune deploy --target rat --ctx 5
```

To rebuild the data (e.g. for ctx=3, after running `rationalize --ctx 3`):

```bash
python3 -m baseline.finetune build --target rat --ctx 3
```

**Deployment is the blocker.** Creating a deployment is a control-plane (ARM)
operation; an API key is a data-plane credential, so `deploy` returns 404 no
matter what. It needs Azure AD credentials with a role like *Cognitive Services
OpenAI Contributor*, or someone who has one, deploying from Azure AI Foundry:
Deployments -> Create -> pick the fine-tuned model -> name it exactly
`automisc-ft-ctx5`, `automisc-ft-ctx3`, or `automisc-ft-rat-ctx5`.

Once deployed, each model is evaluated under **both** inference styles:

```bash
python3 -m baseline.main condition=gpt4o_ft_bare_inf_bare model=automisc-ft-ctx5     num_context_turns=5
python3 -m baseline.main condition=gpt4o_ft_bare_inf_cot  model=automisc-ft-ctx5     num_context_turns=5
python3 -m baseline.main condition=gpt4o_ft_rat_inf_bare  model=automisc-ft-rat-ctx5 num_context_turns=5
python3 -m baseline.main condition=gpt4o_ft_rat_inf_cot   model=automisc-ft-rat-ctx5 num_context_turns=5
```

Delete the fine-tuned deployments in the portal afterwards; they bill hourly
while they exist.

## Running the Qwen tier

### Step 1: distilled rationales (Azure, no GPU)

Only needed for the `ft_rat` arm. **Already done for ctx=5**: all 1,925
utterances are in `data/fine_tuning/rationales/hlqc_rationales_ctx5.json`
(3,850 gpt-4o calls, ~10 hours wall clock). Rationales average 44 words / ~69
tokens per tier.

Checkpoints every 25 utterances, and re-invoking the same command resumes
without re-calling anything, so this is safe to rerun. For another context
length:

```bash
PYTHONPATH=src python -m baseline.rationalize --ctx 3
```

### Step 2: train the adapters (A100)

Two adapter pairs, `bare` and `rat`, each a T1 and a T2 adapter:

```bash
TARGET=bare sbatch scripts/mlerp_local_train.slurm
TARGET=rat  sbatch scripts/mlerp_local_train.slurm
TARGET=rat CTX=3 sbatch scripts/mlerp_local_train.slurm
```

Adapters land in `data/fine_tuning/baseline_local/ctx<N>/<target>/{t1,t2}/`, with
a `train_metadata.json` recording the base model, example counts, LoRA config,
and epochs. About 2 hours per pair.

Needs the **full 40GB A100** (`--gres=gpu:40gb:1`); a 20GB MIG slice will OOM on
a 7B model at these sequence lengths.

### Step 3: the eight evaluation cells

```bash
ARM=zs      INF=bare sbatch scripts/mlerp_local_predict.slurm
ARM=zs      INF=cot  sbatch scripts/mlerp_local_predict.slurm
ARM=fs      INF=bare sbatch scripts/mlerp_local_predict.slurm
ARM=fs      INF=cot  sbatch scripts/mlerp_local_predict.slurm
ARM=ft_bare INF=bare sbatch scripts/mlerp_local_predict.slurm
ARM=ft_bare INF=cot  sbatch scripts/mlerp_local_predict.slurm
ARM=ft_rat  INF=bare sbatch scripts/mlerp_local_predict.slurm
ARM=ft_rat  INF=cot  sbatch scripts/mlerp_local_predict.slurm
```

Or everything at once — two training jobs plus eight dependent evaluation jobs,
so the `ft_*` cells wait for their adapters:

```bash
CTX=5 bash scripts/submit_local_grid.sh
SKIP_TRAIN=1 CTX=5 bash scripts/submit_local_grid.sh   # adapters already exist
```

One job per cell rather than one job for everything: the `inf_cot` cells generate
100-200 tokens per call and take hours, so a single job would exceed the wall
clock. Each cell checkpoints every 25 utterances, so requeues are safe.

Local smoke test, no GPU (a few minutes on CPU):

```bash
PYTHONPATH=src python -m baseline.local_arm predict --arm zs --inf bare --ctx 5 --limit 4 \
    model.base_model=Qwen/Qwen2.5-0.5B-Instruct inference.force_cpu=true
```

## Two settings that are easy to get wrong

**`inference.max_new_tokens: 256`, uniformly.** It is tempting to set a tight
budget for the `inf_bare` cells since a bare label is ~3 tokens. Don't: a
*non-compliant* rationale then gets truncated mid-sentence, `parse_label` returns
`UNKNOWN`, and a compliance finding is silently recorded as a parsing failure.
This is observable — a smoke run at 32 tokens produced 100% `UNKNOWN` on the
few-shot CoT cell, which dropped to 0% at 256. Cost is self-limiting because
compliant bare outputs hit EOS after ~3 tokens.

**Per-arm `max_input_len`.** Measured prompt maxima at ctx=5: zs/ft T1 ~3.0k
tokens, few-shot T1 ~5.4k (the exemplar transcripts dominate). So the config uses
4096 for `zs`/`ft_*` and **8192** for `fs`. A single window of 3072 would silently
truncate few-shot prompts, and because the target utterance sits at the *end* of
the prompt, truncation removes the thing being classified. `local_arm` prints a
`prompt_truncated` count and warns if any prompt hit the cap.

## Evaluation

Runs at the end of each slurm job, or manually:

```bash
python3 -m baseline.eval
```

It discovers every recognised file in `data/annotated/baseline/`, parsing both
`<tier>_<arm>_inf_<style>_ctx<N>.csv` and the legacy names (un-suffixed = ctx5),
and writes:

- `docs/BASELINE_RESULTS.md` — performance tables grouped by MISC tier, context,
  and model tier, plus an instruction-compliance table and the paper's GPT-4.1
  reference numbers.
- `outputs/baseline_eval/comparison.csv` — the performance table as CSV.
- `outputs/baseline_eval/compliance.csv` — rationale-emission and `UNKNOWN` rates.
- `outputs/baseline_eval/<cell>_<t1|t2>_report.csv` — per-code precision/recall/F1.

Two macro-F1 columns: `Macro-F1 (gold)` averages only over codes present in the
gold labels (comparable to the paper), `Macro-F1 (all)` also counts
predicted-but-never-gold codes, each contributing 0.

Because compliance cannot be recovered from a parsed label, every Qwen row also
stores the raw generation (`t1_raw` / `t2_raw`), a rationale-emission flag, and
prompt/generated token counts.

## Code map

| File | Role |
|---|---|
| `src/baseline/main.py` | gpt-4o cell runner (T1→T2 loop, checkpoint/resume, retry) |
| `src/baseline/local_arm.py` | Qwen tier: `train --target`, `predict --arm --inf` |
| `src/baseline/rationalize.py` | Frozen per-ctx distilled rationale targets |
| `src/baseline/fewshot.py` | Per-ctx exemplar builder/loader |
| `src/baseline/finetune.py` | Azure per-(target, ctx) JSONL build, submit/status/deploy |
| `src/baseline/eval.py` | Tier/arm/style parsing, metrics, compliance, report |
| `src/automisc_ft/data.py` | Prompt/target builders (`structure_suffix`, rationale targets) |
| `src/automisc_ft/infer.py` | `TieredAnnotator`, `parse_label`, `emitted_rationale` |
| `src/automisc_ft/train.py` | `train_adapter_pair` — shared by this grid and the 5-fold CV run |
| `conf/baseline_config.yaml` | gpt-4o tier: model/provider, conditions |
| `conf/baseline_ft_local_config.yaml` | Qwen tier: base model, LoRA, per-arm windows |
| `src/components/prompts/templates/*/t{1,2}_bare.j2` | Label-only prompt variants |
| `src/components/prompts/response_formats.py` | `*_bare` label-only schemas |
| `scripts/mlerp_baseline.slurm` | gpt-4o cells + eval (`CONDITIONS`, `CTX`) |
| `scripts/mlerp_local_train.slurm` | One Qwen adapter pair (`TARGET`, `CTX`) |
| `scripts/mlerp_local_predict.slurm` | One Qwen cell (`ARM`, `INF`, `CTX`) |
| `scripts/submit_local_grid.sh` | Whole Qwen tier with training dependencies |

The single-call GRPO ladder is a separate set of modules and does not share any
of the runners above; see the code map in [GRPO.md](GRPO.md).
