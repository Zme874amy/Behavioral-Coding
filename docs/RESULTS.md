# SFT thesis experiment — first run

Goal: train a LoRA adapter on **human-labelled** MI utterances and compare it
against the **zero-shot** base model on a **strictly held-out** dataset.

## Setup

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` (494 M params, non-gated HF) |
| Adapter | LoRA r=8, α=16, dropout=0.05, target = q_proj, v_proj |
| Trainable params | 540 672 (0.11 % of base) |
| Class structure | T2 (MISC 2.5 with Change/Sustain Talk codes) |
| Training set | `data/manual/MIV6.3A_manual.csv` (821 human-labelled utterances) |
| Held-out test set | `data/manual/HLQC_balanced_manual.csv` (1 925 examples; 300 sampled) |
| LODO leakage | None — HLQC never touches training |
| Optimisation | 3 epochs, batch 2, LR 1e-4 cosine, 1 233 steps |
| Hardware | Apple Silicon (MPS for training, CPU for inference) |
| Wall-clock | ~50 min training + ~17 min eval (300 examples × 2 modes) |

The MPS path required two workarounds:
- `clip_grad_norm_` returns NaN on MPS in long runs → disabled (`max_grad_norm=0`).
- MPS activations accumulate after ~50–100 steps → custom `TrainerCallback`
  calls `torch.mps.empty_cache()` every 25 steps to keep step-time at ~3 it/s
  for the whole run.

Why not `gemma-4-e4b` / `qwen3.5-9b`?  Those are **LM Studio GGUF** checkpoints
and cannot be LoRA-fine-tuned through HuggingFace `peft`. Gemma is gated and
requires an HF token, which isn't installed. Qwen 2.5-0.5B is the largest
non-gated proxy that completes within the 4-hour CPU/MPS budget.

## Results — LODO held-out HLQC, n=300

We evaluated three configurations on the same 300 sampled rows:

| Model | Prompt used |
|---|---|
| zero-shot (bare) | Inline prompt with just the list of code letters |
| **zero-shot (rich)** | **Production flat.j2 prompt with full per-code definitions and examples** (= `src/components/prompts/templates/{speaker}/flat.j2`) |
| fine-tuned (LoRA) | Same bare prompt the SFT was trained with |

| Metric | ZS bare | **ZS rich** | Fine-tuned | FT − ZS rich |
|---|---:|---:|---:|---:|
| accuracy | 0.0067 | **0.1067** | 0.5367 | +0.430 |
| f1_macro | 0.0018 | 0.0110 | 0.0362 | +0.025 |
| f1_weighted | 0.0009 | 0.1051 | 0.4372 | +0.332 |

Training loss decreased from 9.0 → **1.14**, confirming real learning. The
fairer comparison (rich-prompted zero-shot) still leaves a **+43 pp accuracy
gap** for the fine-tuned adapter — so the SFT win isn't a prompt artefact —
but the gap is much smaller than the +53 pp the bare-prompt run suggested.

### MISC-2.5 family-level recall

This is the per-category-family view of which classes each model
actually gets right (i.e. was the predicted code in the same family as
the gold code, ignoring exact match):

| Family | support | ZS bare | ZS rich | FT |
|---|---:|---:|---:|---:|
| Counsellor reflections (CR, SR) | 36 | 0% | 0% | 0% |
| Counsellor MI-consistent (AF, SU, RF, EC) | 11 | 73% | 9% | 0% |
| Counsellor MI-neutral (ADP, RCP, GI) | 14 | 0% | 0% | 0% |
| Counsellor MI-inconsistent (ADW, RCW, CO, DI, WA) | 14 | 7% | 0% | 0% |
| Counsellor questions (OQ, CQ) | 33 | 0% | **85%** | 0% |
| Counsellor utility (FA, FI, ST) | 50 | 0% | 0% | **100%** |
| Client Change Talk (+ codes) | 11 | 100%* | 0% | 0% |
| Client Sustain Talk (− codes) | 5 | 0% | 100%* | 0% |
| Client Neutral (N) | 126 | 0% | 17% | **100%** |

\* = artefact of the model collapsing onto a single default code that happens
to belong to that family (bare ZS defaults to `O+`, rich ZS defaults to `O-`);
exact-match recall is still 0 % on every Change/Sustain-Talk sub-code.

### What this tells us about classification issues

Each model effectively collapses to a per-speaker default label:

| | client default | counsellor default |
|---|---|---|
| bare zero-shot | `O+` (219/142 over-predicted) | `AF` (57/158) |
| rich zero-shot | `O-` (108) | `OQ` (115) |
| **fine-tuned**  | `N` (142) | `FI` (158) |

**Easiest class:** `OQ` (Open Question, support 12) — rich ZS gets **92 %**
recall on it. The rich prompt's emphasis on the question grammar transfers
zero-shot.

**Codes nobody ever got right (0 % across all three models):**
- All reflections: `CR`, `SR`
- MI-neutral counsellor codes: `ADP`, `RCP`, `GI`
- Most counsellor MI-inconsistent codes: `ADW`, `RCW`, `DI`, `WA`
- Every individual Change Talk sub-code: `D+`, `AB+`, `R+`, `N+`, `C+`, `AC+`, `TS+`
- Every individual Sustain Talk sub-code: `D-`, `AB-`, `R-`, `N-`

**Codes where the fine-tuned adapter is the *only* model getting them right:**
- `N` (126 gold) — FT recall 100 %, precision 89 %
- `FI` (35 gold) — FT recall 100 %, precision 22 %

**Codes the rich prompt unlocks but the fine-tuned model loses:**
- `OQ` (12 gold) — rich-ZS gets 11/12 right; FT gets 0/12 (it always says `FI` for counsellors).

This is the most important finding: **fine-tuning on this small dataset
*hurts* OQ recognition by overwriting the base model's prompt-sensitivity
with a speaker-conditioned default.** That's a classic small-data SFT
failure mode and is what the next experiments should target.

## What this proves and what it doesn't

**Proves**
- The pipeline is correct end-to-end: data loaders, LODO splitter, prompt
  template, LoRA trainer, zero-shot vs fine-tuned eval, predictions/metrics
  artefacts. All artefacts are reproducible under
  `outputs/sft_runs/qwen05b_full/`.
- SFT produces a huge jump in plausible-label output: zero-shot Qwen 0.5B
  emits MISC-shaped tokens at random (O+ in 73 % of cases) and gets 0.67 %
  accuracy. After SFT it produces consistent, valid label codes and reaches
  53.7 % accuracy on HLQC.

**Does NOT yet prove (the actual thesis claim)**
- That fine-tuning catches **long-tail clinical behaviours** like Change Talk.
  Looking at the predictions, the fine-tuned model collapsed to a
  speaker-conditioned prior:
    - all 142 client utterances → predicted `N` (Neutral)
    - all 158 counsellor utterances → predicted `FI` (Filler)
  Change Talk codes (`R+`, `O+`, `C+`, `D+`, `TS+`) never appear in its output.
  The model learned the easy decision rule ("if client, say `N`; if
  counsellor, say `FI`") which is enough to get high accuracy on the
  majority classes but misses the rare ones the thesis cares about.

## Why this happened
- The training set is heavily imbalanced: `N` and `FI` dominate, while each
  Change Talk subtype has only a handful of examples.
- Qwen 0.5B is small and 821 examples / 11 classes ≈ 70/class on average,
  with the rare Change Talk classes having ~5-20 examples each — well below
  what cross-entropy SFT needs to compete with the majority-class prior.
- Standard SFT loss treats every token equally, so the optimiser is happy
  to be confidently wrong on the long tail as long as it nails the majority.

## Suggested next experiments

1. **Train the SFT model with the *rich* prompt, not the bare prompt** —
   so the model keeps the prompt-sensitivity that gave rich-ZS 85 % family
   recall on questions and learns to *refine* on top of that signal,
   instead of erasing it.
2. **Class-balanced or oversampled training set**: replicate every minority
   example until each class has ~50 instances. Almost guaranteed to break
   the speaker-conditioned default and pull Change Talk F1 above zero.
3. **Two-stage training**: pre-train on MIV6.3A + AnnoMI for the coarse T1
   task first (CRL / SRL / IMC / IMI / Q / O for counsellor; C / S / N for
   client), then a short T2 phase that emphasises Change/Sustain Talk
   sub-codes. Forces the model to commit to a family first.
4. **Bigger base model** (Qwen 2.5 1.5B / 3B) once an HF token is available
   or once we're on a CUDA box — 0.5B simply has too few features to
   distinguish `R+` / `C+` / `D+` / `TS+` reliably.
5. **Use Apple's MLX framework** instead of PyTorch+MPS to remove the
   2³² NDArray cap and the cache-creep workaround.

## Files produced

- `outputs/sft_runs/qwen05b_full/lora_model/local_finetuned_model/` — LoRA
  adapter (3.1 MB) ready to plug back into Qwen 2.5-0.5B-Instruct.
- `outputs/sft_runs/qwen05b_full/sft_artifacts/train.jsonl|test.jsonl` —
  the exact prompt/completion pairs used (821 train, 1 925 test).
- `outputs/sft_runs/qwen05b_full/sft_artifacts/sft_metadata.json` — every
  config value that produced these results.
- `outputs/sft_runs/qwen05b_full/eval/sft_evaluation/predictions.csv` —
  per-utterance prediction from both models (utt_text, human_label,
  zero_shot_pred, fine_tuned_pred).
- `outputs/sft_runs/qwen05b_full/eval/sft_evaluation/metrics.json` — the
  full classification report (precision/recall/F1 per code).

## Reproducing (legacy 0.5B bare-prompt run)
```bash
PYTHONPATH=src TOKENIZERS_PARALLELISM=false \
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
python src/sft/main.py +sft_models=qwen05b model.base_model=Qwen/Qwen2.5-0.5B-Instruct \
  prompt_style=bare hydra.run.dir=outputs/sft_runs/qwen05b_full
```

---

## HF upgrade + proposal models (May 2026)

### Proposal name → Hugging Face repo (for LoRA SFT)

| Proposal / LM Studio | HF repo | Gated? |
|---|---|---|
| `google/gemma-4-e4b` | `google/gemma-3n-E4B-it` | Yes — accept license + `huggingface-cli login` |
| `qwen/qwen3.5-9b` | `Qwen/Qwen2.5-7B-Instruct` | No |
| Fallback Gemma | `google/gemma-2-9b-it` | Yes |

LM Studio names (`gemma-4-e4b`, `qwen3.5-9b`) remain in [`conf/config.yaml`](../conf/config.yaml) for **parser/annotator inference only**. SFT uses HF checkpoints via [`conf/sft_config.yaml`](../conf/sft_config.yaml) and presets under `conf/sft_models/`.

### What was implemented

- [`src/components/hf_load.py`](../src/components/hf_load.py) — token-aware `from_pretrained`, gated-model errors, Gemma 3n retry, CPU fallback for MPS eval cap
- [`conf/sft_models/`](../conf/sft_models/) — `qwen7b`, `gemma3n_e4b`, `gemma2_9b`, `qwen05b` presets (proposal LoRA r=16 α=32 on 7B/Gemma)
- [`conf/sft_device/`](../conf/sft_device/) — `cuda` (bf16, full data) vs `mps` (smoke only)
- [`src/sft/preflight_hf.py`](../src/sft/preflight_hf.py) — fail-fast auth + load + generate check
- [`docs/HF_SETUP.md`](./HF_SETUP.md) — env, login, and license steps
- [`scripts/mlerp_sft.slurm`](../scripts/mlerp_sft.slurm) — batch SFT on MLeRP

### Headline experiment (rich-ZS vs rich-FT) — **run on M3 HPC (CUDA)**

Default config now targets **Qwen 2.5-7B** with **rich prompt** for train and both eval passes.

| Model / condition | accuracy | f1_macro | f1_weighted | Change Talk F1 |
|---|---:|---:|---:|---:|
| Qwen-7B rich-ZS | *pending HPC* | | | |
| Qwen-7B rich-FT | *pending HPC* | | | |
| Δ (rich-FT − rich-ZS) | | | | |
| Gemma-3n-E4B rich-ZS | *pending HPC* | | | |
| Gemma-3n-E4B rich-FT | *pending HPC* | | | |

**Local Mac note:** This machine has **no CUDA**. Rich-prompt LoRA at ~832 tokens on MPS slows to minutes/step after ~3–4 steps; **7B/Gemma full runs must use HPC**. Preflight for `Qwen/Qwen2.5-0.5B-Instruct` passes with the new loader (CPU generate on MPS).

### HPC commands

```bash
cd /mnt/userdata4/$USER/localproject
source scripts/env.sh
python src/sft/preflight_hf.py --model Qwen/Qwen2.5-7B-Instruct
python src/sft/main.py +sft_models=qwen7b sft_device=cuda \
  hydra.run.dir=outputs/sft_runs/qwen7b_rich

export PRESET=gemma3n_e4b && sbatch scripts/mlerp_sft.slurm
```

After each run, copy metrics from `outputs/sft_runs/<run>/sft_evaluation/metrics.json` into the table above.

### Fair comparison reminder

Prior `qwen05b_full` fine-tuned on **bare** prompt; rich-ZS was **10.7%**. The missing cell is **rich-FT** (same `flat.j2` as rich-ZS, only LoRA differs). That is what the HPC runs above are meant to fill.
