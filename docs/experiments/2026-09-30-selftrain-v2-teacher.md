# Self-training v2, the teacher comparison, and the teacher-model search

**Started** 2026-09-30 · **Status** self-training v2 complete (two pools, 3 seeds
each); Qwen2.5-32B teacher NO-GO; off-the-shelf teacher screen complete
(2026-10-04), but its HLQC gate is invalid (corrected 2026-10-05): on HLQC no
open model beats the student, yet on MIV6.3A (report-only) Gemma-4-31B few-shot
reaches macro-F1 0.583 vs the student's 0.457, so the HLQC ranking reflects
HLQC's coding conventions, not teacher quality; the re-run gates on the HLQC
val fold with robust metrics (docs/RERUN_PLAN.md P5); track 2 (fine-tuned large
teacher) not started · **Backbone** `ft1mix_bare` (1 shared adapter, mixed regime) ·
**Context length** 5

## The question

Self-training v1 (Welivita pool) **lowered** T2 macro-F1 by −0.044 (real over 3
seeds). Two questions follow:

1. Can a properly built self-training recipe (the student labels for itself)
   reach or beat the baseline?
2. Does a **stronger teacher** (distillation) label better than the student
   itself? This is run as a separate *comparison* arm, never as "self-training".

Scope rule: HLQC is the only labelled data used to select or calibrate anything.
Welivita's human MITI labels are a **meter** (measure pseudo-label accuracy),
never a selector. The MIV6.3A test set never chooses anything.

## Why v1 failed (diagnosis, 2026-09-30)

| Problem | Evidence |
|---|---|
| Wrong but certain labels | MITI agreement 48% on shared codes; every accepted label had confidence 1.0 (3-sample vote) |
| Class prior inverted | AF / GI / SU = 17% / 17% / 15% of pseudo-labels (human: 2% / 1.5% / 0.9%) |
| Human signal drowned | 6,879 pseudo vs 1,925 human labels (3.6 : 1) |
| Client supervision collapsed | Client share of targets 0.46 → 0.14 |
| Wrong domain, wrong labeller | Welivita is written Q&A; labeller was the 2-adapter model |

## What changed (v2 design)

| Step | v1 | v2 | Basis |
|---|---|---|---|
| Confidence | 3-sample vote | Likelihood over the code set, softmaxed (`TieredAnnotator.score_codes`) | UPS, SoftMatch |
| Threshold | one global 0.8 | per-code τ at precision ≥ 0.8 on HLQC out-of-fold (`selftrain.calibrate`) | FlexMatch |
| Labeller | 1 model | 3-seed self-ensemble, all must agree | Co-teaching, SemiEvol |
| Class balance | rare-code exemption | quotas ∝ HLQC prior^0.5 | CReST, DARP |
| Budget | uncapped | ≤ 1 pseudo per human label, counsellor only | Arazo 2020 |
| Pool | Welivita | Welivita, then in-domain MIV6.3B (test near-duplicates removed) | Du 2021, Kumar 2020 |

Code: `src/automisc_ft/infer.py` (`score_codes`, `predict_row_scored`),
`src/selftrain/{label_pool,select,calibrate,screen_teachers}.py`,
`src/baseline/cv_anchor.py score-fold`, `src/selftrain/expansion_table.py`.

## Results

### Step A — student calibration (HLQC out-of-fold)

- Scorer argmax = greedy on 97.9% (T1) / 97.7% (T2) of 1,925 rows.
- T2 acc 0.735, macro-F1 0.399 [0.348–0.464]. ECE T2 0.23 (overconfident).
- Codes reaching 80% precision: counsellor FI, SR, CQ, OQ, ST, ADP, WA; client N.
  Never reliable: AF, CR, EC, GI, SU, RCW, FA, CO, DI, RF, ADW — the tail codes
  that drive macro-F1.

### Self-training v2, 3 seeds each (`docs/EXPANSION_RESULTS.md`)

| Arm | Pseudo-labels | T2 acc Δ | T2 macro-F1 Δ | Verdict |
|---|---|---|---|---|
| v1 (Welivita) | 6,879 | — | −0.044 | real (harmful) |
| v2 Welivita (ST-1) | 607 counsellor; MITI 0.648 | −0.010 [−0.024, +0.007] | **−0.047 [−0.071, −0.025]** | real (harmful) |
| v2 MIV6.3B (Step F) | 836 counsellor | **+0.023 [+0.006, +0.043]** | −0.030 [−0.046, −0.009] | acc real; F1 under seed floor |

**Mechanism** (ST-1, counsellor codes, pooled over 3×3 seeds):

| Codes | Predicted share | Mean ΔF1 |
|---|---|---|
| Got pseudo-labels (FI, ADP, CQ, OQ, SR) | 0.577 → 0.627 | +0.011 |
| Got none (GI, ST, SU, AF, EC, CR) | 0.415 → 0.360 | **−0.096** |

Cleaner labels (MITI 0.65 vs 0.48) do not fix it. An honest student can only
label what it already knows, so self-training under class imbalance amplifies
the majority: accuracy holds or rises, macro-F1 falls.

### Teacher arm — Qwen2.5-32B-Instruct-AWQ (Step 0)

**Answer-format bug (fixed 2026-10-02).** The first teacher runs scored the
student's `" CODE\n"` string. The teacher was never trained on that, so the
ranking was noise (few-shot counsellor acc 0.067). Fix:
`--teacher-answer-format auto` scores `CODE<eos>` zero-shot and
`{"label": "CODE"}<eos>` few-shot. Argmax = greedy went from 85% to 100%.
Lesson: score the answer in the format the model is prompted to give.

| Labeller (HLQC) | T2 acc | T2 macro-F1 [CI] | Counsellor T2 acc |
|---|---|---|---|
| Student (out-of-fold) | 0.735 | **0.399** [0.348–0.464] | 0.620 |
| 32B few-shot | 0.642 | 0.324 [0.281–0.373] | 0.501 |
| 32B zero-shot | 0.600 | 0.256 [0.226–0.291] | 0.426 |

**NO-GO.** The 32B reaches 80% precision only on the same head codes as the
student (CQ, FI, OQ, SR, ADP), so distilling from it would repeat the majority
amplification. Not trained.

Report-only MIV6.3A row (tier `qwen32b`, never used to choose): few-shot T2
macro-F1 0.533 (learnable 0.561), close to GPT-4o few-shot 0.539 / 0.564 and above
the student's 3-seed mean 0.457 / 0.493.

**Why HLQC and MIV6.3A disagree — session quality ruled out.**

| HLQC sessions | 32B few-shot T2 macro-F1 | Student (out-of-fold) |
|---|---|---|
| High-quality | 0.348 | 0.367 |
| Low-quality | 0.323 | 0.444 |

The 32B is equally weak on both halves. HLQC stays the selection set.

### Teacher-model search (2026-10-03 →)

The search is not limited to Qwen. Each model gets the same few-shot greedy
two-call screen on HLQC (`selftrain.screen_teachers`, ~0.2 s per row).
Criteria: T2 macro-F1 vs the student's 0.399, plus precision on the tail codes.

| Model | Served via | T2 macro-F1 [CI] | T2 acc | Counsellor acc | Job |
|---|---|---|---|---|---|
| Student (reference) | — | **0.399** [0.348–0.464] | 0.735 | 0.620 | — |
| Gemma-4-31B-it qat w4a16 | vLLM | 0.362 [0.325–0.414] | 0.605 | 0.488 | 170513 |
| gpt-oss-20b (reasoning low) | vLLM | 0.303 [0.265–0.350] | 0.593 | 0.410 | 170512 |
| Qwen3.6-27B q8 | Ollama (shared) | 0.362 [0.321–0.412] | 0.617 | 0.487 | 170516 |
| gpt-oss-120b (reasoning low) | Ollama (shared) | 0.305 [0.272–0.349] | 0.573 | 0.417 | 170514 |
| Qwen3-30B-A3B-Instruct-2507 fp16 | Ollama (shared) | 0.298 [0.253–0.346] | 0.530 | 0.437 | 170515 |
| Llama4 Scout 16x17B | Ollama (shared) | 0.209 [0.184–0.242] | 0.558 | 0.371 | 170518 |
| Gemma3-27B | Ollama (shared) | 0.319 [0.280–0.354] | 0.538 | 0.425 | 170517 |
| Qwen3-VL-32B | Ollama (shared) | **invalid** (100% empty answers) | | | 170519 |

Qwen3-VL-32B: the shared `qwen3-vl:32b` tag is the *thinking* build. It ignored
`think: false` and returned empty content within the 48-token budget on every
row. Ollama also serves this architecture one request at a time. A valid run
would need thinking on and ~2k tokens per call, roughly 10+ h serial. Not re-run,
given that every other model sits at or below 0.362.

Gemma3-27B failed to parse at T1 on 7.1% of rows; those fall back to the
speaker's first group.

All other Ollama runs: 0 errors, ≤ 1.1% unparseable, longest prompt ~4.3k tokens
(context 12,288), so none were truncated. 1–4 s per row.

**Model size does not buy MISC coding.** gpt-oss-120b (0.305) scores no better
than gpt-oss-20b (0.303), and Llama4 Scout (109B total) is the weakest. The two
best, Gemma-4-31B and Qwen3.6-27B, tie at 0.362, and both have CIs that overlap
the student without beating it. Their tail-code precision stays far below the
0.8 bar, except EC at 0.67–0.75.

Gemma-4-31B is the best so far. It finds more tail rows (GI recall 0.53, AF
0.65), but labels them with low precision (GI 0.22, AF 0.26, SU 0.10). Only EC
(0.75) and CR (0.55) come close to the 0.8 bar.

### Audit: is the screen pipeline wrong, or is HLQC the wrong yardstick? (2026-10-04)

Raised because GPT-4o few-shot scored well on MIV6.3A (0.539), while every open
model scored ≤ 0.362 on HLQC.

| Check | Result |
|---|---|
| Prompts vs the GPT-4o baseline (`baseline.main`) | **Byte-identical**: 0 diffs over 40 HLQC prompts (system template, few-shot turns, context excerpt, user turn) |
| Output parsing (GPT-4o used enum-constrained structured output; screens use free text + `parse_label`) | Invalid raw codes are ≤ 1.1% at T1 (Gemma3 7.1%) and 2–7% at T2. Most are the prompt's own long forms (ADWP, RCWP, CON, DIR, a known frozen-prompt quirk), and `parse_label` maps all of them correctly. Cost ≤ ~0.01 |
| **Same screen code on MIV6.3A** (report-only pipeline check, job 170529) | Gemma-4-31B: T2 acc 0.726, **macro-F1 0.583** [0.553–0.652]. GPT-4o few-shot: 0.669 / 0.539. Student 3-seed mean: macro-F1 0.457 |

**Why HLQC penalises off-the-shelf models:**
- **Annotation convention, with inconsistent gold.** The top teacher error is
  FI→FA (103–117 rows). HLQC gold labels "okay" / "Yeah." as Filler 34 times and
  as Facilitate 19 times, but the codebook text defines "Mm-hmm"-type
  acknowledgements as FA. Models follow the codebook; the fine-tuned student
  learned HLQC's habit, including FI being the most frequent counsellor code
  (198 vs FA 55). Merging FA into FI narrows the gap from 0.037 to ~0.025.
- **Domain.** HLQC is noisy speech-to-text transcripts, partly low-quality
  sessions. It includes 107 IMI-group rows (ADW, RCW, WA, CO) that MIV6.3A never
  has, while MIV6.3A has many SU/EC/AF rows that models handle well.
- **Home advantage.** The student's 0.399 is out-of-fold, but it was trained on
  the same annotators and conventions.

**Consequence.** HLQC is still valid for *ranking teachers against each other*:
all of them are off-the-shelf, so they meet its conventions on equal footing.
It is NOT a fair *teacher-vs-student gate*. The Step 0 NO-GO for Qwen2.5-32B
(also 0.533 on MIV6.3A) rested on that gate and should be read in this light.

Track 2 (approved): LoRA-fine-tune the best ~27–31B open model on HLQC as the
teacher, Noisy-Student style, with 5-fold calibration. It runs only if no
off-the-shelf model clears the student.

## Running it

```bash
# off-the-shelf screen of a staged safetensors model (vLLM)
MODEL=/mnt/userdata4/jia-wen/models/<name> sbatch scripts/mlerp_teacher_screen.slurm
# screen straight from MLeRP's shared Ollama store (no download, no quota)
OMODEL=gpt-oss:120b THINK=low MAXTOK=1024 PARALLEL=4 \
  sbatch --gres=gpu:40gb:2 scripts/mlerp_teacher_screen_ollama.slurm
# stage a model for likelihood scoring / LoRA (login node)
bash scripts/stage_model.sh <hf-repo>
```

## Infrastructure notes

- **Shared models.** MLeRP has a shared Ollama store at `/apps/ollama/models`
  (~310 GB of GGUF, not counted against the user quota), served with
  `/apps/ollama/bin/ollama-0.24.0`. It is enough for greedy screens. It cannot
  give prompt log-probabilities (needed for likelihood confidence) or be
  LoRA-fine-tuned, so the winning model still has to be staged as safetensors.
- **Storage.** The user quota is ~112 GB. Retrains pass
  `EXTRA="training.save_strategy=no"`, because the HF Trainer otherwise writes a
  final checkpoint plus optimizer state, ~0.5 GB per run. The 2026-10-03 cleanup
  (approved) freed 34 GB, including the Qwen2.5-32B weights.
- **Hardware.** BigCats nodes have 2× A100 40 GB; `--gres=gpu:40gb:2` with TP 2
  fits models up to ~65 GB.

## Verdict (so far)

- Self-training, done properly, does not raise macro-F1 on this task. The
  failure is structural (majority amplification), not label noise, and it holds
  on both an out-of-domain and an in-domain pool.
- Neither the 32B nor any open model screened so far labels HLQC better than
  the fine-tuned 7B student.
- Synthesis remains the only data-expansion route with a real macro-F1 gain
  (+0.037 to +0.071; see `EXPANSION_RESULTS.md`).
