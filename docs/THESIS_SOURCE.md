# Thesis source pack

Compiled 28 September 2026 from the repository at the time of writing. This file is a fact pack for a later drafting model. It is not the thesis.

## Rules for the drafting model

- Do not invent a number, interval, seed, job id, dataset size, or citation. If a figure is not in this file or in a file this file names, leave it out or mark it as not yet measured.
- Do not promote a single-seed point estimate into a claim. The verdict words in the generated tables are the claims.
- Do not compare two runs as an ablation unless this file says they differ by one factor. Training-file overwrites broke several comparisons that look like ablations.
- Rounding in the generated markdown is three decimal places. Do not add digits.
- `docs/EXPERIMENT_LOG.md` has stale rows. The 8 September 2026 self-training row says the scored cell does not exist. The prediction file does exist and is scored below. Prefer prediction files, `docs/BASELINE_RESULTS.md`, and `docs/EXPANSION_RESULTS.md` over older log sentences.
- Generated files, both dated 2026-09-28:
  - `docs/BASELINE_RESULTS.md` — `baseline.eval`. Full grid, paired tests, cross-scheme.
  - `docs/EXPANSION_RESULTS.md` — `selftrain.expansion_table`. One-adapter data expansion. This is the authoritative multi-seed summary for those arms.
  - `outputs/baseline_eval/comparison.csv` — the unrounded grid.
- This compilation recomputed some off-grid point estimates on 28 September 2026 with `src.baseline.eval.score` and, for accuracy deltas only, `_paired_delta_ci`. Those recomputations are labelled below. Gold-macro-F1 deltas for the two-adapter data arms are point estimates. They do not have a paired F1 interval in this file.

## Identity

- Author: Jiawen Liu. Degree: Master of Data Science, Monash University. Submission target in the literature-review proposal: May 2026.
- Supervisors named in the proposal: Levin Kuhlmann (main), Trang Vu (associate).
- Working title, carried from the proposal and still accurate for the executed work: “Bridging the Fidelity Gap: Local Fine-Tuning for Automated Behavioural Coding in Motivational Interviewing.”
- Ethics project ID stated in the proposal: 49359. The proposal says public data during the amendment, with local processing. Do not expand the ethics claim beyond that sentence without the PDF.

## What the proposal planned, and what was actually run

The proposal (May 2026 literature-review proposal) framed three gaps: long-tail codes fail, a proprietary model is a black box that also requires sending session text off site, and expert labels are scarce. Its planned method list is not the method list of this repository.

| Proposal item | Status in this repository |
|---|---|
| Local adaptation of a small model on expert labels a service already holds | Run. Qwen2.5-7B-Instruct, LoRA, HLQC labels, evaluated on MIV6.3A. |
| Rationale before the code (RIFT-style targets) | Run in a different form. Targets are gpt-4o rationales written with the gold label visible (post-hoc distillation), plus GRPO, which is rewarded on the label and format rather than on imitating those rationales. |
| Active learning | Not run. |
| Joint training on MISC and MITI | Not run. Welivita (MITI-mapped) was used as a pseudo-label pool and as a transfer test, not as joint supervised training. |
| Gemma and Qwen3.5 | Not run. The only adapted model is Qwen2.5-7B-Instruct. The synthetic-data teacher is Qwen2.5-32B-Instruct-AWQ. |
| Crank Group coaching data | Not run. |
| Counsellor-wise 5-fold as the primary test | Not the primary test. Primary test is a corpus hold-out: train HLQC, test MIV6.3A. A separate 5-fold on HLQC was run for an AutoMISC-prompt comparison (section below). It is not the MIV result. |
| Fleiss κ ≥ 0.60 as a success bar | Not a result of this project. Cohen’s κ is reported against gold. No new multi-rater study was run. |
| Fine-tuned GPT-4o | Not obtained. GPT-4o cells are in-context only. `docs/BASELINE_RESULTS.md` lists the eight `gpt4o_ft_*` cells as missing. The experiment notes say Azure training does not yield a deployable fine-tuned model. |

Figures in the proposal about human coding time (about 3.5 hours per 30-minute session, about 40 hours to train a coder) and utterance-level human κ (0.31–0.64) are citations inside the proposal, not measurements from this repository. Verify them against the proposal PDF and the papers it cites before using them.

## Research questions the results can answer

These are gap-level questions. Datasets are instruments.

1. Access. Can a small language model, adapted locally on expert labels a service already holds, code motivational-interviewing behaviour well enough that fidelity feedback does not require sending session text to a proprietary model?
2. Rare behaviours. Does that local coder recover behaviours that are rare in a session but linked to outcome (support, affirm, and other sparse codes), or only the behaviours that fill most of a session?
3. Checkable decision. Does training the coder to give a reason before the code make the judgement more accurate, and is that reason actually used, compared with a code and no reason?
4. Labels without a new expert pass. When another expert coding round is too costly, can labels from real unlabelled sessions, or examples written and checked by a larger model, supply the missing rare behaviours?

Cross-scheme transfer, classic classifiers, and the AutoMISC cross-validation are analyses under these questions. They are not extra research questions.

Pipeline structure (adapter count, call count), teacher forcing, and retrieval are tests of whether the *procedure* recovers the tail. They sit under question 2. They did not. The data arms sit under question 4.

## Chapter map (ACL-style names)

Use this order. Do not organise the method chapter as a diary of campaigns.

1. Introduction — fidelity gap, local coding, the four questions.
2. Related work — MISC 2.5, AutoMISC, PEFT, rationale distillation, GRPO, synthetic data and self-training for long-tail labels. State the proposal methods that were not run in one short paragraph, not as if they were contributions.
3. Method — one fixed system, then one change per subsection: task and hierarchy; local LoRA coder; structural matrix (adapters × calls); rationale targets versus label-only targets; GRPO; teacher forcing; data expansion (self-training, synthesis); retrieval. Metrics definitions belong in Experimental Setup, not here.
4. Experimental setup — corpora and sizes, split, hyperparameters, frozen artifacts, metrics, uncertainty, what was not run.
5. Results — mirror the method order, then answer each research question. Lead with the verdict, then the number.
6. Conclusion.
7. Limitations — 10 test conversations, no dev split, post-hoc rationales, overwritten synthetic files, pending jobs, GPT-4o fine-tunes missing, Welivita labels are weak gold.
8. Ethics — proposal ID 49359; public corpora; local weights. Do not invent a data-management claim.
9. Appendix — full per-code tables, missing-cell list, hyperparameter files.

## Task

MISC 2.5 hierarchical coding. Tier 1 (T1) is the group. Tier 2 (T2) is the fine code and must be a child of T1. Counsellor and client have separate vocabularies. A volley is a speaker turn. Context is the previous 3 or 5 volleys. The AutoMISC paper used 3. The local headline setting is 5.

Two calls: a T1 generation, then a T2 generation conditioned on the predicted T1. One call: a single generation emits the rationale (if any) and both labels. `restrict_t2_to_group` is false in `conf/baseline_ft_local_config.yaml`, so parsing allows the full per-speaker T2 vocabulary rather than masking to the predicted group.

## Data

Confirmed with pandas on 28 September 2026 unless noted.

| File | Rows | Role |
|---|---:|---|
| `data/manual/HLQC_balanced_manual.csv` | 1,925 | Supervised training and few-shot exemplars. 10 conversations. Speaker: counsellor 1,040, client 885. Every row has a T2 gold label. |
| `data/manual/MIV6.3A_manual.csv` | 821 | Primary evaluation. 10 conversations. Speaker: counsellor 580, client 241. Every reported MIV number uses these 821 rows. |
| `data/manual/AnnoMI_eval.csv` | 9,699 | Transfer corpus before the shared-code filter. Speaker field is `therapist` / `client`, 133 conversations. Scored subset is below. |
| `data/manual/Welivita_eval.csv` | 20,462 | MITI-mapped counsellor behaviour, weak gold. Also the self-training pool. |

The two primary corpora are disjoint. No evaluation utterance is in a training target. There is no development split: `training.validation_split` is 0.0. Hyperparameters were not tuned per arm.

T2 codes present in MIV gold and absent from HLQC: `TS+`, `AC−`. T2 codes present in HLQC and absent from MIV: `ADW`, `C-`, `CO`, `N+`, `O-`, `RCW`, `WA`.

Support for codes that the data-expansion discussion uses (gold counts):

| Code | HLQC train | MIV eval |
|---|---:|---:|
| SU | 9 | 39 |
| EC | 22 | 39 |
| AF | 23 | 30 |
| GI | 16 | 24 |
| SR | 137 | 134 |
| FI | 198 | 56 |
| CR | 85 | 42 |
| OQ | 90 | 79 |
| TS+ | 0 | 2 |
| AC− | 0 | 1 |

`SU` is 9 of 1,040 counsellor training rows. That is why rare-class oversampling exists in the GRPO config.

### Frozen artifacts

From `docs/EXPERIMENTS.md`:

- Few-shot exemplars, stratified: 6 counsellor T1, 18 counsellor T2, 3 client T1, 14 client T2. Paths: `data/fewshot/exemplars_ctx3.json`, `data/fewshot/exemplars.json` (ctx 5).
- Distilled rationales: `data/fine_tuning/rationales/hlqc_rationales_ctx5.json` and the ctx3 file. gpt-4o saw the prompt plus the gold label and wrote a one-to-two sentence justification. These rationales are post-hoc. They may not be a reason that would have produced the label on its own.
- Out-of-fold predicted T1 for teacher forcing: `data/fine_tuning/oof_t1/hlqc_oof_t1_ctx5.json` (and ctx3). Five conversation folds. The production T1 adapter memorises HLQC, so its in-sample T1 is too clean to use as training noise.

Context length is frozen into exemplars and rationales. A ctx5 rationale can cite text that a ctx3 prompt does not contain. Do not treat context length as a test-time knob independent of those files.

## Model and supervised-training settings

Source: `conf/baseline_ft_local_config.yaml`. The single-call SFT block in `conf/grpo_config.yaml` copies the same optimisation settings.

- Base model: `Qwen/Qwen2.5-7B-Instruct`.
- LoRA: r = 16, alpha = 32, dropout = 0.05. Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- Precision: bf16. Attention: sdpa. `fp16` is false.
- Epochs: 3. Learning rate: 1e-4. Scheduler: cosine. Warmup ratio: 0.05. Weight decay: 0.0. Max grad norm: 1.0. Seed: 42 unless a later job overrides it.
- Batch size 1, gradient accumulation 4. Gradient checkpointing on.
- `max_length` 4096. `max_target_length` 256. The config comment records rationale token lengths on the frozen ctx5 file with this tokenizer: mean 69, p99 138, max 247.
- Inference `max_new_tokens` 256. Input window 4096 for fine-tuned and zero-shot arms, 8192 for few-shot.
- Checkpoint during prediction every 25 utterances. A short file is a partial run. `docs/BASELINE_RESULTS.md` says not to read a partial cell as a result. The coverage table in that file shows no partial cells in the intended grid as of the 28 September regeneration.

## Structural matrix

From `docs/EXPERIMENTS.md`. Two adapters and one call cannot exist.

|  | 2 adapters | 1 adapter |
|---|---|---|
| 2 calls | A: `ft_bare`, `ft_rat`, `grpo_pair_{dec,joint}` | B: `ft1mix_*`, `ft1seq_*`, `grpo_mix_{dec,joint}` |
| 1 call | impossible | F: `sc_ft_bare`, `sc_ft_rat`, `sc_grpo*` |

- `ft_bare` trains on the label only. `ft_rat` trains on distilled rationale plus label.
- `ft1mix` trains one adapter on the mixed union of T1 and T2 examples. `ft1seq` trains T1 then T2 on the same adapter; it forgets T1.
- Inf-Bare asks for the label only. Inf-CoT asks for a rationale first.
- Compare A with B to vary adapter count (calls fixed at 2). Compare B with F to vary call count (adapters fixed at 1). Do not interpret `ft_bare` versus `sc_ft_bare` as a single effect.
- From 28 September 2026 the paired tests in `baseline.eval` use `ft1mix_bare` as the reference (`PAIRED_REFERENCE` in `src/baseline/eval.py`). Earlier write-ups that call `ft_bare` the reference are outdated for the paired tables.

The one-adapter mixed bare-label model at context 5 is the backbone of the data-expansion table. The two-adapter `ft_bare` model remains the backbone of the earlier self-training and synthesis retrains. Those are different systems. Do not pool them.

## GRPO settings

Source: `conf/grpo_config.yaml` for single-call, `conf/grpo_tc_config.yaml` for two-call. The two-call file says its knobs are copied so the arms differ in structure, not in optimisation.

- Init: `sc_ft_bare` for single-call. Cold start (`init_from: none`) is an ablation. Two-call arms warm-start from the matching SFT arm (`ft_bare` for pair, `ft1mix_bare` for mix).
- G = 8 (`num_generations`). Epochs: 2. Learning rate: 1e-6. Temperature: 1.0. Top-p: 1.0. `max_completion_length` 256.
- KL beta: 0.02. Loss: `dr_grpo`. Epsilon: 0.2. `num_iterations`: 1.
- `scale_rewards` must stay `none`. The config explains that group scaling would cancel the rare-class weight.
- Rare-class weighting on by default: power 0.5, floor 0.5, ceiling 3.0, `oversample_max_copies` 2. Weights use training counts only, per speaker.
- Checkpoint selection: conversation hold-out of HLQC, `val_folds` 7, `val_fold` 0, chosen before training because that fold holds 369 rows (19%) and 22 T2 codes. MIV is not read for selection. `val_max_examples` 200. `min_mean_rationale_words` 4.0.
- GRPO seeds in the scored tables are 0, 1, and 2. SFT seed in the yaml is 42.
- Campaign note in `docs/experiments/2026-08-26-two-call-grpo.md`: joint two-call runs used G = 6 rather than G = 8 to fit a 24-hour wall. The checked-in yaml still says 8. Do not silently pick one. Report the yaml as the declared setting and the campaign note as the recorded exception for joint.

Reward components are label correctness, hierarchy consistency, and output format. The exact weights are in `docs/GRPO.md` and the training code. Do not invent weights if they are not re-read from that file.

## Teacher forcing

Defined in `conf/baseline_ft_local_config.yaml` under `training.tf_modes`.

- `full`: T2 always conditioned on gold T1 during training (`p_gold` = 1).
- `zero`: always conditioned on out-of-fold predicted T1 (`p_gold` = 0).
- `dyn`: linear from 1 to 0 across epochs. At 3 epochs the schedule is 1, 0.5, 0.
- At inference, T2 always sees the predicted T1. The mismatch is the thing being measured.
- Cell A (19 August 2026) freezes T1 and retrains only T2. The three cell-A schedules share a staged constant-learning-rate procedure, so `tffull` is not the original `ft_bare`.
- Cell B (8–9 September 2026) rebuilds the one Mix adapter. Seed-1 replicates moved T2 by up to about 0.08, which is why a one-adapter gap below that floor is not treated as a result in the paired tables. Context-3 teacher forcing was abandoned because the disk filled. It is not a result.

## Metrics and uncertainty

From `src/baseline/eval.py` and the headers of the two generated result files.

- Accuracy and Cohen’s κ on the gold label.
- Macro-F1 (gold): every code that appears in the test gold, including `TS+` and `AC−`, which HLQC-trained models cannot have learned. This is the paper-comparable macro-F1.
- Macro-F1 (learnable): drops `TS+` and `AC−`. Use this when comparing HLQC-trained arms with each other.
- Macro-F1 (all): includes codes the model predicted that have no gold support. It is in the tables. Do not make it the headline.
- Bracketed intervals are 95% conversation-clustered bootstrap intervals. Resample unit is the conversation (10 conversations on MIV), not the row. `N_BOOT = 10000`, `BOOT_SEED = 0`, `CI_LEVEL = 0.95`.
- Where a cell has several seeds, `±` is the standard deviation across seeds. The bracket remains the test-sampling interval.
- Paired Δ is arm minus reference on the same utterances. McNemar p is the two-sided exact test on utterances where the arms disagree.
- Grid verdicts in `BASELINE_RESULTS.md`: `ns` if the Δ interval includes 0; `<floor` if the absolute Δ is inside the seed-noise floor; `real` if it clears both. The floor is 1.96 times the standard error of a difference of single-seed runs, from measured seed SDs. If an arm has no replicates, the fallback floor is 0.08. Plain one-adapter baseline T2 accuracy spans only 0.010 across seeds 42, 1, and 2, so that arm’s floor is much smaller than 0.08. Staged teacher-forcing arms still use the 0.08 fallback when they lack their own seed spread.
- Expansion verdicts in `EXPANSION_RESULTS.md`, which are the ones to use for data expansion: `real` requires the same sign at every matched seed, an interval that excludes 0, and a mean difference larger than the seed-noise band. `sign-flips` means the interval excludes 0 but seeds disagree on direction. `single-seed` means the interval excludes 0 but only one seed exists. `ns` means the interval includes 0. `pending` means the arm is not scored.

A difference must clear test-sampling noise and seed noise before it is written as a finding.

## Paper reference (not a result of this repository)

`docs/BASELINE_RESULTS.md` quotes the AutoMISC paper, GPT-4.1, hierarchical, 3 context volleys, clean set:

| Tier | Scope | Metric | Paper |
|---|---|---|---:|
| T1 | counsellor | accuracy | 0.82 |
| T1 | client | accuracy | 0.88 |
| T2 | counsellor | accuracy | 0.68 |
| T2 | counsellor | macro-F1 | 0.42 |
| T2 | client | accuracy | 0.76 |
| T2 | client | macro-F1 | 0.41 |

These are counsellor-or-client figures, not all-speaker figures. Do not place an all-speaker accuracy of 0.67 next to the paper’s 0.68 and call them the same comparison. The closest local all-speaker numbers are in the tables below; the counsellor rows are the ones that line up with the paper’s scope. The paper model is GPT-4.1. This project’s frontier rows are GPT-4o, in context only, and are not that paper number.

## Primary grid — context 5, the headline setting

All figures in this section are copied from `docs/BASELINE_RESULTS.md` (generated 2026-09-28). n = 821 unless a scope row says otherwise (counsellor 580, client 241). Intervals are conversation-clustered 95% CIs.

### T2, all speakers, selected conditions

| Condition | Seeds | Accuracy | κ | Macro-F1 gold | Macro-F1 learnable |
|---|---:|---|---|---|---|
| GPT-4o ZS-CoT | 1 | 0.664 [0.628–0.702] | 0.637 | 0.501 | 0.528 |
| GPT-4o FS-CoT | 1 | 0.680 [0.653–0.714] | 0.655 | 0.543 | 0.559 |
| GPT-4o FS-Bare | 1 | 0.669 [0.636–0.697] | 0.643 | 0.539 | 0.564 |
| Qwen ZS-Bare | 1 | 0.381 [0.351–0.409] | 0.341 | 0.256 | 0.266 |
| Qwen ZS-CoT | 1 | 0.442 [0.413–0.471] | 0.401 | 0.325 | 0.343 |
| Qwen FS-CoT | 1 | 0.470 [0.443–0.497] | 0.434 | 0.394 | 0.418 |
| FT-Bare Inf-Bare (2 adapters) | 1 | 0.660 [0.623–0.693] | 0.629 | 0.424 | 0.458 |
| FT-Bare Inf-CoT | 1 | 0.646 [0.605–0.680] | 0.614 | 0.397 | 0.428 |
| FT-Rat Inf-CoT | 1 | 0.620 [0.574–0.660] | 0.587 | 0.410 | 0.443 |
| FT-Rat Inf-Bare | 1 | 0.510 [0.489–0.535] | 0.472 | 0.377 | 0.391 |
| FT1-Mix-Bare Inf-Bare (1 adapter) | 3 | 0.673 ± 0.005 [0.637–0.707] | 0.643 ± 0.006 | 0.457 ± 0.017 | 0.493 ± 0.019 |
| FT1-Mix-Bare Inf-CoT | 1 | 0.666 [0.634–0.703] | 0.637 | 0.473 | 0.490 |
| FT1-Seq-Bare Inf-Bare | 1 | 0.559 [0.514–0.600] | 0.519 | 0.360 | 0.389 |
| SC FT-Bare (1 call) | 1 | 0.653 [0.614–0.685] | 0.622 | 0.484 | 0.496 |
| SC FT-Rat | 1 | 0.564 [0.526–0.591] | 0.528 | 0.366 | 0.396 |
| SC GRPO | 3 | 0.658 ± 0.003 [0.627–0.688] | 0.629 ± 0.003 | 0.466 ± 0.003 | 0.479 ± 0.002 |
| SC GRPO, no rare-class weighting | 3 | 0.657 ± 0.008 [0.624–0.686] | 0.628 ± 0.009 | 0.464 ± 0.017 | 0.480 ± 0.013 |
| SC GRPO cold start | 1 | 0.435 [0.385–0.483] | 0.395 | 0.332 | 0.344 |
| TC GRPO Pair-Dec | 3 | 0.647 ± 0.001 [0.606–0.682] | 0.615 | 0.397 ± 0.001 | 0.429 ± 0.002 |
| TC GRPO Mix-Dec | 3 | 0.667 ± 0.001 [0.635–0.704] | 0.638 | 0.472 ± 0.003 | 0.489 ± 0.003 |
| TC GRPO Mix-Joint | 3 | 0.661 ± 0.005 [0.630–0.696] | 0.631 | 0.469 ± 0.004 | 0.486 ± 0.004 |
| AG retrieval CoT (on the two-call fine-tune) | 1 | 0.407 [0.378–0.433] | 0.369 | 0.303 | 0.321 |

κ and F1 intervals are in `BASELINE_RESULTS.md`. They are omitted here only to keep the row readable. Do not invent them.

### T2 counsellor, the paper’s scope, context 5

| Condition | Accuracy | Macro-F1 gold |
|---|---|---|
| GPT-4o FS-CoT | 0.666 [0.636–0.702] | 0.543 |
| FT-Bare Inf-Bare | 0.634 [0.603–0.661] | 0.445 |
| FT1-Mix-Bare Inf-Bare (3 seeds) | 0.654 ± 0.001 [0.621–0.687] | 0.497 ± 0.009 |
| SC FT-Bare | 0.622 [0.584–0.654] | 0.459 |
| Qwen ZS-Bare | 0.297 [0.269–0.324] | 0.245 |

Client T2 for FT-Bare Inf-Bare is accuracy 0.722 [0.631–0.798], gold macro-F1 0.402. Client intervals are wide because there are 241 client rows inside 10 conversations.

### T1, context 5, all speakers (selected)

| Condition | Accuracy |
|---|---|
| FT-Bare Inf-Bare | 0.789 [0.757–0.825] |
| FT1-Mix-Bare Inf-Bare (3 seeds) | 0.786 ± 0.006 [0.757–0.816] |
| FT1-Seq-Bare Inf-Bare | 0.602 [0.571–0.630] |
| SC FT-Bare | 0.790 [0.758–0.818] |
| Qwen ZS-Bare | 0.599 (paired table; arm accuracy) |

The sequential one-adapter schedule loses T1. That is the forgetting result. The mixed one-adapter schedule does not.

### Structural contrast (one axis at a time)

Copied from `BASELINE_RESULTS.md`. Learnable macro-F1 is the T2 column.

Context 5, bare-label training:

| Cell | Structure | Inf | T1 acc | T2 acc | T2 learnable F1 |
|---|---|---|---|---|---|
| A | 2 adapters, 2 calls | bare | 0.789 [0.757–0.825] | 0.660 [0.623–0.693] | 0.458 [0.414–0.519] |
| A | 2 adapters, 2 calls | cot | 0.777 [0.743–0.816] | 0.646 [0.605–0.680] | 0.428 [0.382–0.488] |
| B | 1 adapter, mixed | bare | 0.786 ± 0.006 [0.757–0.816] | 0.673 ± 0.005 [0.637–0.707] | 0.493 ± 0.019 [0.448–0.548] |
| B | 1 adapter, mixed | cot | 0.788 [0.760–0.818] | 0.666 [0.634–0.703] | 0.490 [0.434–0.544] |
| B | 1 adapter, sequential | bare | 0.602 [0.571–0.630] | 0.559 [0.514–0.600] | 0.389 [0.343–0.432] |
| F | 1 adapter, 1 call | bare | 0.790 [0.758–0.818] | 0.653 [0.614–0.685] | 0.496 [0.423–0.567] |

Context 3, bare-label, mixed one-adapter, Inf-Bare: T1 acc 0.798 [0.771–0.825], T2 acc 0.687 [0.655–0.718], learnable F1 0.513 [0.463–0.575]. That learnable F1 is the highest single-run figure in the structural grid. It is one seed. It is not the data-expansion result.

Context 5, rationale training, Inf-CoT, two adapters: T2 acc 0.620 [0.574–0.660], learnable F1 0.443. Rationale training did not beat label-only training. Asking a rationale-trained model for a bare label is worse still (T2 acc 0.510).

### Paired tests against FT1-Mix-Bare, context 5, Inf-Bare, T2 accuracy

Reference accuracy is 0.671 on the original (seed-42) run. Δ is arm minus that reference.

| Arm | Arm acc | Δ acc | 95% CI | Verdict |
|---|---:|---:|---|---|
| Qwen ZS-Bare | 0.381 | −0.290 | [−0.324, −0.250] | real |
| Qwen FS-Bare | 0.402 | −0.269 | [−0.316, −0.220] | real |
| FT-Bare Inf-Bare | 0.660 | −0.011 | [−0.028, +0.008] | ns |
| FT-Rat Inf-Bare | 0.510 | −0.161 | [−0.193, −0.127] | real |
| FT1-Seq-Bare | 0.559 | −0.112 | [−0.133, −0.088] | real |
| FT-Bare TF-Full | 0.647 | −0.024 | [−0.051, +0.006] | ns |
| FT-Bare TF-Dyn | 0.635 | −0.037 | [−0.049, −0.024] | below floor |
| FT-Bare TF-Zero | 0.613 | −0.058 | [−0.092, −0.023] | below floor |
| FT1-Mix TF-Full (2 seeds) | 0.610 | −0.061 | [−0.090, −0.031] | below floor |
| FT1-Mix TF-Dyn (2 seeds) | 0.704 | +0.033 | [+0.011, +0.052] | below floor |
| FT1-Mix TF-Zero (2 seeds) | 0.653 | −0.018 | [−0.041, +0.002] | ns |
| SC FT-Bare | 0.653 | −0.018 | [−0.040, +0.001] | ns |

Inf-CoT, context 5, T2, same reference family (reference accuracy 0.666):

| Arm | Arm acc | Δ | Verdict |
|---|---:|---|---|
| FT-Bare Inf-CoT | 0.646 | −0.021 [−0.042, +0.003] | ns |
| FT-Rat Inf-CoT | 0.620 | −0.046 [−0.075, −0.015] | below floor |
| AG retrieval | 0.407 | −0.259 [−0.299, −0.225] | real |
| TC GRPO Pair-Dec (3 seeds) | 0.647 | −0.019 [−0.040, +0.005] | ns |
| TC GRPO Mix-Dec (3 seeds) | 0.666 | +0.000 [−0.005, +0.005] | ns |
| TC GRPO Mix-Joint (3 seeds) | 0.664 | −0.002 [−0.017, +0.012] | ns |
| SC GRPO (3 seeds) | 0.655 | −0.011 [−0.044, +0.029] | ns |
| SC GRPO unweighted (3 seeds) | 0.650 | −0.016 [−0.050, +0.022] | ns |
| SC GRPO cold start | 0.435 | −0.231 [−0.275, −0.191] | real |

Reading that is safe:

- Local fine-tuning beats the untuned Qwen by a large margin. Zero-shot Qwen T2 accuracy 0.381 versus about 0.66–0.67 after LoRA. That gap is `real`.
- One shared adapter (mixed) matches two adapters. The paired gap is `ns`. Sequential training of one adapter does not match them. That gap is `real`.
- One call matches two calls within noise (`SC FT-Bare` is `ns`).
- GRPO matches its supervised warm start. It does not beat it. Cold start collapses (`real` loss). Turning rare-class weighting off does not move the aggregate (`ns`).
- Teacher forcing can change behaviour (next section) without producing a T2 accuracy gain that clears the seed floor.
- Retrieval few-shot on the fine-tuned two-call model, Inf-CoT, drops T2 accuracy from the 0.65 region to 0.407. That loss is `real`. One-adapter retrieval is not scored yet (jobs below).

### Teacher-forcing mechanism (cell A, context 5, Inf-Bare)

From `docs/experiments/2026-08-19-teacher-forcing.md`. These are conditional accuracies, not the marginal paired verdicts above.

| Arm | Out-of-group rate | T2 acc given T1 correct | T2 acc given T1 wrong | Marginal T2 acc |
|---|---:|---:|---:|---:|
| `ft_bare` (unstaged) | 0.0% | 0.836 | 0.000 | 0.660 |
| `ft_bare_tffull` (staged control) | 0.0% | 0.819 | 0.000 | 0.647 |
| `ft_bare_tfdyn` | 15.6% | 0.756 | 0.179 | 0.635 |
| `ft_bare_tfzero` | 13.4% | 0.741 | 0.133 | 0.613 |

T1 accuracy is 0.789281 and κ is 0.753486 for all four arms to six decimals in that note, because they share one T1 adapter. Against the staged control, `tfdyn` gains 0.0378 on rows where T1 is wrong and loses 0.0499 on rows where T1 is right, net −0.0122. `tfzero` gains 0.0280 and loses 0.0621, net −0.0341. The note states that T1 is wrong on 21% of rows and right on 79%. The override does not pay for the damage on the majority.

### Recovery probe (two-call)

`docs/experiments/2026-08-26-two-call-grpo.md` summarises the probe on `mix_dec` seed 0: given a wrong injected T1, T2 is correct on only about 0.07 of rows and emits a code inside the wrong group on about 0.84 of rows. Those two figures are written with a tilde in the note. Do not add more decimal places. The probe output file is not in the local tree. The definition is in `src/baseline/recovery.py`: recovery rate is P(T2 correct | wrong T1); follow rate is P(T2 is a child of the wrong T1).

The same note’s conclusion, which the paired tables above support: decoupled and joint are a wash, mix stays at least as strong as pair, and RL does not beat SFT. The note’s sentence that `ft1mix_bare` is the single best T2 cell was written before the data-expansion runs. It is not the current ranking.

### Faithfulness probe

From `BASELINE_RESULTS.md`. A rationale from a different utterance is forced as a prefix. Net flip = swap flip minus control flip. Donor-match is the rate of landing on the donor’s code.

T2, context 5:

| Arm | Control flip | Swap flip | Net flip | Donor-match |
|---|---:|---:|---:|---:|
| `sc_ft_rat` | 0.041 | 0.739 | +0.698 | 0.668 |
| `sc_grpo` seed 0 | 0.010 | 0.808 | +0.798 | 0.721 |
| `sc_zs` | 0.066 | 0.868 | +0.803 | 0.757 |

The generated file says to read `sc_zs` first. A model with no rationale training has the highest net flip, so a forced prefix steering the next tokens is what an autoregressive model does. The absolute flip is not evidence that training made the rationale load-bearing. What survives is the comparison between arms.

## Data expansion — one adapter, the current main comparison

Source: `docs/EXPANSION_RESULTS.md`, generated 2026-09-28. Every arm is a retrain of FT1-Mix-Bare. Baseline is the same recipe on HLQC only, three seeds. Matched seeds are 42, 1, and 2.

### Official verdicts, T2, all speakers

| Arm | Seeds | T2 acc | Δ acc [95% CI] | Acc verdict | Macro-F1 | Δ F1 [95% CI] | F1 verdict |
|---|---:|---|---|---|---|---|---|
| Synthesis v2 proto (dialogue windows) | 3/3 | 0.687 ± 0.023 | +0.014 [+0.001, +0.025] | sign-flips | 0.510 ± 0.023 | +0.053 [+0.025, +0.077] | real |
| Synthesis v3, HLQC topic mix | 1/1 | 0.698 | +0.027 [−0.003, +0.055] | ns | 0.506 | +0.047 [+0.014, +0.094] | single-seed |
| Synthesis v3, MIV6.3A topic mix | 1/1 | 0.689 | +0.018 [−0.024, +0.048] | ns | 0.528 | +0.069 [+0.008, +0.135] | single-seed |
| Self-training | 0/0 | — | — | pending | — | — | pending |
| Synthesis v1 | 0/0 | — | — | pending | — | — | pending |
| Synthesis v2 boundary | 0/0 | — | — | pending | — | — | pending |

Speaker split for proto macro-F1: counsellor 0.520 ± 0.008, Δ +0.022 [−0.005, +0.057], `ns`. Client 0.499 ± 0.050, Δ +0.086 [+0.025, +0.140], `real`. The F1 gain is client-driven.

T1 accuracy for proto: 0.797 ± 0.021, Δ +0.011 [−0.003, +0.022], `ns`. The added data does not show a T1 cost on this test.

Safe claim: on the one-adapter backbone, dialogue-window synthetic data raised gold macro-F1 by about five points, and that lift is stable in sign across three seeds and is concentrated on the client. It did not raise accuracy in a way that survives the seed check, because seed 42 gained and seed 2 lost. Do not write “accuracy improved to 0.713.” That number is seed 42 alone.

### Per-seed point estimates (not the verdict)

Recomputed 28 September 2026 with `eval.score` on the prediction CSVs, no bootstrap. The expansion table is the claim. These rows exist so a drafter can see why the accuracy verdict is `sign-flips`.

| Run | File | T2 acc | κ | Gold F1 | Learnable F1 | T1 acc |
|---|---|---:|---:|---:|---:|---:|
| HLQC-only, seed 42 (published `ft1mix`) | `data/annotated/baseline/qwen_ft1mix_bare_inf_bare_ctx5.csv` | 0.6711 | 0.6416 | 0.4593 | 0.4961 | 0.7881 |
| HLQC-only, seed 1 | `data/annotated/base_1a_s1/...` | 0.6687 | 0.6378 | 0.4384 | 0.4735 | 0.7783 |
| HLQC-only, seed 2 | `data/annotated/base_1a_s2/...` | 0.6784 | 0.6499 | 0.4727 | 0.5105 | 0.7905 |
| Proto, seed 42 | `data/annotated/synth_v2_proto_1a/...` | 0.7125 | 0.6875 | 0.5310 | 0.5534 | 0.8210 |
| Proto, seed 1 | `data/annotated/synth_v2_proto_1a_s1/...` | 0.6784 | 0.6511 | 0.5127 | 0.5271 | 0.7832 |
| Proto, seed 2 | `data/annotated/synth_v2_proto_1a_s2/...` | 0.6699 | 0.6420 | 0.4853 | 0.5108 | 0.7856 |
| Topic mix, HLQC topics, seed 42 | `data/annotated/synth_v3_train_1a/...` | 0.6979 | 0.6720 | 0.5061 | 0.5266 | 0.7905 |
| Topic mix, MIV topics, seed 42 | `data/annotated/synth_v3_eval_1a/...` | 0.6894 | 0.6635 | 0.5281 | 0.5436 | 0.7808 |

Seed labels: the unsuffixed run is the yaml default seed 42, and `comparison.csv` stores it with a missing seed. Seeds 1 and 2 in `comparison.csv` match the `base_1a_s1` and `base_1a_s2` files to four decimals (0.6687 and 0.6784). The retrain logs do not echo the integer seed. Do not claim a log line that is not there. The two later baselines are not copies of seed 42: their T2 predictions differ from the published file.

Accuracy deltas, proto minus its matched baseline: seed 42 +0.0414, seed 1 +0.0097, seed 2 −0.0085. Same sign is required for a `real` accuracy claim. It fails.

### True positives on MIV, proto minus matched one-adapter baseline

Aligned on `conv_id|corp_utt_idx`, n = 821. Integer counts, 28 September 2026. A positive number means the proto model got more gold instances of that code right.

| Code | Gold n | Seed 42 | Seed 1 | Seed 2 |
|---|---:|---:|---:|---:|
| SU | 39 | +7 | +9 | +8 |
| AF | 30 | +7 | +11 | +5 |
| EC | 39 | +6 | −2 | +1 |
| GI | 24 | −1 | +11 | −1 |
| SR | 134 | +13 | −17 | −3 |
| FI | 56 | −4 | −4 | −11 |
| CR | 42 | −2 | −8 | −7 |
| R+ | 32 | +6 | +5 | −1 |
| R− | 25 | +4 | +6 | +3 |
| N | 125 | −6 | −7 | −2 |

`SU` and `AF` rise on all three seeds. `SR` rises only on seed 42 and falls on seed 1. `FI` falls on all three. Do not describe the seed-42 `SR` gain as the effect.

### What those one-adapter models were trained on

Counts are pandas lengths of the CSVs on disk on 28 September 2026. `synth_target == 1` marks a synthetic row. A row with a non-null T2 gold label is a loss target. HLQC contributes 1,925 labelled rows in each augmented file.

| On-disk file | Rows | Synthetic rows | Labelled rows (HLQC + synthetic labels) | Extra labelled beyond HLQC |
|---|---:|---:|---:|---:|
| `data/synth/aug/Qwen2_5-32B-Instruct-AWQ_proto.csv` | 2,641 | 716 | 2,277 | 352 |
| `data/synth/aug/Qwen2_5-32B-Instruct-AWQ_proto_train.csv` (HLQC topic mix) | 2,773 | 848 | 2,334 | 409 |
| `data/synth/aug/Qwen2_5-32B-Instruct-AWQ_proto_eval.csv` (MIV topic mix) | 2,713 | 788 | 2,312 | 387 |
| `data/synth/aug/Qwen2_5-32B-Instruct-AWQ_boundary.csv` | 2,570 | 645 | 2,217 | 292 |
| `data/synth/aug/Qwen2_5-32B-Instruct-AWQ.csv` (v1) | 4,556 | 862 labelled + 1,769 unlabelled setup | 2,787 | 862 |

The one-adapter proto retrains used the 2,641-row proto file (352 labelled synthetic targets). That file is a 26 September re-verification overwrite. It is not the file the two-adapter proto model was trained on. The 26 September analysis recorded that earlier file as 2,653 rows and 346 labelled synthetic targets. That earlier file is not on disk now. Do not treat 2,653 as a count of the current CSV, and do not treat the two-adapter proto score and the one-adapter proto score as an adapter-count ablation.

The current boundary CSV (2,570 rows, 292 extra labels) is also a later overwrite. The two-adapter boundary model was trained earlier. The 26 September analysis recorded that training file as 2,616 rows with 653 labelled synthetic targets. Same warning: not an ablation against the current file, and the one-adapter boundary retrain is still pending.

Generation settings that are recorded, not re-derived here: the teacher is Qwen2.5-32B-Instruct-AWQ. Proto and topic-mix jobs used 50 dialogue windows. v1 asked for 20 examples of each of 36 codes and 80 of `SU`, `EC`, `AF`, `GI`, `TS+`, `AC−` (1,080 requested last turns). Raw v1 file on disk: `data/synth/raw/Qwen2_5-32B-Instruct-AWQ.csv`, 3,313 rows. Temperature and the exact keep rule should be copied from `src/synth/` and the teacher logs if the drafter needs them. This pack does not re-quote diversity statistics or the self-judge scores. Those live in `outputs/synth/teacher_169724.log`, `teacher_169725.log`, and the judge CSVs under `outputs/synth/`. Do not invent them. The judge is the same 32B model scoring its own text. Do not call it an independent rater even if those scores are later copied in.

## Data expansion — two adapters, single seed, already scored

These runs use `ft_bare` (two adapters, Inf-Bare, context 5), not the one-adapter backbone. One seed (yaml default 42; the meta sidecar records `seed: null`). They are not in the expansion table, which correctly lists the one-adapter repeats as pending.

Recomputed 28 September 2026 against `data/annotated/baseline/qwen_ft_bare_inf_bare_ctx5.csv`, aligned n = 821. Accuracy intervals are conversation-clustered paired 95% CIs from `_paired_delta_ci` (`N_BOOT = 10000`, seed 0). Gold-F1 deltas are point estimates only.

Reference: T2 acc 0.6602, gold F1 0.4240, learnable F1 0.4579, T1 acc 0.7893.

| Arm | Job (meta) | T2 acc | Δ T2 acc [95% CI] | Gold F1 (point) | Δ gold F1 (point) | T1 acc | Δ T1 acc [95% CI] |
|---|---|---:|---|---:|---:|---:|---|
| Self-train | 167401, finished 2026-09-08 | 0.6955 | +0.0353 [+0.0162, +0.0536] | 0.4316 | +0.0076 | 0.8234 | +0.0341 [+0.0241, +0.0449] |
| Synth v1 | 167868 (campaign note) | 0.6468 | −0.0134 [−0.0447, +0.0183] | 0.4436 | +0.0196 | 0.7747 | −0.0146 [−0.0358, +0.0036] |
| Synth v2 proto | two-adapter retrain | 0.6711 | +0.0110 [−0.0152, +0.0388] | 0.4698 | +0.0459 | 0.7868 | −0.0024 [−0.0249, +0.0244] |
| Synth v2 boundary | two-adapter retrain | 0.6114 | −0.0487 [−0.0913, −0.0073] | 0.4452 | +0.0212 | 0.7613 | −0.0280 [−0.0507, +0.0000] |

Self-training accuracy intervals exclude 0 on both tiers. The gold-F1 change is a point estimate of +0.008 and must not be called significant, because this compilation did not compute its interval. Net T2 correctness is +29 utterances (0.0353 × 821). Of the per-code true-positive changes versus `ft_bare`: SR +20 (65 → 85 of 134), SU +5 (8 → 13 of 39), AF +5 (21 → 26 of 30), GI +4, R+ −8. The accuracy gain is mostly simple reflection, not the rarest codes.

Self-training file `data/selftrain/aug/welivita_mi_ft_bare_ctx5.csv`: 21,391 rows, 8,804 with a T2 label, 12,587 unlabelled. Labelled speaker split: counsellor 7,563, client 1,241. Subtracting HLQC (1,040 counsellor, 885 client) leaves 6,523 counsellor and 356 client accepted pseudo-labels, 6,879 in total. Pool file `data/selftrain/pseudo/welivita_mi_ft_bare_ctx5.csv`: 20,462 rows. The keep rule used in the campaign notes is hierarchy-valid and confidence at least 0.8, with a cap of 300 on head codes and no cap on `SU`, `EC`, `AF`, `GI`, `TS+`, `AC−`. Re-read `src/selftrain/select.py` before stating a threshold as code rather than as this note.

Synth v1 on two adapters: accuracy interval includes 0. Gold F1 point estimate is up. Not a significant accuracy result.

Synth v2 proto on two adapters: accuracy interval includes 0. Gold F1 point estimate 0.470 versus 0.424. No paired F1 interval in this pack. Do not import the one-adapter `real` F1 verdict onto this run. The training file was overwritten; see the warning above.

Synth v2 boundary on two adapters: T2 accuracy fell, and the accuracy interval excludes 0 on the negative side. That is a loss on the single seed that was trained, on a file that was later overwritten. The one-adapter repeat is pending. Do not generalise it further.

## Classic classifiers

Scored 28 September 2026 with `eval.score` on `data/annotated/classic/`, point estimates, no bootstrap in this pack. Same 821 MIV rows, context 5, child-code mask as recorded in the analysis notes. These models are not in `baseline.eval`’s grid.

| Model | T2 acc | T2 κ | T2 gold F1 | T2 learnable F1 | T1 acc |
|---|---:|---:|---:|---:|---:|
| n-gram logreg, class-weighted | 0.4641 | 0.4074 | 0.2097 | 0.2264 | 0.6054 |
| n-gram SVM, class-weighted | 0.4519 | 0.3928 | 0.1931 | 0.2085 | 0.5822 |
| n-gram logreg | 0.4385 | 0.3732 | 0.1451 | 0.1567 | 0.5773 |
| RoBERTa-base, class-weighted | 0.4409 | 0.3901 | 0.1845 | 0.1993 | 0.6760 |
| Frozen Qwen embeddings, logreg, class-weighted | 0.3703 | 0.3154 | 0.1822 | 0.1967 | 0.5493 |

Best classic T2 accuracy in this set is 0.464, against FT-Bare at 0.660 on the same test. RoBERTa has the highest classic T1 accuracy (0.676). Client κ for the n-gram arms was previously summarised as 0.028–0.143; that range is not recomputed in this pack, so do not add a new client-κ figure without scoring it.

## Cross-scheme transfer

From `docs/BASELINE_RESULTS.md`. The model is the two-adapter FT-Bare Inf-Bare context-5 coder, or zero-shot, with no retraining on the target corpus. Metrics use the shared vocabulary only. Intervals are conversation-clustered.

AnnoMI. Client T1, shared codes C, S, N, n = 4,807, coverage 100%, OOV predictions 0%:

| Arm | Accuracy | Macro-F1 shared |
|---|---|---|
| ZS-Bare | 0.709 [0.669–0.747] | 0.470 [0.436–0.502] |
| FT-Bare | 0.716 [0.673–0.757] | 0.498 [0.456–0.543] |

The two accuracy intervals overlap. Fine-tuning does not clearly move AnnoMI client T1.

AnnoMI. Counsellor T2, shared codes OQ, CQ, SR, CR, GI, n = 2,628:

| Arm | Accuracy | Macro-F1 shared | OOV pred |
|---|---|---|---:|
| ZS-Bare | 0.281 [0.249–0.314] | 0.305 [0.282–0.327] | 55.1% |
| FT-Bare | 0.504 [0.471–0.539] | 0.528 [0.499–0.558] | 21.7% |

Welivita / MITI, weak gold, counsellor T2, shared codes GI, ADW, CR, SU, AF, CQ, DI, SR, ADP, OQ, CO, EC, WA, n = 1,501. The self-trained arm is excluded here because it trained on this corpus.

| Arm | Accuracy | Macro-F1 shared | OOV pred |
|---|---|---|---:|
| ZS-Bare | 0.142 [0.119–0.167] | 0.183 [0.153–0.209] | 54.5% |
| FT-Bare | 0.289 [0.258–0.320] | 0.296 [0.264–0.322] | 15.1% |

`EXPANSION_RESULTS.md` repeats the zero-shot rows only. Its ladder has one fine-tuned rung: MIV6.3A cross-corpus, FT1-Mix baseline, counsellor T2 accuracy, seed 42, full MISC vocabulary, 0.653, with the interval printed as unavailable. One-adapter AnnoMI and Welivita runs are pending (jobs 170070–170073).

Do not compare the AnnoMI T2 accuracy of 0.504 with the MIV all-speaker accuracy of 0.660 as if they were the same label set. AnnoMI here is five shared counsellor codes.

## AutoMISC-prompt 5-fold on HLQC

This is not the MIV test. Job 168271, predictions written 16 September 2026 15:32 UTC. Original AutoMISC prompts, context 3, five folds on the 1,925 HLQC rows. The prediction files were not in the local working tree when this pack was compiled. The figures below are the three-decimal values recorded from that scoring. Do not add precision. Do not place them in a table next to MIV numbers without saying the test set is different.

| Slice | n | ZS acc | FT acc | ZS κ | FT κ | Pipe F1 ZS | Pipe F1 FT | Gold F1 ZS | Gold F1 FT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 counsellor | 1040 | 0.604 | 0.752 | 0.506 | 0.689 | — | — | 0.522 | 0.719 |
| T1 client | 885 | 0.756 | 0.877 | 0.349 | 0.463 | — | — | 0.534 | 0.626 |
| T1 all | 1925 | 0.674 | 0.809 | 0.597 | 0.755 | — | — | 0.526 | 0.688 |
| T2 counsellor | 1040 | 0.342 | 0.598 | 0.287 | 0.551 | 0.227 | 0.424 | 0.240 | 0.448 |
| T2 client | 885 | 0.698 | 0.862 | 0.250 | 0.418 | 0.161 | 0.315 | 0.196 | 0.383 |
| T2 all | 1925 | 0.506 | 0.719 | 0.420 | 0.653 | — | — | 0.221 | 0.419 |

The paper’s counsellor T2 macro-F1 is 0.42 and its accuracy is 0.68, on a different model and the paper’s test. The HLQC fine-tune’s pipeline F1 on counsellor T2 is 0.424, close to that F1, with accuracy 0.598. Client change-bucket and sustain-bucket support-weighted F1, as recorded from `comparison.txt`: change n = 89, ZS 0.449, FT 0.500; sustain n = 40, ZS 0.115, FT 0.304. Those are not a binary change-versus-sustain classifier.

## Missing cells and jobs that had not started as of this compilation

`docs/BASELINE_RESULTS.md` lists 27 intended grid cells with no scored file. Among them: all eight `gpt4o_ft_*` cells; Qwen `ft_rat` at context 3; several `ft1mix_rat` and `ft1seq_rat` context-3 cells; two-call GRPO at context 3; two-call GRPO cold-start and unweighted ablations. The full list is in that file under Coverage. Do not fill them.

`docs/EXPANSION_RESULTS.md` “Not yet available” (one-adapter repeats): self-training seeds 42, 1, 2; synthesis v1 seeds 42, 1, 2; synthesis v2 boundary seeds 42, 1, 2; both topic-mix arms at seeds 1 and 2; SocialDial-rule arm; synthetic-only proto; boundary with the verifier off; retrieval on FT1-Mix, Inf-Bare and Inf-CoT.

Slurm queue on MLeRP, checked 28 September 2026 in the afternoon (Australia/Melbourne). All of the following were `PENDING` with elapsed 0. They are not results.

| Jobs | Name | What the 28 September log says they are |
|---|---|---|
| 170061–170066 | selftrain_retrain | Seeds 1–2 for self-train, v1, and v2-boundary on the one-adapter backbone |
| 170067–170069 | selftrain_retrain | SocialDial-rule retrains, after 170052 |
| 170052 | synth_teacher | SocialDial generation, still pending |
| 170070–170073 | xs1a_* | One-adapter cross-scheme on AnnoMI and Welivita, baseline and proto |
| 170075–170079 | cv1a_f0–f4 | One-adapter HLQC cross-validation anchor |
| 170080–170081 | qwen_agentic | Retrieval on `ft1mix_bare` |
| 170044–170051, 170053 | selftrain_retrain | Also pending; the log row does not itemise these ids separately from the consolidation campaign |

Context-3 teacher forcing was not completed. Active learning, joint MISC–MITI supervised training, Gemma, Qwen3.5, and Crank Group data were not run.

## Claims that survive, and claims that do not

Survive, with the caveat attached:

- A 7B model adapted with LoRA on 1,925 HLQC labels codes MIV6.3A far above the same model with no adapter. T2 accuracy moves from about 0.38 (zero-shot, context 5) to about 0.66–0.67. The paired test calls that gap real.
- On counsellor T2, the paper’s scope, GPT-4o few-shot chain-of-thought at context 5 reaches accuracy 0.666 and gold macro-F1 0.543. The local one-adapter model, three seeds, reaches accuracy 0.654 and gold macro-F1 0.497. The local two-adapter model, one seed, reaches accuracy 0.634 and gold macro-F1 0.445. State the comparison with those scopes. Do not say the local model matches the paper’s 0.68 / 0.42 without saying the paper number is GPT-4.1 on the paper’s split and the local number is a different corpus pair.
- One mixed adapter matches two adapters. One call matches two calls. Sequential training of a single adapter does not. GRPO matches supervised fine-tuning and does not beat it. Cold-start GRPO fails. These are the paired verdicts.
- Distilled rationales do not improve T2 accuracy over label-only targets. A forced-prefix probe does not show that trained rationales are more load-bearing than an untrained model’s continuation.
- Teacher forcing teaches the model to sometimes disagree with a wrong group (T2 accuracy given wrong T1 rises from 0 to 0.179 on `tfdyn`) and still loses marginal accuracy, because most groups are right.
- On the one-adapter backbone, proto synthetic windows produce a real gold-macro-F1 gain of +0.053 [+0.025, +0.077] across three seeds, driven by the client. Accuracy sign-flips across seeds.
- On the two-adapter backbone, one self-training seed raises T2 accuracy by +0.035 [+0.016, +0.053], mostly simple reflection, with gold macro-F1 essentially flat on the point estimate.
- On AnnoMI’s five shared counsellor codes, the two-adapter model reaches T2 accuracy 0.504 [0.471–0.539] against zero-shot 0.281. On Welivita’s weak gold it reaches 0.289 [0.258–0.320] against zero-shot 0.142. Both are well below the MIV figure, on different label sets.
- Classic n-gram and encoder classifiers on the same MIV test top out at T2 accuracy 0.464.

Do not claim:

- That seed-42 proto accuracy 0.713 is the result. The three-seed accuracy verdict is `sign-flips`.
- That topic-mix F1 gains are established. They are `single-seed`.
- That self-training, v1, or boundary have been replicated on the one-adapter backbone. They are pending.
- That two-adapter proto and one-adapter proto differ only by adapter count. The training files differ, and one of them was overwritten.
- That GRPO discovered a useful reason. Accuracy did not rise, and the faithfulness probe does not isolate training.
- That retrieval helps. On the scored two-call cell it hurts, and the one-adapter retrieval jobs have not run.
- Any number for jobs 170044–170081.

## Source index

| Need | File |
|---|---|
| Arm definitions | `docs/EXPERIMENTS.md` |
| Full grid, paired tests, cross-scheme, faithfulness | `docs/BASELINE_RESULTS.md` |
| One-adapter multi-seed data expansion | `docs/EXPANSION_RESULTS.md` |
| Campaign dates, including stale rows | `docs/EXPERIMENT_LOG.md` |
| SFT hyperparameters | `conf/baseline_ft_local_config.yaml` |
| Single-call GRPO | `conf/grpo_config.yaml`, `docs/GRPO.md` |
| Two-call GRPO | `conf/grpo_tc_config.yaml`, `docs/GRPO_TWOCALL.md`, `docs/experiments/2026-08-26-two-call-grpo.md` |
| Teacher forcing | `docs/experiments/2026-08-19-teacher-forcing.md` |
| Unrounded grid | `outputs/baseline_eval/comparison.csv` |
| Metric code | `src/baseline/eval.py` (`N_BOOT`, `PAIRED_REFERENCE`) |
| Proposal (planned, not all executed) | the May 2026 literature-review proposal PDF |
