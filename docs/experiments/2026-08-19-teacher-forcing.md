# Teacher forcing on the Tier-1 conditioning

**Started** 2026-08-19 · **Status** ctx5 complete for cells A and B; ctx3
abandoned · **Arms** `ft_bare_tf{full,dyn,zero}` (cell A, 2 adapters),
`ft1mix_bare_tf{full,dyn,zero}` (cell B, 1 adapter) · **Context length** 5

## The question

Every two-call arm trains its T2 adapter conditioned on the **gold** T1 label
([`data.py`](../../src/automisc_ft/data.py) `build_tier_examples`) but runs it
conditioned on the **predicted** one
([`infer.py`](../../src/automisc_ft/infer.py) `predict_row`). At ctx5 the T1
adapter is 78.9% accurate, so about one T2 call in five is conditioned on a group
spec the model never saw paired with that utterance during training.

Does closing that mismatch help, and what does it cost?

## Why now: the ceiling is already visible

Scoring the existing grid by whether T1 was correct (the new
`T2 accuracy by Tier-1 correctness` table in
[BASELINE_RESULTS.md](../BASELINE_RESULTS.md)) shows something sharper than
expected. For `ft_bare` at ctx5, Inf-Bare:

| | value |
|---|---|
| T1 error rate | 21.1% |
| T2 accuracy when T1 was correct | 0.836 |
| T2 accuracy when T1 was wrong | **0.000** (0 of 173) |
| T2 accuracy, marginal | 0.660 |
| T2 emitted outside the predicted group | **0.0%** |

`0.000` is not a rounding artifact and not constrained decoding — decoding is
unconstrained (`restrict_t2_to_group: false`). Three facts compose into it:

1. The T1 groups **partition** the T2 vocabulary — no T2 code belongs to two groups.
2. The T2 prompt lists **only the predicted group's** codes.
3. The fine-tuned model picks from that list **173 times out of 173**.

So when T1 is wrong the gold T2 sits in a group the prompt never showed, and T2
cannot be right unless the model emits a code it was not offered. It never does.

That gives an exact ceiling:

```
T2_acc  ≈  T1_acc × P(T2 correct | T1 correct)
0.660   ≈  0.789  × 0.836
```

T2 accuracy is fully determined by T1 accuracy and within-group T2 accuracy.
Every gain from better T2 modelling is capped by T1 until something teaches the
model that the spec can be wrong.

Worth noting the untrained model is not like this: zero-shot Qwen emits
out-of-group codes 5.6% of the time. **Fine-tuning drove spec-obedience to
exactly 100%.** That is the behaviour this campaign tries to move.

## What varies

`p_gold` — the probability that a T2 training example is conditioned on the gold
T1 rather than an out-of-fold predicted one, evaluated once per epoch.

| Arm | schedule | p_gold at E=3 |
|---|---|---|
| `ft_bare_tffull` | `constant(1.0)` | `[1.0, 1.0, 1.0]` |
| `ft_bare_tfdyn` | `linear(1.0 → 0.0)` | `[1.0, 0.5, 0.0]` |
| `ft_bare_tfzero` | `constant(0.0)` | `[0.0, 0.0, 0.0]` |

The completion target is always the gold `t2_label_GT`. Only the conditioning
moves, and no rows are dropped, so all three see identical example counts.

**Terminology.** Teacher forcing is defined at the *token* level, and at that
level all three arms — `tfzero` included — are fully teacher forced. What varies
is one field in the prompt, which is the *stage* level. In prose this is
**training the downstream stage on predicted rather than gold upstream inputs**,
the standard fix for pipeline error propagation; scheduled sampling (Bengio et
al. 2015) is the citation for the decaying variant, applied by analogy one level
up. No arm here is unsupervised.

## Two design decisions

**Predicted T1 must be out of fold.** The production T1 adapter trained on all
1,925 HLQC rows for 3 epochs, so its predictions on those same rows are partly
memorised — far cleaner than the ~79% it manages on held-out data. Training
against them would understate the exposure roughly threefold and flatten the
contrast. So [`oof_t1.py`](../../src/baseline/oof_t1.py) splits the corpus into
K conversation-level folds, trains a T1-only adapter per fold on the others, and
predicts the held-out fold.

**K=5, not 3.** HLQC holds only ten conversations. At K=3 a fold adapter trains
on 57–77% of the corpus against the production adapter's 100%, so its predictions
are *noisier* than the ones met at inference — the mirror of the in-sample
problem. K=5 lifts that to 74–86%.

**`tffull` is trained, not aliased to `ft_bare`.** A declining rate needs the
dataset rebuilt between epochs, so `tfdyn` runs as three chained 1-epoch stages,
which forces a constant LR and no warmup (three staged cosine cycles are not one
3-epoch cosine). All three arms therefore share the staged procedure, and
`tffull` is the matched control that makes the `dyn` number attributable.
`ft_bare` vs `ft_bare_tffull` prices the staging itself, for free.

## Running it

```bash
CTX=5 bash scripts/submit_tf_axis.sh        # ctx5 end to end
CTX=3 bash scripts/submit_tf_axis.sh        # after ctx5 reads cleanly
DRYRUN=1 CTX=5 bash scripts/submit_tf_axis.sh
```

**Gate on calibration between the fold predictions and the T2 trainings:**

```bash
PYTHONPATH=src python -m baseline.oof_t1 report --ctx 5
```

Pooled out-of-fold T1 accuracy on HLQC should land near the 0.789 the production
adapter scores on the evaluation corpus. Far above it and the exposure being
trained on is gentler than the one met at inference, which makes every number
below a lower bound rather than an estimate. Record the figure here either way.

**Recorded, ctx5 (2026-08-19).** Coverage 1925/1925. Pooled out-of-fold T1
accuracy **0.8229** against the 0.7893 reference — gap **+0.034**, inside the
0.05 threshold, so the exposure is calibrated and the contrast is interpretable
rather than a lower bound. Per fold: 0.765 / 0.852 / 0.755 / 0.891 / 0.859; the
spread is expected, since conversation-level folds over ten unequal
conversations give unequal training fractions.

The exposure that reaches training: **341 of 1925 examples (17.7%)** are
conditioned on a T1 group that does not contain the gold T2 label, versus the
**21.1%** T1 error rate met at inference. Slightly gentler, in the direction that
would understate rather than overstate any benefit.

## Cells

| Stage | Count | Artifact |
|---|---:|---|
| Out-of-fold T1 adapters | 5 per ctx | `data/fine_tuning/oof_t1/ctx{N}/fold{k}/t1/` |
| Out-of-fold T1 predictions | 5 per ctx | `data/fine_tuning/oof_t1/hlqc_oof_t1_ctx{N}.json` |
| Staged T2 adapters (cell A) | 3 per ctx | `data/fine_tuning/baseline_local/ctx{N}/bare_tf{mode}/t2/` |
| Staged shared adapters (cell B) | 3 | `data/fine_tuning/baseline_local_1a/ctx{N}/mixed/bare_tf{mode}/` |
| Prediction cells (Inf-Bare) | 3 per cell | `data/annotated/baseline/qwen_{ft_bare,ft1mix_bare}_tf{mode}_inf_bare_ctx{N}.csv` |

## Results

### ctx5, Inf-Bare (2026-08-20)

| Arm | Out-of-group | T2 acc \| T1 ok | T2 acc \| T1 wrong | T2 acc (marginal) |
|---|---:|---:|---:|---:|
| `ft_bare` (unstaged reference) | 0.0% | 0.836 | 0.000 | 0.660 |
| `ft_bare_tffull` (staged control) | 0.0% | 0.819 | 0.000 | 0.647 |
| `ft_bare_tfdyn` | **15.6%** | 0.756 | **0.179** | 0.635 |
| `ft_bare_tfzero` | **13.4%** | 0.741 | **0.133** | 0.613 |

T1 accuracy is 0.789281 and kappa 0.753486 for all four arms to six decimals —
the shared-T1-by-path design confirmed on the numbers rather than asserted.

**The mechanism engaged.** Out-of-group emission went from an absolute 0.0% to
13-16%, and `T2 acc | T1 wrong` came off the floor for the first time in the
whole grid: 0.000 → 0.179. The model learned that the spec can be wrong, and its
overrides land on the correct code often enough to matter.

**It still made things worse.** Decomposing against the staged control, where the
staging cancels:

| Arm | gain on wrong-T1 rows | loss on right-T1 rows | net |
|---|---:|---:|---:|
| `tfdyn` | +0.0378 | −0.0499 | **−0.0122** |
| `tfzero` | +0.0280 | −0.0621 | **−0.0341** |

(net matches the observed marginal change to four decimals). Teaching the model
to distrust the spec on the 21% of rows where it is wrong also taught it to
distrust the spec on the 79% where it is right, and the majority dominates. This
is the trade the split table exists to expose: marginal accuracy alone would have
shown a dull 0.647 → 0.635 → 0.613 and hidden both halves of it.

**`dyn` beats `zero` on both subgroups**, which contradicts the prediction made
when planning this — that at E=3 `dyn` would collapse onto `zero` because its
final epoch sees the same conditioning. It does not. Keeping one epoch of pure
gold conditioning preserves more spec-trust (0.756 vs 0.741) *and* produces
better overrides (0.179 vs 0.133). The schedule shape matters more than the
endpoint, which is an argument for the per-step ε that was deferred.

**Staging costs 0.013.** `ft_bare` 0.660 → `ft_bare_tffull` 0.647 at identical
p_gold=1, from the constant LR and per-stage optimizer resets alone. That is the
same order as the `tfdyn` effect (−0.012), which is exactly why the matched
staged control was worth training: without it, `dyn` vs the published `ft_bare`
would have read as −0.025 and been half LR artifact.

### ctx5, cell B — 1 adapter, 2 calls (2026-09-08)

Same axis on `ft1mix_bare`, where **one** LoRA serves both calls. The predicted-T1
store is the same frozen ctx5 artifact cell A used, so the conditioning noise is
held fixed and only adapter count varies.

| Arm | T1 acc | T2 acc | T2 kappa | T2 F1(learn) | T2 \| T1 wrong | Out-of-group |
|---|---:|---:|---:|---:|---:|---:|
| `ft1mix_bare` (unstaged reference) | 0.7881 | 0.6711 | 0.6416 | **0.4961** | 0.000 | 0.1% |
| `ft1mix_bare_tffull` (staged control) | **0.7247** | 0.6102 | 0.5754 | 0.4268 | 0.000 | 0.0% |
| `ft1mix_bare_tfdyn` | **0.8063** | **0.7040** | **0.6690** | 0.4682 | 0.126 | 7.9% |
| `ft1mix_bare_tfzero` | 0.7808 | 0.6529 | 0.6196 | 0.4529 | **0.222** | 9.6% |

### Seed-1 replicate (2026-09-09) — the result does not survive

| Arm | T2 s42 | T2 s1 | mean | **spread** | T1 s42 | T1 s1 | mean | spread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `tffull` | 0.6102 | 0.6906 | 0.6504 | **0.0804** | 0.7247 | 0.7905 | 0.7576 | 0.0658 |
| `tfdyn` | 0.7040 | 0.6638 | 0.6839 | 0.0402 | 0.8063 | 0.8124 | 0.8094 | 0.0061 |
| `tfzero` | 0.6529 | 0.6663 | 0.6596 | 0.0134 | 0.7808 | 0.7552 | 0.7680 | 0.0256 |

**Largest within-arm spread from changing the seed alone: 0.0804 T2. Largest
between-arm gap of the two-seed means: 0.0335.** Noise exceeds signal by 2.4x,
and the ordering inverts: at seed 42 `tfdyn` (0.7040) beat `tffull` (0.6102); at
seed 1 `tffull` (0.6906) beats `tfdyn` (0.6638). The seed-42 control was a bad
draw, exactly as suspected. **No teacher-forcing effect in cell B is
distinguishable from run-to-run variance.**

This also retracts the coupling claim as originally written. `tffull` holds
`p_gold=1` in both seeds — identical T2 conditioning — and still moved T1 by 6.6
points. So the T1 swing is instability of the shared adapter, **not** a response
to re-conditioning T2. What survives is narrower and still worth knowing: the
1-adapter regime is unstable across seeds at fixed configuration, where cell A's
T1 is bit-identical by construction. One thing resists the noise — `tfdyn`'s T1
is 0.8063/0.8124 (spread 0.0061) and above the base arm in both seeds — but that
is two runs, and it is not what the campaign set out to test.

**By extension, cell A's verdict is weaker than it reads.** Its arms differ by
0.013–0.047 T2 on a single seed each, which is inside the noise band just
measured on the sibling structure. Cell A retrains only T2 rather than the whole
adapter, so it is plausibly steadier, but that is an assumption — no cell-A arm
has been replicated.

### What the single-seed run had suggested

**T1 moves, and that is the finding.** In cell A the T1 column is identical to
four decimals across all four arms — verified stronger than that: the T1
prediction is the same on **821/821 rows**, because `resolve_adapters` hands
every cell-A variant the same frozen T1 adapter file (`ff502bace06e`) and
decoding is greedy. Cell A therefore doubles as a control proving the T1
evaluation path is deterministic. In cell B there is no separate T1 adapter to
pin: one file serves both calls, and T1 accuracy swings **8 points**
(0.7247–0.8063) between arms whose T1 training data is byte-identical. Only the
T2 conditioning differed. That interference is exactly what cell A's
architecture makes impossible, and it is the reason cell B was run.

**Do not read the +0.094 over the control at face value.** `tffull` runs
`p_gold=1`, so its T2 conditioning is identical to the unstaged reference and the
staged procedure is the only difference — yet it lost 6.3 points of T1 and 6.1 of
T2. Staging cost cell A 1.3 points of T2 and *zero* T1. Either staging is far more
damaging when one adapter carries both tiers under a constant LR with optimizer
resets, or that run drew badly. Both readings inflate `tfdyn`'s margin.

**And it is a single seed.** Arms sharing identical T1 data landed anywhere from
0.7247 to 0.8063 on T1. That spread *is* the coupling result, but it also puts
run-to-run variance at several points, which is wider than `tfdyn`'s +0.033 over
the unstaged reference. A second replicate (seed 1, all three arms) was submitted
2026-09-08 to separate systematic effect from draw.

**Macro-F1 disagrees.** The unstaged `ft1mix_bare` is best on F1(learnable) at
0.4961 against `tfdyn`'s 0.4682. So `tfdyn`'s gain is in accuracy and kappa —
head classes — not in the tail. It is not uniformly better.

### ctx3

Abandoned. Out-of-fold T1 finished (merged store written 2026-08-20 03:10). All
three T2 trainings then failed on 2026-08-20: `tffull` (161092) and `tfdyn`
(161094) each completed stage 2 of 3 and died in `_train_tier` as stage 3
started — the SLURM logs truncate at 76 KB so the exception itself is gone,
but `/mnt/userdata4` is at 50/50 GB with **zero bytes free**, which is enough
to kill a LoRA save. `tfzero` (161096) exited in 0s with an empty log, almost
certainly lion's 4-job cap colliding with the other two. The three prediction
jobs sat `PENDING` / `DependencyNeverSatisfied` until cancelled 2026-08-24.
No ctx3 cell was scored. The disk was later cleared and expanded to 100 GB, and
the three trainings were resubmitted 2026-09-08 (167410/167412/167414) — then
cancelled ~20 min in to concentrate the remaining budget on ctx5 and on cell B.
`ARM_CTXS` now scopes every `tf*` arm to ctx5, so the coverage audit reports a
closed decision rather than an open gap. Resubmit is
`SKIP_OOF=1 CTX=3 bash scripts/submit_tf_axis.sh`.

### Reading guide

| Reading | What it means |
|---|---|
| `Out-of-group` rises above 0% under `tfzero` | The mechanism engaged: the model learned the spec is fallible |
| `T2 acc \| T1 wrong` rises off 0.000 | It engaged *usefully* — overrides land on the right code |
| `T2 acc \| T1 ok` falls | The cost side: the 21% of mis-conditioned training rows also taught it to distrust correct specs |
| Marginal T2 accuracy | Only the net of those two, and the least informative number here |

Both of the first two happened; the third outweighed them.

Record pooled out-of-fold T1 accuracy, and `ft_bare` vs `ft_bare_tffull`, before
reading anything else: the first says whether the exposure was calibrated, the
second says how much of any gap is the staging rather than the axis.

## Run record

**2026-08-19, ctx5.** Out-of-fold stages ran as 5 fold trainings (~48 min each)
and 5 fold predictions (~3 min each — T1-only greedy decode is far cheaper than a
two-call cell). Staged T2 trainings and their prediction cells followed.

Two infrastructure problems surfaced, both worth keeping:

**The `panther` QOS cannot run GPU jobs.** `sacctmgr` reports
`MaxTRESPerJob = gres/gpu=0` for panther. A GPU job submitted there is *accepted*
and then pends forever with reason `QOSMaxGRESPerJob` — it never fails, so it
looks like an ordinary queue wait. `submit_tf_axis.sh` originally alternated
placement across lion and panther to dodge lion's 4-job cap; that silently
stranded half the fold trainings. Everything now goes to lion.
[`submit_grpo_ladder.sh`](../../scripts/submit_grpo_ladder.sh) still routes GRPO
*training* jobs to panther and its comment claims panther "differs only in a
7-day wall" — that will strand the next ladder run and has not been changed here.

**The out-of-fold store had a write race.** All five prediction jobs originally
appended to one shared JSON: each loaded the store at startup and rewrote the
whole file at the end, so with four running concurrently under lion, whichever
finished last erased the folds that landed while it ran. It presented as fold 1
being absent despite its job reporting `COMPLETED`, and fold 0 holding 375 of its
489 rows. Fixed by giving each fold its own file
(`hlqc_oof_t1_ctx{N}_fold{k}.json`), with `load_oof_t1` merging them and `report`
writing the merged view — one writer per file rather than synchronisation. The
predictions were simply re-run, since the fold *adapters* were unaffected. The
same one-writer-per-file rule is why run metadata is a per-cell `.meta.json`
sidecar rather than a shared ledger.

**2026-08-20, ctx3.** Out-of-fold T1 completed. T2 trainings failed (see
Results). Prediction jobs cancelled 2026-08-24.

**2026-08-24.** ctx5 prediction CSVs and the cluster's `baseline.eval` report
copied onto the Mac. `docs/BASELINE_RESULTS.md` had still been the 2026-08-19
pre-TF generate.

**2026-09-08, cell B.** The axis ported to the 1-adapter regime. `--tf` now
accepts `--regime mixed`, and `train_single_staged` rebuilds *both* tiers each
stage (T1 unchanged, T2 at that stage's `p_gold`) before chaining through
`init_adapter_dir`; there is no separate T2 adapter to isolate. Three trainings
at ~92 min each — 963 steps per stage against cell A's 482, because each stage
sees 1,925 T1 plus 1,925 T2 examples.

Two things surfaced while building it. The CPU smoke test produced **NaN**
adapters: on Apple Silicon the trainer forces `max_grad_norm = 0.0` to dodge a
PyTorch MPS bug, which disables gradient clipping outright, and the staged
constant-LR recipe then diverges on a toy set. Re-run with MPS disabled — so
clipping matches the cluster's CUDA path — it gives three distinct, finite
adapters with monotonically growing magnitude. **Local smoke tests of staged
training are not trustworthy on the Mac.** That diagnosis was slowed by a second
bug: `train_single_staged` logged its stage banners with `log.info`, but this
module logs through hydra's logger, which is unconfigured under the `local_arm`
CLI, so they were swallowed. `train_t2_staged` had used `print` deliberately for
exactly that reason. Both now print.

**2026-09-08, seeds made real.** `run_local_fine_tuning` never passed a seed to
`TrainingArguments`, so HF's default of 42 governed LoRA init and the data
sampler on every run ever made here, and `cfg.training.seed` only shuffled row
order. Now threaded through, defaulting to 42 so every prior run reproduces
unchanged. `local_arm` gained `--seed`, which gives a replicate its own adapter
tree and a `_seed{N}` result file — the convention `sc_arm` already used and
`baseline.eval` already parses and aggregates.

## Verdict

**Do not replace `ft_bare` with any of these arms.** Closing the gold/predicted
T1 mismatch taught the model that the group spec is fallible, which is exactly
what the split table was built to detect: out-of-group emission left 0%, and
`T2 acc | T1 wrong` left 0.000. The cost on the 79% of rows where the spec is
right outweighed that gain. Against the staged control, `tfdyn` is −0.012 T2
accuracy and `tfzero` is −0.034; `dyn` is the less-bad of the two on both
subgroups, so the schedule shape is doing work the endpoint does not. Staging
itself costs 0.013, which is why the matched control was not optional.

That is a **cell A** result: 2 adapters, ctx5, Inf-Bare.

**Cell B is a null too, and for a sharper reason.** The seed-1 replicate
(2026-09-09) returned `tffull` at T2 0.6906 against its seed-42 0.6102 — the
control was a bad draw, and the ordering inverts between seeds. Changing only
the seed moves an arm by up to 0.0804 T2, while the arms' two-seed means span
0.0335. Nothing in cell B is distinguishable from noise.

The coupling claim is retracted as first written: `tffull` keeps `p_gold=1` in
both seeds and still swung T1 6.6 points, so the swing is shared-adapter
instability rather than a response to re-conditioning T2. The narrower survivor:
**the 1-adapter regime is unstable across seeds at fixed configuration**, where
cell A's T1 is bit-identical by construction. Anything comparing 1-adapter arms
at n=1 — including the `ft1mix` vs `ft1seq` rows in the main grid — is reading
differences smaller than its own noise floor.

That applies to cell A's verdict as well: its 0.013–0.047 spreads are single-seed
and sit inside the band measured here. The negative result there is probably
right (the mechanism numbers — out-of-group 0% → 15.6%, `T2|T1-wrong` 0.000 →
0.179 — are far too large to be noise) but the *size* of the accuracy cost is
not established.

ctx3 never produced a number and has been scoped out. The next experiment, if
any, is a per-step ε rather than another constant or linear `p_gold` at E=3.
