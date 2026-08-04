# GRPO ladder — runbook

One question: **does reasoning the model discovers for itself beat reasoning
copied from gpt-4o, and does either beat no reasoning at all** — particularly
on rare codes, under the HLQC to MIV6.3A distribution shift.

## Why this experiment, given the baseline results

The two-call grid in [BASELINE_RESULTS.md](BASELINE_RESULTS.md) settled two
things and opened one.

**Accuracy is solved; coverage is not.** Qwen `FT-Bare` reaches T2-all accuracy
0.660, inside the gpt-4o band of 0.639-0.699. Its Macro-F1 (gold) is 0.424
against gpt-4o's 0.536-0.544. A 7B fine-tune closes the accuracy gap by getting
the head classes right and loses on the tail. Every number this experiment
cares about is therefore a macro-F1, not an accuracy.

**Imitated reasoning actively hurts.** `FT-Rat` is worse than `FT-Bare` in every
cell (T2-all 0.510 and 0.620 against 0.660), while chain-of-thought helps the
un-tuned model a lot (T2-all zero-shot 0.381 to 0.442). Reasoning has value for
this model, but training it to reproduce rationales that gpt-4o wrote *after*
being shown the gold label is not how to install it. Those rationales are
post-hoc by construction and need not describe anything that would
independently produce the label.

That is the gap GRPO addresses: reinforce the rollouts that land the right code
and keep whatever reasoning survives, rather than copying reasoning that was
never load-bearing.

**Which codes are actually rare.** Grounded in counts, not intuition. The
codes that are scarce in HLQC training but common in MIV6.3A evaluation are
`SU` (9 train, 39 eval), `EC` (22, 39), `AF` (23, 30) and `GI` (16, 24). `SU`
is the extreme: 0.9% of counsellor training rows against 6.7% of evaluation
rows. Note that `RCP`, which looks rare, has **zero** gold occurrences in
either corpus and is not a target. Separately, `TS+` (2) and `AC-` (1) occur in
MIV6.3A but never in HLQC, so no arm trained on HLQC can predict them; they
enter Macro-F1 (gold) as guaranteed zeros, which is why `baseline.eval` also
reports Macro-F1 (learnable) with them excluded.

## The ladder

All five arms share one prompt format, so they differ in training signal alone.

| Arm | Reasoning comes from | Role |
|---|---|---|
| `sc_ft_bare` | nothing (label targets) | floor |
| `sc_zs` | prompt only | reference |
| `sc_fs` | prompt plus HLQC exemplars | reference (Phase 3) |
| `sc_ft_rat` | copied gpt-4o rationale | imitation baseline |
| `sc_grpo` | self-discovered, reward-filtered | **headline** |
| gpt-4o (two-call, already run) | — | frontier ceiling |

The two comparisons that answer the question are `sc_grpo` vs `sc_ft_bare`
(does reasoning help) and `sc_grpo` vs `sc_ft_rat` (does discovered reasoning
beat imitated).

## Single-call format

The two-call pipeline predicts T1 with one adapter, then prompts a second
adapter conditioned on that prediction. That shape is bad for RL: T2's reward
depends on a T1 prediction that is itself moving, so there are two coupled
non-stationary problems. Here one generation produces everything:

```json
{"rationale": "...", "t1": "CRL", "t2": "CR"}
```

One rollout is one sequence with one scalar reward covering both tiers. Prefill
is roughly what the two calls cost together (measured at ctx5: joint counsellor
prompt ~4.3k tokens against 3.0k + 1.4k), so the saving is one generation per
utterance rather than two.

Because the format differs, **single-call numbers are not comparable to the
two-call table**. `baseline.eval` prints them as a separate block. The two
supervised arms are retrained in single-call form for exactly this reason: the
ladder has to be internally format-matched or the headline comparisons measure
the format change instead of the training signal.

Context is **ctx5 for Phase 1**. Not because ctx5 is better — the gpt-4o rows
say ctx3 wins on T2 — but because holding it fixed makes the port checkable: if
single-call `sc_ft_bare` lands near the two-call ctx5 T2-all accuracy of 0.660,
the port is validated rather than assumed. ctx3 is a Phase 3 sweep.

## Reward

Per rollout, in `single_call.score_completion`:

| Component | Value | Condition |
|---|---:|---|
| format | 0.1 | parses as JSON, both codes in the speaker's vocabulary, rationale at least 5 words |
| hierarchy | 0.1 | the emitted `t2` is a child of the emitted `t1` |
| t1 | 0.4 | `t1` matches gold |
| t2 | 0.4 | `t2` matches gold |

## Three things that are easy to get wrong

**1. Rare-class weighting is a no-op under default GRPO.** Every rollout in a
group answers the same prompt, so they share one gold label and therefore one
class weight. Under the default `scale_rewards="group"` the advantage is
`(w*r - mean(w*r)) / std(w*r)` and `w` cancels top and bottom exactly. The run
would look correctly configured, cost the same GPU hours, and quietly produce
the unweighted result — which would then be compared against the genuine
unweighted ablation and show no difference. The config therefore pins
`scale_rewards: none` with `loss_type: dr_grpo`, and
`grpo._check_weighting_is_live` refuses to start if that is ever changed back.

Weighting is applied two ways, which compound: an inverse-frequency reward
weight `clip((N/(K*n_c))**0.5, 0.5, 3.0)` on the gold T2 code, and
oversampling of rare-gold prompts (capped at 2 copies). Both are computed from
the **training slice only**, so the validation label distribution does not leak
into the objective. In practice `SU` gets weight 2.53 against `FI` at 0.54.

**2. Initialisation has to be able to sometimes be right.** GRPO learns from
within-group reward *variance*. If all G rollouts are wrong, the advantage is
zero and so is the gradient. A cold Qwen2.5-7B sits at 0.38 T2 accuracy, so
most groups would carry no signal at all. `sc_grpo` therefore starts from the
`sc_ft_bare` adapter. The Phase 2 cold-start run exists to demonstrate this
rather than assert it.

**3. The reward can be farmed.** Format is worth 0.1 and the labels 0.8, so
dropping the rationale is a cheap trade the optimiser can find.
`RationaleCollapseGuard` aborts the run if the mean rationale falls below 4
words, and a KL penalty (`beta: 0.02`) anchors the policy to the `sc_ft_bare`
reference. With PEFT, TRL uses the adapter-disabled base as that reference, so
it costs no extra memory. Neither of these proves the rationale matters — the
Phase 3 rationale-swap probe is what actually decides that.

## Checkpoint selection

On a **conversation-level** split of the training corpus (fold 0 of 7), never on
MIV6.3A. Conversation-level rather than row-level because utterances from one
session share context windows, so a row-level split would put validation text
inside training prompts. Selection uses the mean of the two macro-F1s, not
accuracy, since accuracy is the thing the ladder has already saturated.

HLQC has only 10 conversations, so the split is coarse and fold sizes vary from
35 to 503 rows. Fold 0 was fixed before any training ran, on the grounds that
it holds out 369 rows spanning 22 distinct T2 codes that all also occur in the
training slice. That is selection on label coverage, not on any model's score.

## Phases and budget

Estimates below are the planning figures. **Run `calibrate` first** and re-cost
against `projected_run_hours` before submitting the ladder; the planning numbers
assume a vLLM prefix-cache hit rate that nobody has measured on this prompt
shape, so treat them as within a factor of two until then.

| Phase | Runs | GPU-h | Cut? |
|---|---|---:|---|
| 0 — env and calibration | venv, `sc_ft_bare`, 50-prompt timing | ~3 | never |
| 1 — the ladder | `sc_ft_bare`, `sc_ft_rat`, `sc_grpo` x 3 seeds, 5 predictions | ~24 | never |
| 2 — why it worked | weighting off x 3 seeds, cold start x 1 | ~25 | keep if possible |
| 3 — reviewer defence | few-shot, faithfulness probe, ctx3 replication | ~11 | cut first |

The dominant cost is the policy forward/backward over ~4.3k-token prompts, not
generation. If the budget bites, the lever is `num_generations` (8 to 6) and
prompts per epoch, not the rollout engine.

## Running it

The GRPO arm needs its own environment. `scripts/env.sh` reuses the read-only
MLeRP DSKS conda env (torch 2.4.0) that every other part of the pipeline runs
against, and vLLM pins its own torch build; the two cannot share.

```bash
bash scripts/setup_grpo_env.sh          # once, on a login node
source scripts/setup_grpo_env.sh use    # every session
```

MLeRP's home directory is a symlink to `/mnt/userdata4/$USER` under a **50GB
quota**, and `/tmp` is a 30GB shared root filesystem with ~2GB free. The setup
script therefore points `HF_HOME` at the existing cache (which already holds
Qwen2.5-7B; a fresh path would silently re-download 15GB), keeps `TMPDIR` on the
same volume as the venv, installs with `--no-cache-dir`, and preflights free
space. If it reports too little disk, the safe thing to delete is optimizer
state from **completed** runs — it can only resume an interrupted run and is
useless for inference:

```bash
find outputs data -type f \( -name optimizer.pt -o -name scheduler.pt \
     -o -name rng_state.pth \) -delete
rm -rf ~/.cache/pip ~/.conda/pkgs/*
```

Then, from the repo root:

```bash
# Phase 0
STAGE=sft TARGET=bare sbatch scripts/mlerp_grpo.slurm
STAGE=calibrate       sbatch scripts/mlerp_grpo.slurm

# Phase 1
STAGE=sft TARGET=rat                     sbatch scripts/mlerp_grpo.slurm
for s in 0 1 2; do
  STAGE=grpo SEED=$s                     sbatch scripts/mlerp_grpo.slurm
done
STAGE=predict ARM=sc_ft_bare             sbatch scripts/mlerp_grpo.slurm
STAGE=predict ARM=sc_ft_rat              sbatch scripts/mlerp_grpo.slurm
STAGE=predict ARM=sc_grpo SEED=0         sbatch scripts/mlerp_grpo.slurm

# Phase 2
STAGE=grpo SEED=0 WEIGHTING=off          sbatch scripts/mlerp_grpo.slurm
STAGE=grpo SEED=0 INIT=cold              sbatch scripts/mlerp_grpo.slurm
STAGE=predict ARM=sc_grpo_unw  SEED=0    sbatch scripts/mlerp_grpo.slurm
STAGE=predict ARM=sc_grpo_cold SEED=0    sbatch scripts/mlerp_grpo.slurm

# Phase 3
STAGE=predict ARM=sc_zs                  sbatch scripts/mlerp_grpo.slurm
STAGE=predict ARM=sc_fs                  sbatch scripts/mlerp_grpo.slurm
STAGE=faith   ARM=sc_grpo SEED=0         sbatch scripts/mlerp_grpo.slurm
STAGE=faith   ARM=sc_ft_rat              sbatch scripts/mlerp_grpo.slurm
STAGE=predict ARM=sc_grpo SEED=0 CTX=3   sbatch scripts/mlerp_grpo.slurm

# Score everything, both ladders, into docs/BASELINE_RESULTS.md
PYTHONPATH=src python -m baseline.eval
```

Only `grpo` and `calibrate` need a full A100: they colocate a vLLM engine with
the policy, and 30% of a 20GB MIG slice cannot even hold the 7B weights.
`predict`, `sft` and `faith` are single-sequence HF decoding at ~16GB, so they
fit a slice and should be submitted with `--gres=gpu:3g.20gb:1`. This is worth
doing rather than tidy: BigCats routinely has every 40GB card allocated while
the MIG slices sit idle, and the two prediction arms above ran immediately on
slices instead of waiting behind a six-hour queue.

The `sc_fs` prompt carries one exemplar per T1 group — six for the counsellor,
which at ctx5 adds ~2.6k tokens to an already ~4.3k-token prompt. That fits
`inference.max_input_len: 8192`, but not by much. Truncation here is right-side,
so it would remove the utterance being coded rather than the exemplars and the
model would answer a question it was never asked; `sc_arm predict` therefore
counts truncated prompts and warns. If that warning ever fires, raise
`max_input_len` rather than trusting the run.

Each GRPO variant writes to its own adapter directory and its own result
filename (`sc_grpo`, `sc_grpo_unw`, `sc_grpo_cold`). They must not be allowed to
share: Phase 2 would overwrite Phase 1 and the ablation would end up compared
against itself.

## Reading the faithfulness probe

The probe substitutes the rationale the model generated for a *different*
utterance with a different gold code, forces it into this utterance's answer,
and lets the model continue. Forcing a prefix is itself an intervention, so it
is run against a control that forces the model's own rationale back in; the
reported effect is the difference.

- **High net flip rate** — the label followed the substituted rationale, so the
  reasoning was load-bearing. This is what licenses a claim about *reasoning*
  rather than about labels.
- **Net rate near zero** — the model had already decided from the utterance and
  the rationale is narration.
- **Donor-match rate** distinguishes steering from disturbance: drifting to a
  random third code shows the swap confused the model, landing on the donor's
  own code shows it actually followed the substituted reasoning.

## Code map

| Path | Role |
|---|---|
| [conf/grpo_config.yaml](../conf/grpo_config.yaml) | every hyperparameter, with the reasoning for the non-obvious ones |
| [src/baseline/single_call.py](../src/baseline/single_call.py) | prompt construction, parsing, reward, class weights |
| [src/baseline/sc_arm.py](../src/baseline/sc_arm.py) | supervised arms: `train` and `predict` |
| [src/baseline/sc_infer.py](../src/baseline/sc_infer.py) | single-call annotator |
| [src/baseline/grpo.py](../src/baseline/grpo.py) | GRPO training, validation selection, `calibrate` |
| [src/baseline/faithfulness.py](../src/baseline/faithfulness.py) | rationale-swap probe |
| [src/baseline/eval.py](../src/baseline/eval.py) | scores both ladders, aggregates seeds |
| `src/components/prompts/templates/*/t12.j2` | the joint T1+T2 prompt |
| [scripts/setup_grpo_env.sh](../scripts/setup_grpo_env.sh) | the vLLM venv |
| [scripts/mlerp_grpo.slurm](../scripts/mlerp_grpo.slurm) | job submission |
