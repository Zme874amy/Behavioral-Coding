# Two-call GRPO — runbook (cells A and B under RL)

One question: **does the structure a policy is optimised in — one call or two,
one adapter or two — change whether RL-discovered reasoning pays off**, and does
the coupling that `sc_grpo` avoided by going single-call actually cost anything.

## Why this experiment

The single-call ladder ([GRPO.md](GRPO.md)) put GRPO in exactly one cell of the
structural matrix ([EXPERIMENTS.md](EXPERIMENTS.md) §1): 1 call, 1 adapter. It
chose that cell to dodge a real problem — in the two-call flow, T2's reward
depends on a T1 prediction that is itself moving, "two coupled non-stationary
problems". And its headline was equivocal: GRPO only drew *level* with
`sc_ft_bare` (T2 0.658 vs 0.653), so "reasoning beats no reasoning" is still open.

The SFT grid fills all four cells; the RL grid had one. This campaign fills the
other two, so `A-grpo`/`B-grpo`/`sc_grpo` form the same adapter×call ablation
under RL that `ft_bare`/`ft1mix_bare`/`sc_ft_bare` form under SFT. And it runs
each cell under **two credit-assignment schemes**, so the coupling single-call
sidestepped can be priced rather than assumed.

## The four arms

`grpo_<regime>_<credit>`, each warm-started from the matching SFT arm (GRPO needs
an init that can already sometimes be right — a cold 7B carries no group signal):

| Arm | Cell | Adapters | Warm-start | Credit |
|---|---|---|---|---|
| `grpo_pair_dec` | A | 2 (t1, t2) | `ft_bare` pair | decoupled |
| `grpo_pair_joint` | A | 2 (t1, t2) | `ft_bare` pair | joint |
| `grpo_mix_dec` | B | 1 (shared) | `ft1mix_bare` | decoupled |
| `grpo_mix_joint` | B | 1 (shared) | `ft1mix_bare` | joint |

Read **A against B** for the cost of collapsing two adapters into one under RL,
**B against `sc_grpo`** for the cost of collapsing two calls into one, and
**dec against joint** for whether the coupling matters. All are cot (they emit a
rationale), run at 3 seeds, aggregated like `sc_grpo`.

## Reward

Each call emits the two-call shape `{"explanation": ..., "label": "CR"}`, parsed
by `automisc_ft.infer.parse_label` — identical to the SFT two-call arms, so "did
RL help this cell" is not a parser artefact. Scored in
[src/baseline/two_call.py](../src/baseline/two_call.py):

**Decoupled** — each call is its own GRPO problem, scored alone to a max of 1.0
(the `sc_grpo` advantage scale):

| Call | format | hierarchy | label |
|---|---:|---:|---:|
| T1 | 0.1 | — | 0.9 |
| T2 | 0.1 | 0.1 (t2 ∈ conditioning-t1 group) | 0.8 |

**Joint** — the (T1, T2) pair is scored once with the *single-call* four-part
reward (format 0.1, hierarchy 0.1, t1 0.4, t2 0.4, max 1.0), reconstructed from
the two texts, so the joint arm sits on the same scale as `sc_grpo`. That one
scalar is the shared advantage broadcast over both turns' tokens.

## The two credit schemes

**Decoupled** ([grpo_tc.py](../src/baseline/grpo_tc.py)) makes each call
stationary and reuses the TRL GRPO machinery almost verbatim. The pair regime
runs two stages — a T1 adapter, then a T2 adapter; the mix regime runs one stage
over an interleaved T1+T2 dataset updating one shared adapter, each prompt scored
by its own tier's reward. **T2 conditions on the frozen out-of-fold predicted
T1** (`data/fine_tuning/oof_t1/hlqc_oof_t1_ctx{N}.json`, the same store the SFT
teacher-forcing arms use), so the conditioning distribution does not move under
the T2 optimisation and the `tfzero` exposure the pipeline meets at inference is
held fixed. Full coverage of the training slice is asserted, so no row silently
falls back to gold.

**Joint** ([two_call_grpo.py](../src/baseline/two_call_grpo.py)) keeps the calls
coupled: it samples T1, then T2 given the *sampled* T1, scores the pair once, and
applies the group-centred advantage to both turns. A two-turn trajectory does not
fit TRL's one-prompt-one-reward trainer, so this is a self-contained GRPO loop
over `transformers` + `peft` (advantage `r − mean_g r`, no std division; k3 KL to
the adapter-disabled base; dr_grpo constant-length normalisation). For the pair
regime the T1 tokens' log-probs come from adapter `t1` and the T2 tokens' from
`t2`, so **one shared advantage updates both adapters at once**.

## Three things that are easy to get wrong (per call)

1. **Rare-class weighting is a no-op under std scaling.** Every rollout of one
   prompt shares one gold label and one weight, so `scale_rewards=none` is pinned
   and `grpo_tc._check_weighting_is_live` refuses to start otherwise. Weighting is
   applied to the tier carrying the gold T2 code (the T2 stage for pair, the T2
   records for mix).
2. **Warm-start, not cold.** Each arm starts from its SFT adapter; cold-start is
   the Phase-2 ablation that demonstrates the need rather than asserting it.
3. **The rationale can be farmed.** Format and hierarchy are cheap; the labels
   carry the rest. A collapse guard aborts if the mean rationale falls through the
   floor, and the KL anchor holds the policy near the SFT reference.

Plus one that is specific to the two-call decoupled T2: **the conditioning must
be frozen and fully covered.** An incomplete oof store would let uncovered rows
fall back to gold, quietly training T2 at a gentler exposure than it meets at
inference — asserted, not trusted.

## Reading the recovery probe

Accuracy on the real pipeline cannot tell a T2 that reasons about the utterance
from one that just emits whatever the injected group makes salient — they score
the same whenever the T1 is right. The [recovery probe](../src/baseline/recovery.py)
holds the trained T2 fixed and varies only the injected T1:

- **gold T1** — the ceiling given the right group,
- **predicted T1** — the pipeline number,
- **wrong T1** — a uniformly random incorrect group.

- **recovery rate** = P(T2 correct | wrong T1). High means T2 re-read the utterance.
- **follow rate** = P(T2 is a child of the wrong T1). High means it obeyed the label.

A high recovery with a low follow rate is the resilient policy the reward is
meant to produce. The probe is **unique to the two-call cells**: `sc_grpo` has no
committed T1 to corrupt, which is itself part of the B-vs-F story. Run on the four
GRPO arms and on `ft_bare` / `ft1mix_bare` as the untuned baseline.

## Phases and budget

Run `grpo_tc calibrate` first — the single-call ladder measured 16.3 GPU-h/run,
but the two-call shape (a second call, and for joint a G-way fan-out with a T2
continuation each) has not been measured. Decoupled reuses the frozen oof store,
so no fold trainings are added.

| Phase | Runs | Cut? |
|---|---|---|
| 0 — calibrate pair + mix | 2 short | never |
| 1 — ladder: 4 arms × 3 seeds | pair-dec is 2 stages/seed | never |
| 2 — ablations: weighting-off ×3, cold-start ×1, per cell | ctx5 only | keep if possible |
| 3 — recovery probe: 4 arms + 2 SFT baselines | eval-only, MIG slices | cheap |
| 4 — ctx3 replication | 1 seed/arm | cut first |

## Running it

The GRPO venv, not `scripts/env.sh`. Preconditions: the SFT warm-starts and the
oof_t1 store must exist.

```bash
# preconditions (once, per ctx)
PYTHONPATH=src python -m baseline.local_arm train --target bare --regime pair  --ctx 5
PYTHONPATH=src python -m baseline.local_arm train --target bare --regime mixed --ctx 5
bash scripts/submit_tf_axis.sh                      # freezes the oof_t1 store

# calibrate, then the whole campaign with dependencies wired
CTX=5 CALIBRATE=1 bash scripts/submit_grpo_tc.sh
CTX=5 ABLATE=1    bash scripts/submit_grpo_tc.sh    # Phase 1+2+3, then rescore

# one job at a time
STAGE=train REGIME=pair CREDIT=dec   SEED=0 sbatch scripts/mlerp_grpo_tc.slurm
STAGE=train REGIME=mix  CREDIT=joint SEED=0 sbatch scripts/mlerp_grpo_tc.slurm
STAGE=predict  ARM=grpo_pair_dec SEED=0 sbatch --gres=gpu:3g.20gb:1 scripts/mlerp_grpo_tc.slurm
STAGE=recovery ARM=grpo_pair_dec SEED=0 sbatch --gres=gpu:3g.20gb:1 scripts/mlerp_grpo_tc.slurm

# score both ladders into docs/BASELINE_RESULTS.md
PYTHONPATH=src python -m baseline.eval
```

`train` and `calibrate` need a full 40GB card; `predict` and `recovery` are
two-call HF decoding at ~16GB and fit a 20GB MIG slice. Each variant writes to
its own adapter directory and result filename (`_unw`, `_cold`), so Phase 2 can
never overwrite Phase 1.

## Smoke test (CPU, tiny model)

```bash
PYTHONPATH=src python -m baseline.grpo_tc train --regime mix --credit dec --seed 0 --ctx 3 \
  --limit 4 model.base_model=Qwen/Qwen2.5-0.5B-Instruct inference.force_cpu=true
```

## Code map

| Path | Role |
|---|---|
| [conf/grpo_tc_config.yaml](../conf/grpo_tc_config.yaml) | hyperparameters, copied from the single-call config |
| [src/baseline/two_call.py](../src/baseline/two_call.py) | per-tier prompts, the reward split, the joint scorer |
| [src/baseline/grpo_tc.py](../src/baseline/grpo_tc.py) | decoupled training, prediction, calibration |
| [src/baseline/two_call_grpo.py](../src/baseline/two_call_grpo.py) | the joint-trajectory GRPO loop |
| [src/baseline/recovery.py](../src/baseline/recovery.py) | the T2-recovery probe |
| [scripts/mlerp_grpo_tc.slurm](../scripts/mlerp_grpo_tc.slurm) | one job (STAGE-driven) |
| [scripts/submit_grpo_tc.sh](../scripts/submit_grpo_tc.sh) | the whole campaign, dependency-wired |
| [docs/experiments/2026-08-26-two-call-grpo.md](experiments/2026-08-26-two-call-grpo.md) | campaign write-up |
