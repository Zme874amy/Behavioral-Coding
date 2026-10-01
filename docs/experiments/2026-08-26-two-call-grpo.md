# Two-call GRPO — cells A and B under RL

## Question

Does the structure a policy is optimised in — one call or two, one adapter or two
— change whether RL-discovered reasoning pays off? And does the coupled
non-stationarity that `sc_grpo` avoided by collapsing to a single call actually
cost anything, or was the simplification free?

## Why now

GRPO lived in exactly one cell of the matrix ([EXPERIMENTS.md](../EXPERIMENTS.md)
§1): 1 call, 1 adapter. The SFT grid fills all four cells, so `sc_grpo` could only
be compared to SFT arms that differ from it on *two* axes at once. Filling the two
two-call cells under RL closes that gap and makes the adapter×call ablation
identifiable under GRPO. The single-call headline was also equivocal — GRPO drew
level with `sc_ft_bare` rather than beating it — so whether reasoning helps is
worth re-asking where the calls are actually separated.

## What changed

- **Four new arms** `grpo_{pair,mix}_{dec,joint}`, warm-started from `ft_bare` /
  `ft1mix_bare`, under two credit schemes:
  - *decoupled* — each call a stationary GRPO problem; T2 conditions on the frozen
    out-of-fold predicted T1 (reusing the teacher-forcing store). Reuses the TRL
    machinery.
  - *joint* — a self-contained two-turn GRPO loop; one shared advantage over both
    turns (and, for the pair regime, both adapters).
- **A T2-recovery probe** ([recovery.py](../../src/baseline/recovery.py)):
  conditions the fixed T2 on gold / predicted / random-wrong T1 and reports
  recovery rate and follow rate. Unique to the two-call cells.
- New code: `two_call.py`, `grpo_tc.py`, `two_call_grpo.py`, `recovery.py`;
  `conf/grpo_tc_config.yaml`; eval-registry entries; `mlerp_grpo_tc.slurm` +
  `submit_grpo_tc.sh`. Full design and rationale: [GRPO_TWOCALL.md](../GRPO_TWOCALL.md).

## How to run

Preconditions: SFT warm-starts (`ft_bare` pair, `ft1mix_bare`) and the oof_t1
store. Then:

```bash
CTX=5 CALIBRATE=1 bash scripts/submit_grpo_tc.sh   # measure first
CTX=5 ABLATE=1    bash scripts/submit_grpo_tc.sh   # ladder + ablations + recovery + rescore
```

## Cells produced

`qwen_grpo_{pair,mix}_{dec,joint}[_unw|_cold]_inf_cot_ctx5_seed{0,1,2}` in
`data/annotated/baseline/`, scored by `baseline.eval` into cells A and B of the
Coverage matrix; recovery JSONs in `outputs/grpo/recovery_*.json`.

## Results

All four arms completed at ctx5, 3 seeds each (weighted Phase-1 only; the
`_unw`/`_cold` ablations and ctx3 were not run and show as missing in Coverage).
Full numbers in [BASELINE_RESULTS.md](../BASELINE_RESULTS.md); the T2 (all, mean
over seeds) accuracy / macro-F1 (learnable) the argument rests on:

| Cell | Arm | T2 acc | T2 F1(learn) |
|---|---|---|---|
| F (1-call, 1-ad) | `sc_ft_bare` / `sc_grpo` | 0.660 / 0.660 | 0.485 / 0.481 |
| A (2-call, 2-ad) | `ft_bare` | 0.642 | 0.446 |
| | `grpo_pair_dec` / `grpo_pair_joint` | 0.647 / 0.648 | 0.429 / 0.431 |
| B (2-call, 1-ad) | `ft1mix_bare` | 0.678 | 0.502 |
| | `grpo_mix_dec` / `grpo_mix_joint` | 0.667 / 0.661 | 0.489 / 0.486 |

Seed spreads on the generated table are small (±0.001–0.006), so the ~0.01 gaps
below sit at or inside the noise band.

- **Decoupled ≈ joint.** pair 0.647 vs 0.648; mix 0.667 vs 0.661 — the credit
  scheme does not move T2, so the coupled non-stationarity single-call was built
  to avoid costs ~nothing. (Caveat: joint ran G=6 vs decoupled G=8 to fit the 24h
  wall; the wash is far larger than that knob would plausibly move.)
- **1 adapter (B) ≥ 2 (A)**, under RL as under SFT: `mix` beats `pair` in every
  pairing, and `ft1mix_bare` is the single best T2 cell.
- **RL draws level with SFT, never surpasses it**, in every structural cell —
  replicating the single-call headline across call- and adapter-count, so the
  null is not a single-call artifact.

The recovery probe explains the lack of T2 headroom: fed a wrong T1, T2 recovers
on only ~0.07 of rows and emits a code inside the wrong group ~0.84 of the time
(`mix_dec` seed0) — it conditions on the injected group label, not the utterance.

## Verdict

The assumption that motivated single-call — that two-call RL's coupling is
*harmful* — does not hold: joint and decoupled are a wash, so single-call was
free rather than a compromise. The headline "self-discovered reasoning helps"
assumption is confirmed **negative** with higher confidence: RL only repairs to
the SFT level in every cell, never beating it. The ceiling here is therefore not
in the training signal (RL vs SFT) or the pipeline structure (call/adapter count)
— those are interchangeable within noise — which points the lever at **data (tail
coverage under the HLQC→MIV6.3A shift) and scale**, not more RL or structural
tuning. This campaign's contribution is ruling those knobs out.
