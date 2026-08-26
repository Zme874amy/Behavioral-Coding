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

Pending — not yet run. Numbers are left to `baseline.eval`; this doc will quote
the comparisons it argues from (A vs B, B vs `sc_grpo`, dec vs joint, each vs its
SFT warm-start, and the recovery probe) once cells land.

## Verdict

Pending.
