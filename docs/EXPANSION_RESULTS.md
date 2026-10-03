# Data-expansion results (1-adapter backbone)

*Generated 2026-10-03 by `selftrain.expansion_table` — edit that module, not this file.*

Every arm below is a retrain of **FT1-Mix-Bare** (one shared adapter, two calls), differing only in its training set, and is scored on the same 821 MIV6.3A gold utterances. The baseline is the same recipe on human HLQC data alone, run at 3 seeds.

**How to read.** `Seeds` is matched/run: the difference is taken at matched training seeds (42↔42, 1↔1, 2↔2) and averaged. The bracket is a 95% CI from resampling the evaluation *conversations* (the same resample applied to every seed of both arms). `real` needs all three: the same sign in every matched seed, a CI that excludes 0, and a mean difference larger than the seed-noise band. `single-seed` means the CI excludes 0 but there is only one matched seed — not yet a claim. `ns` means the CI includes 0. `sign-flips` means seeds disagree on the direction.

## Main arms — T2, all speakers

| Arm | vs | Seeds | T2 acc | Δ acc [95% CI] | Verdict | T2 Macro-F1 | Δ Macro-F1 [95% CI] | Verdict |
|---|---|---:|---:|---:|---|---:|---:|---|
| Self-training (Welivita pseudo-labels) | baseline | 3/3 | 0.682 ± 0.010 | +0.009 [-0.010, +0.029] | ns | 0.413 ± 0.033 | -0.044 [-0.074, -0.022] | real |
| Synthesis v1 (one-shot utterances) | baseline | 3/3 | 0.692 ± 0.028 | +0.019 [-0.005, +0.038] | ns | 0.508 ± 0.027 | +0.052 [+0.007, +0.079] | real |
| Synthesis v2 proto (dialogue windows) | baseline | 3/3 | 0.687 ± 0.023 | +0.014 [+0.001, +0.025] | sign-flips | 0.510 ± 0.023 | +0.053 [+0.025, +0.077] | real |
| Synthesis v2 boundary | baseline | 3/3 | 0.686 ± 0.005 | +0.013 [-0.014, +0.036] | ns | 0.494 ± 0.011 | +0.037 [+0.000, +0.077] | real |
| Synthesis v3, HLQC topic mix | baseline | 3/3 | 0.685 ± 0.020 | +0.013 [-0.011, +0.033] | ns | 0.502 ± 0.009 | +0.045 [+0.023, +0.069] | real |
| Synthesis v3, MIV6.3A topic mix | baseline | 3/3 | 0.690 ± 0.007 | +0.017 [-0.008, +0.037] | ns | 0.528 ± 0.002 | +0.071 [+0.032, +0.105] | real |

## Ablations — each against the arm it differs from by one factor

| Arm | vs | Seeds | T2 acc | Δ acc [95% CI] | Verdict | T2 Macro-F1 | Δ Macro-F1 [95% CI] | Verdict |
|---|---|---:|---:|---:|---|---:|---:|---|
| v3 HLQC mix + SocialDial rule | synth_v3_train | 3/3 | 0.681 ± 0.012 | -0.004 [-0.015, +0.009] | ns | 0.509 ± 0.014 | +0.007 [-0.013, +0.027] | ns |
| v2 proto, synthetic rows only (no human) | baseline | 1/1 | 0.531 | -0.140 [-0.185, -0.090] | single-seed | 0.398 | -0.061 [-0.128, -0.003] | single-seed |
| v2 boundary, keep-hard (verifier off) | synth_v2_bound | 1/1 | 0.621 | -0.065 [-0.090, -0.041] | single-seed | 0.508 | +0.002 [-0.064, +0.063] | ns |

- **v3 HLQC mix + SocialDial rule** — isolates the rule: same generator and topic mix, rule on
- **v2 proto, synthetic rows only (no human)** — SocialDial's human / synthetic / both split; read with baseline (human) and v2 proto (both)
- **v2 boundary, keep-hard (verifier off)** — isolates the verifier filter: same windows, unverified labels kept

## Self-training v2 and the 32B-teacher comparison

Same pipeline for every row: calibrated per-code thresholds, prior-aligned quotas, at most one pseudo label per human label, counsellor rows only. Only the labeller differs (the student's 3-seed self-ensemble vs the 32B teacher).

| Arm | vs | Seeds | T2 acc | Δ acc [95% CI] | Verdict | T2 Macro-F1 | Δ Macro-F1 [95% CI] | Verdict |
|---|---|---:|---:|---:|---|---:|---:|---|
| Self-training v2 (Welivita) | baseline | 3/3 | 0.663 ± 0.019 | -0.010 [-0.024, +0.007] | ns | 0.410 ± 0.023 | -0.047 [-0.071, -0.025] | real |
| Self-training v2, round 2 (Welivita) | baseline | 0/0 | — | — | pending | — | — | pending |
| Self-training v2 (MIV6.3B pool) | baseline | 3/3 | 0.695 ± 0.009 | +0.023 [+0.006, +0.043] | real | 0.426 ± 0.021 | -0.030 [-0.046, -0.009] | <floor |
| 32B-teacher labels (Welivita) | baseline | 0/0 | — | — | pending | — | — | pending |
| 32B-teacher labels (MIV6.3B pool) | baseline | 0/0 | — | — | pending | — | — | pending |

### Teacher vs self on the same pool

| Arm | vs | Seeds | T2 acc | Δ acc [95% CI] | Verdict | T2 Macro-F1 | Δ Macro-F1 [95% CI] | Verdict |
|---|---|---:|---:|---:|---|---:|---:|---|
| Teacher vs self (Welivita) | selftrain_v2 | 0/0 | — | — | pending | — | — | pending |
| Teacher vs self (MIV6.3B) | selftrain_v2b | 0/0 | — | — | pending | — | — | pending |


## Retrieval on the same backbone

| Arm | vs | Seeds | T2 acc | Δ acc [95% CI] | Verdict | T2 Macro-F1 | Δ Macro-F1 [95% CI] | Verdict |
|---|---|---:|---:|---:|---|---:|---:|---|
| Retrieval few-shot on FT1-Mix, Inf-Bare | baseline | 1/1 | 0.667 | -0.004 [-0.033, +0.027] | ns | 0.495 | +0.036 [-0.042, +0.098] | ns |
| Retrieval few-shot on FT1-Mix, Inf-CoT | baseline_cot | 1/1 | 0.671 | +0.005 [-0.020, +0.033] | ns | 0.500 | +0.028 [-0.032, +0.073] | ns |

## Where the T2 change comes from — by speaker

| Arm | Speaker | Seeds | Macro-F1 | Δ Macro-F1 [95% CI] | Verdict |
|---|---|---:|---:|---:|---|
| Self-training (Welivita pseudo-labels) | counsellor | 3/3 | 0.482 ± 0.019 | -0.015 [-0.049, +0.020] | ns |
| Self-training (Welivita pseudo-labels) | client | 3/3 | 0.339 ± 0.053 | -0.074 [-0.146, -0.028] | real |
| Synthesis v1 (one-shot utterances) | counsellor | 3/3 | 0.523 ± 0.029 | +0.026 [-0.007, +0.059] | ns |
| Synthesis v1 (one-shot utterances) | client | 3/3 | 0.492 ± 0.026 | +0.079 [+0.002, +0.121] | real |
| Synthesis v2 proto (dialogue windows) | counsellor | 3/3 | 0.520 ± 0.008 | +0.022 [-0.005, +0.057] | ns |
| Synthesis v2 proto (dialogue windows) | client | 3/3 | 0.499 ± 0.050 | +0.086 [+0.025, +0.140] | real |
| Synthesis v2 boundary | counsellor | 3/3 | 0.545 ± 0.006 | +0.048 [+0.008, +0.085] | real |
| Synthesis v2 boundary | client | 3/3 | 0.440 ± 0.027 | +0.026 [-0.033, +0.091] | ns |
| Synthesis v3, HLQC topic mix | counsellor | 3/3 | 0.533 ± 0.027 | +0.036 [+0.009, +0.064] | real |
| Synthesis v3, HLQC topic mix | client | 3/3 | 0.469 ± 0.016 | +0.056 [+0.021, +0.102] | real |
| Synthesis v3, MIV6.3A topic mix | counsellor | 3/3 | 0.538 ± 0.014 | +0.041 [+0.004, +0.074] | real |
| Synthesis v3, MIV6.3A topic mix | client | 3/3 | 0.517 ± 0.013 | +0.104 [+0.043, +0.165] | real |
| v3 HLQC mix + SocialDial rule | counsellor | 3/3 | 0.528 ± 0.010 | -0.005 [-0.028, +0.013] | ns |
| v3 HLQC mix + SocialDial rule | client | 3/3 | 0.488 ± 0.019 | +0.019 [-0.028, +0.061] | ns |
| v2 proto, synthetic rows only (no human) | counsellor | 1/1 | 0.381 | -0.120 [-0.178, -0.055] | single-seed |
| v2 proto, synthetic rows only (no human) | client | 1/1 | 0.416 | +0.002 [-0.106, +0.102] | ns |
| v2 boundary, keep-hard (verifier off) | counsellor | 1/1 | 0.495 | -0.045 [-0.087, +0.003] | ns |
| v2 boundary, keep-hard (verifier off) | client | 1/1 | 0.522 | +0.052 [-0.078, +0.159] | ns |
| Self-training v2 (Welivita) | counsellor | 3/3 | 0.459 ± 0.026 | -0.039 [-0.063, -0.010] | real |
| Self-training v2 (Welivita) | client | 3/3 | 0.358 ± 0.035 | -0.055 [-0.112, -0.002] | real |
| Self-training v2 (MIV6.3B pool) | counsellor | 3/3 | 0.473 ± 0.004 | -0.024 [-0.051, +0.013] | ns |
| Self-training v2 (MIV6.3B pool) | client | 3/3 | 0.376 ± 0.042 | -0.037 [-0.063, -0.008] | <floor |
| Retrieval few-shot on FT1-Mix, Inf-Bare | counsellor | 1/1 | 0.506 | +0.005 [-0.030, +0.053] | ns |
| Retrieval few-shot on FT1-Mix, Inf-Bare | client | 1/1 | 0.484 | +0.069 [-0.076, +0.172] | ns |
| Retrieval few-shot on FT1-Mix, Inf-CoT | counsellor | 1/1 | 0.510 | +0.021 [-0.015, +0.079] | ns |
| Retrieval few-shot on FT1-Mix, Inf-CoT | client | 1/1 | 0.490 | +0.035 [-0.082, +0.110] | ns |

## T1 check — does the added data cost Tier-1?

| Arm | Seeds | T1 acc | Δ acc [95% CI] | Verdict |
|---|---:|---:|---:|---|
| Self-training (Welivita pseudo-labels) | 3/3 | 0.805 ± 0.009 | +0.019 [+0.001, +0.039] | real |
| Synthesis v1 (one-shot utterances) | 3/3 | 0.794 ± 0.018 | +0.009 [-0.003, +0.021] | ns |
| Synthesis v2 proto (dialogue windows) | 3/3 | 0.797 ± 0.021 | +0.011 [-0.003, +0.022] | ns |
| Synthesis v2 boundary | 3/3 | 0.793 ± 0.005 | +0.008 [-0.005, +0.019] | ns |
| Synthesis v3, HLQC topic mix | 3/3 | 0.790 ± 0.018 | +0.005 [-0.006, +0.015] | ns |
| Synthesis v3, MIV6.3A topic mix | 3/3 | 0.791 ± 0.015 | +0.005 [-0.007, +0.016] | ns |
| v3 HLQC mix + SocialDial rule | 3/3 | 0.788 ± 0.009 | -0.002 [-0.015, +0.011] | ns |
| v2 proto, synthetic rows only (no human) | 1/1 | 0.687 | -0.101 [-0.140, -0.054] | single-seed |
| v2 boundary, keep-hard (verifier off) | 1/1 | 0.784 | -0.013 [-0.034, +0.009] | ns |
| Self-training v2 (Welivita) | 3/3 | 0.796 ± 0.013 | +0.010 [-0.003, +0.023] | ns |
| Self-training v2 (MIV6.3B pool) | 3/3 | 0.805 ± 0.009 | +0.019 [+0.005, +0.036] | real |
| Retrieval few-shot on FT1-Mix, Inf-Bare | 1/1 | 0.780 | -0.009 [-0.027, +0.010] | ns |
| Retrieval few-shot on FT1-Mix, Inf-CoT | 1/1 | 0.786 | -0.002 [-0.016, +0.012] | ns |

## Cross-scheme generalization (shared codes only)

| Model | Corpus | Level | Speaker | n | Accuracy [95% CI] | Macro-F1 shared [95% CI] | OOV pred |
|---|---|---|---|---:|---:|---:|---:|
| Zero-shot (no adapter) | annomi | T1 | client | 4807 | 0.709 [0.669–0.747] | 0.470 [0.436–0.502] | 0.0% |
| Zero-shot (no adapter) | annomi | T2 | counsellor | 2628 | 0.281 [0.249–0.314] | 0.305 [0.282–0.327] | 55.1% |
| Zero-shot (no adapter) | welivita | T2 | counsellor | 1501 | 0.142 [0.119–0.167] | 0.183 [0.153–0.209] | 54.5% |
| Baseline FT1-Mix (human HLQC only) | annomi | T1 | client | 4807 | 0.714 [0.672–0.754] | 0.498 [0.457–0.540] | 0.0% |
| Baseline FT1-Mix (human HLQC only) | annomi | T2 | counsellor | 2628 | 0.510 [0.477–0.543] | 0.526 [0.496–0.554] | 19.8% |
| Baseline FT1-Mix (human HLQC only) | welivita | T2 | counsellor | 1501 | 0.280 [0.247–0.313] | 0.292 [0.262–0.319] | 14.1% |
| Synthesis v2 proto (dialogue windows) | annomi | T1 | client | 4807 | 0.726 [0.683–0.765] | 0.545 [0.500–0.590] | 0.0% |
| Synthesis v2 proto (dialogue windows) | annomi | T2 | counsellor | 2628 | 0.524 [0.492–0.557] | 0.532 [0.501–0.561] | 19.3% |
| Synthesis v2 proto (dialogue windows) | welivita | T2 | counsellor | 1501 | 0.268 [0.238–0.302] | 0.307 [0.273–0.335] | 19.0% |

## Generalization ladder — FT1-Mix baseline, counsellor T2 accuracy

Read top to bottom: each rung adds one kind of shift. The cross-scheme rungs are scored on the shared codes only, so they are not on exactly the same vocabulary as the first two.

| Rung | Vocabulary | Accuracy [95% CI] |
|---|---|---:|
| HLQC, in-distribution 5-fold CV | full MISC | 0.621 [0.584–0.663] |
| MIV6.3A, cross-corpus (seed 42) | full MISC | 0.653 [0.615–0.692] |
| AnnoMI, cross-scheme | shared codes | 0.510 [0.477–0.543] |
| Welivita/MITI, cross-scheme (weak gold) | shared codes | 0.280 [0.247–0.313] |

## Not yet available

- Self-training v2, round 2 (Welivita) — seed 1
- Self-training v2, round 2 (Welivita) — seed 2
- Self-training v2, round 2 (Welivita) — seed 42
- 32B-teacher labels (Welivita) — seed 1
- 32B-teacher labels (Welivita) — seed 2
- 32B-teacher labels (Welivita) — seed 42
- 32B-teacher labels (MIV6.3B pool) — seed 1
- 32B-teacher labels (MIV6.3B pool) — seed 2
- 32B-teacher labels (MIV6.3B pool) — seed 42
- Teacher vs self (Welivita) — seed 1
- Teacher vs self (Welivita) — seed 2
- Teacher vs self (Welivita) — seed 42
- Teacher vs self (MIV6.3B) — seed 1
- Teacher vs self (MIV6.3B) — seed 2
- Teacher vs self (MIV6.3B) — seed 42
