# Data-split review (2026-10-05)

**Question.** Every MISC result so far trains on HLQC gold and tests on MIV6.3A gold. That protocol
was inherited, not chosen: the May 2026 proposal planned a within-corpus 5-fold CV as the primary
test, and no document records why the project switched to the corpus hold-out. This review states
what a good split must do, measures the candidates, and gives a justified choice for **every**
dataset.

Evidence code: `src/eda/split_eval.py` (`PYTHONPATH=src python -m eda.split_eval`). Manifests:
`src/eda/splits.py` → `data/splits/`.

## 1. What a split has to satisfy

| # | Criterion | Why it matters here |
|---|---|---|
| C1 | **Test labels are the most reliable we have** | a noisy test penalises correct models (the 2026-10-04 teacher audit showed exactly that on HLQC) |
| C2 | **Test matches the target use** | RQ1: "a small model, adapted locally **on expert labels a service already holds**". The target is a service coding its own sessions; the AutoMISC target is MIBot chatbot transcripts |
| C3 | **No leakage** | split by session; duplicate sessions on the same side; no test-set tuning |
| C4 | **Enough training data, incl. rare codes** | only 2,746 human MISC labels exist in total |
| C5 | **Statistical power** | the test must resolve the differences we report |
| C6 | **Comparability** | with AutoMISC's GPT-4.1 numbers (all 821 MIV utterances) and with our past results |
| C7 | **Model selection without the test** | checkpoint and hyperparameters chosen before seeing test labels |

## 2. Evidence

### E1. Which labels are most reliable? (C1)

| Gold set | Who coded it | Reliability evidence | Problems found (DATASETS.md §4) |
|---|---|---|---|
| `misc.miv63a.gold` | 3 interns + 1 grad student, consensus, aligned to MI clinicians | Fleiss κ ≥ 0.6 after alignment | FI 9.7% (above 5%) but genuine pleasantries; no other rule violations found |
| `misc.hlqc.gold` | AutoMISC team | none reported | FI catch-all (64% of acknowledgements, 19% vs the 5% ceiling); under-codes change talk vs AnnoMI experts (κ 0.36); 6 T1/T2 contradictions; ASR text |
| `miti.casaa.gold` | MITI developers' lab (reference coding) | the reference standard | MITI scheme; only 6 exact MISC T2 codes (311 gold, 18 clean sessions) |

→ **MIV6.3A is the most reliable MISC test.** HLQC is not fit to be a test set. CASAA is the best *external* test.

### E2. Is the 10-session MIV test representative? (C2)

- All 10 sessions come from the thesis's **MI target population** ("low-confidence-or-discordant": confidence ≤ 5, or confidence > importance). The 58 high-confidence sessions are outside the study's scope.
- The thesis describes them as "the first 10 conversations" of MIBot v6.3A.
- Against the other 105 target-population sessions, nothing differs significantly:

| Measure | Test (10) | Rest of target population | Mann-Whitney p |
|---|---:|---:|---:|
| utterances per session | 82.1 | 80.6 | 0.80 |
| Δ confidence | +1.90 | +1.71 | 0.72 |
| Δ readiness | +0.10 | +0.79 | 0.15 |

→ **Representative of the population it should represent.** This corrects an earlier note in DATASETS.md that compared against all 163 other sessions.

### E3. Statistical power of the test (C5)

Main model (`ft1mix_bare`, seed 42) on the MIV test, with a session-cluster bootstrap:

| Speaker | T2 macro-F1 | 95% CI | Dropping one session | Codes with < 5 test examples |
|---|---:|---|---|---|
| counsellor | 0.501 | [0.435, 0.602] | — | FA, DI, RF (3 of 14) |
| client | 0.414 | [0.322, 0.522] | 0.368–0.474 | AB+, TS+, N−, AC− (4 of 13) |

→ **The test is thin.** Absolute intervals are 0.17–0.20 wide (paired differences are tighter), and one session moves client macro-F1 by ±0.05. Any protocol must keep all 821 MIV utterances scored. Shrinking the test to fit a held-out dev set from MIV would make this worse.

### E4. Does in-domain training data help? Proxy experiment (C2, C4)

TF-IDF + logistic regression, one classifier per speaker. It isolates data/convention match, not model capacity. All MIV scores cover all 821 utterances.

| Protocol | Counsellor macro-F1 | Counsellor acc | Client macro-F1 | Client acc |
|---|---:|---:|---:|---:|
| **P0** train HLQC → test MIV (current) | 0.260 | 0.462 | 0.083 | 0.502 |
| **P2** MIV 5-fold CV, MIV data only | **0.474** | **0.647** | 0.192 | **0.577** |
| **P3** HLQC + other MIV folds → MIV fold | 0.442 | 0.612 | **0.209** | 0.552 |
| **P1** train MIV → test HLQC (reversed) | 0.127 | 0.284 | 0.109 | 0.673 |

- About 650 in-domain utterances beat 1,925 HLQC utterances by a wide margin (counsellor macro-F1 0.47 vs 0.26). The convention and register gap (DATASETS.md §5.3) costs more than the extra data is worth.
- Adding HLQC to in-domain data helps client codes slightly and costs counsellor codes slightly.
- **Template leakage ruled out:** only 14 of 652 longer MIV utterances recur verbatim in another fold (closing lines such as "Thank you for our conversation today.").

### E5. Does pooling hurt a third domain? External check on CASAA (C2)

Counsellor only, CASAA's 18 clean sessions:

| Training data | CASAA T2 macro-F1 (6 exact codes) | CASAA T2 acc | CASAA T1 acc |
|---|---:|---:|---:|
| HLQC | 0.178 | 0.219 | 0.457 |
| MIV | 0.144 | 0.170 | 0.413 |
| **HLQC + MIV** | **0.234** | **0.235** | **0.492** |

→ **Pooled training generalises best** to an independent spoken-MI domain. Pooling does not trade away robustness.

## 3. Candidates against the criteria

| Protocol | C1 test reliable | C2 target match | C3 no leak | C4 training data | C5 power | C6 comparability | Verdict |
|---|---|---|---|---|---|---|---|
| P0 HLQC → MIV (current) | ✓ | cold-start transfer only (service has no labels) | ✓ | 1,925, wrong convention | 821 test | ✓ with all past results | keep as **secondary** "cold-start" setting |
| P1 MIV → HLQC | ✗ HLQC labels noisy | ✗ | ✓ | 821 | 1,925 noisy | ✗ | **reject** |
| P2 MIV-only 5-fold CV | ✓ | ✓ | ✓ (session folds) | ~650 per fold | 821 test | ✓ with AutoMISC (all 821) | dominated by P3 on client and on CASAA |
| **P3 HLQC + MIV 5-fold CV** | ✓ | ✓ matches RQ1 as worded | ✓ (session folds; HLQC val fold for selection) | ~2,580 per fold | 821 test | ✓ with AutoMISC; new numbers for fine-tuned arms | **recommended primary** |
| P4 pooled 20-session CV, test on both | ✗ half the test is HLQC | partly | ✓ | ~2,200 | 2,746 (mixed quality) | ✗ | reject as headline |

**Costs of P3.** Five training runs per arm instead of one. Fine-tuned results so far were obtained under P0 and stay valid as the cold-start setting, but they are not P3 numbers. Zero-/few-shot baselines are unaffected: they don't train, and they are already scored on all 821.

**The proxy's limits.** A 7B model brings MI knowledge from pretraining and few-shot prompts, so the size of the P0 → P3 gap will differ. The direction, plus E1–E3, is what the recommendation rests on. **Confirm on the 7B model** with one P3 run of `ft1mix_bare` (5 folds × 1 seed) before switching the headline.

## 4. Decision per dataset

| Dataset | Role | Split | Justification |
|---|---|---|---|
| `misc.miv63a.gold` | **test** (P0); test folds + in-domain training (P3) | P0: all 10 test. P3: 5 session folds (seed 42; the folds `automisc_ft` planned), each session tested once | most reliable MISC labels (E1); target population (E2); every utterance scored (E3) |
| `misc.hlqc.gold` | **training only** | all 10 in training; checkpoint selection on HLQC val fold 0 of 7 (369 rows), fixed before training | never a test set (E1); a selection fold that never touches MIV (C7) |
| `miti.casaa.gold` | **external test only** | 18 sessions; Emmy = high_121 and Rounder = high_072 excluded | reference-quality labels from a third domain (E1, E5); too small to train on |
| `annomi.gold` | **external test** for MISC models (counsellor main behaviour, client C/S/N at T1); own-scheme split only for AnnoMI-scheme work | transfer test: exclude 15/21/44/53 (HLQC train copies), plus 48 transcripts if the model saw `pool.hlqc`; score questions at T1 (AnnoMI "open" ≠ MISC OQ). Own scheme: 91/22/20 transcripts, stratified by quality × annotator, with the 7 ten-rater transcripts (best labels) in test | expert labels, but a different scheme and coding convention, so a transfer test, not a MISC test; the cleanest labels go to the test side |
| `welivita.gold` | **weak pool** (self-training / distillation), weak transfer test; own-scheme split only for Welivita-scheme work | own scheme: 1,604/196/200 dialogues, duplicate clusters kept together, labels = 7,152 stage-I-agreed sentences | crowd κ 0.34, written forum domain, and self-training with it hurt (−0.047), so it should not be a headline test |
| `pool.hlqc`, `pool.miv63a`, `pool.miv63b` | unlabelled pools (training side only) | drop the 10 MIV test sessions; drop the 45 redundant HLQC duplicates; drop the pool twins of any corpus used as a test | leakage control (C3) |
| `synth.*` | training augmentation only | never a test | generator-assigned labels; see SYNTHETIC_DATA.md |

## 5. What has to happen next

1. **Your decision:** adopt P3 as the primary MISC protocol, with P0 kept as the cold-start setting. The manifest exists already: `data/splits/misc_pooled_cv.json`, marked PROPOSED.
2. If yes: one confirmation run of `ft1mix_bare` under P3 (5 folds) on MLeRP, then re-run the arms we want to claim under P3 (baseline, synthesis v2/v3).
3. Independent of the decision: rerun the AnnoMI transfer test with the exclusions and T1 questions.
