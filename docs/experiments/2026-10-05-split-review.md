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
| `annomi.gold` | **external test** for MISC models (counsellor main behaviour, client C/S/N at T1); own-scheme split only for AnnoMI-scheme work | transfer test: exclude 15/21/44/53 (HLQC train copies), plus 48 transcripts if the model saw `pool.hlqc`; score questions at T1 (AnnoMI "open" ≠ MISC OQ). Own scheme: see §7.1 (series-level, 97/15/21) | expert labels, but a different scheme and coding convention, so a transfer test, not a MISC test; the cleanest labels go to the test side |
| `welivita.gold` | **weak pool** (self-training / distillation), weak transfer test; own-scheme split only for Welivita-scheme work | own scheme: see §7.2 (1,602/203/195 dialogues) | crowd κ 0.34, written forum domain, and self-training with it hurt (−0.047), so it should not be a headline test |
| `pool.hlqc`, `pool.miv63a`, `pool.miv63b` | unlabelled pools (training side only) | drop the 10 MIV test sessions; drop the 45 redundant HLQC duplicates; drop the pool twins of any corpus used as a test | leakage control (C3) |
| `synth.*` | training augmentation only | never a test | generator-assigned labels; see SYNTHETIC_DATA.md |

## 5. What has to happen next (updated in §8)

1. **Your decision:** adopt P3 as the primary MISC protocol, with P0 kept as the cold-start setting. The manifest exists already: `data/splits/misc_pooled_cv.json`, marked PROPOSED.
2. If yes: one confirmation run of `ft1mix_bare` under P3 (5 folds) on MLeRP, then re-run the arms we want to claim under P3 (baseline, synthesis v2/v3).
3. Independent of the decision: rerun the AnnoMI transfer test with the exclusions and T1 questions.

## 6. Leakage audit (follow-up, same day)

*Does pooled CV, or any other split, leak?* Each channel was checked in the code and the data. Evidence: `eda.split_eval.pooled_cv_leakage`, `annomi_grouping`, `welivita_grouping`.

### 6.1 MISC: current protocol (P0) and proposed pooled CV (P3)

| # | Leakage channel | P0 (current) | P3 (pooled CV) | Evidence | Mitigation |
|---|---|---|---|---|---|
| L1 | Same session in train and test | no | no: folds are by session | manifest check: each MIV session tested once, never trained on in its own fold | built into `splits.check` |
| L2 | Same person in train and test | no | no | 10 distinct participants; no participant id repeats among the 173 MIV6.3A sessions | — |
| L3 | Same session in another corpus | HLQC-train copies in AnnoMI/CASAA | same | DATASETS.md §5.1 | exclusion lists (unchanged) |
| L4 | Near-identical utterances across the split (chatbot templates) | n/a | small | 18 of 532 longer counsellor utterances have a ≥ 0.9-similar one in another fold (closing/continuation templates); 14 of 652 exact | accepted: the deployed chatbot repeats them too. Out-of-domain tests (CASAA, AnnoMI) measure generalisation beyond this chatbot |
| L5 | **Test-informed design decisions** | **yes, already** | yes | the retriever's rare-code list (SU, EC, AF, GI) and the self-training "no cap" codes were picked because they are common in MIV6.3A; synthesis `v3_mivmix` uses MIV6.3A topic counts; the main setting (`ft1mix_bare`, ctx 5) was chosen by comparing MIV scores | disclose for P0. For P3, freeze the current config and rebuild every such statistic from the fold's training sessions (`misc_pooled_cv.leakage_rules`) |
| L6 | Checkpoint / hyperparameter selection on the test | no (HLQC val fold) | no (same HLQC val fold) | `baseline/grpo.py`, THESIS_SOURCE checkpoint rule | no MIV session is used for selection |
| L7 | Few-shot exemplars from the test | no (HLQC only) | no | `baseline/fewshot.py` | — |
| L8 | Pools containing test sessions | handled | handled | `selftrain.label_pool` drops all 10 MIV sessions and eval-like utterances | keep excluding all 10 in P3 too |
| L9 | Codes seen in only one fold | n/a | AC− (1), RF (1) | they can't be learned in-domain when that fold is test | not leakage. Score pooled out-of-fold predictions once, not per fold |
| L10 | Same coders in train and test | n/a | yes: MIV's coding team labels both sides | by design | this is RQ1's scenario (a service's own coders). Report it as such; CASAA/AnnoMI give coder-independent checks |

**Answer: pooled CV adds no leak of sessions, people or text beyond small template overlap (L4).** Its real risk is reusing decisions that were tuned on MIV (L5), and that risk exists **already** in the current protocol. The rules in `misc_pooled_cv.json` close it for P3: freeze the config, recompute statistics per fold, pool the scoring.

### 6.2 Other datasets

| Dataset | Channel | Found | Proxy effect | Decision |
|---|---|---|---|---|
| AnnoMI | **parts of one video series / good-bad versions of one role-play split across train/dev/test** | 31 multi-transcript series (e.g. "Daryl interviews Ricky 1–3", "The Effective / Ineffective Physician"); **13 of them were split** by the old transcript-level manifest | none measurable on coarse labels (proxy 0.715 by transcript vs 0.718 by series) | **fixed:** `annomi_own` is now series-level (97/15/21 transcripts) |
| AnnoMI | same annotator in train and test | yes (10 annotators, 11–13 transcripts each) | unseen annotators −0.01 to −0.02; untestable for reflection subtype (the proxy is near chance there) | keep annotator stratification and report per-annotator scores. The 7 ten-rater transcripts (majority labels) are test |
| Welivita | same opening post (one question, several answers) across splits | 316 thread pairs | small (0.475 by dialogue vs 0.469 by cluster) | already split by cluster |
| Welivita | source shift (CounselChat professionals vs Reddit peers) | — | in-source 0.47 vs cross-source 0.36–0.38 | stratify by source (done); report cross-source as a robustness check |
| CASAA | same client across transcripts | the five training sessions have five different clients | — | none needed (test only) |

## 7. Protocols for the other coding schemes

The MISC protocol is §4. Each other scheme gets its own protocol, chosen by the same criteria (C1–C7).

### 7.1 AnnoMI scheme (main behaviour, subtypes, client talk type)

- **Data:** `annomi.gold` only (133 transcripts).
- **Split:** `annomi_own`, by video series, stratified by series MI quality. Train 97, dev 15, test 21 transcripts.
- **Why:**
  - C3: series grouping removes the shared-client/story channel.
  - C1: the test holds the 7 ten-rater transcripts, the only multi-rater labels, so the cleanest labels are tested.
  - C5: 21 transcripts; report per-annotator scores, because single-annotator labels carry a strong annotator effect.
  - C7: dev is used for selection.
- **Not used for training:** HLQC overlaps (15/21/44/53 and the 48 pool twins) only matter when a model is trained on HLQC and tested on AnnoMI. For AnnoMI-scheme training they are ordinary AnnoMI transcripts.

### 7.2 Welivita scheme (15 MITI-derived listener codes)

- **Data:** `welivita.gold`; labels = the 7,152 stage-I-agreed listener sentences (crowd κ 0.34 overall, so only agreed labels are trusted).
- **Split:** `welivita_own`, by same-post cluster, stratified by source. Train 1,602, dev 203, test 195 dialogues.
- **Plus:** cross-source robustness (train CounselChat → test Reddit, and the reverse).
- **Why:** C1 agreed labels only; C3 cluster grouping; C2 source stratification, plus cross-source because the shift is large.

### 7.3 MITI 4 (`miti_scheme.json`)

- **Human MITI data:** CASAA only (20 transcripts, 18 clean). It is too small to train on and it is the reference standard, so it is **test only**.
- **Training options, in order of recommendation:**
  - **A.** A MISC-trained model whose output is mapped MISC → MITI (`quality.MISC_TO_MITI`). This needs no MITI training data. MITI's Seek can't be produced, because MISC folds permission-seeking into EC.
  - **B.** Welivita's train split (MITI-derived) → CASAA: cross-domain (written → spoken) with weak labels.
  - **C.** MI-TAGS (MITI 4.2, 242 sessions), if obtained, deduplicated against HLQC, AnnoMI and CASAA first.
- **Why:** C1 (CASAA is reference quality) and C4 (no trainable human MITI corpus exists).

### 7.4 Cross-scheme transfer tests (MISC model on another scheme)

These involve no training on the target scheme, so there is no split, only exclusions:
- **AnnoMI:** drop 15/21/44/53 (and the 48 pool twins if the model saw `pool.hlqc`); score counsellor main behaviour and questions at T1, and client C/S/N at T1.
- **Welivita:** the weak test, on the shared codes.
- **CASAA:** the 18 clean sessions.

## 8. Next steps (updated)

1. **Your decision:** P3 (`misc_pooled_cv`) as the primary MISC protocol, P0 kept as cold-start. P3 adds no session, person or text leakage beyond small template overlap. Its rules (frozen config, per-fold statistics, pooled scoring) also close the test-informed-decision channel that P0 already has.
2. If yes: a 7B confirmation run of `ft1mix_bare` under P3 (5 folds), then the arms we want to claim.
3. In any case: disclose L5 for the existing P0 results, and rerun the AnnoMI transfer test with the exclusions and T1 questions.
