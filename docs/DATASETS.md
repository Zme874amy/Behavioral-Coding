# Datasets: catalogue, overlap map, cleaning, splits

The single reference for every **available** dataset in this project. Each dataset has one
canonical ID, used everywhere from now on. The live version, with every figure and
table, is [`notebooks/datasets_eda.ipynb`](../notebooks/datasets_eda.ipynb). The logic is in
`src/eda/` (`registry`, `overlap`, `clean`, `splits`, `stats`), and the split manifests are in
`data/splits/*.json`. Built 2026-10-04.

Gated or unobtainable corpora (MI-TAGS, BiMISC, Pérez-Rosas 2016, 7 Cups) and corpora
considered then excluded (KMI, IC-AnnoMI, MIDAS) are covered in
[experiments/2026-10-04-dataset-survey.md](experiments/2026-10-04-dataset-survey.md), not here.
**Our own generated (synthetic) sets are documented separately in [SYNTHETIC_DATA.md](SYNTHETIC_DATA.md)**,
because they change with every synthesis campaign. Nothing on this page depends on them: overlap scores
and split manifests here are computed from real data only.

## 1. Names: which file to use

Each corpus lives in several files. Pick the file whose *use it for* matches the task. *Derived from* gives the lineage, so the same data is never counted twice.

| Dataset ID | File | Level | Labels | Use it for | Derived from |
|---|---|---|---|---|---|
| `misc.miv63a.gold` | `data/manual/MIV6.3A_manual.csv` | utterance | MISC T1/T2 human gold (*_GT) + GPT-4.1 (*_auto) | TEST set of every MISC model | pool.miv63a (10 sessions) |
| `misc.hlqc.gold` | `data/manual/HLQC_balanced_manual.csv` | utterance | MISC T1/T2 human gold | TRAIN set; few-shot exemplars | pool.hlqc (10 sessions) |
| `pool.hlqc` | `data/HLQC.csv` | turn (volley) | none (session high/low in id) | raw turns | Pérez-Rosas et al. 2019 ASR transcripts |
| `pool.hlqc` | `data/parsed/HLQC_parsed.csv` | utterance | none | unlabelled pool (retrieval, self-training) | data/HLQC.csv, split by the AutoMISC parser |
| `pool.miv63a` | `data/MIV6.3A.csv` | turn (volley) | none | raw turns | MIBot v6.3A study logs |
| `pool.miv63a` | `data/parsed/MIV6.3A_parsed.csv` | utterance | none | unlabelled pool (must drop the 10 test sessions) | data/MIV6.3A.csv, parser |
| `pool.miv63a` | `data/2024-11-14-MIV6.3A-...merged.csv` | participant | readiness / importance / confidence, demographics | session outcomes (join 'Participant id' = conv_id) | MIBot v6.3A surveys |
| `pool.miv63b` | `data/MIV6.3B.csv` | turn (volley) | none | raw turns | MIBot v6.3B study logs |
| `pool.miv63b` | `data/parsed/MIV6.3B_parsed.csv` | utterance | none | unlabelled pool (self-training) | data/MIV6.3B.csv, parser |
| `pool.miv63b` | `data/2024-11-19-MIV6.1B_...merged.csv` | participant | survey outcomes | session outcomes (file name says 6.1B) | MIBot surveys |
| `annomi.gold` | `data/external/annomi/AnnoMI-full.csv` | turn x annotator | AnnoMI attributes per annotator | agreement analysis (7 ten-rater transcripts) | official release (Wu et al.) |
| `annomi.gold` | `data/external/annomi/AnnoMI-simple.csv` | turn | AnnoMI main behaviour / talk type | own-scheme train/dev/test | official release |
| `annomi.gold` | `data/AnnoMI.csv` | turn | MISC-mapped AnnoMI labels (AutoMISC) | source of AnnoMI_eval (different row order) | AnnoMI-full, mapped by AutoMISC |
| `annomi.gold` | `data/manual/AnnoMI_eval.csv` | turn | MISC gold on shared codes only | cross-scheme TEST of MISC models | data/AnnoMI.csv (baseline.prep_crossscheme) |
| `annomi.gold` | `data/parsed/AnnoMI_parsed.csv` | utterance | none | utterance-split text | AnnoMI, parser |
| `welivita.gold` | `data/external/welivita/MI_Dataset.csv` | sentence | ann1, ann2, judge stages, final label | agreement analysis; own-scheme split | official release (Welivita & Pu) |
| `welivita.gold` | `data/external/welivita_mi_parsed.csv` | sentence | final label + MISC map (weak_t2) | self-training pool | MI_Dataset.csv (selftrain.ingest) |
| `welivita.gold` | `data/manual/Welivita_eval.csv` | sentence | MISC gold on shared codes | cross-scheme TEST of MISC models | welivita_mi_parsed.csv (baseline.prep_crossscheme) |
| `miti.casaa.gold` | `data/external/casaa/CASAA_eval.csv` | utterance | MITI codes + MISC map where exact | MITI / MISC TEST (18 clean sessions) | CASAA PDFs (baseline.prep_casaa) |
| `miti.casaa.gold` | `data/external/casaa/casaa_turns.csv` | turn | raw code cell + coder notes | audit of the parse | CASAA PDFs |
| `miti.casaa.gold` | `data/external/casaa/casaa_globals.csv` | session | MITI global ratings | session-level analysis (9 sessions) | CASAA PDFs |

Traps:
- `data/HLQC.csv` and `data/parsed/HLQC_parsed.csv` are the unlabelled pool. The train set is `data/manual/HLQC_balanced_manual.csv`.
- `data/AnnoMI.csv` is in a different row order from the official release.
- Welivita's labels are the original authors' in all three files.

## 2. Catalogue

| ID | Domain | Modality / transcription | Sessions | Units (couns / client) | Scheme | Annotators / reliability | Role |
|---|---|---|---:|---|---|---|---|
| `misc.miv63a.gold` | smoking cessation | chatbot text (LLM counsellor, Prolific participants) | 10 | 821 (580 / 241) | MISC 2.5 + AutoMISC AC± and T1 grouping (§3), both speakers | 4 trained coders, consensus; Fleiss κ ≥ 0.6 after alignment | **test** |
| `misc.hlqc.gold` | mixed health (alcohol, diet, exercise, smoking) | spoken demo videos, **ASR** (20% of units have sentence punctuation) | 10 | 1,925 (1,040 / 885) | MISC 2.5 + AutoMISC AC± and T1 grouping (§3), both speakers | AutoMISC team; no κ reported; FI/FA inconsistent | **train** |
| `pool.hlqc` | mixed health | spoken, ASR | 257 (154 high / 103 low) | 30,610 utts | none (session high/low only) | — | unlabelled pool |
| `pool.miv63a` | smoking | chatbot text | 173 | 13,692 utts | none (+ readiness / importance / confidence pre, post, week-later) | — | unlabelled pool (minus test) |
| `pool.miv63b` | smoking | chatbot text | 165 | 11,771 utts | none | — | unlabelled pool |
| `annomi.gold` | mixed (23 alcohol, 19 smoking, 8 medication, …) | spoken demos, **manual** transcription | 133 (110 high / 23 low) | 9,699 turns (4,882 / 4,817) | AnnoMI: main behaviour + subtypes; client change/neutral/sustain | 10 experts: 126 transcripts single-annotated, 7 by all 10. **Measured** Fleiss κ: main 0.74, question type 0.74, reflection type 0.50, input type 0.51, client 0.47 | own-scheme train/dev/test; MISC transfer test |
| `welivita.gold` | mental-health peer support | written forum (1,000 CounselChat + 1,000 Reddit threads) | 2,000 | 20,462 (17,261 listener / 3,201 seeker) | MITI-derived, 15 listener codes | 2 MTurk workers + 2 expert judge stages. **Measured** ann1 vs ann2 κ 0.34 (raw 0.41) | own-scheme (agreed subset); weak transfer test |
| `miti.casaa.gold` | mixed (alcohol, smoking, diabetes, IPV, parenting) | spoken training demos, manual | 20 | 1,346 (687 / 659) | MITI 4 (counsellor only) + session globals (9) | MITI developers' reference coding | **test only** |

Licences:

| Dataset(s) | Licence |
|---|---|
| AutoMISC sets (MIV, HLQC) | research use |
| AnnoMI | public, no explicit licence |
| Welivita | CC BY-NC-SA 3.0 |
| CASAA | training material; research use with citation, not redistributed |

## 3. Coding schemes (every handbook code)

Code lists come from the **handbooks**, not from the data, so a real class that no dataset contains is still listed:
- **MISC 2.5:** [Houck et al. 2010](https://casaa.unm.edu/assets/docs/misc25.pdf) (counsellor categories p.16, No Code p.14, client categories pp.38–41, summary groups pp.47–48).
- **MITI 4.2.1:** [Moyers, Manuel & Ernst 2015](https://casaa.unm.edu/assets/docs/miti4_21.pdf).
- **Welivita:** Welivita & Pu 2022, Table 1.
- **AnnoMI:** Wu et al. 2023, §4.

Encoded in `eda.registry.HANDBOOK`, and compared with our vocabulary and the data by `eda.registry.handbook_check()` (notebook §3). Checked 2026-10-05.

### MISC 2.5 (target scheme)

**Counsellor:** 17 categories in the handbook. Advise and Raise Concern split by permission, giving 19 codes, **all 19 in our vocabulary**.

| Code | Name | Manual summary group | Our T1 (AutoMISC grouping) | Real gold examples |
|---|---|---|---|---|
| ADP | Advise with permission | MICO | IMC | HLQC, MIV |
| ADW | Advise without permission | MIIN | IMI | HLQC only |
| AF | Affirm | MICO | CRL | HLQC, MIV |
| CO | Confront | MIIN | IMI | HLQC only |
| DI | Direct | MIIN | IMI | HLQC, MIV |
| EC | Emphasize control | MICO | CRL | HLQC, MIV |
| FA | Facilitate | — | O | HLQC, MIV |
| FI | Filler | — | O | HLQC, MIV |
| GI | Giving information | — | IMC | HLQC, MIV |
| OQ | Open question | MICO | Q | HLQC, MIV |
| CQ | Closed question | — | Q | HLQC, MIV |
| RCP | Raise concern with permission | MICO | IMC | **none** |
| RCW | Raise concern without permission | MIIN | IMI | HLQC only |
| SR | Simple reflection | MICO | SRL | HLQC, MIV |
| CR | Complex reflection | MICO | CRL | HLQC, MIV |
| RF | Reframe | — | CRL | HLQC, MIV |
| SU | Support | MICO | CRL | HLQC, MIV |
| ST | Structure | — | O | HLQC, MIV |
| WA | Warn | MIIN | IMI | HLQC only |

**Client:** Follow/Neutral/Ask, plus 7 change-language types × valence.

| Handbook code | Name | Ours | Real gold examples (+ / −) |
|---|---|---|---|
| FN | Follow / Neutral / Ask | `N` | HLQC, MIV |
| C± | Commitment | `C±` | + HLQC, MIV / − HLQC only |
| R± | Reason | `R±` | both, both |
| D± | Desire | `D±` | both, both |
| A± | Ability | `AB±` | both, both |
| N± | Need | `N±` | + HLQC only / − both |
| TS± | Taking steps | `TS±` | + MIV only / **− none** |
| O± | Other | `O±` | + both / − HLQC only |

**Where we depart from the handbook:**

| Item | Handbook | Ours | Consequence |
|---|---|---|---|
| **AC± (Activation)** | **not in MISC 2.5** | AutoMISC addition; thesis footnote 1 on "17 client codes": "Although not listed in the MISC 2.5, we include Activation+/- … based on definitions in [49]" = Miller & Rollnick, *Motivational Interviewing*, 4th ed. 2023. Not in MISC 2.1 either | Present in our gold (HLQC AC+ 23; MIV AC+ 11, AC− 1). The handbook codes "offering alternatives" as Commitment (C+). When comparing with other MISC data, map AC± to C± |
| Counsellor T1 groups (CRL, SRL, IMC, IMI, Q, O) | not in the handbook; its groups are **MICO / MIIN** (+ neutral) | AutoMISC's grouping by semantic similarity (thesis §3) | IMC contains GI and CRL contains RF, which are neither MICO nor MIIN in the manual. Don't call T1 "the MISC hierarchy" |
| No Code (NC) | exists (uncodable utterances) | not in our vocabulary | Fine for our data. A model can't abstain |
| Reflection valence (SR/CR +, −, 0, ±) | required | dropped | We code SR/CR without valence |
| 7 global ratings | Acceptance, Empathy, Direction, Autonomy Support, Collaboration, Evocation; client Self-Exploration | not used | Not in any of our datasets |

### MITI 4.2.1 (CASAA): 10 behaviour codes + 4 globals, all present in CASAA

| MITI code | MISC mapping |
|---|---|
| SR | SRL/SR, exact |
| CR | CRL/CR, exact |
| AF | CRL/AF, exact |
| Emphasize (autonomy) | CRL/EC, exact |
| GI | IMC/GI, exact |
| Confront | IMI/CO, exact |
| Q | Q at T1 only (MITI 4 does not split open/closed) |
| Persuade, Persuade with Permission, Seek (collaboration) | none |
| *NC, SAME* | **not MITI codes**: CASAA transcript conventions (uncoded; continuation). NC → O at T1 |
| Globals | Cultivating Change Talk, Softening Sustain Talk, Partnership, Empathy (9 CASAA sessions) |

### Welivita (Table 1 of the paper): 15 labels, all present in the data

| Welivita code | MISC |
|---|---|
| Give Information | GI |
| Advise without / with Permission | ADW / ADP |
| Complex / Simple Reflection | CR / SR |
| Support | SU |
| Affirm | AF |
| Closed / Open Question | CQ / OQ |
| Direct | DI |
| Confront | CO |
| Emphasize Autonomy | EC |
| Warn | WA |
| Self-Disclose, Other | none (authors' additions, not MITI) |

"Give Information" includes giving an opinion, which is broader than MISC GI. Our ingest map (`selftrain.ingest.MITI_TO_MISC_T2`) also lists Reframe and Structure. Neither exists in Welivita's scheme, so they're dead entries; harmless, because no row carries them.

### AnnoMI (§4 of the paper): all attributes present in the data

Question / Input / Reflection are **separate attributes that can co-occur**, and one Main Behaviour is picked per utterance.

| AnnoMI attribute | MISC mapping |
|---|---|
| question: open / closed | OQ / CQ |
| reflection: simple / complex | SR / CR |
| input: information | GI |
| input: advice | ADP or ADW (permission not recorded) |
| input: negotiation/goal-setting, options; main: other | none |
| client: change / sustain | C / S at T1 only |
| client: neutral | N (= FN) |

**Coverage gaps that matter for training (real gold only):**

| Gap | Codes |
|---|---|
| Handbook classes with **no real example anywhere** | RCP, TS− |
| In MIV test, absent from HLQC train | TS+, AC− |
| In HLQC train, absent from MIV test | ADW, CO, RCW, WA, C−, N+, O− |
| Thin in HLQC train | SU 9, RF 9, DI 12, CO 15, GI 16 |

## 4. Annotation quality: what can hurt training

Notebook §4; code in `src/eda/quality.py`. This mainly compares the **train** set (`misc.hlqc.gold`) with the **test** set
(`misc.miv63a.gold`). Evidence comes from five sources:
- rule checks quoted from the MISC 2.5 manual;
- the same text labelled differently;
- **independent coders on the same sessions**: AnnoMI's experts on 4 shared sessions, and the CASAA MITI reference on Emmy = `high_121`;
- train-vs-test label shift;
- the main model's (`ft1mix_bare`, 3 seeds) errors on the test set.

### 4.1 Findings

| # | Issue | Evidence | Effect on the MIV6.3A test |
|---|---|---|---|
| 1 | **HLQC uses FI as a catch-all** | 64% (77/121) of standalone "okay"/"yeah"/"mm-hmm" are coded FI, but the manual (p.22) says FA. FI is 19% of counsellor utterances, against the manual's 5% ceiling. Only 7% of HLQC FI are pleasantries (39% acknowledgements, 20% fragments), vs 68% pleasantries in the test. "okay" is FI 24 / FA 10, "yeah" FI 14 / FA 15 | small: test pleasantries are still predicted FI 95% of the time. But HLQC-based scores (CV, teacher gate) penalise codebook-correct models |
| 2 | **HLQC under-codes client change talk** | against AnnoMI experts on the same 185 client turns, κ 0.36 (AnnoMI's own inter-rater κ is 0.47). In 28 turns HLQC says neutral where AnnoMI says change. HLQC client labels are 85% neutral vs 52% in the test | **large**: 32% of gold change/sustain talk is predicted neutral. O+ recall 0.03, R+ 0.54, R− 0.51 |
| 3 | **Annotator groups draw the simple/complex line differently** | CASAA MITI reference: 9 of 16 HLQC SR turns are CR. MIV coders call long chatbot paraphrases (median 18 words) SR. Inside HLQC, SR and CR are the same length (11 vs 12 words) | **largest single error**: test SR recall 0.42, and 51% of test SRs are predicted CR |
| 4 | **The test set's MI-consistent codes are nearly absent from training** | SU 9 rows (0.9% vs 6.7% in test), EC 22 (2.1% vs 6.7%), AF 2.2% vs 5.2%. Permission-seeking, which the manual codes EC (p.16), appears once in HLQC, coded FI | SU recall 0.39 (→ CR, AF), EC 0.33 (→ SU, CQ, FI) |
| 5 | **Train-only codes** | ADW/RCW/WA/CO are 95 rows (9% of training counsellor labels), 96–100% from low-quality sessions, and never occur in the test. Client C−/N+/O−: 13 rows | ~7 spurious predictions per run (RCW/WA for GI) |
| 6 | **ASR transcripts and segmentation** | WER of HLQC against manual transcripts: 9–19% (AnnoMI), 25% (CASAA). 78% of HLQC gold questions have no "?" (test: 2%). 36% of utterances start with and/but/or/so/because (test: 0.5%), and 19% are ≤ 3 words | indirect: fragments get FI; questions must be detected from words alone |
| 7 | **Training codes concentrated in one session** | GI 75% from `high_121`; DI 75% from `low_001`; ADP 51% from `low_031` | per-class CV unstable; the model learns one session's style |
| 8 | **AnnoMI "open" ≠ MISC OQ** | of 114 shared questions, AnnoMI calls 68 "open" that HLQC codes CQ, following the MISC rule ("could you…", specific-fact questions = CQ). 44 of those come from AnnoMI annotator 3 | affects the AnnoMI transfer test: its OQ/CQ gold is not MISC-consistent |

Other checks HLQC passes:
- **CQ rules:** questions starting "do/could/is…" are coded CQ 40/41 times, and scale questions CQ 6/6.
- **Counsellor main behaviour vs AnnoMI:** 72% agreement per turn (κ 0.58), in the expected range between expert groups.
- **Reflection type vs AnnoMI:** simple/complex agrees in 17/21 cases.

### 4.2 Possible actions (none applied; they change training data, so they need your decision)

| Issue | Option |
|---|---|
| 1 | Relabel standalone acknowledgements FI → FA in a derived train copy. Treat HLQC-CV numbers as convention-dependent |
| 2 | Re-annotate HLQC client turns, or reweight non-neutral client labels |
| 3 | Agree a written SR/CR guideline and adjudicate. Also report reflections at T1 (SRL vs CRL) |
| 4 | Targeted SU / EC / permission-seeking examples (human or synthetic) |
| 5 | Downweight, or train ADW/RCW/WA/CO at T1 only |
| 6 | Re-segment HLQC, or use corrected transcripts (MI-TAGS, if obtained) |
| 8 | Score AnnoMI questions at T1 (Q) only |

## 5. Overlap map: who overlaps whom

Overlap was detected by 5-gram containment between conversations, ignoring shingles found in more than
5 conversations. Unrelated sessions score ≤ 0.02. Copies of one video (manual vs ASR
transcript) score 0.13–0.95. "→" means "is the same session as".

| Overlap | Sessions | Consequence |
|---|---|---|
| **HLQC train ⊂ HLQC pool** (same ids) | all 10 | expected (subset) |
| **MIV6.3A test ⊂ MIV6.3A pool** (same ids) | all 10 | pool users must drop them. `selftrain.label_pool` already does, plus templated-line near-dupes |
| **HLQC train → HLQC pool under other ids** | low_026→high_084 (0.95), low_001→low_027 (0.93) & low_040 (0.90), low_080→low_019 (0.87) & low_098 (0.59), low_031→low_054 (0.82) | the pool holds extra copies of training sessions. low_026/high_084 also has **conflicting** low/high quality labels |
| **HLQC train → AnnoMI** | low_001→53 (0.69), low_080→15 (0.67), high_099→21 (0.66), low_033→44 (0.62) | **AnnoMI transfer-test results so far include 4 training sessions.** Exclude 15, 21, 44, 53 |
| **HLQC pool → AnnoMI** | 48 AnnoMI transcripts match ≥ 1 of 63 pool sessions (68 pairs, up to 0.85) | exclude those 48 when the model also saw `pool.hlqc` |
| **HLQC train → CASAA** | high_121 → Emmy's First Encounter (0.40) | Emmy is excluded from the CASAA test |
| **HLQC pool → CASAA** | high_072 → The Rounder (0.28; HLQC holds only ~900 words) | Rounder is excluded from the CASAA test |
| **HLQC pool → HLQC pool** (re-uploads of the same video) | 53 pairs (51 ≥ 0.5, e.g. low_003 = low_005, low_008 = low_048 at 1.00), 4 with conflicting high/low labels | 45 redundant sessions listed for dropping |
| **Welivita → Welivita** | 356 dialogue pairs ≥ 0.1. **316 share the opening post but have different replies** (one CounselChat question answered by several therapists); **30 are true duplicates** (e.g. 99 = 1856); 10 overlap partially | labels are on the replies, so all are kept; each cluster stays in one split |
| MIV6.3A test ↔ anything else | none at session level; 22 of 652 long test utterances recur verbatim in `pool.miv63b` (templated chatbot lines) | handled by `label_pool._drop_eval_like` |
| AnnoMI ↔ AnnoMI, CASAA ↔ CASAA, MIV ↔ MIV, CASAA ↔ AnnoMI, MIV ↔ HLQC/AnnoMI | none (curated distinct videos; one participant per MIV session) | — |

## 6. Cleaning and processing audit (`eda.clean`)

No data file was modified. Leakage and duplicate handling is done with exclusion lists in
`data/splits/`, so every past result reproduces from the originals.

| Issue | Where | Size | Status |
|---|---|---|---|
| T1 contradicts T2 group | `misc.hlqc.gold` (**train**) | 6 rows: high_117#142 Q/ST, high_121#120 CRL/ST, high_127#7 O/OQ, low_031#156 O/OQ, low_031#179–180 O/CQ | **needs decision.** Proposed: set T1 = group of T2 in a derived copy (all six look like correct T2s) |
| Identical short utterances with conflicting T2 | HLQC train 9/27 ("okay", "yeah", "mmhmm": FI vs FA), MIV test 1/7, AnnoMI 4/5, Welivita 19/38 | — | annotation noise: see §4 (finding 1). Report, do not relabel |
| Duplicate sessions with conflicting high/low quality label | `pool.hlqc` | 4 pairs (high_025 = low_086, high_084 = low_026, high_036 = low_051, …) | **needs decision** if session quality is ever used as a label |
| Near-duplicate sessions | `pool.hlqc` | 51 pairs; 45 redundant ids | exclusion list (`pools.json`) |
| Duplicate dialogues | `welivita.gold` | 9 at ≥ 0.9; 356 pairs at ≥ 0.1 | clusters kept in one split |
| Empty text | `pool.miv63a` | 2 rows | harmless; consumers already skip empty text |
| Topic labels with stray whitespace | `annomi.gold` | release-level | stripped in `eda.registry` |
| No casing / punctuation (ASR) | HLQC train 58% / 80%, HLQC pool 58% / 90% | — | expected. It's a domain gap to MIV (15% uncased) and AnnoMI (4% uncased), not a cleaning target |
| Multi-code turns, SAME rows, 2 client rows with a code | `miti.casaa.gold` | 50 / 8 / 2 | handled: no gold assigned |
| Uncoded rows | Welivita 450 listener ("-") + all seekers; CASAA clients | — | expected |
| Row count 1,925 vs paper 1,924 | `misc.hlqc.gold` | 1 | off-by-one; no duplicate rows found |
| Missing turn 25 | 3 CASAA PDFs | — | source gap, not a parse loss |

## 7. EDA highlights (details in the notebook)

- **Register gap:** HLQC train is ASR speech (14% of units contain fillers, 20% have sentence punctuation). The MIV test is typed chatbot text (0.4% fillers, 83% punctuation). AnnoMI and CASAA are manual speech transcripts.
- **Unit size:**

  | Dataset | Units / session (median) | Words / unit (median) |
  |---|---:|---:|
  | HLQC gold | 228 | 8 |
  | MIV gold | 78.5 | 12 |
  | AnnoMI | 47 | 9 (turns, p95 57) |
  | CASAA | 76 | 17 (turns) |
  | Welivita | 9 | 16 |

- **Client talk is mostly neutral** everywhere: 85% in HLQC train, 52% in MIV test, 64% in AnnoMI. Change/sustain sub-codes are all tail codes in both MISC sets.
- **AnnoMI annotator effect:** on identical utterances, the complex-reflection share ranges from 0.19 to 0.82 across annotators, and the change-talk share from 0.14 to 0.38. Annotator 3 is the outlier (κ 0.65 vs ≥ 0.79 against the leave-one-out majority). Each annotator single-handedly coded 11–13 transcripts.
- **Welivita:** only 7,152 of 17,261 listener segments have crowd agreement. Agreement is lowest for reflections (CR 261/1,294 agreed) and advice with permission (57/484).
- **MIV outcomes:** confidence rises pre → post (4.27 → 5.62). The 10 test sessions gained more confidence than the rest (+1.90 vs +1.31), so the test set leans toward successful sessions.

## 8. Split plan (`python -m eda.splits` → `data/splits/*.json`)

| Manifest | Scheme | Train | Dev | Test | Exclusions |
|---|---|---|---|---|---|
| `misc_main` (**unchanged**) | MISC 2.5 | `misc.hlqc.gold` (10) | 5-fold HLQC CV (`baseline.oof_t1._split`, seed 42) | `misc.miv63a.gold` (10); `miti.casaa.gold` on 6 exact T2 codes + T1 (18 sessions) | CASAA Emmy, Rounder. AnnoMI 15/21/44/53 for any HLQC-trained model; 48 AnnoMI transcripts if the model saw `pool.hlqc` |
| `annomi_own` | AnnoMI | 91 transcripts | 22 | 20 (incl. the 7 ten-rater transcripts, majority labels) | transcript-level, stratified by quality × annotator; MISC-transfer exclusions as above |
| `welivita_own` | Welivita MITI-derived | 1,604 dialogues | 196 | 200 | dialogue-level, duplicate clusters in one split, stratified by source. Labels: the 7,152 stage-I-agreed listener rows |
| `casaa_test` | MITI 4 | — | — | 18 sessions | Emmy, Rounder |
| `pools` | — | `pool.hlqc`: drop the 45 redundant duplicates; `pool.miv63a`: drop the 10 test sessions; `pool.miv63b`: as is | — | — | `exclude_if_annomi_is_test` (63 HLQC pool ids), `exclude_if_casaa_is_test` (2) |

The manifests are byte-identical across reruns and `PYTHONHASHSEED` values. Built-in checks:
- no conversation sits in two splits;
- no Welivita duplicate cluster spans splits;
- CASAA Emmy is not in a test split;
- the MIV pool exclusion covers every test session.

## 9. What this changes for results already reported

- **AnnoMI transfer numbers** (e.g. T2 accuracy 0.504 for the 2-adapter model) include 4 transcripts that are HLQC training sessions. Rerun excluding 15/21/44/53 before quoting them again.
- **AnnoMI OQ/CQ transfer scores** compare against an "open" that is broader than MISC OQ (§4, finding 8). Report AnnoMI questions at T1.
- **HLQC-based scores** (HLQC CV, the teacher-screen gate) depend on HLQC's FI convention (§4, finding 1). This confirms the 2026-10-04 teacher audit.
- The MIV6.3A test set and all main-setting results are free of leakage: no test session appears in any training source. But the main errors trace to train/test convention and prior differences (§4, findings 2–4).
- The CASAA results to come should use `casaa_test.json` (18 sessions).

## Reproduce

```bash
PYTHONPATH=src python -m baseline.prep_casaa          # once: parse CASAA PDFs
PYTHONPATH=src python -m eda.splits                   # manifests + leak checks
jupyter nbconvert --to notebook --execute --inplace notebooks/datasets_eda.ipynb
```
