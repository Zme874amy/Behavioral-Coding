# Datasets: catalogue, overlap map, cleaning, splits

The single reference for every **available** dataset in this project. Each dataset has one
canonical ID, used everywhere from now on. The live version, with every figure and
table, is [`notebooks/datasets_eda.ipynb`](../notebooks/datasets_eda.ipynb). The logic is in
`src/eda/` (`registry`, `overlap`, `clean`, `splits`, `stats`), and the split manifests are in
`data/splits/*.json`. Built 2026-10-04.

Gated or unobtainable corpora (MI-TAGS, BiMISC, Pérez-Rosas 2016, 7 Cups) and corpora
considered then excluded (KMI, IC-AnnoMI, MIDAS) are covered in
[experiments/2026-10-04-dataset-survey.md](experiments/2026-10-04-dataset-survey.md), not here.

## 1. Names: canonical ID → files

| Canonical ID | File(s) | What the file really is |
|---|---|---|
| `misc.miv63a.gold` | `data/manual/MIV6.3A_manual.csv` | **TEST set.** 10 MIBot v6.3A sessions with expert MISC consensus (`*_GT`); `*_auto` = AutoMISC GPT-4.1 |
| `misc.hlqc.gold` | `data/manual/HLQC_balanced_manual.csv` | **TRAIN set.** 10 HLQC sessions (5 high, 5 low) with MISC labels |
| `pool.hlqc` | `data/HLQC.csv` / `data/parsed/HLQC_parsed.csv` | All 257 HLQC sessions, volley-level / utterance-level, **unlabelled**. Contains the 10 train sessions |
| `pool.miv63a` | `data/MIV6.3A.csv` / `data/parsed/MIV6.3A_parsed.csv` / `data/2024-11-14-MIV6.3A-…merged.csv` | All 173 MIBot v6.3A sessions, **unlabelled**, plus outcome surveys (join on `Participant id` = `conv_id`). **Contains the 10 test sessions** |
| `pool.miv63b` | `data/MIV6.3B.csv` / `data/parsed/MIV6.3B_parsed.csv` / `data/2024-11-19-MIV6.1B_…merged.csv` | A different MIBot run, 165 sessions, unlabelled (the outcome file is named "6.1B") |
| `annomi.gold` | `data/external/annomi/AnnoMI-{full,simple}.csv` (official) / `data/AnnoMI.csv` / `data/manual/AnnoMI_eval.csv` / `data/parsed/AnnoMI_parsed.csv` | One corpus in five files. `data/AnnoMI.csv` is AutoMISC's MISC-mapped copy in a **different row order** (join on `conv_id` + `utterance_id`). `AnnoMI_eval.csv` is our cross-scheme test file built from it |
| `welivita.gold` | `data/external/welivita/MI_Dataset.csv` (official) / `data/external/welivita_mi_parsed.csv` / `data/manual/Welivita_eval.csv` | One corpus, three files. Labels are **Welivita & Pu's own** (100% match to the release), not ours |
| `miti.casaa.gold` | `data/external/casaa/{CASAA_eval,casaa_turns,casaa_globals}.csv` | Parsed by `baseline.prep_casaa` from the CASAA PDFs (gitignored) |
| `synth.v1` | `data/synth/aug/Qwen2_5-32B-Instruct-AWQ.csv` | **Every `synth/aug` file = the 1,925 HLQC train rows + synthetic rows** (`conv_id` starts `synth:`) |
| `synth.v2_proto` / `synth.v2_boundary` | `…_proto.csv` / `…_boundary.csv` | v2 ontology windows (prototype / confusable-pair focus) |
| `synth.v3_hlqcmix` | `…_proto_train.csv` | v3, HLQC topic mix. Despite "train" in the name, this is a dataset version, not a split |
| `synth.v3_mivmix` | `…_proto_eval.csv` | v3, MIV6.3A topic mix. **This is training data**: "eval" names the topic mix, not a split |

## 2. Catalogue

| ID | Domain | Modality / transcription | Sessions | Units (couns / client) | Scheme | Annotators / reliability | Role |
|---|---|---|---:|---|---|---|---|
| `misc.miv63a.gold` | smoking cessation | chatbot text (LLM counsellor, Prolific participants) | 10 | 821 (580 / 241) | MISC 2.5, two tiers, both speakers | 4 trained coders, consensus; Fleiss κ ≥ 0.6 after alignment | **test** |
| `misc.hlqc.gold` | mixed health (alcohol, diet, exercise, smoking) | spoken demo videos, **ASR** (20% of units have sentence punctuation) | 10 | 1,925 (1,040 / 885) | MISC 2.5, two tiers, both speakers | AutoMISC team; no κ reported; FI/FA inconsistent | **train** |
| `pool.hlqc` | mixed health | spoken, ASR | 257 (154 high / 103 low) | 30,610 utts | none (session high/low only) | — | unlabelled pool |
| `pool.miv63a` | smoking | chatbot text | 173 | 13,692 utts | none (+ readiness / importance / confidence pre, post, week-later) | — | unlabelled pool (minus test) |
| `pool.miv63b` | smoking | chatbot text | 165 | 11,771 utts | none | — | unlabelled pool |
| `annomi.gold` | mixed (23 alcohol, 19 smoking, 8 medication, …) | spoken demos, **manual** transcription | 133 (110 high / 23 low) | 9,699 turns (4,882 / 4,817) | AnnoMI: main behaviour + subtypes; client change/neutral/sustain | 10 experts: 126 transcripts single-annotated, 7 by all 10. **Measured** Fleiss κ: main 0.74, question type 0.74, reflection type 0.50, input type 0.51, client 0.47 | own-scheme train/dev/test; MISC transfer test |
| `welivita.gold` | mental-health peer support | written forum (1,000 CounselChat + 1,000 Reddit threads) | 2,000 | 20,462 (17,261 listener / 3,201 seeker) | MITI-derived, 15 listener codes | 2 MTurk workers + 2 expert judge stages. **Measured** ann1 vs ann2 κ 0.34 (raw 0.41) | own-scheme (agreed subset); weak transfer test |
| `miti.casaa.gold` | mixed (alcohol, smoking, diabetes, IPV, parenting) | spoken training demos, manual | 20 | 1,346 (687 / 659) | MITI 4 (counsellor only) + session globals (9) | MITI developers' reference coding | **test only** |
| `synth.*` | by topic mix | generated dialogue windows (Qwen2.5-32B) | 862 / 50 / 48 / 50 / 50 | 2,631 / 716 / 645 / 848 / 788 | MISC 2.5, generator-labelled, verifier-checked | — | train augmentation only |

Licences:

| Dataset(s) | Licence |
|---|---|
| AutoMISC sets (MIV, HLQC) | research use |
| AnnoMI | public, no explicit licence |
| Welivita | CC BY-NC-SA 3.0 |
| CASAA | training material; research use with citation, not redistributed |
| synthetic | ours |

## 3. Coding schemes (every code)

**MISC 2.5** (target scheme; `automisc_ft.data`):

| Speaker | T1 group | T2 codes |
|---|---|---|
| counsellor | CRL | CR (complex reflection), AF (affirm), SU (support), RF (reframe), EC (emphasise control) |
| counsellor | SRL | SR (simple reflection) |
| counsellor | IMC (MI-consistent) | ADP (advise with permission), RCP (raise concern with permission), GI (giving information) |
| counsellor | IMI (MI-inconsistent) | ADW (advise without permission), CO (confront), DI (direct), RCW (raise concern without permission), WA (warn) |
| counsellor | Q | OQ (open), CQ (closed) |
| counsellor | O | FA (facilitate), FI (filler), ST (structure) |
| client | C (change talk) | O+ D+ AB+ R+ N+ C+ AC+ TS+ (other, desire, ability, reasons, need, commitment, activation, taking steps) |
| client | S (sustain talk) | O− D− AB− R− N− C− AC− TS− |
| client | N | N (neutral) |

**MITI 4 (CASAA)** → MISC:

| MITI code | MISC mapping |
|---|---|
| SR | SRL/SR, exact |
| CR | CRL/CR, exact |
| AF | CRL/AF, exact |
| Emphasize | CRL/EC, exact |
| GI | IMC/GI, exact |
| Confront | IMI/CO, exact |
| Q | Q at T1 only (MITI 4 does not split open/closed) |
| NC (not coded) | O at T1 only |
| Persuade, PwP (persuade with permission), Seek (seeking collaboration) | none |
| SAME | CASAA convention for a turn that continues the previous coded utterance; no gold |

**Welivita (MITI-derived, 15 codes)** → MISC (`selftrain.ingest.MITI_TO_MISC_T2`):

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
| Self-Disclose, Other | none |

**AnnoMI** → MISC:

| AnnoMI code | MISC mapping |
|---|---|
| question: open / closed | OQ / CQ |
| reflection: simple / complex | SR / CR |
| therapist input: information | GI |
| therapist input: advice | ADP or ADW (permission not recorded) |
| therapist input: negotiation, options; other | none |
| client: change / sustain | C / S at T1 only |
| client: neutral | N |

**Code coverage gaps that matter for training:**

| Gap | Codes |
|---|---|
| In MIV test, absent from HLQC train | TS+, AC− |
| Absent from MIV test | ADW, RCW, WA, CO |
| Absent from every dataset | RCP |
| Thin in HLQC train | SU 9, RF 9, DI 12, CO 15, GI 16 |

## 4. Overlap map: who overlaps whom

Overlap was detected by 5-gram containment between conversations, ignoring shingles found in more than
5 conversations. Unrelated sessions score ≤ 0.02. Copies of one video (manual vs ASR
transcript) score 0.13–0.95. "→" means "is the same session as".

| Overlap | Sessions | Consequence |
|---|---|---|
| **HLQC train ⊂ HLQC pool** (same ids) | all 10 | expected (subset) |
| **MIV6.3A test ⊂ MIV6.3A pool** (same ids) | all 10 | pool users must drop them. `selftrain.label_pool` already does, plus templated-line near-dupes |
| **HLQC train → HLQC pool under other ids** | low_026→high_084 (0.95), low_001→low_027 (0.90) & low_040 (0.87), low_031→low_054 (0.82), low_080→low_019 (0.80) & low_098 (0.53) | the pool holds extra copies of training sessions. low_026/high_084 also has **conflicting** low/high quality labels |
| **HLQC train → AnnoMI** | low_001→53 (0.67), high_099→21 (0.66), low_033→44 (0.62), low_080→15 (0.60) | **AnnoMI transfer-test results so far include 4 training sessions.** Exclude 15, 21, 44, 53 |
| **HLQC pool → AnnoMI** | 48 AnnoMI transcripts match ≥ 1 of 63 pool sessions (68 pairs, up to 0.85) | exclude those 48 when the model also saw `pool.hlqc` |
| **HLQC train → CASAA** | high_121 → Emmy's First Encounter (0.40) | Emmy is excluded from the CASAA test |
| **HLQC pool → CASAA** | high_072 → The Rounder (0.28; HLQC holds only ~900 words) | Rounder is excluded from the CASAA test |
| **HLQC pool → HLQC pool** | 53 pairs (51 ≥ 0.5, e.g. low_003 = low_005, low_008 = low_048 at 1.00), 4 with conflicting high/low labels | 45 redundant sessions listed for dropping |
| **Welivita → Welivita** | 356 dialogue pairs ≥ 0.1 (the same CounselChat thread under two ids, e.g. 99 = 1856) | split by duplicate cluster |
| **synthetic → real** | v2/v3 windows copy HLQC-train exemplar phrases verbatim (≤ 0.20). Indirectly that touches AnnoMI 44 and CASAA Emmy, the HLQC-train copies | training-side only; the test exclusions above cover it |
| **synth.v3_hlqcmix ↔ synth.v3_mivmix** | windows 19 and 33 near-identical (0.91) | the two v3 arms share 2 windows |
| **synth.v1 → synth.v1** | 15 pairs (mode collapse), 3 redundant | listed for dropping |
| **synth.v3_mivmix ← MIV6.3A test** | topic *proportions* measured on the test set (no text) | documented ablation; report as such |
| MIV6.3A test ↔ anything else | none at session level; 22 of 652 long test utterances recur verbatim in `pool.miv63b` (templated chatbot lines) | handled by `label_pool._drop_eval_like` |
| AnnoMI ↔ AnnoMI, CASAA ↔ AnnoMI, MIV ↔ HLQC/AnnoMI | none | — |

## 5. Cleaning and processing audit (`eda.clean`)

No data file was modified. Leakage and duplicate handling is done with exclusion lists in
`data/splits/`, so every past result reproduces from the originals.

| Issue | Where | Size | Status |
|---|---|---|---|
| T1 contradicts T2 group | `misc.hlqc.gold` (**train**) | 6 rows: high_117#142 Q/ST, high_121#120 CRL/ST, high_127#7 O/OQ, low_031#156 O/OQ, low_031#179–180 O/CQ | **needs decision.** Proposed: set T1 = group of T2 in a derived copy (all six look like correct T2s) |
| Identical short utterances with conflicting T2 | HLQC train 9/27 ("okay", "yeah", "mmhmm": FI vs FA), MIV test 1/7, AnnoMI 4/5, Welivita 19/38 | — | annotation noise. Report, do not relabel |
| Duplicate sessions with conflicting high/low quality label | `pool.hlqc` | 4 pairs (high_025 = low_086, high_084 = low_026, high_036 = low_051, …) | **needs decision** if session quality is ever used as a label |
| Near-duplicate sessions | `pool.hlqc` | 51 pairs; 45 redundant ids | exclusion list (`pools.json`) |
| Duplicate dialogues | `welivita.gold` | 9 at ≥ 0.9; 356 pairs at ≥ 0.1 | clusters kept in one split |
| Redundant synthetic windows | `synth.v1` | 3 | exclusion list (`synthetic.json`) |
| Empty text | `pool.miv63a` | 2 rows | harmless; consumers already skip empty text |
| Topic labels with stray whitespace | `annomi.gold` | release-level | stripped in `eda.registry` |
| No casing / punctuation (ASR) | HLQC train 58% / 80%, HLQC pool 58% / 90% | — | expected. It's a domain gap to MIV (15% uncased) and AnnoMI (4% uncased), not a cleaning target |
| Multi-code turns, SAME rows, 2 client rows with a code | `miti.casaa.gold` | 50 / 8 / 2 | handled: no gold assigned |
| Uncoded rows | Welivita 450 listener ("-") + all seekers; CASAA clients; synthetic context rows (~50%) | — | expected |
| Row count 1,925 vs paper 1,924 | `misc.hlqc.gold` | 1 | off-by-one; no duplicate rows found |
| Missing turn 25 | 3 CASAA PDFs | — | source gap, not a parse loss |

## 6. EDA highlights (details in the notebook)

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
- **Synthetic diversity:** v1 has Self-BLEU 0.263 and 31% near-duplicates, against 0.037–0.066 and 6–9% for v2/v3.

## 7. Split plan (`python -m eda.splits` → `data/splits/*.json`)

| Manifest | Scheme | Train | Dev | Test | Exclusions |
|---|---|---|---|---|---|
| `misc_main` (**unchanged**) | MISC 2.5 | `misc.hlqc.gold` (10) | 5-fold HLQC CV (`baseline.oof_t1._split`, seed 42) | `misc.miv63a.gold` (10); `miti.casaa.gold` on 6 exact T2 codes + T1 (18 sessions) | CASAA Emmy, Rounder. AnnoMI 15/21/44/53 for any HLQC-trained model; 48 AnnoMI transcripts if the model saw `pool.hlqc` |
| `annomi_own` | AnnoMI | 91 transcripts | 22 | 20 (incl. the 7 ten-rater transcripts, majority labels) | transcript-level, stratified by quality × annotator; MISC-transfer exclusions as above |
| `welivita_own` | Welivita MITI-derived | 1,604 dialogues | 196 | 200 | dialogue-level, duplicate clusters in one split, stratified by source. Labels: the 7,152 stage-I-agreed listener rows |
| `casaa_test` | MITI 4 | — | — | 18 sessions | Emmy, Rounder |
| `pools` | — | `pool.hlqc`: drop the 45 redundant duplicates; `pool.miv63a`: drop the 10 test sessions; `pool.miv63b`: as is | — | — | `exclude_if_annomi_is_test` (63 HLQC pool ids), `exclude_if_casaa_is_test` (2) |
| `synthetic` | — | train only | — | never | 3 redundant v1 windows; 2 windows shared by the v3 mixes |

The manifests are byte-identical across reruns and `PYTHONHASHSEED` values. Built-in checks:
- no conversation sits in two splits;
- no Welivita duplicate cluster spans splits;
- CASAA Emmy is not in a test split;
- the MIV pool exclusion covers every test session.

## 8. What this changes for results already reported

- **AnnoMI transfer numbers** (e.g. T2 accuracy 0.504 for the 2-adapter model) include 4 transcripts that are HLQC training sessions. Rerun excluding 15/21/44/53 before quoting them again.
- The MIV6.3A test set and all main-setting results are unaffected: no test session appears in any training source.
- The CASAA results to come should use `casaa_test.json` (18 sessions).

## Reproduce

```bash
PYTHONPATH=src python -m baseline.prep_casaa          # once: parse CASAA PDFs
PYTHONPATH=src python -m eda.splits                   # manifests + leak checks
jupyter nbconvert --to notebook --execute --inplace notebooks/datasets_eda.ipynb
```
