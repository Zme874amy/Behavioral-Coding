# Datasets: catalogue, overlap map, cleaning, splits

The single reference for every **available** dataset in this project. Each dataset has one
canonical ID, used everywhere from now on. Analysis tables cover one dataset each (§4); cross-dataset results are only in §5. The live version, with every figure and
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

One table per dataset. Pick the file whose *use it for* matches the task. *Derived from* gives the lineage, so the same data is never counted twice.

**`misc.miv63a.gold`**

| File | Level | Labels | Use it for | Derived from |
|---|---|---|---|---|
| `data/manual/MIV6.3A_manual.csv` | utterance | MISC T1/T2 human gold (*_GT) + GPT-4.1 (*_auto) | TEST set of every MISC model | pool.miv63a (10 sessions) |

**`misc.hlqc.gold`**

| File | Level | Labels | Use it for | Derived from |
|---|---|---|---|---|
| `data/manual/HLQC_balanced_manual.csv` | utterance | MISC T1/T2 human gold | TRAIN set; few-shot exemplars | pool.hlqc (10 sessions) |

**`pool.hlqc`**

| File | Level | Labels | Use it for | Derived from |
|---|---|---|---|---|
| `data/HLQC.csv` | turn (volley) | none (session high/low in id) | raw turns | Pérez-Rosas et al. 2019 ASR transcripts |
| `data/parsed/HLQC_parsed.csv` | utterance | none | unlabelled pool (retrieval, self-training) | data/HLQC.csv, split by the AutoMISC parser |

**`pool.miv63a`**

| File | Level | Labels | Use it for | Derived from |
|---|---|---|---|---|
| `data/MIV6.3A.csv` | turn (volley) | none | raw turns | MIBot v6.3A study logs |
| `data/parsed/MIV6.3A_parsed.csv` | utterance | none | unlabelled pool (must drop the 10 test sessions) | data/MIV6.3A.csv, parser |
| `data/2024-11-14-MIV6.3A-...merged.csv` | participant | readiness / importance / confidence, demographics | session outcomes (join 'Participant id' = conv_id) | MIBot v6.3A surveys |

**`pool.miv63b`**

| File | Level | Labels | Use it for | Derived from |
|---|---|---|---|---|
| `data/MIV6.3B.csv` | turn (volley) | none | raw turns | MIBot v6.3B study logs |
| `data/parsed/MIV6.3B_parsed.csv` | utterance | none | unlabelled pool (self-training) | data/MIV6.3B.csv, parser |
| `data/2024-11-19-MIV6.1B_...merged.csv` | participant | survey outcomes | session outcomes (file name says 6.1B) | MIBot surveys |

**`annomi.gold`**

| File | Level | Labels | Use it for | Derived from |
|---|---|---|---|---|
| `data/external/annomi/AnnoMI-full.csv` | turn x annotator | AnnoMI attributes per annotator | agreement analysis (7 ten-rater transcripts) | official release (Wu et al.) |
| `data/external/annomi/AnnoMI-simple.csv` | turn | AnnoMI main behaviour / talk type | own-scheme train/dev/test | official release |
| `data/AnnoMI.csv` | turn | MISC-mapped AnnoMI labels (AutoMISC) | source of AnnoMI_eval (different row order) | AnnoMI-full, mapped by AutoMISC |
| `data/manual/AnnoMI_eval.csv` | turn | MISC gold on shared codes only | cross-scheme TEST of MISC models | data/AnnoMI.csv (baseline.prep_crossscheme) |
| `data/parsed/AnnoMI_parsed.csv` | utterance | none | utterance-split text | AnnoMI, parser |

**`welivita.gold`**

| File | Level | Labels | Use it for | Derived from |
|---|---|---|---|---|
| `data/external/welivita/MI_Dataset.csv` | sentence | ann1, ann2, judge stages, final label | agreement analysis; own-scheme split | official release (Welivita & Pu) |
| `data/external/welivita_mi_parsed.csv` | sentence | final label + MISC map (weak_t2) | self-training pool | MI_Dataset.csv (selftrain.ingest) |
| `data/manual/Welivita_eval.csv` | sentence | MISC gold on shared codes | cross-scheme TEST of MISC models | welivita_mi_parsed.csv (baseline.prep_crossscheme) |

**`miti.casaa.gold`**

| File | Level | Labels | Use it for | Derived from |
|---|---|---|---|---|
| `data/external/casaa/CASAA_eval.csv` | utterance | MITI codes + MISC map where exact | MITI / MISC TEST (18 clean sessions) | CASAA PDFs (baseline.prep_casaa) |
| `data/external/casaa/casaa_turns.csv` | turn | raw code cell + coder notes | audit of the parse | CASAA PDFs |
| `data/external/casaa/casaa_globals.csv` | session | MITI global ratings | session-level analysis (9 sessions) | CASAA PDFs |

Traps:
- `data/HLQC.csv` and `data/parsed/HLQC_parsed.csv` are the unlabelled pool. The train set is `data/manual/HLQC_balanced_manual.csv`.
- `data/AnnoMI.csv` is in a different row order from the official release.
- Welivita's labels are the original authors' in all three files.

## 2. Catalogue

A catalogue for looking things up, one row per dataset; the analysis of each dataset is in §4.

| ID | Domain | Modality / transcription | Sessions | Units (couns / client) | Scheme | Annotators / reliability | Role |
|---|---|---|---:|---|---|---|---|
| `misc.miv63a.gold` | smoking cessation | chatbot text (LLM counsellor, Prolific participants) | 10 | 821 (580 / 241) | MISC 2.5 + AutoMISC AC± and T1 grouping (§3), both speakers | 4 trained coders, consensus; Fleiss κ ≥ 0.6 after alignment | **test** |
| `misc.hlqc.gold` | mixed health (alcohol, diet, exercise, smoking) | spoken demo videos, **ASR** (20% of units have sentence punctuation) | 10 | 1,925 (1,040 / 885) | MISC 2.5 + AutoMISC AC± and T1 grouping (§3), both speakers | AutoMISC team; no κ reported; FI/FA inconsistent | **train** |
| `pool.hlqc` | mixed health | spoken, ASR | 257 (154 high / 103 low) | 30,610 utts | none (session high/low only) | — | unlabelled pool |
| `pool.miv63a` | smoking | chatbot text | 173 | 13,692 utts | none (+ readiness / importance / confidence pre, post, week-later) | — | unlabelled pool (minus test) |
| `pool.miv63b` | smoking | chatbot text | 165 | 11,771 utts | none | — | unlabelled pool |
| `annomi.gold` | mixed (23 alcohol, 19 smoking, 8 medication, …) | spoken demos, **manual** transcription | 133 (110 high / 23 low) | 9,699 turns (4,882 / 4,817) | AnnoMI: main behaviour + subtypes; client change/neutral/sustain | 10 experts: 126 transcripts single-annotated, 7 by all 10. **Measured** Fleiss κ: main 0.74, question type 0.74, reflection type 0.50, input type 0.51, client 0.47 | own-scheme train/dev/test; MISC transfer test |
| `welivita.gold` | mental-health peer support | written forum (1,000 CounselChat + 1,000 Reddit threads) | 2,000 | 20,462 (17,261 listener / 3,201 seeker) | MITI variant (15 listener codes; maps to MITI 4.2.1, §3) | 2 MTurk workers + 2 expert judge stages. **Measured** ann1 vs ann2 κ 0.34 (raw 0.41) | own-scheme (agreed subset); weak transfer test |
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

### Welivita (Table 1 of the paper): 15 labels, all present in the data, **a MITI variant**

Welivita & Pu adapted their labels from MITI 2.0 and 4.2.1. Open/closed questions, Direct, Warn and Support come from earlier MITI versions; Self-Disclose and Other are their additions. So it is not a separate scheme. It maps to **MITI 4.2.1** (`registry.WELIVITA_TO_MITI`, each row citing the manual) and to MISC:

| Welivita code | MITI 4.2.1 | MITI mapping, with manual basis | MISC |
|---|---|---|---|
| Closed / Open Question | Q | exact: MITI 4 does not split open/closed | CQ / OQ |
| Simple / Complex Reflection | SR / CR | exact | SR / CR |
| Affirm | AF | exact (MITI 4.2.1 Affirm is stricter) | AF |
| Emphasize Autonomy | Emphasize | exact | EC |
| Confront | Confront | exact | CO |
| Advise with Permission | Persuade with Permission | exact: E.4.c | ADP |
| Advise without Permission | Persuade | exact: E.4.b (advice without autonomy emphasis) | ADW |
| Warn | Confront | exact: E.4.g.2 lists "warning" | WA |
| Support | not coded | exact: p.26 "statements of support … are no longer coded" (Welivita's example nearly copies the manual's) | SU |
| Other | not coded | exact: §F greetings and off-topic | — |
| Give Information | GI | approximate: Welivita GI includes opinions, which MITI codes as Persuade | GI |
| Direct | Persuade | approximate: imperatives are advice (Persuade); with disapproval, Confront | DI |
| Self-Disclose | — | none: Persuade only when used to persuade (needs context) | — |

- **Coverage of listener labels:** 57% exact, 34% approximate, 8% none.
- **Gap:** MITI's **Seek** has no Welivita counterpart.
- **Caveat:** "Give Information" includes giving an opinion, which is broader than both MITI GI and MISC GI. Our MISC ingest map (`selftrain.ingest.MITI_TO_MISC_T2`) also lists Reframe and Structure, which are dead entries: no Welivita row carries them.

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

## 4. Per-dataset findings

One subsection per dataset. Every table in it describes that dataset only. Cross-dataset results are in §5. Notebook §4–§9 have the full tables and plots.

### 4.1 `misc.hlqc.gold`: HLQC balanced subset (TRAIN)

**Profile:** 10 sessions (5 high, 5 low quality); 1,925 utterances (1,040 counsellor / 885 client); median 228 utterances per session (range 35–318); median 8 words per utterance. Speech-to-text transcript: 42% of utterances have uppercase, 20% have sentence punctuation, 14% contain fillers (um/uh/mm-hmm).

**Labels.**
- **Counsellor:** FI 198, SR 137, CQ 132, OQ 90, CR 85, ADP 81, ST 76, FA 55, ADW 31, RCW 25, WA 24, AF 23, EC 22, GI 16, CO 15, DI 12, SU 9, RF 9.
- **Client:** N 756 (85%), AB+ 27, R+ 25, AC+ 23, R− 12, O− 10, D− 8, D+ 6, AB− 6, O+ 3, C+ 3, N− 3, N+ 2, C− 1.

**Quality checks (MISC 2.5 manual rules)**

| Check | Result |
|---|---|
| Standalone acknowledgements ("okay", "yeah", "mm-hmm", "right"): manual says FA (p.22) | **77 of 121 (64%) coded FI.** "okay" FI 24 / FA 10; "yeah" FI 14 / FA 15 |
| FI share of counsellor utterances: manual ceiling 5% (p.24) | **19%** (198/1,040) |
| What FI contains | 39% acknowledgements, 20% fragments of ≤ 3 words, 35% longer statements, **7% pleasantries** (what FI is for) |
| Questions starting "do/could/is…" coded CQ (p.28) | 40 of 41 ✓ |
| Scale questions coded CQ (p.28) | 6 of 6 ✓ |
| Permission-seeking coded EC (p.16) | 1 instance, coded FI ✗ |
| Gold questions with no "?" in the text | **78%** (ASR drops punctuation) |
| Repeated short utterances with conflicting labels | 9 of 27 (all FI vs FA) |
| T1 contradicts the T2 group | 6 rows (high_117#142 Q/ST, high_121#120 CRL/ST, high_127#7 O/OQ, low_031#156 O/OQ, low_031#179–180 O/CQ): **needs decision** |

**Segmentation:**
- 2.66 utterances per counsellor turn;
- 36% of utterances start with and/but/or/so/because (cut mid-sentence);
- 19% are ≤ 3 words;
- SR : CR = 137 : 85, with SR and CR the same length (median 11 vs 12 words).

**Session concentration.** Several codes come mostly from one session:

| Code | Share from the top session |
|---|---|
| GI | 75% from `high_121` |
| DI | 75% from `low_001` |
| ADP | 51% from `low_031` |
| WA | 67% from `low_001` |

ADW, CO and DI come 100% from low-quality sessions, and RCW and WA 96%.

**Other:** 1,925 rows vs 1,924 in the paper (off by one; no duplicate rows).

### 4.2 `misc.miv63a.gold`: MIBot v6.3A expert consensus (TEST)

**Profile:** 10 chatbot sessions; 821 utterances (580 counsellor / 241 client); median 78.5 utterances per session (range 37–119); median 12 words per utterance. Typed text: 85% uppercase, 83% sentence punctuation, 0.4% fillers.

**Labels.**
- **Counsellor:** SR 134, OQ 79, CQ 75, FI 56, CR 42, ADP 42, EC 39, SU 39, AF 30, GI 24, ST 13, FA 4, DI 2, RF 1.
- **Client:** N 125 (52%), R+ 32, R− 25, AC+ 11, AB− 10, O+ 10, D+ 9, C+ 6, D− 5, AB+ 3, TS+ 2, N− 2, AC− 1.

**Quality checks**

| Check | Result |
|---|---|
| FI share | 9.7%, above the 5% ceiling, but **68% of it is real pleasantries** ("Hello!", "Thank you for our conversation today") |
| FA | almost never used (4 rows): the chatbot does not backchannel |
| Questions starting "do/could/is…" coded CQ | 60 of 60 ✓ |
| Utterances ending "?" coded as reflections | 1 of 159 |
| Gold questions with no "?" | 2% |
| Repeated short utterances with conflicting labels | 1 of 7 ("I'm glad to hear that": FI / AF / SU) |

**Segmentation:** 3.79 utterances per counsellor turn (long multi-sentence chatbot turns); 0.5% start mid-sentence. Reflections are long, polished paraphrases: **SR median 18 words**, CR 15.5.

**Selection and representativeness:** all 10 sessions come from the thesis's MI target population ("low-confidence-or-discordant"; the 58 high-confidence sessions are outside the study). Against the other 105 target-population sessions they don't differ in length, Δ confidence (+1.90 vs +1.71) or Δ readiness (Mann-Whitney p ≥ 0.15).

### 4.3 `annomi.gold`: AnnoMI

**Profile:** 133 professionally transcribed demonstrations (110 high, 23 low quality); 9,699 turns (4,882 therapist / 4,817 client); median 47 turns per transcript; median 9 words per turn. Topics: alcohol 23, smoking 19, medication 8, …

**Labels.**
- **Therapist main behaviour:** other 1,586, question 1,386, reflection 1,296, input 614.
- **Subtypes:** open 954 / closed 702; simple 849 / complex 695; information 455, negotiation 148, advice 145, options 45.
- **Client:** neutral 3,102, change 1,174, sustain 541.

**Agreement among its own annotators** (7 transcripts coded by all 10, Fleiss κ):

| Attribute | κ |
|---|---|
| main behaviour | 0.74 |
| question type | 0.74 |
| input type | 0.51 |
| reflection type | **0.50** |
| client talk | **0.47** |

**Annotator effect.** On identical utterances, the complex-reflection share ranges from 0.19 to 0.82 across annotators, and the change-talk share from 0.14 to 0.38. Annotator 3 is the outlier (κ 0.65 against the leave-one-out majority; the others are ≥ 0.79). Each annotator alone coded 11–13 transcripts.

**Cleaning:** topic labels have stray whitespace (stripped in the loader). The open/closed convention differs from MISC: see §5.2.

### 4.4 `welivita.gold`: Welivita & Pu peer-support forums

**Profile:** 2,000 written threads (1,000 CounselChat, 1,000 Reddit); 20,462 sentences (17,261 listener / 3,201 seeker); median 16 words.

**Labels** (listener, final): Give Information 4,856, Advise without Permission 2,285, Self-Disclose 1,390, Complex Reflection 1,294, Support 1,233, Affirm 945, Closed Question 905, Direct 898, Other 805, Simple Reflection 556, Advise with Permission 484, Open Question 476, Confront 318, Emphasize Autonomy 253, Warn 113. 450 listener sentences have no label ("-").

**Agreement:** ann1 vs ann2 κ 0.34 (raw 0.41). Only 7,152 of 17,261 listener sentences have crowd agreement, and agreement is lowest for Complex Reflection (261 of 1,294) and Advise with Permission (57 of 484).

**Duplicate threads:** of 356 overlapping thread pairs, **316 share the opening post with different replies** (one CounselChat question answered by several therapists), **30 are true duplicates** (e.g. 99 = 1856), and 10 overlap partially.

### 4.5 `miti.casaa.gold`: CASAA MITI 4 coded transcripts

**Profile:** 20 transcripts; 1,346 rows (687 counsellor / 659 client); median 76 rows per transcript. Manual transcripts, counsellor coded only.

**Labels** (single-code counsellor rows): CR 160, Q 107, NC 90, SR 83, Confront 52, GI 51, Persuade 30, Seek 25, AF 15, Emphasize 8, PwP 1. Also 50 multi-code turns (no gold), 8 SAME continuations, and 2 client rows carrying a code (a source quirk). MITI global ratings exist for 9 sessions.

**Cleaning:** turn 25 is missing in 3 source PDFs (a source gap). Two sessions duplicate HLQC sessions (§5.1).

### 4.6 Unlabelled pools

| Pool | Profile | Issues |
|---|---|---|
| `pool.hlqc` | 257 sessions (154 high / 103 low), 30,610 utterances; ASR (10% have sentence punctuation) | 53 near-duplicate session pairs (re-uploads of the same video; 45 redundant ids), 4 of them with **conflicting high/low labels** (high_025 = low_086, high_084 = low_026, high_036 = low_051, …). It also holds 6 copies of HLQC train sessions under other ids (§5.1) |
| `pool.miv63a` | 173 sessions, 13,692 utterances; outcome surveys (confidence 4.27 → 5.62 pre → post) | contains the 10 test sessions (excluded by `selftrain.label_pool`); 2 empty rows |
| `pool.miv63b` | 165 sessions, 11,771 utterances | 22 of 652 long MIV6.3A test utterances recur verbatim (templated chatbot lines; removed by `label_pool._drop_eval_like`) |

No data file was modified. Duplicates and leaks are handled by the exclusion lists in `data/splits/`.

## 5. Comparisons

Each table here compares datasets on purpose. Its title says which.

### 5.1 Overlap map: which sessions appear in more than one dataset

5-gram containment between conversations, ignoring phrases found in more than 5 conversations. Unrelated sessions score ≤ 0.02, and copies of one video (manual vs ASR transcript) score 0.13–0.95. "→" means "is the same session as".

| Overlap | Sessions | Consequence |
|---|---|---|
| **HLQC train ⊂ HLQC pool** (same ids) | all 10 | expected (subset) |
| **MIV6.3A test ⊂ MIV6.3A pool** (same ids) | all 10 | pool users must drop them; `selftrain.label_pool` does |
| **HLQC train → HLQC pool under other ids** | low_026→high_084 (0.95), low_001→low_027 (0.93) & low_040 (0.90), low_080→low_019 (0.87) & low_098 (0.59), low_031→low_054 (0.82) | extra copies of training sessions |
| **HLQC train → AnnoMI** | low_001→53 (0.69), low_080→15 (0.67), high_099→21 (0.66), low_033→44 (0.62) | **AnnoMI transfer-test results so far include 4 training sessions.** Exclude 15, 21, 44 and 53 |
| **HLQC pool → AnnoMI** | 48 AnnoMI transcripts match ≥ 1 of 63 pool sessions (68 pairs, up to 0.85) | exclude them when the model also saw `pool.hlqc` |
| **HLQC train → CASAA** | high_121 → Emmy's First Encounter (0.40) | Emmy excluded from the CASAA test |
| **HLQC pool → CASAA** | high_072 → The Rounder (0.28) | Rounder excluded from the CASAA test |
| MIV6.3A test ↔ anything else | none at session level | — |

Within-dataset duplicates are listed in each dataset's subsection (§4.4, §4.6). AnnoMI, CASAA and MIV have none: AnnoMI and CASAA are curated distinct videos, and every MIV session is a different participant.

### 5.2 Same sessions, different coders: HLQC labels vs AnnoMI experts and the CASAA MITI reference

Utterances were aligned word by word and compared per reference turn, because AnnoMI and CASAA code whole turns.

| Comparison | n | Agreement | κ | Reference group's own κ |
|---|---:|---:|---:|---:|
| HLQC vs AnnoMI: counsellor main behaviour | 194 turns | 72% | 0.58 | 0.74 |
| HLQC vs AnnoMI: client talk type | 185 turns | 75% | **0.36** | 0.47 |
| HLQC OQ/CQ vs AnnoMI open/closed | 114 | 40% | 0.07 | 0.74 |
| HLQC SR/CR vs AnnoMI simple/complex | 21 | 81% | 0.54 | 0.50 |
| HLQC vs CASAA MITI reference | 34 turns | 44% | 0.32 | (reference) |

- **Client talk:** HLQC says neutral in 28 turns where AnnoMI says change talk.
- **Open/closed:** HLQC follows the MISC rule ("could you…", specific-fact questions = CQ). AnnoMI calls 68 such questions "open", 44 of them from annotator 3, so **AnnoMI "open" is not MISC OQ**.
- **Reflections:** the MITI developers code 9 of 16 HLQC "simple" reflection turns as complex.
- **Transcription:** word error rate of HLQC's ASR against the manual transcripts is 9–19% (AnnoMI sessions) and 25% (CASAA Emmy).

### 5.3 Train (`misc.hlqc.gold`) vs test (`misc.miv63a.gold`)

| Aspect | Train (HLQC) | Test (MIV6.3A) |
|---|---|---|
| Transcript | ASR speech: 20% punctuation, 14% fillers | typed chatbot text: 83% punctuation, 0.4% fillers |
| Words per utterance (median) | 8 | 12 |
| Client labels neutral | 85% | 52% |
| R+ / R− share of client labels | 2.8% / 1.4% | 13.3% / 10.4% |
| SU / EC / AF share of counsellor labels | 0.9% / 2.1% / 2.2% | 6.7% / 6.7% / 5.2% |
| FI meaning | acknowledgements and fragments (7% pleasantries) | pleasantries (68%) |
| SR median length | 11 words | 18 words |
| Codes only in this set | ADW, RCW, WA, CO (95 rows = 9% of counsellor labels); C−, N+, O− | TS+, AC− |

### 5.4 Impact on the main model (`ft1mix_bare`, trained on HLQC, tested on MIV6.3A, 3 seeds)

| Quality issue | Effect on the test |
|---|---|
| SR/CR line differs between coder groups (§4.1, §4.2, §5.2) | **largest single error:** SR recall 0.42; 51% of test SRs predicted CR |
| HLQC under-codes change talk (§5.2, §5.3) | 32% of gold change/sustain talk predicted neutral; O+ recall 0.03, R+ 0.54, R− 0.51 |
| SU / EC / permission-seeking missing from training (§4.1, §5.3) | SU recall 0.39 (→ CR, AF); EC 0.33 (→ SU, CQ, FI) |
| Train-only codes (§5.3) | ~7 spurious ADW/RCW/WA/CO predictions per run |
| HLQC FI catch-all (§4.1) | small: test pleasantries still predicted FI 95% of the time. HLQC-based scores (CV, teacher gate) are biased instead |

### 5.5 Possible actions (none applied; they change training data, so they need your decision)

| Issue | Option |
|---|---|
| SR/CR line | agree a written SR/CR guideline and adjudicate; also report reflections at T1 (SRL vs CRL) |
| change talk | re-annotate HLQC client turns, or reweight non-neutral client labels |
| SU / EC | targeted SU / EC / permission-seeking examples (human or synthetic) |
| train-only codes | downweight, or train ADW/RCW/WA/CO at T1 only |
| FI catch-all | relabel standalone acknowledgements FI → FA in a derived train copy |
| ASR | re-segment HLQC, or use corrected transcripts (MI-TAGS, if obtained) |
| AnnoMI open ≠ OQ | score AnnoMI questions at T1 (Q) only |
| HLQC T1/T2 rows; high/low conflicts | decide (see §4.1, §4.6) |

## 6. Split plan and justification (`python -m eda.splits` → `data/splits/*.json`)

The splits were reviewed rather than inherited: see [experiments/2026-10-05-split-review.md](experiments/2026-10-05-split-review.md) for the criteria, the evidence and the candidates. In short:
- **MIV6.3A has the most reliable MISC labels**, so it stays the test set. HLQC is never a test set.
- In a proxy experiment, **in-domain training data matters more than size**: about 650 MIV utterances beat 1,925 HLQC utterances (counsellor macro-F1 0.47 vs 0.26). Pooled HLQC + MIV also generalises best to CASAA.
- The test is thin (CI width 0.17–0.20), so every protocol keeps all 821 MIV utterances scored.

**Lucky draws and exemplars** (split review §9–§10): every manifest carries `variance_rules`:
- ≥ 3 matched training seeds;
- paired differences;
- session-bootstrap CIs;
- no single random split or fold draw as a headline.

`misc_main` and `misc_pooled_cv` also carry `exemplar_rules`:
- few-shot, retrieval and synthesis exemplars come from the training side only;
- prompted arms use ≥ 3 exemplar draws;
- 22 of 41 few-shot exemplars come from HLQC sessions that are AnnoMI/CASAA test transcripts, so the AnnoMI/CASAA exclusions apply to **any** model that saw HLQC gold by training, prompt, retrieval or synthetic data.

The manifests are byte-identical across reruns and `PYTHONHASHSEED` values. Built-in checks:
- no conversation sits in two splits;
- in pooled CV, each MIV session is tested exactly once and never trained on in its own fold;
- no Welivita duplicate cluster spans splits;
- no AnnoMI video series spans splits;
- each manifest has its own seeded RNG, so changing one split never shifts another;
- CASAA Emmy is not in a test split;
- the MIV pool exclusion covers every test session.

### 6.1 `misc_main`: cold-start transfer (current reported setting)

| Split | Data | Why |
|---|---|---|
| train | `misc.hlqc.gold` (10 sessions) | the only other MISC gold. HLQC is never used as a test set because its labels are noisy |
| checkpoint selection | HLQC val fold 0 of 7 (369 rows), fixed before training | MIV is never read for selection |
| test | `misc.miv63a.gold` (10 sessions, all 821 utterances); `miti.casaa.gold` on 6 exact T2 codes + T1 (18 sessions) | MIV: most reliable labels, target population. CASAA: reference-quality external test |
| exclude | CASAA Emmy and Rounder; AnnoMI 15/21/44/53 for any HLQC-trained model; 48 AnnoMI transcripts if the model saw `pool.hlqc` | duplicate sessions |

This answers: "can a model trained only on public labels code a new service's sessions?"

### 6.2 `misc_pooled_cv`: **PROPOSED primary** (pending your decision and a 7B confirmation run)

| Split | Data | Why |
|---|---|---|
| folds | **headline: leave-one-session-out** (10 folds, no randomness). **Exploratory: 5-fold, re-drawn with each training seed** (seed-tied) | every MIV utterance is tested exactly once, so scores still cover all 821 and stay comparable with AutoMISC. A single fold draw would be a lucky draw: the client gain ranged from +0.001 to +0.045 across 20 draws |
| train (fold k) | all `misc.hlqc.gold` + the MIV sessions outside fold k (9 under LOSO, 8 under 5-fold) | RQ1 as worded: adapting on labels the service already holds. Best client and best external (CASAA) proxy scores |
| checkpoint selection | HLQC val fold, as in 6.1 | no MIV session is used for selection |
| external test / exclude | as in 6.1 | — |

Cost: LOSO is 10 runs per arm per seed; seed-tied 5-fold is 5. P0 results stay valid as the cold-start setting.

**Leakage** (audit in the split review, §6): folds are by session and by participant, and no text crosses folds beyond 18 of 532 template-like counsellor lines. The real risk is reusing decisions tuned on MIV, a risk that exists already under P0: the retriever's rare-code list, the self-training caps, `v3_mivmix` topic counts, and the choice of main setting. Rules in the manifest:
- freeze the current config;
- rebuild priors, caps, topic mixes and retrieval indexes from each fold's training sessions;
- score the pooled out-of-fold predictions once, paired across arms.

### 6.3 `annomi_own`: AnnoMI scheme

**5-fold CV grouped by video series** (parts of one session and good/bad versions of one role-play stay in one fold), stratified by series MI quality. Every transcript is tested once; for test fold k, dev is fold k+1. Folds hold 21–37 transcripts.

**Why CV and not one split:** a single 15% split moved proxy macro-F1 by SD 0.02–0.03, with a range of about 0.1. Also report the score on the 7 ten-rater transcripts (majority labels, the cleanest) and per-annotator scores (annotator effect).

**For MISC models, AnnoMI is an external transfer test:** counsellor main behaviour, client C/S/N, and questions at T1 only, with the exclusions in 6.1.

### 6.4 `welivita_own`: Welivita scheme

**5-fold CV grouped by same-post cluster**, stratified by source (~400 dialogues per fold; dev = the next fold). Labels are the 7,152 stage-I-agreed sentences. Report cross-source robustness (CounselChat → Reddit and the reverse): the proxy drops from 0.47 in-source to 0.36–0.38 across sources.

In MISC work it is a weak pool: crowd κ 0.34, written forum domain, and self-training with it hurt (−0.047). **In MITI work it is the training set** (6.4b), because its labels map to MITI 4.2.1.

### 6.4b `miti_scheme`: MITI 4

**Two MITI datasets:** Welivita (a MITI variant, mapped to MITI 4.2.1 in §3) for training and selection, and CASAA (the reference coding) as **test only**.

Recommended training: **pooled in MITI space.** Welivita's agreed labels (Self-Disclose dropped) plus the MISC gold, both mapped to MITI. Proxy on CASAA, every route trained on MITI-mapped labels:

| Route | CASAA MITI macro-F1 (excl. Seek) |
|---|---:|
| MISC gold only | 0.246 |
| Welivita only | 0.303 |
| **pooled** | **0.335** |

Selection uses the `welivita_own` folds. Seek can't be learned from either source. MI-TAGS, if obtained, would join after deduplication.

### 6.5 `casaa_test`

External test only: 18 sessions (Emmy and Rounder excluded). It is too small to train on, and its labels are the reference standard.

### 6.6 `pools`

| Pool | Action |
|---|---|
| `pool.hlqc` | drop the 45 redundant duplicates; `exclude_if_annomi_is_test` (63 ids); `exclude_if_casaa_is_test` (2) |
| `pool.miv63a` | drop the 10 test sessions. The 58 high-confidence sessions are outside the MI target population: usable as unlabelled data, but not representative of the test |
| `pool.miv63b` | use as is |

All pools are training-side only.

## 7. What this changes for results already reported

- **AnnoMI transfer numbers** (e.g. T2 accuracy 0.504 for the 2-adapter model) include 4 HLQC training sessions. Rerun excluding 15/21/44/53.
- **AnnoMI OQ/CQ transfer scores** compare against an "open" that is broader than MISC OQ (§5.2). Report AnnoMI questions at T1.
- **HLQC-based scores** (HLQC CV, the teacher-screen gate) depend on HLQC's FI convention (§4.1). This confirms the 2026-10-04 teacher audit.
- The MIV6.3A test set is leak-free, but the main model's largest errors trace to train/test convention and prior differences (§5.4). The proposed pooled CV (§6.2) addresses this directly.
- The CASAA results to come should use `casaa_test.json` (18 sessions).

## Reproduce

```bash
PYTHONPATH=src python -m baseline.prep_casaa          # once: parse CASAA PDFs
PYTHONPATH=src python -m eda.splits                   # manifests + leak checks
jupyter nbconvert --to notebook --execute --inplace notebooks/datasets_eda.ipynb
```
