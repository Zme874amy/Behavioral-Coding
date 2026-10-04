# External MI-coded dataset survey (2026-10-04)

> The organised catalogue of the datasets we actually use (names, overlaps, cleaning, splits) is [../DATASETS.md](../DATASETS.md). This page records the external search behind it.

**Question.** Which other MISC-, MITI- or MI-coded corpora actually exist and can be
used? How good are their labels? Which ones duplicate what we already hold
(MIV6.3A, HLQC, AnnoMI, Welivita)?

**Method.** For each candidate I checked that the data really exists: repo tree,
access route, file contents, and label columns. I did not stop at the paper's
"we release" sentence. Where the raw annotator columns allow it, I computed
agreement myself (`scripts/audit_external_datasets.py`). Where they don't, I quote
the paper's reliability figures and spot-check labels by eye. Downloads went to a
session scratch dir, not into `data/`.

## Verdict

| Rank | Dataset | Scheme | Size | Label quality | Access | Overlap with ours | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | **MI-TAGS** (Cohen et al., LREC-COLING 2024) | MITI 4.2: 10 codes + "structure statement". CLEAR 1.0 client tags (change / counter-change / neutral). Plus session globals | 242 sessions, 15,627 utts (8,389 T / 7,238 C) | κ vs expert psychologist 0.74 / 0.84 / 0.68 (3 trained undergrads, after 3 calibration rounds). CLEAR κ 0.90. **Most sessions single-annotated**, with the expert's codes used where present. The public sample already has debatable rows: "Well, I do exercise a lot more." → Neutral | **Gated**: [Google request form](https://advanced-reality-lab.github.io/MI-TAGS/) with an NDA (no redistribution). Only a 10-row sample is public | **140 of the 242 sessions are HLQC sessions**, re-transcribed by hand where the video was still available. Verified: HLQC `high_040` / `high_083` utterances appear in the sample. At least one AnnoMI title too ("The Ineffective Physician") | **Get it.** It is the only large MITI corpus with expert-calibrated utterance codes. It is also a cleaner re-annotation of HLQC sessions, which is where our HLQC FI/FA problems and noisy ASR live. Must dedup against AnnoMI before using AnnoMI as a transfer test |
| 2 | **CASAA MITI 4 coded training transcripts** (UNM / Moyers, Miller) | MITI 4: GI, Persuade, PwP, Q, SR, CR, AF, Seek, Emphasize, Confront | 20 coded transcripts, ≈650 counsellor codes (regex lower bound) | **Reference coding by the MITI developers.** Not crowd or student coded. Rich in non-adherent and tail codes: Confront ≈40, Persuade ≈30, Seek ≈45 | Free PDFs on [casaa.unm.edu/tools/miti.html](https://casaa.unm.edu/tools/miti.html). Training material; no explicit data licence, so treat as research or eval use with citation | Some videos (Moyers' "Helping People Change" series) may also be in HLQC, AnnoMI or MI-TAGS as YouTube copies | **Use as a small, clean calibration and eval set** (counsellor only). Needs PDF parsing; code letters sit at the line end |
| 3 | **AnnoMI-full** (already held as AnnoMI) | MISC-derived: Q (open/closed), R (simple/complex), input (info/advice/negotiation/options), other. Client: change/neutral/sustain | 133 transcripts, 9,699 utts | Measured on the 7 transcripts coded by all 10 experts: Fleiss κ main behaviour 0.74, question type 0.74, **reflection type 0.50**, input type 0.51, **client talk 0.47**. Annotator 3 is an outlier (κ 0.65 vs ≥0.79 for the rest). **Strong annotator effect** on identical utterances: complex share of reflections ranges **0.19–0.82** across annotators, change-talk share 0.14–0.38. The other 126 transcripts each have a single annotator | Public, GitHub | We already use it | **Keep, but treat SR/CR and client labels as noisy.** `annotator_id` is a free covariate: weight by annotator reliability, or stratify evaluation by annotator. The `advice` subtype (246) could add an ADV label we don't currently map |
| 4 | **Welivita MI** (already held) | MITI-derived, 15 codes, written forum text | 17,261 listener utts | Measured: crowd ann1 vs ann2 **raw 0.41, κ 0.34**. Final label goes through 2 expert-judge stages | Public, CC BY-NC-SA 3.0 | We already use it | Keep as weak gold, as now. The original files carry the per-stage columns: restricting to `stage I agreed label` (7,152 listener utts where the two crowd annotators agreed) gives a higher-precision subset |
| 5 | MIDAS (Spanish MI, NAACL 2025) | MITI/ITEM: **question and reflection only**, span-level | 74 convs, 1,506 counsellor turns (884 Q / 415 R spans) | ICC 0.92 on 5 conversations (code counts, not utterance κ). 3 MI counsellors | Public, [MichiganNLP/MIDAS](https://github.com/MichiganNLP/MIDAS) | None | Low value: Spanish, and no SR/CR or OQ/CQ split |
| 6 | KMI (Korean, NAACL 2025) | 8 MITI-style therapist codes; client unlabelled | 1,000 synthetic dialogues, 9,558 therapist utts | **Labels are a T5 forecaster's plan, not human codes.** Experts checked 210 samples (96%) but excluded "General". Spot check: 157 of the 776 "General" utterances open the way a reflection does (e.g. "I see you were worried about…"), and translation artifacts appear | Public, CC BY 4.0 | None | Skip. Synthetic, skewed (GI 87, Advise 43), weaker than our own v2/v3 synthesis |
| 7 | IC-AnnoMI (NLPAICS 2024) | AnnoMI's coarse labels | 97 ChatGPT rewrites of AnnoMI train dialogues (7,668 augmented utts) | Augmented utterances **inherit** the source AnnoMI label. Only dialogue-level ratings, by a single expert (MI_psych mean 3.3/4); no κ | Public, GitHub | Derived from AnnoMI | Skip. Same labels, unverified text |

**Exist but not obtainable:**

- **BiMISC** (LREC-COLING 2024): real Dutch MI with English post-edits, 80 sessions, 8,572 utts, multi-label MISC 2.1. Its repo has been a placeholder since 2024-04 ("content will come soon"), and the paper reports no κ.
- **Pérez-Rosas 2016** (277 sessions, MITI): reported as no longer available.
- **7 Cups MI** (Hsu et al. 2022; 734 chats, 17 codes, α 0.31–1.0): only through a collaboration agreement.
- **PAIR** (SR/CR/NR reflections): code and weights only.
- **HLQC MISC labels** beyond our 10-session balanced file: not public.

**Not MI-coded:**

- **OnCoCo 1.0**: German/English, its own 66-category scheme, partly synthetic.
- **CaiTI**: response scoring.
- **HF "MITI"**: a surgical-robotics dataset.
- **counsel-chat-miti-st-dpo**: LLM-judged preference pairs.

## Recommendations

1. **Request MI-TAGS access.** It needs the user's own name, affiliation and intended
   use, plus agreement to the NDA, so the user has to submit the form. Once it is in:
   map MITI → our T2 codes (Q needs open/closed resolution; Persuade → ADW/ADP;
   Seek / Emphasize → EC). Then compare its labels on the 140 HLQC sessions against
   `HLQC_balanced_manual` wherever the sessions coincide. That puts a number on how
   noisy HLQC's own labels are.
2. **Parse the CASAA coded transcripts** into an eval file. It is the only gold made by
   the scheme's authors, and it covers the non-adherent codes that every other corpus
   lacks.
3. **AnnoMI:** add annotator-aware handling (per-annotator reliability weights, or
   drop annotator 3 from training) before using its SR/CR or client labels as targets.
4. **Dedup by video title and normalised text** across HLQC, AnnoMI, MI-TAGS and CASAA
   before any cross-corpus train/test split.

## Reproduce

```bash
python scripts/audit_external_datasets.py /tmp/mi_dataset_audit
```

## Follow-up (same day): MI-TAGS access, same-scheme pairs, CASAA parsed

### Has anyone outside the MI-TAGS authors' lab actually got the data?

There's no evidence anyone has. The user's own form request has gone unanswered.

Semantic Scholar lists 13 papers citing MI-TAGS (queried 2026-10-04), and I read the
full text of every one available on arXiv. **Only the authors' own lab used the data**:
Yosef, Zisquit, Cohen, Klomek, Bar and Friedman (npj Mental Health Research 2025)
fine-tuned a 13B model on 6,000 MI-TAGS samples. Everyone else cites it without
using the data:

- CAMI and MIThinker use only its GPT-4o MITI prompt.
- Kong & Moon (2025) and Mahmood et al. (MIBot 2025) mention it in related work.
- StratCBT (2026) lists it in a dataset comparison table.
- The rest are surveys.

Releases are also lopsided: the GitHub repo holds a 10-row sample, and the Zenodo
record holds only the paper PDF.

**Next step:** email the authors directly. bencohen3@gmail.com is the README contact,
and Kfir Bar and Doron Friedman (Reichman University) are the senior authors. In the
same email, ask whether the 140 HLQC-derived sessions can be shared alone.

### More searching, aimed at same-scheme pairs or splittable corpora

None of these adds usable MI-coded data:

| Lead | What it is | Why it doesn't help |
|---|---|---|
| MIRROR @ IberLEF 2026 | MITI 4.2.1 shared task: SR/CR, OQ/CQ, PE/GI | Test sets are hidden and Spanish; the seeds are 3×200 rows. Binary pairs only |
| AutoMISC NLPAI4Health 2025 | Says it releases 506 MISC-labelled transcripts | The 506 are **GPT-4.1 labels** (AnnoMI, HLQC, MIBot). Its human labels are the 821 + 1,924 we already hold |
| EMMI (2024) | Multimodal annotations on AnnoMI + HLQC | No new MI codes |
| Boosting-with-MI (Welivita 2023) | MI-adherent rephrasing | Same Welivita gold plus automatic augmentation |
| CoachLah (2026) | 36,852 real health-coaching utterances | Behaviour-goal labels only. Could serve as an *unlabelled* pool |
| "Say it aloud" (Zenodo) | Change-talk counts for 16 participants | No transcripts |
| 7 Cups (Hsu 2022), Pérez-Rosas 2016/17 (MITI, 277 sessions), BiMISC (MISC, 8.5k utts) | Real MI-coded corpora | Request-only or unreleased. Contacts: x.sun2@uva.nl / j.pei@uva.nl (BiMISC), vperezr@txstate.edu (Pérez-Rosas, now at Texas State) |

**What we can actually train and evaluate on, by scheme:**

| Scheme | Train | Eval | Note |
|---|---|---|---|
| MISC 2.5 | HLQC_balanced_manual (1,924) | MIV6.3A (821) **+ CASAA mapped** (T2 311 / T1 484, clean sessions only) | CASAA is a second MISC-compatible test set, but covers only 6 T2 codes |
| MITI 4.x | **none public**; Welivita (MITI-derived, written forum text) is the closest | CASAA (670 counsellor turns) | MI-TAGS would complete this pair |
| AnnoMI scheme | AnnoMI split by transcript | AnnoMI held-out transcripts | 133 sessions is enough for grouped CV. Stratify by `annotator_id`, because the annotator effect is large |
| Welivita MITI | Welivita `stage I agreed` (7,152) split by dialogue | same, held out | Big enough, but forum text and weak gold |

### CASAA parsed: `src/baseline/prep_casaa.py`

The parser works on word coordinates (pdfplumber). It handles:

- notes columns, and coder notes inside the code cell ("GI / Persuade ruled out because…" gives GI only);
- parenthesised remarks, and words that wrap within a cell ("Empha / size");
- the speaker column drifting across pages;
- a turn-number typo in the source ("Q C" for "2 C");
- MITI global ratings typed after the last turn, which go to `casaa_globals.csv` (9 sessions).

**Output:** 20 transcripts, 1,329 turns (670 counsellor), giving 1,346 eval rows in the
`load_manual` schema plus the columns `miti_codes`, `align`, `miti_turn` and
`hlqc_overlap`. Files are gitignored under `data/external/casaa/`; the transcripts are
CASAA's.

- **Single-code counsellor rows:** CR 160, Q 107, NC 90, SR 83, Confront 52, GI 51, Persuade 30, Seek 25, AF 15, Emphasize 8, PwP 1. SAME (8) marks continuations.
- **Multi-code turns:** 32 split by sentence where the sentence count equals the code count; 50 left as multi-label with no gold.
- **MISC gold:** T2 369 (CR 160, SR 83, CO 52, GI 51, AF 15, EC 8). T1 566, which adds Q→Q and NC→O.

**Checks:**

- Turn numbers are continuous, except 3 PDFs that skip turn 25 in the source itself.
- No code tokens left in any utterance text, and all 7 counsellor turns without a code are genuine backchannels.
- 18 of 18 randomly sampled turns match the source PDF text, codes and notes.
- `load_manual` reads the file.

**Overlap:** CASAA *Emmy's First Encounter* is HLQC `high_121` (0.55 5-gram hit rate
against the ASR text), which is **in our training set**. *The Rounder* is HLQC
`high_072`, which holds only its first ~900 words. Both carry `hlqc_overlap`. Excluding
them leaves T2 gold 311 (CR 126, SR 74, CO 52, GI 43, AF 13, EC 3) and T1 gold 484.
No overlap with AnnoMI or MIV6.3.

```bash
pip install pdfplumber
PYTHONPATH=src python -m baseline.prep_casaa   # downloads the 20 PDFs, then parses
```
