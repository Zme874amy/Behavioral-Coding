# External MI-coded dataset survey (2026-10-04)

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
