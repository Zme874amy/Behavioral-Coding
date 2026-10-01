# `src/synth` — synthetic MISC training data

## Versioning rule (read before editing a prompt)

**Never overwrite a prompt builder in place.** Each generation of the prompt is a
separate experimental condition; overwriting one destroys the ability to reproduce
or compare the dataset it produced.

- `prompts_v1.py` — frozen. The one-shot, code-conditioned prompt (1,080 samples).
- `prompts_v2.py` — frozen. Scenario-conditioned dialogue windows, with the
  7-dimension ontology as it was actually run.
- `generate.py` — the **current** generation (v3): SpeechDialogueFactory dialogue
  script + Readiness Ruler conditioning + optional SocialDial rule step.

To change the prompt: add `prompts_v4.py`, point `generate.py` at it, and leave the
earlier modules untouched.

This rule exists because v1 and v2 were both overwritten in place while `src/synth/`
was still untracked, leaving no git history. They were reconstructed from captured
renderings and cross-checked against the conditioning columns retained in
`data/synth/raw/*.csv`; the reconstruction notices in each module record that.

## Pipeline

    ontology.py   scenario seed   topic (corpus-measured) + Readiness Ruler
          v
    generate.py   script -> [rule] -> 8-12 turn window, every utterance labelled
          v
    segment.py    split turns into MISC thought units, reusing the production
                  parser prompt (components/parser.py). Required: MISC codes a
                  thought unit, not a speaker turn, and ~50% of generated turns
                  carry more than one.
          v
    verify.py     independent re-code of every unit; dedupe; drop anything
                  resembling the evaluation set; emit an augmented train CSV
          v
    metrics.py    diversity scorecard (distinct-n, Self-BLEU, length KS, novelty)
    judge.py      0-100 consistency / coherence / naturalness, with a real
                  transcript reference column

`codes.py` is the single source of truth for the code vocabulary. The spec YAML
writes four IMI codes as ADWP/CON/DIR/RCWP; the MISC 2.5 manual, the AutoMISC
thesis, `automisc_ft.data` and the gold corpora all write them ADW/CO/DI/RCW.
`codes.py` renders the codebook with the valid abbreviation, constrains generated
codes to the valid vocabulary, and normalises anything that slips through.

## Gate before GPU

Run `metrics.py` and `judge.py` on a small pilot and compare against the real
corpus column before spending a retrain. The v1 failure (Self-BLEU 0.199 against
0.002 for real text) was visible in a 50-window pilot and would have been caught
before a 12-hour retrain.
