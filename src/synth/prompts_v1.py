"""FROZEN: the v1 generation prompt (the 1,080-sample one-shot run).

RECONSTRUCTION NOTICE. The original `generate.py` that produced the v1 dataset was
overwritten in place on 2026-09-22 and `src/synth/` had never been committed, so
there is no git history for it. This module restores the prompt verbatim from the
rendered output captured while the code was live, with the quota constants and the
response schema alongside. The per-sample conditioning is independently recoverable
from `data/synth/raw/Qwen2_5-32B-Instruct-AWQ.csv`, which retains a `target_code`
column covering all 36 Tier-2 codes.

Kept so the three generations can be compared and reproduced rather than one
replacing the next. Do not edit: add `prompts_v4.py` instead.

Known failure of this design, measured in `synth.metrics`: conditioning on nothing
but a target code collapsed the output. Self-BLEU 0.199 against 0.002 for the real
corpus, distinct-2 0.348 against 0.498, median utterance 15 words against 8, 10.9%
exact duplicates, and 15.6% of utterances mentioning smoking against 1.4% in HLQC.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel

# Quotas as run: 20 samples per code, x4 for the starved tail -> 1,080 samples.
PER_CODE = 20
RARE_MULTIPLIER = 4
RARE_CODES = ["SU", "EC", "AF", "GI", "TS+", "AC-"]
TEMPERATURE = 0.9


class SynthTurn(BaseModel):
    speaker: str          # "counsellor" or "client"
    text: str


class SynthSample(BaseModel):
    context: List[SynthTurn]
    utterance: str


def build_gen_messages(speaker: str, group: str, code: str, spec_text: str) -> List[dict]:
    """The v1 prompt. `spec_text` was the raw spec-YAML block for the target's
    Tier-1 group, i.e. the target code's siblings and their definitions.

    Note: injecting the raw spec text is also how the ADWP/CON/DIR/RCWP
    abbreviation mismatch reached the model (see `synth.codes`).
    """
    system = (
        "You are an expert in Motivational Interviewing (MI) and the MISC 2.5 "
        "coding scheme. You write short, realistic MI counselling excerpts for "
        "training a behavioural-coding model. Definitions for the relevant codes:\n\n"
        f"{spec_text}\n\n"
        "Write JSON with two fields: `context` (a list of 1-3 preceding turns, "
        "alternating counsellor/client, that set up the moment) and `utterance` "
        f"(the final {speaker} utterance). The final utterance MUST be a clear, "
        f"unambiguous example of the MISC 2.5 code '{code}'. Keep it natural and "
        "concise; do not mention codes or MISC in the text."
    )
    user = f"Generate one example whose final {speaker} utterance is coded '{code}'."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


__all__ = ["build_gen_messages", "SynthSample", "SynthTurn",
           "PER_CODE", "RARE_MULTIPLIER", "RARE_CODES", "TEMPERATURE"]
