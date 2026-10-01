"""Single source of truth for MISC code vocabulary in the synthesis pipeline.

Why this module exists. The spec YAML at `components/prompts/specs/` writes four
IMI codes as ADWP / CON / DIR / RCWP. The MISC 2.5 manual (Houck, Moyers, Miller,
Glynn & Hallgren), the AutoMISC thesis taxonomy (Fig. 3.2), `automisc_ft.data`'s
group maps, and the gold corpora all write them ADW / CO / DI / RCW. The project
already guards the difference for its own arms via `sft.eval.LABEL_ALIASES` —
whose only four entries are exactly these — but the v2 synthesis prompt injected
the raw spec text and typed the generated codes as free-form `str`, bypassing that
guard. Result on the v2 pilot: 4.2-5.4% invalid proposals, and RCW absent from the
synthetic data entirely while RCWP appeared.

This module fixes that in three ways:
  * `codebook()` renders the definitions with the VALID abbreviation, so the prompt
    and the schema cannot disagree;
  * `T1Code` / `T2Code` constrain generation to the valid vocabulary (SDF §3.1.1
    "constrained decoding"), reusing the pattern in `prompts/response_formats.py`;
  * `normalise_code()` is a backstop for anything that still slips through —
    spec abbreviations, "Name (CODE)" forms, and bare full names.
"""
from __future__ import annotations

import re
from typing import Dict, Literal, Optional, Tuple

from automisc_ft.data import (
    CLIENT_GROUPS, COUNSELLOR_GROUPS,
    t1_codes_for_speaker, t2_codes_for_speaker,
)
from components.prompts.loader import load_spec
from sft.eval import LABEL_ALIASES

# ---------------------------------------------------------------- vocabulary
T1_CODES: Tuple[str, ...] = tuple(dict.fromkeys(list(COUNSELLOR_GROUPS) + list(CLIENT_GROUPS)))
T2_CODES: Tuple[str, ...] = tuple(dict.fromkeys(
    [c for codes in COUNSELLOR_GROUPS.values() for c in codes]
    + [c for codes in CLIENT_GROUPS.values() for c in codes]))

# Constrained decoding: the model cannot emit anything outside these sets.
T1Code = Literal[T1_CODES]  # type: ignore[valid-type]
T2Code = Literal[T2_CODES]  # type: ignore[valid-type]

SPEAKER_OF: Dict[str, str] = {}
for _c in (c for codes in COUNSELLOR_GROUPS.values() for c in codes):
    SPEAKER_OF[_c] = "counsellor"
for _c in (c for codes in CLIENT_GROUPS.values() for c in codes):
    SPEAKER_OF.setdefault(_c, "client")


def group_of(code: str, speaker: str) -> Optional[str]:
    groups = COUNSELLOR_GROUPS if speaker == "counsellor" else CLIENT_GROUPS
    for g, codes in groups.items():
        if code in codes:
            return g
    return None


# ------------------------------------------------------- codebook rendering
_CODE_IN_SPEC = re.compile(r'\*\*(.+?)\s*\(([A-Za-z+\-]+)\)\*\*')


def _fix_abbrevs(text: str) -> str:
    """Rewrite spec-YAML abbreviations to the valid label (ADWP->ADW, ...)."""
    for wrong, right in LABEL_ALIASES.items():
        text = re.sub(rf'\({re.escape(wrong)}\)', f'({right})', text)
    return text


def codebook() -> str:
    """The full two-speaker codebook, with valid abbreviations only."""
    parts = []
    for spk in ("counsellor", "client"):
        parts.append(f"### {spk.upper()} codes (Tier-1 group -> Tier-2 codes)")
        for group, text in load_spec(spk, "t2").items():
            parts.append(f"[{group}]\n{_fix_abbrevs(str(text))}")
    return "\n".join(parts)


def _name_to_code() -> Dict[str, str]:
    """Full code name (lowercased) -> valid abbreviation, e.g. 'affirm' -> 'AF'."""
    out: Dict[str, str] = {}
    for spk in ("counsellor", "client"):
        for _, text in load_spec(spk, "t2").items():
            for name, abbrev in _CODE_IN_SPEC.findall(_fix_abbrevs(str(text))):
                out[name.strip().lower()] = abbrev
    return out


NAME_TO_CODE: Dict[str, str] = _name_to_code()
# The client N group states "just use 'N'" in prose rather than the bold
# **Name (CODE)** form the parser above looks for; thesis Fig. 3.3 labels it
# "Neutral (N)". Added explicitly so a model answering "Neutral" resolves.
NAME_TO_CODE.setdefault("neutral", "N")
NAME_TO_CODE.setdefault("neutral talk", "N")


# ------------------------------------------------------------- normalisation
def normalise_code(raw, speaker: str, tier: str) -> Optional[str]:
    """Coerce a model-emitted code onto the valid vocabulary, or None.

    Handles: valid codes; spec abbreviations via LABEL_ALIASES; trailing
    "Name (CODE)" forms; and bare full names ("Affirm", "Open Question").
    Returns None if it cannot be resolved for this speaker and tier.
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    m = re.search(r'\(([A-Za-z+\-]+)\)\s*$', s)   # "Reasons (R-)" -> "R-"
    if m:
        s = m.group(1)
    s = LABEL_ALIASES.get(s, s)
    allowed = t1_codes_for_speaker(speaker) if tier == "t1" else t2_codes_for_speaker(speaker)
    if s in allowed:
        return s
    hit = NAME_TO_CODE.get(s.lower())
    return hit if hit in allowed else None


__all__ = ["T1_CODES", "T2_CODES", "T1Code", "T2Code", "SPEAKER_OF", "group_of",
           "codebook", "normalise_code", "NAME_TO_CODE"]
