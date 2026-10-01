"""Tools the MISC agent can call, plus the deterministic gate.

Everything here reuses the taxonomy/codebook already in the tree, so the agent's
"world" can never drift from what the rest of the pipeline believes:
  - ``define`` -> the codebook definition text (prompts/specs yaml), keyed by the
    T1 group, so it returns a code together with its SIBLINGS. That is exactly the
    material the critique step needs to separate e.g. SU from AF.
  - ``check``  -> the deterministic hierarchy + vocabulary gate, from
    automisc_ft.data. This is code, not the model: it guarantees a well-formed
    hierarchical label regardless of how the reasoning went.
  - ``retrieve`` -> nearest labelled HLQC examples, rendered as a compact text
    observation, delegating selection to the existing RetrievalFewshotProvider.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from automisc_ft.data import (
    CLIENT_GROUPS,
    COUNSELLOR_GROUPS,
    t1_codes_for_speaker,
    t2_codes_for_group,
    t2_codes_for_speaker,
)
from components.prompts.loader import load_spec

_SPEC_CACHE: dict = {}


def _groups(speaker: str) -> dict:
    return COUNSELLOR_GROUPS if speaker == "counsellor" else CLIENT_GROUPS


def _spec(speaker: str) -> dict:
    if speaker not in _SPEC_CACHE:
        _SPEC_CACHE[speaker] = load_spec(speaker, "t2")
    return _SPEC_CACHE[speaker]


def parent_group(speaker: str, t2_code: str) -> Optional[str]:
    for g, codes in _groups(speaker).items():
        if t2_code in codes:
            return g
    return None


def define(speaker: str, code: str) -> str:
    """Codebook definition(s) for a code or a T1 group (with its siblings)."""
    spec = _spec(speaker)
    if code in spec:                       # `code` is a T1 group
        return f"Definitions for T1 group {code}:\n{spec[code]}"
    g = parent_group(speaker, code)
    if g and g in spec:
        return (f"{code} is a child of T1 group {g}. "
                f"Definitions for {g} (with sibling codes):\n{spec[g]}")
    return f"(no codebook definition found for '{code}')"


def check(speaker: str, t1: str, t2: str) -> Tuple[bool, str]:
    """Deterministic gate: is (t1, t2) a valid, in-vocab, hierarchical label?"""
    t1s = t1_codes_for_speaker(speaker)
    t2s = t2_codes_for_speaker(speaker)
    if t1 not in t1s:
        return False, f"T1 '{t1}' is not a valid {speaker} T1 code. Choose one of {t1s}."
    if t2 not in t2s:
        return False, f"T2 '{t2}' is not a valid {speaker} T2 code."
    children = t2_codes_for_group(speaker, t1)
    if t2 not in children:
        return False, (f"T2 '{t2}' is not a child of T1 '{t1}'. "
                       f"The valid children of {t1} are {children}.")
    return True, "ok"


def snap_to_hierarchy(speaker: str, t1: str, t2: str) -> Tuple[str, str]:
    """Last-resort coercion to a valid (t1, t2) when revises are exhausted.

    Keeps a valid T1; snaps T2 to a child of it (its own group if t2 is valid but
    mismatched, else the group's first child). Never invents an out-of-vocab code.
    """
    t1s = t1_codes_for_speaker(speaker)
    if t1 not in t1s:
        pg = parent_group(speaker, t2)
        t1 = pg if pg else t1s[0]
    children = t2_codes_for_group(speaker, t1)
    if t2 not in children:
        t2 = children[0] if children else t2
    return t1, t2


def retrieve(retriever, speaker: str, tier: str, t1_label: Optional[str],
             max_items: int = 6) -> str:
    """Nearest labelled examples as a compact observation string."""
    exs = retriever.retrieve_examples(speaker, tier, t1_label)[:max_items]
    if not exs:
        return "(no examples found)"
    key = "t1_label" if tier == "t1" else "t2_label"
    lines = [f'- "{e["utterance"]}" -> {e[key]}' for e in exs]
    header = (f"Nearest labelled {speaker} examples"
              + (f" within T1 group {t1_label}" if tier == "t2" and t1_label else "")
              + f" ({tier.upper()}):")
    return header + "\n" + "\n".join(lines)
