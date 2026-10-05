"""MISC 2.5 as used in this project (AutoMISC's variant).

Two-tier: T1 is AutoMISC's grouping (CRL/SRL/IMC/IMI/Q/O for counsellors,
C/S/N for clients), T2 the fine code. The manual's own summary groups are
MICO/MIIN; AutoMISC's T1 groups are not in the manual. AC+/- is an AutoMISC
extension (thesis footnote 1), not a MISC 2.5 code.
"""
from typing import Dict, List, Optional

COUNSELLOR_GROUPS: Dict[str, List[str]] = {
    "CRL": ["CR", "AF", "SU", "RF", "EC"],
    "SRL": ["SR"],
    "IMC": ["ADP", "RCP", "GI"],
    "IMI": ["ADW", "CO", "DI", "RCW", "WA"],
    "Q": ["OQ", "CQ"],
    "O": ["FA", "FI", "ST"],
}
CLIENT_GROUPS: Dict[str, List[str]] = {
    "C": ["O+", "D+", "AB+", "R+", "N+", "C+", "AC+", "TS+"],
    "S": ["O-", "D-", "AB-", "R-", "N-", "C-", "AC-", "TS-"],
    "N": ["N"],
}
GROUPS = {"counsellor": COUNSELLOR_GROUPS, "client": CLIENT_GROUPS}

COUNSELLOR_T1: List[str] = list(COUNSELLOR_GROUPS)
CLIENT_T1: List[str] = list(CLIENT_GROUPS)
COUNSELLOR_T2: List[str] = [c for cs in COUNSELLOR_GROUPS.values() for c in cs]
CLIENT_T2: List[str] = [c for cs in CLIENT_GROUPS.values() for c in cs]

CHANGE_TALK_T2: List[str] = list(CLIENT_GROUPS["C"])
SUSTAIN_TALK_T2: List[str] = list(CLIENT_GROUPS["S"])

# The v1 prompt specs and flat.j2 spell four IMI codes the long way. Parsers map
# them back; prompt set v2 uses the canonical codes (docs/RERUN_PLAN.md P2).
LABEL_ALIASES: Dict[str, str] = {"ADWP": "ADW", "RCWP": "RCW", "CON": "CO", "DIR": "DI"}

# Our client codes vs the handbook's names: N is Follow/Neutral (FN), AB is Ability (A).
HANDBOOK_NAMES: Dict[str, str] = {"N": "FN", **{f"AB{v}": f"A{v}" for v in "+-"}}
EXTENSIONS: Dict[str, str] = {
    "AC+": "AutoMISC addition (Activation; thesis footnote 1 cites Miller & Rollnick, Motivational Interviewing, "
           "4th ed. 2023, mobilising change talk); not in MISC 2.5 or MISC 2.1. "
           "In the manual, 'offering alternatives' is Commitment (C+).",
    "AC-": "AutoMISC addition (Activation-); not in MISC 2.5.",
}
MICO = {"AF", "ADP", "EC", "RCP", "SU", "OQ", "SR", "CR"}   # manual p.47 (sMICO incl. OQ + reflections)
MIIN = {"ADW", "CO", "DI", "RCW", "WA"}                       # manual pp.47-48

# Real MISC 2.5 classes with no gold example in any real dataset (DATASETS.md section 3).
NEVER_SEEN_REAL = ("RCP", "TS-")

HANDBOOK = {
    "counsellor": ["ADP", "ADW", "AF", "CO", "DI", "EC", "FA", "FI", "GI", "OQ", "CQ",
                   "RCP", "RCW", "SR", "CR", "RF", "SU", "ST", "WA"],
    "client": ["FN"] + [f"{c}{v}" for c in ("C", "R", "D", "A", "N", "TS", "O") for v in "+-"],
    "either": ["NC"],
    "globals": ["Acceptance", "Empathy", "Direction", "Autonomy Support", "Collaboration", "Evocation",
                "Self-Exploration (client)"],
    "notes": "SR/CR require a valence (+/-/0/+-) in the manual; Ask is part of Follow/Neutral (FN).",
}


def t1_codes(speaker: str) -> List[str]:
    return COUNSELLOR_T1 if speaker == "counsellor" else CLIENT_T1


def t2_codes(speaker: str) -> List[str]:
    return COUNSELLOR_T2 if speaker == "counsellor" else CLIENT_T2


def t1_of(speaker: str, t2: str) -> Optional[str]:
    """The T1 group of a T2 code for that speaker, or None if it is not one of ours."""
    for g, cs in GROUPS[speaker].items():
        if t2 in cs:
            return g
    return None


def canonical(code: str) -> str:
    """Resolve a v1 long-form alias (ADWP -> ADW, ...) to the canonical code."""
    return LABEL_ALIASES.get(code, code)


# Full names as the prompts spell them (spec YAML v2 / t1 templates), plus the
# few variants models produce. Used only to resolve an answer written as a name
# ("Affirm", "Simple Reflection") onto its code; a name shared by several codes
# (client "Desire" = D+ or D-) resolves only when the caller's allowed set makes
# it unique (the T2 call allows one T1 group).
CODE_NAMES: Dict[str, List[str]] = {
    "CR": ["complex reflection"], "AF": ["affirm", "affirmation"], "SU": ["support"],
    "RF": ["reframe"], "EC": ["emphasize control", "emphasise control"],
    "SR": ["simple reflection"],
    "ADP": ["advise with permission"], "RCP": ["raise concern with permission"],
    "GI": ["giving information", "give information"],
    "ADW": ["advise without permission"], "CO": ["confront"], "DI": ["direct"],
    "RCW": ["raise concern without permission"], "WA": ["warn"],
    "OQ": ["open question"], "CQ": ["closed question"],
    "FA": ["facilitate"], "FI": ["filler"], "ST": ["structure"],
    "N": ["neutral", "follow/neutral"],
    **{f"{c}{s}": [n, f"{n}{s}"] for c, n in (("D", "desire"), ("AB", "ability"), ("R", "reasons"),
                                              ("N", "need"), ("C", "commitment"), ("AC", "activation"),
                                              ("TS", "taking steps"), ("O", "other")) for s in "+-"},
    "CRL": ["c-reflective", "complex reflective"], "SRL": ["s-reflective", "simple reflective"],
    "IMC": ["imperative-mico"], "IMI": ["imperative-miin"], "Q": ["question"],
}


def code_from_name(text: str, allowed) -> Optional[str]:
    """The allowed code whose full name `text` is, if exactly one matches."""
    t = " ".join(str(text).lower().replace("_", " ").split()).strip(" .:*`\"'")
    hits = [c for c in allowed if t in CODE_NAMES.get(c, ())]
    return hits[0] if len(hits) == 1 else None
