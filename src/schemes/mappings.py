"""Cross-scheme mappings. Unmapped codes return None; nothing defaults silently."""
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------- Welivita
# Welivita & Pu (2022) Table 1: 15 listener labels adapted from MITI 2.0 / 4.2.1.
WELIVITA_HANDBOOK = {
    "counsellor": ["Closed Question", "Open Question", "Simple Reflection", "Complex Reflection",
                   "Give Information", "Advise with Permission", "Affirm", "Emphasize Autonomy", "Support",
                   "Advise without Permission", "Confront", "Direct", "Warn", "Self-Disclose", "Other"],
    "client": [],
    "notes": "A MITI variant: labels adapted from MITI 2.0 and 4.2.1 (open/closed questions, Direct, Warn, Support from "
             "earlier MITI versions) plus Self-Disclose and Other. Maps to MITI 4.2.1 via WELIVITA_TO_MITI. Seekers are not coded.",
}

# Welivita label -> (MITI 4.2.1 code, exact|approx|none, manual basis).
# exact 57% / approximate 34% / none 8% of listener labels (docs/experiments/2026-10-05-split-review.md section 12).
WELIVITA_TO_MITI: Dict[str, Tuple[Optional[str], str, str]] = {
    "Closed Question": ("Q", "exact", "MITI 4.2.1 does not split open/closed"),
    "Open Question": ("Q", "exact", "MITI 4.2.1 does not split open/closed"),
    "Simple Reflection": ("SR", "exact", ""),
    "Complex Reflection": ("CR", "exact", ""),
    "Affirm": ("AF", "exact", "MITI 4.2.1 Affirm is stricter than earlier versions (p.26)"),
    "Emphasize Autonomy": ("Emphasize", "exact", ""),
    "Confront": ("Confront", "exact", ""),
    "Advise with Permission": ("PwP", "exact", "E.4.c: permission asked/given or autonomy-supportive preface"),
    "Advise without Permission": ("Persuade", "exact", "E.4.b: advice/suggestions without autonomy emphasis"),
    "Warn": ("Confront", "exact", "E.4.g.2 lists 'warning' under Confront"),
    "Support": ("NC", "exact", "p.26: statements of support are no longer coded ('I know it's really hard to stop smoking')"),
    "Other": ("NC", "exact", "F: greetings and off-topic statements are not coded"),
    "Give Information": ("GI", "approx", "Welivita GI includes opinions; MITI codes unsolicited opinions as Persuade (E.4.b)"),
    "Direct": ("Persuade", "approx", "imperatives are advice (Persuade); with disapproval they are Confront"),
    "Self-Disclose": (None, "none", "Persuade only when used to persuade (E.4.b), otherwise not coded: needs context"),
}

# Welivita label -> MISC 2.5 counsellor T2 (the corpus's own MISC reading; used for
# weak supervision and MISC-space transfer). Self-Disclose and Other have no MISC code.
WELIVITA_TO_MISC_T2: Dict[str, str] = {
    "Give Information": "GI",
    "Advise without Permission": "ADW",
    "Advise with Permission": "ADP",
    "Complex Reflection": "CR",
    "Simple Reflection": "SR",
    "Support": "SU",
    "Affirm": "AF",
    "Closed Question": "CQ",
    "Open Question": "OQ",
    "Direct": "DI",
    "Confront": "CO",
    "Emphasize Autonomy": "EC",
    "Warn": "WA",
}

# ---------------------------------------------------------------- MISC <-> MITI
# MISC 2.5 counsellor T2 -> MITI 4.2.1. "NC" = MITI does not code it.
MISC_TO_MITI: Dict[str, str] = {
    "SR": "SR", "CR": "CR", "OQ": "Q", "CQ": "Q", "GI": "GI", "AF": "AF", "EC": "Emphasize",
    "CO": "Confront", "FA": "NC", "FI": "NC", "ST": "NC", "ADW": "Persuade", "RCW": "Persuade",
    "WA": "Persuade", "DI": "Persuade", "ADP": "PwP", "RCP": "PwP",
    "SU": "NC",   # MITI 4.2.1 p.26: statements of support are no longer coded
}
# MISC codes with no MITI counterpart without context (returned as None).
MISC_TO_MITI_UNMAPPED: Dict[str, str] = {
    "RF": "Reframe: MITI 4.2.1 has no reframe code; usually a complex reflection, sometimes Persuade",
}

# Canonical MITI code (as parsed from the CASAA transcripts) -> (MISC T1, MISC T2 or None).
# Q and NC fix only the T1 group: MITI does not split open/closed or the O codes.
MITI_TO_MISC: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    "SR": ("SRL", "SR"), "CR": ("CRL", "CR"), "AF": ("CRL", "AF"),
    "Emphasize": ("CRL", "EC"), "GI": ("IMC", "GI"), "Confront": ("IMI", "CO"),
    "Q": ("Q", None), "NC": ("O", None),
}

# ---------------------------------------------------------------- MISC -> AnnoMI
# MISC counsellor T2 -> AnnoMI main behaviour (AnnoMI 'open' is not MISC OQ; compare questions at T1).
MISC_TO_ANNOMI: Dict[str, str] = {
    "OQ": "question", "CQ": "question", "SR": "reflection", "CR": "reflection",
    "GI": "therapist_input", "ADP": "therapist_input", "ADW": "therapist_input",
}


def misc_to_miti(code: str) -> Optional[str]:
    """MITI 4.2.1 code for a MISC counsellor T2 code, or None if it has none."""
    return MISC_TO_MITI.get(code)


def welivita_to_miti(label: str) -> Optional[str]:
    return WELIVITA_TO_MITI.get(label, (None,))[0]


def miti_to_misc(code: str) -> Tuple[Optional[str], Optional[str]]:
    return MITI_TO_MISC.get(code, (None, None))
