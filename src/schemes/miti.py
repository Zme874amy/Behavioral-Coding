"""MITI 4.2.1 behaviour codes (counsellor only; clients are not coded)."""
from typing import Dict, List

CODES: List[str] = ["GI", "Persuade", "PwP", "Q", "SR", "CR", "AF", "Seek", "Emphasize", "Confront"]
NAMES: Dict[str, str] = {
    "GI": "Giving information", "Persuade": "Persuade", "PwP": "Persuade with permission",
    "Q": "Question (open/closed not split)", "SR": "Simple reflection", "CR": "Complex reflection",
    "AF": "Affirm", "Seek": "Seeking collaboration", "Emphasize": "Emphasizing autonomy", "Confront": "Confront",
}
# MITI 4.2.1 summary groupings used for session-level scores.
RELATIONAL = {"AF", "Seek", "Emphasize"}
TECHNICAL_REFLECTIONS = {"SR", "CR"}
MI_ADHERENT = {"Seek", "AF", "Emphasize"}
MI_NON_ADHERENT = {"Confront", "Persuade"}
# Transcript conventions in the CASAA coded transcripts (not manual codes).
TRANSCRIPT_CODES = {"NC": "not coded (structure, greeting, facilitate)",
                    "SAME": "continues the previous coded utterance"}

HANDBOOK = {
    "counsellor": ["GI", "Persuade", "Persuade with Permission", "Q", "SR", "CR", "AF", "Seek",
                   "Emphasize", "Confront"],
    "client": [],
    "globals": ["Cultivating Change Talk", "Softening Sustain Talk", "Partnership", "Empathy"],
    "notes": "Clients are not coded; Q is not split into open/closed; uncodable utterances get no code.",
}
