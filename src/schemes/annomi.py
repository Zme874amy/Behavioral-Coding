"""AnnoMI utterance attributes (Wu et al. 2023, Sec. 4)."""

CLIENT_TALK = ["change", "neutral", "sustain"]
MAIN_BEHAVIOUR = ["question", "input", "reflection", "other"]

HANDBOOK = {
    "counsellor": ["question:open", "question:closed", "reflection:simple", "reflection:complex",
                   "input:information", "input:advice", "input:options", "input:negotiation/goal-setting",
                   "main:question", "main:input", "main:reflection", "main:other"],
    "client": ["change", "neutral", "sustain"],
    "notes": "Question/Input/Reflection are separate attributes that can co-occur in one utterance; "
             "a single Main Behaviour is chosen per utterance.",
}
