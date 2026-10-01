"""FROZEN: the v2 generation prompt and ontology (scenario-conditioned windows).

RECONSTRUCTION NOTICE. `generate.py` and `ontology.py` were overwritten in place on
2026-09-26 and `src/synth/` had never been committed, so there is no git history.
This module restores both verbatim from the rendered output captured while the code
was live. The ontology value lists were independently confirmed against the data
they produced: `data/synth/raw/Qwen2_5-32B-Instruct-AWQ_proto_seg.csv` retains
`domain` (14 values), `stage` (5), `affect` (7), `setting` (5), `session` (2),
`register` (5), `style` (3) and `focus_codes` per window.

Do not edit: add `prompts_v4.py` instead.

What v2 fixed, and what it did not. It cleared the v1 mode collapse (Self-BLEU
0.199 -> 0.047, exact duplicates 10.9% -> 4.1%, distinct-2 0.348 -> 0.549) and,
once the segmentation stage was added, matched the real median utterance length of
8 words. What it did not have was provenance: every value below came from model
pretraining or author judgement rather than a cited source, which is why v3
replaced `stage` (Transtheoretical Model, uncited) with the Readiness Ruler from
the AutoMISC thesis section 2.3.1, and replaced the invented affect/setting/
register/style enums with SpeechDialogueFactory's dialogue-script components.
"""
from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel

TEMPERATURE = 1.0
TURNS = 10
FOCUS_PER_WINDOW = 4
EXEMPLARS_PER_CODE = 3
RARE_CODES = ["SU", "EC", "AF", "GI", "TS+", "AC-", "RF", "ST", "WA", "CO"]
RARE_WEIGHT = 3.0

# --- the v2 ontology, as run (values confirmed against the generated data) -----
DOMAINS: List[str] = [
    "quitting smoking", "cutting down alcohol", "taking medication as prescribed",
    "managing type-2 diabetes through diet", "getting regular exercise",
    "improving sleep habits", "reducing gambling", "quitting vaping",
    "managing chronic pain without over-relying on opioids",
    "sticking with physiotherapy after an injury",
    "engaging with mental-health support", "losing weight",
    "reducing screen time", "attending regular health check-ups",
]
STAGES: List[str] = [            # Transtheoretical Model -- uncited, removed in v3
    "precontemplation (not seeing a problem yet)",
    "contemplation (weighing it up, genuinely torn)",
    "preparation (starting to plan a change)",
    "action (recently started changing)",
    "maintenance (holding a change, worried about slipping)",
]
AFFECTS: List[str] = ["ambivalent", "defensive", "resigned and flat",
                      "cautiously hopeful", "irritated at being asked",
                      "embarrassed", "distracted and hurried"]
SETTINGS: List[str] = ["a GP clinic appointment", "a telehealth phone call",
                       "a community health drop-in",
                       "a hospital bedside conversation",
                       "a pharmacy consultation room"]
SESSIONS: List[str] = ["a first session", "a follow-up session"]
REGISTERS: List[str] = ["terse, answering in a few words",
                        "rambling, going off on tangents",
                        "disfluent, with false starts and fillers",
                        "guarded, giving little away", "talkative and candid"]
STYLES: List[str] = ["warm and unhurried", "brisk and practical",
                     "matter-of-fact and clinical"]

CONFUSABLE: Dict[str, str] = {
    "SU": "AF", "AF": "SU", "CR": "SR", "SR": "CR", "RF": "CR", "EC": "SU",
    "ADP": "ADW", "ADW": "ADP", "RCP": "RCW", "RCW": "RCP", "GI": "ADP",
    "OQ": "CQ", "CQ": "OQ", "FA": "FI", "FI": "FA", "ST": "GI", "DI": "ADW",
    "C+": "N", "D+": "C+", "AB+": "R+", "R+": "N+", "N+": "D+", "AC+": "TS+",
    "C-": "N", "D-": "C-", "AB-": "R-", "R-": "N-", "N-": "D-", "AC-": "TS-",
    "TS+": "AC+", "TS-": "AC-",
}


class ScenarioBrief(BaseModel):
    client_persona: str
    client_wants: str
    client_fears: str
    tension: str
    arc: str


class DialogueTurn(BaseModel):
    speaker: str
    text: str
    t1: str          # free-form in v2; this is what let invalid codes through
    t2: str


class DialogueWindow(BaseModel):
    turns: List[DialogueTurn]


def describe_seed(domain: str, stage: str, affect: str, setting: str,
                  session: str, register: str, style: str) -> str:
    return (
        f"The conversation takes place during {setting}. It is {session}. "
        f"The topic is {domain}. The client is at the {stage} stage and "
        f"comes across as {affect}; they speak in a way that is {register}. "
        f"The counsellor's manner is {style}."
    )


REGISTER_RULE = (
    "CRITICAL -- match the register of real transcribed counselling speech:\n"
    "- Real utterances are SHORT: the median is about 8 words. Most turns are one "
    "short sentence or a fragment.\n"
    "- Include natural spoken features: backchannels ('yeah', 'mm hm', 'okay', "
    "'right'), false starts, self-corrections, trailing off, incomplete sentences.\n"
    "- Do NOT write polished, articulate, therapist-textbook prose. Do NOT make "
    "every counsellor line a perfectly-formed reflection.\n"
    "- Vary sentence openings; avoid repeating the same stock phrasings.\n"
)


def build_brief_messages(seed_description: str) -> List[dict]:
    system = (
        "You are an expert in Motivational Interviewing (MI). You design brief, "
        "realistic clinical scenarios that will be used to script a counselling "
        "excerpt. Be specific and concrete; avoid generic filler."
    )
    user = (
        f"{seed_description}\n\n"
        "Write a short scenario brief as JSON with these fields:\n"
        "- client_persona: one or two sentences about who this person is (age band, "
        "life situation, how they came to be here). Make them a specific individual.\n"
        "- client_wants: what they actually want, in their own terms\n"
        "- client_fears: what worries them about changing\n"
        "- tension: the specific ambivalence at the heart of this conversation\n"
        "- arc: how the excerpt should progress in a few beats\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_window_messages(seed_description: str, brief: ScenarioBrief,
                          focus: List[str], variant: str, codebook: str,
                          exemplar_block: str, turns: int = TURNS) -> List[dict]:
    """The v2 window prompt. `codebook` was the raw two-speaker spec text."""
    if variant == "boundary":
        pairs = ", ".join(f"{c} (easily confused with {CONFUSABLE.get(c, '?')})" for c in focus)
        focus_rule = (
            f"This excerpt must contain utterances coded: {pairs}.\n"
            "IMPORTANT: make these instances genuinely BORDERLINE -- the kind a "
            "trained coder would have to think about, sitting close to the boundary "
            "with the confusable code named above. Do not write textbook-clear "
            "examples. Label each with the code you believe is actually correct."
        )
    else:
        focus_rule = (
            f"This excerpt must contain clear instances of these codes: {', '.join(focus)}.\n"
            "They should arise naturally from the conversation, not be forced in."
        )

    system = (
        "You are an expert Motivational Interviewing coder and dialogue writer. "
        "You write realistic MI counselling excerpts and annotate every utterance "
        f"with MISC 2.5 codes.\n\n{codebook}\n\n{REGISTER_RULE}"
    )
    user = (
        f"SCENARIO\n{seed_description}\n\n"
        f"BRIEF\n- Client: {brief.client_persona}\n- Wants: {brief.client_wants}\n"
        f"- Fears: {brief.client_fears}\n- Tension: {brief.tension}\n- Arc: {brief.arc}\n\n"
        f"TASK\nWrite a {turns}-turn excerpt of this session, alternating counsellor "
        "and client (an utterance = one thought unit; split multi-part turns).\n"
        f"{focus_rule}{exemplar_block}\n\n"
        "Return JSON: {\"turns\": [{\"speaker\": \"counsellor\"|\"client\", "
        "\"text\": ..., \"t1\": <Tier-1 code>, \"t2\": <Tier-2 code>}, ...]}\n"
        "Every turn must carry the t1/t2 codes valid for that speaker. Do not "
        "mention MISC, codes, or coding inside the dialogue text itself."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


__all__ = ["build_brief_messages", "build_window_messages", "describe_seed",
           "ScenarioBrief", "DialogueTurn", "DialogueWindow", "REGISTER_RULE",
           "DOMAINS", "STAGES", "AFFECTS", "SETTINGS", "SESSIONS", "REGISTERS",
           "STYLES", "CONFUSABLE", "RARE_CODES", "TEMPERATURE", "TURNS"]
