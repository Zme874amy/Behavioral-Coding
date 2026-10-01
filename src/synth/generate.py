"""Scenario-conditioned generation of labelled MI dialogue windows.

Pipeline, with each stage traced to its source:

  1. scenario seed            `synth.ontology` -- topic (corpus-measured) +
                              Readiness Ruler (thesis section 2.3.1, Table 2.2)
  2. dialogue script          SpeechDialogueFactory section 3.1.2: Scene /
                              Narrative Flow / Character Behaviors / Emotional
                              Progression. SDF reports this intermediate
                              representation "significantly improves coherence".
  3. rule (optional)          SocialDial section 3.1 rule form
                              `IF f1=k AND ... THEN behaviour`, generated first
                              and prefixed to the dialogue prompt (their section
                              3.3 two-step). Instantiated with MISC codes rather
                              than social norms.
  4. dialogue simulation      SDF section 3.1.3: the whole window in one pass,
                              structured JSON, every utterance annotated.

Codes are constrained to the valid MISC vocabulary via `synth.codes` (SDF's
"constrained decoding"). v2 typed them as free-form `str`, which let the model
emit spec-YAML abbreviations and full names that were then silently dropped.

Usage:
    PYTHONPATH=src python -m synth.generate --variant proto --mix train \
        --windows 50 --model <served-name> --out data/synth/raw/new.csv
    PYTHONPATH=src python -m synth.generate --dry-run --windows 3 --out /tmp/s.csv
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from synth.codes import SPEAKER_OF, T1Code, T2Code, codebook, group_of
from synth.ontology import ScenarioSeed, sample_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
HLQC = REPO_ROOT / "data/manual/HLQC_balanced_manual.csv"

# Codes with the least gold support; docs name SU/EC/AF/GI as the starved tail.
RARE_DEFAULT = ["SU", "EC", "AF", "GI", "TS+", "AC-", "RF", "ST", "WA", "CO", "RCW", "DI"]

# Sibling confusions for the `boundary` variant. Same-Tier-1 pairs are taken from
# the group structure in `automisc_ft.data`; the permission pairs (ADP/ADW,
# RCP/RCW) are the distinction MISC 2.5 makes explicit for those codes.
CONFUSABLE: Dict[str, str] = {
    "SU": "AF", "AF": "SU", "CR": "SR", "SR": "CR", "RF": "CR", "EC": "SU",
    "ADP": "ADW", "ADW": "ADP", "RCP": "RCW", "RCW": "RCP", "GI": "ADP",
    "OQ": "CQ", "CQ": "OQ", "FA": "FI", "FI": "FA", "ST": "GI", "DI": "ADW",
    "CO": "RCW", "WA": "RCW",
    "C+": "N", "D+": "C+", "AB+": "R+", "R+": "N+", "N+": "D+", "AC+": "TS+",
    "C-": "N", "D-": "C-", "AB-": "R-", "R-": "N-", "N-": "D-", "AC-": "TS-",
    "TS+": "AC+", "TS-": "AC-", "O+": "R+", "O-": "R-", "N": "O+",
}


class DialogueScript(BaseModel):
    """SDF section 3.1.2 blueprint. Field names are the paper's."""
    scene: str                    # environmental context and character relationships
    narrative_flow: str           # dialogue progression through distinct phases
    character_behaviors: str      # interaction patterns and communication styles
    emotional_progression: str    # affective development with natural transitions


class MIRule(BaseModel):
    """SocialDial section 3.1 rule form, MISC vocabulary."""
    rule: str


class DialogueTurn(BaseModel):
    speaker: Literal["counsellor", "client"]
    text: str
    t1: T1Code
    t2: T2Code


class DialogueWindow(BaseModel):
    turns: List[DialogueTurn]


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


# ------------------------------------------------------------------ helpers
def load_real_exemplars() -> Dict[str, List[str]]:
    """Real HLQC utterances per T2 code -- style anchors for register matching."""
    try:
        df = pd.read_csv(HLQC)
    except Exception:
        return {}
    out: Dict[str, List[str]] = {}
    for code, grp in df.dropna(subset=["t2_label_GT"]).groupby("t2_label_GT"):
        out[str(code)] = [str(t).strip() for t in grp["utt_text"].dropna() if str(t).strip()][:200]
    return out


def sample_focus_codes(rng: random.Random, n: int, rare: List[str]) -> List[str]:
    codes = list(SPEAKER_OF)
    weights = [3.0 if c in set(rare) else 1.0 for c in codes]
    picked: List[str] = []
    while len(picked) < n and codes:
        c = rng.choices(codes, weights=weights, k=1)[0]
        if c not in picked:
            picked.append(c)
    return picked


# ------------------------------------------------------------------ prompts
def build_script_messages(seed: ScenarioSeed) -> List[dict]:
    """Stage 2 -- SDF section 3.1.2 dialogue script."""
    system = (
        "You are an expert in Motivational Interviewing. You write the blueprint for "
        "a counselling excerpt before it is scripted. Be specific and concrete; the "
        "client should be a particular person, not a type."
    )
    user = (
        f"{seed.describe()}\n\n"
        "Write the blueprint as JSON with exactly these fields:\n"
        "- scene: the environmental context and the relationship between counsellor and client\n"
        "- narrative_flow: how the excerpt progresses, through distinct phases\n"
        "- character_behaviors: the interaction patterns and communication styles of each speaker\n"
        "- emotional_progression: how affect develops, with natural transitions\n\n"
        "Let the Readiness Ruler scores drive this: a client who rates importance high "
        "but confidence low is not the same person as one who rates both low."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_rule_messages(seed: ScenarioSeed, focus: List[str]) -> List[dict]:
    """Stage 3 -- SocialDial rule, stated over Readiness Ruler + MISC codes."""
    system = (
        "You state a single mechanism linking what the counsellor does to how the "
        "client responds, in Motivational Interviewing.\n\n" + codebook()
    )
    user = (
        f"{seed.describe()}\n\n"
        f"Codes this excerpt should exercise: {', '.join(focus)}.\n\n"
        "Write ONE rule in exactly this form, as JSON {\"rule\": \"...\"}:\n"
        "  IF <client factor> = <value> AND <counsellor behaviour> = <MISC code>, "
        "THEN <how the client responds> = <MISC code>.\n\n"
        "Use the Readiness Ruler dimensions (importance / confidence / readiness) for "
        "the client factor, and MISC code abbreviations for the behaviours. State a "
        "mechanism a clinician would recognise, not a restatement of the codes."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_window_messages(seed: ScenarioSeed, script: DialogueScript, focus: List[str],
                          variant: str, exemplars: Dict[str, List[str]], n_ex: int,
                          rng: random.Random, turns: int,
                          rule: Optional[str] = None) -> List[dict]:
    """Stage 4 -- SDF section 3.1.3 single-pass simulation, every turn annotated."""
    ex_lines = []
    for c in focus:
        pool = exemplars.get(c, [])
        if pool:
            picked = rng.sample(pool, min(n_ex, len(pool)))
            ex_lines.append(f"  {c} ({SPEAKER_OF.get(c, '?')}): " +
                            " | ".join(f'"{p}"' for p in picked))
    ex_block = ("\nReal examples of how these codes actually sound in transcripts "
                "(match this register, do not copy them):\n" + "\n".join(ex_lines)
                ) if ex_lines else ""

    if variant == "boundary":
        pairs = ", ".join(f"{c} (easily confused with {CONFUSABLE.get(c, '?')})" for c in focus)
        focus_rule = (
            f"The excerpt must contain utterances coded: {pairs}.\n"
            "IMPORTANT: make these instances genuinely BORDERLINE -- the kind a trained "
            "coder would have to think about, sitting close to the boundary with the "
            "confusable code named above. Label each with the code you believe is correct."
        )
    else:
        focus_rule = (f"The excerpt must contain clear instances of these codes: "
                      f"{', '.join(focus)}.\nThey should arise naturally, not be forced in.")

    rule_block = (f"RULE\n{rule}\n\nThe excerpt must realise this rule at least once.\n\n"
                  if rule else "")

    system = ("You are an expert Motivational Interviewing coder and dialogue writer. "
              "You write realistic MI counselling excerpts and annotate every utterance "
              f"with MISC 2.5 codes.\n\n{codebook()}\n\n{REGISTER_RULE}")
    user = (
        f"{rule_block}"
        f"SCENARIO\n{seed.describe()}\n\n"
        f"SCRIPT\n  Scene: {script.scene}\n  Narrative Flow: {script.narrative_flow}\n"
        f"  Character Behaviors: {script.character_behaviors}\n"
        f"  Emotional Progression: {script.emotional_progression}\n\n"
        f"TASK\nWrite a {turns}-turn excerpt of this session, alternating counsellor and "
        "client (an utterance = one thought unit; split multi-part turns).\n"
        f"{focus_rule}{ex_block}\n\n"
        'Return JSON: {"turns": [{"speaker": "counsellor"|"client", "text": ..., '
        '"t1": <Tier-1 code>, "t2": <Tier-2 code>}, ...]}\n'
        "Every turn must carry the t1/t2 codes valid for that speaker. Do not mention "
        "MISC, codes, or coding inside the dialogue text itself."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ------------------------------------------------------------------ teacher
def call_teacher(messages, model, provider, temperature, base_url, schema):
    from components.utils import call_chat_model

    kwargs = {"base_url": base_url} if base_url else {}
    res = call_chat_model(messages=messages, model=model, provider=provider,
                          temperature=temperature, response_format=schema, **kwargs)
    return res if isinstance(res, BaseModel) else schema(**res)


def _canned(seed: ScenarioSeed, focus: List[str], turns: int):
    script = DialogueScript(scene="[dry-run scene]", narrative_flow="[flow]",
                            character_behaviors="[behaviors]",
                            emotional_progression="[progression]")
    ts = []
    for i in range(turns):
        c = focus[i % len(focus)]
        spk = SPEAKER_OF.get(c, "counsellor")
        ts.append(DialogueTurn(speaker=spk, text=f"[dry-run {spk} turn {i} for {c}]",
                               t1=group_of(c, spk) or "O", t2=c))
    return script, DialogueWindow(turns=ts)


# ------------------------------------------------------------------ driver
def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", choices=["proto", "boundary"], default="proto")
    ap.add_argument("--mix", choices=["train", "eval"], default="train",
                    help="topic mix: HLQC (train) or MIV6.3A (eval) proportions")
    ap.add_argument("--rule", action="store_true",
                    help="generate a SocialDial-style rule and condition on it")
    ap.add_argument("--windows", type=int, default=50)
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--focus-per-window", type=int, default=4)
    ap.add_argument("--exemplars", type=int, default=3)
    ap.add_argument("--rare-codes", nargs="*", default=RARE_DEFAULT)
    ap.add_argument("--model", default="Qwen2.5-32B-Instruct-AWQ")
    ap.add_argument("--provider", default="vllm_server")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    seeds = sample_seeds(args.windows, mix=args.mix, seed=args.seed)
    exemplars = load_real_exemplars() if not args.dry_run else {}

    rows: List[dict] = []
    uid = 0
    n_ok = n_fail = 0
    for i, seed in enumerate(tqdm(seeds, desc=f"windows[{args.variant}/{args.mix}]", unit="win")):
        focus = sample_focus_codes(rng, args.focus_per_window, list(args.rare_codes))
        rule_text = None
        try:
            if args.dry_run:
                script, window = _canned(seed, focus, args.turns)
            else:
                script = call_teacher(build_script_messages(seed), args.model,
                                      args.provider, args.temperature, args.base_url,
                                      DialogueScript)
                if args.rule:
                    rule_text = call_teacher(build_rule_messages(seed, focus), args.model,
                                             args.provider, args.temperature,
                                             args.base_url, MIRule).rule
                window = call_teacher(
                    build_window_messages(seed, script, focus, args.variant, exemplars,
                                          args.exemplars, rng, args.turns, rule_text),
                    args.model, args.provider, args.temperature, args.base_url,
                    DialogueWindow)
        except Exception as e:
            n_fail += 1
            print(f"  window {i} failed: {type(e).__name__}: {str(e)[:120]}")
            continue
        n_ok += 1
        conv_id = f"synth:{args.variant}:{args.mix}:{i}"
        for j, t in enumerate(window.turns):
            rows.append({
                "conv_id": conv_id, "speaker": t.speaker,
                "corp_utt_idx": uid, "conv_vol_idx": j, "conv_utt_idx": j,
                "vol_text": t.text, "utt_text": t.text,
                "t1_proposed": t.t1, "t2_proposed": t.t2,
                "t1_label_GT": None, "t2_label_GT": None,
                "synth_target": True, "variant": args.variant, "mix": args.mix,
                "rule": rule_text or "", "focus_codes": "|".join(focus),
                "scene": script.scene, "narrative_flow": script.narrative_flow,
                "character_behaviors": script.character_behaviors,
                "emotional_progression": script.emotional_progression,
                **seed.as_dict(),
            })
            uid += 1

    out = Path(args.out) if Path(args.out).is_absolute() else REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nvariant={args.variant} mix={args.mix} rule={args.rule}: {n_ok} windows ok, "
          f"{n_fail} failed -> {len(rows)} candidate utterances -> {out}")


if __name__ == "__main__":
    main()
