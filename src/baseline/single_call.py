"""Single-call (joint T1 + T2) format shared by the GRPO ladder.

The two-call pipeline in `automisc_ft.infer.TieredAnnotator` predicts T1 with
one adapter, then prompts a second adapter conditioned on that prediction. That
shape is awkward for RL: T2's reward depends on a T1 prediction that is itself
moving, so there are two coupled non-stationary problems instead of one.

Here the model emits a rationale and both labels in one generation:

    {"rationale": "...", "t1": "CRL", "t2": "CR"}

One rollout is therefore one sequence with one scalar reward covering both
tiers, and the prefill cost is roughly what the two calls cost together
(measured at ctx5: joint counsellor prompt ~4.3k tokens against 3.0k + 1.4k).

Every arm of the single-call ladder -- `sc_ft_bare`, `sc_ft_rat`, `sc_grpo` --
builds prompts through `build_messages`, so the only thing that differs between
them is the training signal, never the input the model sees.

Reward (`score_completion`), per rollout:

    +0.1  format     parses, both codes in the speaker's vocabulary, and (when
                     a rationale is expected) a rationale of at least
                     MIN_RATIONALE_WORDS words
    +0.1  hierarchy  the emitted t2 is a child of the emitted t1
    +0.4  t1 correct
    +0.4  t2 correct

optionally scaled by a rare-class weight. See `class_weights` for why that
scaling only works with `scale_rewards="none"`.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from automisc_ft.data import (
    CLIENT_GROUPS,
    COUNSELLOR_GROUPS,
    t1_codes_for_speaker,
    t2_codes_for_group,
    t2_codes_for_speaker,
)
from components.context import build_context_excerpt
from components.prompts.loader import render_prompt, render_user_prompt
from sft.eval import LABEL_ALIASES

# A rationale shorter than this does not count as one. Only t1/t2 carry most of
# the reward, so the cheapest degenerate policy is to emit `"rationale": ""` and
# collect the labels; this puts a floor under that. It is a floor, not a proof
# of reasoning -- the rationale-swap probe in `baseline.faithfulness` is what
# actually tests whether the rationale is load-bearing.
MIN_RATIONALE_WORDS = 5

REWARD_FORMAT = 0.1
REWARD_HIERARCHY = 0.1
REWARD_T1 = 0.4
REWARD_T2 = 0.4

UNKNOWN = "UNKNOWN"


# -----------------------------------------------------------------------------
# Prompt construction
# -----------------------------------------------------------------------------
def build_messages(
    df: pd.DataFrame,
    row_pos: int,
    context_mode: str,
    num_context_turns: int,
    emit_rationale: bool = True,
    fewshot_messages: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build the joint T1+T2 chat messages for one utterance.

    `emit_rationale` switches only the Output Format block of the template, so
    the taxonomy the model reads is byte-identical whether it is being asked
    for a rationale or not. That matters for the FT-Bare / FT-Rat contrast:
    the arms must differ in target shape alone.
    """
    row = df.iloc[row_pos]
    speaker = row["speaker"]
    transcript = build_context_excerpt(df, row_pos, context_mode, num_context_turns)
    system = render_prompt(
        speaker=speaker, structure="t12", emit_rationale=emit_rationale
    )
    user = render_user_prompt(
        transcript=transcript, speaker=speaker, utterance=row["utt_text"]
    )
    messages = [{"role": "system", "content": system}]
    if fewshot_messages:
        messages += fewshot_messages
    messages.append({"role": "user", "content": user})
    return messages


def build_target(t1: str, t2: str, rationale: Optional[str] = None) -> str:
    """The supervised completion target, in the shape the prompt asks for."""
    payload = {"t1": t1, "t2": t2}
    if rationale is not None:
        payload = {"rationale": rationale, "t1": t1, "t2": t2}
    return json.dumps(payload, ensure_ascii=False)


# -----------------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------------
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _normalise(tok: object) -> str:
    if not isinstance(tok, str):
        return ""
    return tok.strip("`*_:()[]{}\"' \t\n").strip().rstrip(".,;:")


def _resolve(cand: object, allowed: Sequence[str]) -> Optional[str]:
    """Map a raw model value onto the canonical vocabulary, or None.

    Three tolerances, all of which the two-call `automisc_ft.infer.parse_label`
    also grants. Matching its leniency is not cosmetic: if the two-call arms got
    credit for a given output shape, the single-call arms have to as well, or
    the format comparison measures parser strictness instead of the model.

      1. Exact match on the canonical code.
      2. Alias match. The T2 spec YAML writes the MIIN codes as
         ADWP/RCWP/CON/DIR while the label vocabulary uses ADW/RCW/CO/DI.
      3. Token scan inside the value, for the very common habit of answering
         with the full name, ``"Simple Reflection (SR)"``. The scan runs from
         the end because that is where the parenthesised code sits.
    """
    c = _normalise(cand)
    if not c:
        return None
    if c in allowed:
        return c
    alias = LABEL_ALIASES.get(c)
    if alias and alias in allowed:
        return alias

    for tok in reversed([t for t in re.split(r"[\s,;/|]+", str(cand)) if t]):
        t = _normalise(tok)
        if t in allowed:
            return t
        alias = LABEL_ALIASES.get(t)
        if alias and alias in allowed:
            return alias
    return None


def _field(text: str, key: str) -> Optional[str]:
    """Last `"key": value` occurrence, for output that is nearly-JSON."""
    hits = re.findall(
        rf'["\']?{key}["\']?\s*[:=]\s*["\']?([A-Za-z][A-Za-z0-9+\-]*)',
        text,
        flags=re.IGNORECASE,
    )
    return hits[-1] if hits else None


def _rationale_field(text: str) -> str:
    """Rationale text from output that is not valid JSON.

    Two shapes, tried in order: a quoted value (broken JSON that still quotes
    its strings) and an unquoted one (`rationale: the counsellor ...`, which a
    model drifting out of JSON tends to produce). The unquoted form has to stop
    at a newline, since without a closing quote there is nothing else to bound
    it. Missing this case would report a real rationale as absent and show up
    as a false instruction-compliance failure.
    """
    quoted = re.search(
        r'["\']?(?:rationale|explanation)["\']?\s*[:=]\s*["\'](.*?)["\']\s*(?:[,}]|$)',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if quoted:
        return quoted.group(1).strip()
    unquoted = re.search(
        r'["\']?(?:rationale|explanation)["\']?\s*[:=]\s*([^\n"\']+)',
        text,
        flags=re.IGNORECASE,
    )
    return unquoted.group(1).strip().rstrip(",}") if unquoted else ""


def parse_completion(text: str, speaker: str) -> Dict[str, object]:
    """Extract (rationale, t1, t2) from a generation.

    Returns `t1`/`t2` as canonical codes or `UNKNOWN`, plus `json_ok` recording
    whether the output was well-formed JSON. Generation is unconstrained, so
    malformed output is a real outcome that has to be measured rather than
    silently repaired -- but a model that emits valid labels inside slightly
    broken JSON should still be scored on the labels, hence the regex fallback.
    """
    t1_allowed = t1_codes_for_speaker(speaker)
    t2_allowed = t2_codes_for_speaker(speaker)
    out = {"rationale": "", "t1": UNKNOWN, "t2": UNKNOWN, "json_ok": False}
    if not text:
        return out

    cleaned = _FENCE.sub("", text).strip()
    match = _OBJECT.search(cleaned)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                out["json_ok"] = True
                rationale = obj.get("rationale") or obj.get("explanation") or ""
                out["rationale"] = rationale.strip() if isinstance(rationale, str) else ""
                out["t1"] = _resolve(obj.get("t1"), t1_allowed) or UNKNOWN
                out["t2"] = _resolve(obj.get("t2"), t2_allowed) or UNKNOWN
                if out["t1"] != UNKNOWN and out["t2"] != UNKNOWN:
                    return out
        except (json.JSONDecodeError, AttributeError):
            pass

    # Fallback for output that carries the right labels in the wrong wrapper.
    if out["t1"] == UNKNOWN:
        out["t1"] = _resolve(_field(cleaned, "t1"), t1_allowed) or UNKNOWN
    if out["t2"] == UNKNOWN:
        out["t2"] = _resolve(_field(cleaned, "t2"), t2_allowed) or UNKNOWN
    if not out["rationale"]:
        out["rationale"] = _rationale_field(cleaned)
    return out


def rationale_word_count(rationale: str) -> int:
    return len([w for w in re.split(r"\s+", rationale) if any(c.isalpha() for c in w)])


def is_hierarchically_consistent(speaker: str, t1: str, t2: str) -> bool:
    groups = COUNSELLOR_GROUPS if speaker == "counsellor" else CLIENT_GROUPS
    if t1 not in groups or t2 == UNKNOWN:
        return False
    return t2 in groups[t1]


# -----------------------------------------------------------------------------
# Reward
# -----------------------------------------------------------------------------
def score_completion(
    text: str,
    speaker: str,
    t1_gold: str,
    t2_gold: str,
    expect_rationale: bool = True,
) -> Dict[str, object]:
    """Score one rollout. Returns the total plus every component, so the
    trainer can log which part of the reward is actually moving."""
    parsed = parse_completion(text, speaker)
    t1, t2 = parsed["t1"], parsed["t2"]
    n_words = rationale_word_count(parsed["rationale"])

    well_formed = (
        parsed["json_ok"]
        and t1 != UNKNOWN
        and t2 != UNKNOWN
        and (not expect_rationale or n_words >= MIN_RATIONALE_WORDS)
    )
    hierarchical = is_hierarchically_consistent(speaker, t1, t2)
    t1_hit = t1 == t1_gold
    t2_hit = t2 == t2_gold

    total = (
        REWARD_FORMAT * well_formed
        + REWARD_HIERARCHY * hierarchical
        + REWARD_T1 * t1_hit
        + REWARD_T2 * t2_hit
    )
    return {
        "reward": float(total),
        "format": float(well_formed),
        "hierarchy": float(hierarchical),
        "t1_hit": float(t1_hit),
        "t2_hit": float(t2_hit),
        "rationale_words": n_words,
        "t1": t1,
        "t2": t2,
        "rationale": parsed["rationale"],
        "json_ok": parsed["json_ok"],
    }


# -----------------------------------------------------------------------------
# Rare-class emphasis
# -----------------------------------------------------------------------------
def class_weights(
    labels: Sequence[str],
    floor: float = 0.5,
    ceiling: float = 3.0,
    power: float = 0.5,
) -> Dict[str, float]:
    """Inverse-frequency weight per gold T2 code: `clip((N/(K*n_c))**power)`.

    Applied by multiplying a rollout's reward. That only has an effect when
    GRPO's advantage is NOT divided by the within-group standard deviation.
    Every rollout in a group answers the same prompt and so shares one gold
    label and one weight; with the default `scale_rewards="group"` the
    advantage is `(w*r - mean(w*r)) / std(w*r)` and `w` cancels exactly, top
    and bottom, leaving the weighting a silent no-op. With
    `scale_rewards="none"` it is `w * (r - mean(r))`, so the weight survives as
    gradient magnitude, which is the intent. `baseline.grpo` asserts this.

    Weights are computed per speaker, since the counsellor and client
    vocabularies are disjoint.
    """
    counts = Counter(l for l in labels if isinstance(l, str) and l)
    if not counts:
        return {}
    n_total = sum(counts.values())
    k = len(counts)
    weights = {}
    for code, n in counts.items():
        raw = (n_total / (k * n)) ** power
        weights[code] = float(min(max(raw, floor), ceiling))
    return weights


def oversample_counts(
    labels: Sequence[str],
    weights: Dict[str, float],
    max_copies: int = 3,
) -> Dict[str, int]:
    """How many times each gold code's prompts should appear per epoch.

    Separate from the reward weight and solves a different problem: with `SU`
    at 9 of 1040 counsellor rows, a whole epoch contains barely a handful of
    gradient steps that involve it at all. Note the two mechanisms compound --
    a code with weight 2.5 and 3 copies contributes roughly 7x the gradient of
    an unweighted single copy -- so `max_copies` is deliberately small.
    """
    del labels
    return {code: int(min(max(round(w), 1), max_copies)) for code, w in weights.items()}


def speaker_class_weights(
    df: pd.DataFrame,
    label_col: str = "t2_label_GT",
    **kwargs,
) -> Dict[str, Dict[str, float]]:
    """`class_weights` computed independently for each speaker."""
    out = {}
    for speaker in ("counsellor", "client"):
        sub = df[df["speaker"] == speaker]
        out[speaker] = class_weights(sub[label_col].tolist(), **kwargs)
    return out


__all__ = [
    "UNKNOWN",
    "MIN_RATIONALE_WORDS",
    "build_messages",
    "build_target",
    "parse_completion",
    "rationale_word_count",
    "is_hierarchically_consistent",
    "score_completion",
    "class_weights",
    "oversample_counts",
    "speaker_class_weights",
    "t1_codes_for_speaker",
    "t2_codes_for_speaker",
    "t2_codes_for_group",
]
