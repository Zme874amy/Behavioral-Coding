"""Two-call (T1 then T2|T1) format shared by the two-call GRPO cells.

`baseline.single_call` collapses both tiers into one generation so RL sees one
sequence and one scalar reward. The two-call cells deliberately do not: they run
the production `automisc_ft.infer.TieredAnnotator` flow -- a Tier-1 call, then a
Tier-2 call conditioned on the predicted T1 -- and reinforce it. That is the
coupled, non-stationary shape `docs/GRPO.md` says single-call was created to
avoid, and pricing that coupling is the whole point of the two-call ladder.

Each call emits the *two-call* completion shape, not the joint JSON:

    {"explanation": "...", "label": "CR"}

parsed by `automisc_ft.infer.parse_label` (code) and `emitted_rationale`
(rationale presence), exactly as the SFT two-call arms are parsed. Keeping the
parser identical to the SFT arms is what makes "did RL help this cell" a
question the numbers can answer rather than a parser artefact.

Two reward regimes live here, and they are scaled differently on purpose:

* **Decoupled per-call** (`score_t1`, `score_t2`). Each call is its own GRPO
  problem, so each is scored alone to a max of 1.0 -- the same advantage scale
  `sc_grpo` sees. The label carries most of it (T1 0.9, T2 0.8) because format
  and hierarchy are cheap to satisfy and the interesting signal is the code.

* **Joint trajectory** (`score_joint`). The T1 and T2 generations are scored
  together with the SAME four-part reward `single_call.score_completion` uses
  (format 0.1, hierarchy 0.1, t1 0.4, t2 0.4, max 1.0), reconstructed from the
  two texts. This keeps the joint arm directly comparable to `sc_grpo`, which is
  why the joint arm is worth running at all.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from automisc_ft.data import (
    CLIENT_GROUPS,
    COUNSELLOR_GROUPS,
    build_messages_t1,
    build_messages_t2,
    t1_codes_for_speaker,
    t2_codes_for_speaker,
)
from automisc_ft.infer import emitted_rationale, parse_label

# `emitted_rationale` counts a reply as carrying a rationale at >= 4 prose words;
# the format point is gated on that same threshold so the reward and the
# compliance flag downstream never disagree about whether a rationale was there.
MIN_RATIONALE_WORDS = 4

UNKNOWN = "UNKNOWN"

# Decoupled per-call weights. Each call maxes at 1.0.
DEC_FORMAT = 0.1
DEC_HIERARCHY = 0.1        # T2 only
DEC_T1 = 0.9
DEC_T2 = 0.8

# Joint-trajectory weights. The pair maxes at 1.0, matching single_call.
JOINT_FORMAT = 0.1
JOINT_HIERARCHY = 0.1
JOINT_T1 = 0.4
JOINT_T2 = 0.4


# -----------------------------------------------------------------------------
# Prompt construction (thin wrappers so the trainer imports one module)
# -----------------------------------------------------------------------------
def t1_messages(
    df: pd.DataFrame, row_pos: int, context_mode: str, ctx: int,
    structure_suffix: str = "",
) -> List[Dict[str, str]]:
    """Tier-1 chat messages for one utterance (production prompt, untouched)."""
    return build_messages_t1(df, row_pos, context_mode, ctx, structure_suffix)


def t2_messages(
    df: pd.DataFrame, row_pos: int, t1_label: str, context_mode: str, ctx: int,
    structure_suffix: str = "",
) -> List[Dict[str, str]]:
    """Tier-2 chat messages conditioned on `t1_label` (the injected group spec)."""
    return build_messages_t2(df, row_pos, t1_label, context_mode, ctx, structure_suffix)


# -----------------------------------------------------------------------------
# Hierarchy
# -----------------------------------------------------------------------------
def is_child(speaker: str, t1: str, t2: str) -> bool:
    """Whether `t2` is a fine code inside the `t1` group for this speaker."""
    groups = COUNSELLOR_GROUPS if speaker == "counsellor" else CLIENT_GROUPS
    if t1 not in groups or t2 == UNKNOWN:
        return False
    return t2 in groups[t1]


def _rationale_ok(text: str, expect_rationale: bool) -> bool:
    """Format's rationale gate: satisfied trivially when no rationale is expected."""
    if not expect_rationale:
        return True
    return emitted_rationale(text, min_words=MIN_RATIONALE_WORDS)


# -----------------------------------------------------------------------------
# Decoupled per-call rewards
# -----------------------------------------------------------------------------
def score_t1(
    text: str, speaker: str, t1_gold: str, expect_rationale: bool = True,
) -> Dict[str, object]:
    """Score one Tier-1 rollout, alone, to a max of 1.0.

    format (parses to a valid T1 code, and carries a rationale when one is
    expected) + t1 correct.
    """
    t1 = parse_label(text, t1_codes_for_speaker(speaker))
    well_formed = t1 != UNKNOWN and _rationale_ok(text, expect_rationale)
    t1_hit = t1 == t1_gold
    total = DEC_FORMAT * well_formed + DEC_T1 * t1_hit
    return {
        "reward": float(total),
        "format": float(well_formed),
        "t1_hit": float(t1_hit),
        "t1": t1,
        "emitted_rationale": bool(emitted_rationale(text, MIN_RATIONALE_WORDS)),
    }


def score_t2(
    text: str, speaker: str, t1_cond: str, t2_gold: str,
    expect_rationale: bool = True,
) -> Dict[str, object]:
    """Score one Tier-2 rollout, alone, to a max of 1.0.

    format + hierarchy (the emitted t2 is a child of the *conditioning* t1, i.e.
    the group whose spec was injected into this prompt) + t2 correct. Hierarchy
    is measured against the conditioning t1, not the gold one, because that is
    the group the model was actually shown -- rewarding consistency with a group
    it never saw would be incoherent.
    """
    t2 = parse_label(text, t2_codes_for_speaker(speaker))
    well_formed = t2 != UNKNOWN and _rationale_ok(text, expect_rationale)
    hierarchical = is_child(speaker, t1_cond, t2)
    t2_hit = t2 == t2_gold
    total = DEC_FORMAT * well_formed + DEC_HIERARCHY * hierarchical + DEC_T2 * t2_hit
    return {
        "reward": float(total),
        "format": float(well_formed),
        "hierarchy": float(hierarchical),
        "t2_hit": float(t2_hit),
        "t2": t2,
        "emitted_rationale": bool(emitted_rationale(text, MIN_RATIONALE_WORDS)),
    }


# -----------------------------------------------------------------------------
# Joint-trajectory reward (comparable to single_call.score_completion)
# -----------------------------------------------------------------------------
def score_joint(
    t1_text: str, t2_text: str, speaker: str, t1_gold: str, t2_gold: str,
    expect_rationale: bool = True,
) -> Dict[str, object]:
    """Score a (T1, T2) trajectory once, to a max of 1.0.

    Reconstructs the single-call four-part reward from the two generations so a
    joint-trajectory arm sits on the same scale as `sc_grpo`: format is the AND
    of both calls being well formed, hierarchy is t2 under the *sampled* t1, and
    the two label points are 0.4 each. The single scalar returned here is what
    the joint trainer broadcasts as the shared advantage over both completions.
    """
    t1 = parse_label(t1_text, t1_codes_for_speaker(speaker))
    t2 = parse_label(t2_text, t2_codes_for_speaker(speaker))
    well_formed = (
        t1 != UNKNOWN
        and t2 != UNKNOWN
        and _rationale_ok(t1_text, expect_rationale)
        and _rationale_ok(t2_text, expect_rationale)
    )
    hierarchical = is_child(speaker, t1, t2)
    t1_hit = t1 == t1_gold
    t2_hit = t2 == t2_gold
    total = (
        JOINT_FORMAT * well_formed
        + JOINT_HIERARCHY * hierarchical
        + JOINT_T1 * t1_hit
        + JOINT_T2 * t2_hit
    )
    return {
        "reward": float(total),
        "format": float(well_formed),
        "hierarchy": float(hierarchical),
        "t1_hit": float(t1_hit),
        "t2_hit": float(t2_hit),
        "t1": t1,
        "t2": t2,
        "t1_emitted_rationale": bool(emitted_rationale(t1_text, MIN_RATIONALE_WORDS)),
        "t2_emitted_rationale": bool(emitted_rationale(t2_text, MIN_RATIONALE_WORDS)),
    }


__all__ = [
    "MIN_RATIONALE_WORDS",
    "UNKNOWN",
    "t1_messages",
    "t2_messages",
    "is_child",
    "score_t1",
    "score_t2",
    "score_joint",
]
