"""Build hierarchical (T1 / T2) prompt-completion examples for the AutoMISC
fine-tuning experiment and assign conversation-level cross-validation folds.

Every example uses the ORIGINAL AutoMISC prompts (`render_prompt` /
`render_user_prompt`) and the shared context builder
(`components.context.build_context_excerpt`) so the input the model sees is
identical to the production annotator. The only thing fine-tuning changes is
the model weights.

Schema of the manual CSV (e.g. data/manual/MIV6.3A_manual.csv):
    conv_id, speaker, corp_vol_idx, conv_vol_idx, corp_utt_idx, conv_utt_idx,
    utt_text, t1_label_GT, t2_label_GT
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from components.context import build_context_excerpt
from components.prompts.loader import render_prompt, render_user_prompt
from sft.data import (
    CLIENT_T1,
    CLIENT_T2,
    COUNSELLOR_T1,
    COUNSELLOR_T2,
    _normalize_speaker,
    build_completion,
)

# -----------------------------------------------------------------------------
# Tier-1 grouping -> candidate Tier-2 codes. Mirrors the spec YAMLs and is used
# to (optionally) restrict T2 parsing to the predicted T1 group.
# -----------------------------------------------------------------------------
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


def t1_codes_for_speaker(speaker: str) -> List[str]:
    return COUNSELLOR_T1 if speaker == "counsellor" else CLIENT_T1


def t2_codes_for_speaker(speaker: str) -> List[str]:
    return COUNSELLOR_T2 if speaker == "counsellor" else CLIENT_T2


def t2_codes_for_group(speaker: str, t1_label: str) -> List[str]:
    groups = COUNSELLOR_GROUPS if speaker == "counsellor" else CLIENT_GROUPS
    return groups.get(t1_label, t2_codes_for_speaker(speaker))


# -----------------------------------------------------------------------------
# Loading & fold assignment.
# -----------------------------------------------------------------------------
def load_manual(csv_path: str | Path) -> pd.DataFrame:
    """Load the manual CSV, normalize speakers, and order rows so that the
    positional index matches conversation/volley/utterance order (required by
    `build_context_excerpt`, which uses `df.iloc`)."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Manual data file not found: {path}")
    df = pd.read_csv(path)
    df["speaker"] = df["speaker"].map(_normalize_speaker)
    df["conv_id"] = df["conv_id"].astype(str)
    df["utt_text"] = df["utt_text"].astype(str)
    sort_cols = [c for c in ("corp_utt_idx", "conv_vol_idx", "conv_utt_idx") if c in df.columns]
    if sort_cols:
        df = df.sort_values(["conv_id"] + sort_cols, kind="stable")
    df = df.reset_index(drop=True)
    return df


def assign_folds(df: pd.DataFrame, n_folds: int, seed: int = 42) -> Dict[str, int]:
    """Deterministically assign each conversation to a CV fold.

    Conversations are shuffled with `seed` then round-robin assigned so fold
    sizes are as balanced as possible (important when there are few, large
    conversations).
    """
    import random

    conv_ids = sorted(df["conv_id"].unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(conv_ids)
    fold_of: Dict[str, int] = {}
    for i, cid in enumerate(conv_ids):
        fold_of[cid] = i % n_folds
    return fold_of


# -----------------------------------------------------------------------------
# Example construction.
# -----------------------------------------------------------------------------
@dataclass
class TierExample:
    conv_id: str
    row_pos: int          # positional (iloc) index into the full df
    speaker: str
    tier: str             # "t1" or "t2"
    label: str            # gold label (completion target)
    messages: List[Dict[str, str]] = field(default_factory=list)
    # Distilled rationale for the `ft_rat` arm. When set, the completion target
    # becomes the rationale + label JSON instead of the bare label.
    explanation: Optional[str] = None


def _valid_label(label, allowed: List[str]) -> Optional[str]:
    if not isinstance(label, str):
        return None
    label = label.strip()
    return label if label in allowed else None


def build_messages_t1(
    df: pd.DataFrame,
    row_pos: int,
    context_mode: str,
    num_context_turns: int,
    structure_suffix: str = "",
    fewshot_messages: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build the Tier-1 chat messages.

    `structure_suffix` selects the prompt variant: "" is the original
    rationale-first template, "_bare" the label-only one. `fewshot_messages` is
    spliced between the system prompt and the target utterance.
    """
    row = df.iloc[row_pos]
    speaker = row["speaker"]
    transcript = build_context_excerpt(df, row_pos, context_mode, num_context_turns)
    system = render_prompt(speaker=speaker, structure=f"t1{structure_suffix}")
    user = render_user_prompt(transcript=transcript, speaker=speaker, utterance=row["utt_text"])
    messages = [{"role": "system", "content": system}]
    if fewshot_messages:
        messages += fewshot_messages
    messages.append({"role": "user", "content": user})
    return messages


def build_messages_t2(
    df: pd.DataFrame,
    row_pos: int,
    t1_label: str,
    context_mode: str,
    num_context_turns: int,
    structure_suffix: str = "",
    fewshot_messages: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build the Tier-2 chat messages, conditioned on a Tier-1 label."""
    row = df.iloc[row_pos]
    speaker = row["speaker"]
    transcript = build_context_excerpt(df, row_pos, context_mode, num_context_turns)
    system = render_prompt(speaker=speaker, structure=f"t2{structure_suffix}", label=t1_label)
    user = render_user_prompt(transcript=transcript, speaker=speaker, utterance=row["utt_text"])
    messages = [{"role": "system", "content": system}]
    if fewshot_messages:
        messages += fewshot_messages
    messages.append({"role": "user", "content": user})
    return messages


def build_tier_examples(
    df: pd.DataFrame,
    row_positions: List[int],
    tier: str,
    context_mode: str,
    num_context_turns: int,
    structure_suffix: str = "",
    rationales: Optional[Dict[str, dict]] = None,
) -> List[TierExample]:
    """Build training examples for a single tier over the given rows.

    For T2, the spec is injected for the GOLD T1 group (`t1_label_GT`) so the
    model is trained on the same conditional structure the pipeline uses at
    inference (where it injects the PREDICTED T1 group).

    When `rationales` is given (the frozen distilled-rationale store, keyed by
    `corp_utt_idx` as a string), each example carries the matching explanation
    and the completion target becomes rationale + label. Rows with no rationale
    on file are skipped, since a rationale target cannot be fabricated.
    """
    if tier not in {"t1", "t2"}:
        raise ValueError(f"tier must be 't1' or 't2', got {tier!r}")

    examples: List[TierExample] = []
    n_missing_rationale = 0
    for pos in row_positions:
        row = df.iloc[pos]
        speaker = row["speaker"]
        if speaker not in {"counsellor", "client"}:
            continue

        if tier == "t1":
            label = _valid_label(row.get("t1_label_GT"), t1_codes_for_speaker(speaker))
            if label is None:
                continue
            messages = build_messages_t1(
                df, pos, context_mode, num_context_turns, structure_suffix
            )
        else:
            t1_gold = _valid_label(row.get("t1_label_GT"), t1_codes_for_speaker(speaker))
            label = _valid_label(row.get("t2_label_GT"), t2_codes_for_speaker(speaker))
            if t1_gold is None or label is None:
                continue
            messages = build_messages_t2(
                df, pos, t1_gold, context_mode, num_context_turns, structure_suffix
            )

        explanation = None
        if rationales is not None:
            entry = rationales.get(str(row.get("corp_utt_idx")))
            explanation = (entry or {}).get(f"{tier}_explanation")
            if not explanation:
                n_missing_rationale += 1
                continue

        examples.append(
            TierExample(
                conv_id=str(row["conv_id"]),
                row_pos=int(pos),
                speaker=speaker,
                tier=tier,
                label=label,
                messages=messages,
                explanation=explanation,
            )
        )
    if n_missing_rationale:
        print(
            f"WARNING: skipped {n_missing_rationale} {tier} rows with no frozen "
            "rationale; run `python -m baseline.rationalize` for this context length"
        )
    return examples


def build_rationale_completion(explanation: str, label: str) -> str:
    """Completion target for the `ft_rat` arm: rationale then label, as JSON.

    Shaped to match the structured-output schema the gpt-4o CoT conditions emit
    (`{"explanation": ..., "label": ...}`) so both tiers of the experiment are
    scored off the same output format, and so `infer.parse_label` can read it.
    The leading space / trailing newline mirror `build_completion`.
    """
    return " " + json.dumps({"explanation": explanation, "label": label}) + "\n"


def examples_to_prompt_completion(
    examples: List[TierExample], tokenizer
) -> List[Dict[str, str]]:
    """Render each example's chat messages into the model's chat template and
    pair it with the completion target, ready for `run_local_fine_tuning`.

    Examples carrying an `explanation` get the rationale + label JSON target;
    the rest get the bare label.

    The prompt already contains the model's special tokens (via
    `apply_chat_template`), so the trainer must tokenize it with
    `add_special_tokens=False` (which it does).
    """
    rows: List[Dict[str, str]] = []
    for ex in examples:
        prompt = tokenizer.apply_chat_template(
            ex.messages, tokenize=False, add_generation_prompt=True
        )
        if ex.explanation:
            completion = build_rationale_completion(ex.explanation, ex.label)
        else:
            completion = build_completion(ex.label)
        rows.append(
            {
                "prompt": prompt,
                "completion": completion,
                "conv_id": ex.conv_id,
                "speaker": ex.speaker,
                "tier": ex.tier,
                "label": ex.label,
            }
        )
    return rows
