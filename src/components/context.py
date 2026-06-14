"""Shared conversational-context construction for the annotator and the
fine-tuning pipeline.

The logic here is the single source of truth for how prior-volley context is
assembled into the `transcript` block that gets rendered into `user_prompt.j2`.
Both the production `Annotator` (LM Studio / OpenAI inference) and the
`automisc_ft` HuggingFace pipeline call this so the prompt the model sees is
byte-identical whether we are zero-shot annotating or fine-tuning.
"""
from __future__ import annotations

from typing import Literal

import pandas as pd

ContextMode = Literal["all", "cumulative", "interval"]


def build_context_excerpt(
    df: pd.DataFrame,
    utt_idx: int,
    context_mode: ContextMode,
    num_context_turns: int = 0,
) -> str:
    """Return the formatted transcript excerpt for the utterance at positional
    index ``utt_idx`` in ``df``.

    Args:
        df: Utterance-level dataframe with columns ``conv_id``,
            ``conv_vol_idx``, ``conv_utt_idx``, ``speaker``, ``utt_text``.
        utt_idx: Positional (``iloc``) index of the target utterance.
        context_mode: ``all`` (whole conversation), ``cumulative`` (everything
            up to and including the target), or ``interval`` (the previous
            ``num_context_turns`` volleys plus the current volley up to the
            target).
        num_context_turns: Number of prior volleys to include for
            ``interval`` mode.

    Returns:
        A newline-joined transcript where consecutive same-speaker utterances
        are merged into a single ``"speaker: text"`` segment.
    """
    row = df.iloc[utt_idx]
    conv_id = row["conv_id"]
    conv_utt_idx = row["conv_utt_idx"]
    conv_vol_idx = row["conv_vol_idx"]
    conv_df = df[df["conv_id"] == conv_id].reset_index(drop=True)

    if context_mode == "all":
        context_df = conv_df
    elif context_mode == "cumulative":
        context_df = conv_df[conv_df["conv_utt_idx"] <= conv_utt_idx]
    elif context_mode == "interval":
        vol_start = max(0, conv_vol_idx - num_context_turns)
        prev_vol_df = conv_df[
            (conv_df["conv_vol_idx"] >= vol_start)
            & (conv_df["conv_vol_idx"] < conv_vol_idx)
        ]
        curr_vol_df = conv_df[
            (conv_df["conv_vol_idx"] == conv_vol_idx)
            & (conv_df["conv_utt_idx"] <= conv_utt_idx)
        ]
        context_df = pd.concat([prev_vol_df, curr_vol_df])
    else:
        raise ValueError(f"Invalid context mode: {context_mode}")

    formatted_segments = []
    prev_speaker = None
    segment = ""
    for _, ctx_row in context_df.iterrows():
        speaker = ctx_row["speaker"]
        text = ctx_row["utt_text"]
        if speaker != prev_speaker:
            if segment:
                formatted_segments.append(segment.strip())
            segment = f"{speaker}: {text}"
        else:
            segment += f" {text}"
        prev_speaker = speaker

    if segment:
        formatted_segments.append(segment.strip())

    return "\n".join(formatted_segments)
