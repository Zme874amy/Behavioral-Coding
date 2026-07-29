"""Generate frozen rationale training targets for the HLQC fine-tuning set.

The `ft_rat` adaptation arm trains on (rationale + label) targets rather than
bare labels, but no corpus ships human-written rationales. This module distils
them from gpt-4o: for every HLQC utterance and both tiers, gpt-4o is shown the
original AutoMISC prompt plus the gold label and asked to write the 1-2 sentence
justification for that label. The result is frozen to JSON so training runs are
reproducible and the (paid) generation happens exactly once.

Rationales are frozen PER CONTEXT LENGTH. A rationale written with 5 volleys of
transcript can cite material that is invisible at 3 volleys, so reusing one file
across context settings would produce targets the model cannot justify from its
own input.

Because these are post-hoc justifications conditioned on the gold label they may
be unfaithful to any reasoning that would actually produce the label; that is a
known limitation of rationale distillation and belongs in the write-up.

Usage:
    PYTHONPATH=src .venv/bin/python -m baseline.rationalize --ctx 5
    PYTHONPATH=src .venv/bin/python -m baseline.rationalize --ctx 5 --limit 4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from components.context import build_context_excerpt
from components.prompts.loader import render_prompt, render_user_prompt
from components.utils import call_chat_model

REPO_ROOT = Path(__file__).resolve().parents[2]
HLQC_PATH = REPO_ROOT / "data" / "manual" / "HLQC_balanced_manual.csv"
RATIONALE_DIR = REPO_ROOT / "data" / "fine_tuning" / "rationales"

CONTEXT_MODE = "interval"
DEFAULT_CONTEXT_TURNS = 5
DEFAULT_MODEL = "gpt-4o"
DEFAULT_PROVIDER = "azure"
CHECKPOINT_EVERY = 25

GOLD_LABEL_SUFFIX = (
    "\n\n## Gold Label\nThe correct label for this utterance is **{label}**. "
    "Write the 1-2 sentence explanation that justifies this label."
)


def rationales_path(num_context_turns: int) -> Path:
    return RATIONALE_DIR / f"hlqc_rationales_ctx{num_context_turns}.json"


def _rationale_schema():
    """The response schema for a bare explanation.

    Defined lazily so importing this module does not require pydantic model
    construction at import time.
    """
    from baseline.fewshot import _Rationale

    return _Rationale


def load_rationales(num_context_turns: int) -> dict:
    """Load the frozen rationale file, or an empty dict if it does not exist."""
    path = rationales_path(num_context_turns)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _call_with_retry(messages, model: str, provider: str, max_retries: int = 6) -> str:
    delay = 2.0
    for attempt in range(max_retries):
        try:
            res = call_chat_model(
                messages=messages,
                model=model,
                provider=provider,
                response_format=_rationale_schema(),
                temperature=0.0,
            )
            return res["explanation"]
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"    retry {attempt + 1} after error: {e}", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError("unreachable")


def _generate_one(
    df: pd.DataFrame,
    row_pos: int,
    tier: str,
    label: str,
    num_context_turns: int,
    model: str,
    provider: str,
) -> str:
    """Generate the rationale for one (utterance, tier) pair.

    The prompt is the original rationale-full AutoMISC template (`t1`/`t2`, not
    the `_bare` variants) so the distilled explanation is stylistically matched
    to the `inf_cot` prompt the fine-tuned model will be evaluated under.
    """
    row = df.iloc[row_pos]
    speaker = row["speaker"]
    if tier == "t1":
        system_prompt = render_prompt(speaker=speaker, structure="t1")
    else:
        system_prompt = render_prompt(
            speaker=speaker, structure="t2", label=row["t1_label_GT"]
        )

    transcript = build_context_excerpt(df, row_pos, CONTEXT_MODE, num_context_turns)
    user_prompt = render_user_prompt(
        transcript=transcript, speaker=speaker, utterance=row["utt_text"]
    )
    user_prompt += GOLD_LABEL_SUFFIX.format(label=label)

    return _call_with_retry(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        provider=provider,
    )


def generate(
    num_context_turns: int,
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    limit: int | None = None,
    checkpoint_every: int = CHECKPOINT_EVERY,
) -> dict:
    """Generate (or resume) the rationale file for one context length.

    Rows already present in the on-disk file are skipped, so an interrupted run
    can simply be re-invoked. Keys are `corp_utt_idx` as strings (JSON object
    keys are always strings).
    """
    df = pd.read_csv(HLQC_PATH).reset_index(drop=True)
    out_path = rationales_path(num_context_turns)
    store = load_rationales(num_context_turns)
    if store:
        print(f"Resuming from {out_path}: {len(store)} utterances already done")

    n_total = len(df) if limit is None else min(int(limit), len(df))
    done_since_save = 0

    def save() -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(store, f, indent=2)

    try:
        for row_pos in range(n_total):
            row = df.iloc[row_pos]
            key = str(row["corp_utt_idx"])
            entry = store.get(key)
            t1_label = row["t1_label_GT"]
            t2_label = row["t2_label_GT"]
            if not isinstance(t1_label, str) or not isinstance(t2_label, str):
                continue
            if entry and entry.get("t1_explanation") and entry.get("t2_explanation"):
                continue

            entry = entry or {
                "corp_utt_idx": int(row["corp_utt_idx"]),
                "speaker": row["speaker"],
                "t1_label": t1_label,
                "t2_label": t2_label,
            }
            if not entry.get("t1_explanation"):
                entry["t1_explanation"] = _generate_one(
                    df, row_pos, "t1", t1_label, num_context_turns, model, provider
                )
            if not entry.get("t2_explanation"):
                entry["t2_explanation"] = _generate_one(
                    df, row_pos, "t2", t2_label, num_context_turns, model, provider
                )
            store[key] = entry

            done_since_save += 1
            if done_since_save >= checkpoint_every:
                save()
                done_since_save = 0
                print(f"  checkpoint: {len(store)}/{n_total} utterances", flush=True)
    finally:
        save()

    print(f"Wrote {len(store)} rationale pairs to {out_path}")
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ctx",
        type=int,
        default=DEFAULT_CONTEXT_TURNS,
        help="number of prior context volleys; one frozen file per setting",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument(
        "--limit", type=int, default=None, help="only process the first N utterances"
    )
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")

    generate(
        num_context_turns=args.ctx,
        model=args.model,
        provider=args.provider,
        limit=args.limit,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
