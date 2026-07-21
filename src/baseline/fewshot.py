"""Few-shot exemplar construction and loading for the AutoMISC baseline study.

Exemplars are drawn from the held-out HLQC_balanced_manual.csv (never from the
MIV6.3A evaluation set) with stratified selection: one exemplar per T1 category
for the T1 prompts, and one exemplar per T2 code (within its T1 group) for the
T2 prompts. Rationales for the "few-shot with rationales" condition are
generated once with gpt-4o conditioned on the gold label and frozen to a JSON
file so runs are reproducible.

Exemplar transcripts are built with the same number of prior context volleys as
the run that consumes them, so each context setting has its own frozen file:
data/fewshot/exemplars_ctx<N>.json (the original exemplars.json is the ctx5 file).

Run as a script to (re)build an exemplar file:
    PYTHONPATH=src .venv/bin/python -m baseline.fewshot --ctx 3
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from components.context import build_context_excerpt
from components.prompts.loader import render_prompt, render_user_prompt

REPO_ROOT = Path(__file__).resolve().parents[2]
FEWSHOT_DIR = REPO_ROOT / "data" / "fewshot"
HLQC_PATH = REPO_ROOT / "data" / "manual" / "HLQC_balanced_manual.csv"

SEED = 42
CONTEXT_MODE = "interval"
DEFAULT_CONTEXT_TURNS = 5


def exemplars_path(num_context_turns: int) -> Path:
    """Frozen exemplar file for a given context length. The pre-ctx-suffix
    file (exemplars.json) was built with 5 turns and doubles as the ctx5 file."""
    suffixed = FEWSHOT_DIR / f"exemplars_ctx{num_context_turns}.json"
    if suffixed.exists():
        return suffixed
    legacy = FEWSHOT_DIR / "exemplars.json"
    if num_context_turns == 5 and legacy.exists():
        return legacy
    return suffixed

# Valid T2 codes per T1 group (from the prompt specs / response formats).
T2_GROUPS = {
    "counsellor": {
        "CRL": ["CR", "AF", "SU", "RF", "EC"],
        "SRL": ["SR"],
        "IMC": ["ADP", "RCP", "GI"],
        "IMI": ["ADW", "RCW", "WA", "DI", "CO"],
        "Q": ["OQ", "CQ"],
        "O": ["FA", "FI", "ST"],
    },
    "client": {
        "C": ["D+", "AB+", "R+", "N+", "C+", "AC+", "TS+", "O+"],
        "S": ["D-", "AB-", "R-", "N-", "C-", "AC-", "TS-", "O-"],
        "N": ["N"],
    },
}


class _Rationale(BaseModel):
    explanation: str


def _pick_exemplar(df: pd.DataFrame, candidates: pd.DataFrame, num_context_turns: int) -> dict | None:
    """Pick one exemplar row, preferring mid-length utterances."""
    if candidates.empty:
        return None
    preferred = candidates[
        candidates["utt_text"].str.len().between(20, 400)
    ]
    pool = preferred if not preferred.empty else candidates
    row = pool.sample(1, random_state=SEED).iloc[0]
    transcript = build_context_excerpt(
        df, int(row.name), CONTEXT_MODE, num_context_turns
    )
    return {
        "transcript": transcript,
        "speaker": row["speaker"],
        "utterance": row["utt_text"],
        "t1_label": row["t1_label_GT"],
        "t2_label": row["t2_label_GT"],
    }


def build_exemplars(num_context_turns: int = DEFAULT_CONTEXT_TURNS) -> dict:
    df = pd.read_csv(HLQC_PATH).reset_index(drop=True)
    exemplars = {}
    for speaker, groups in T2_GROUPS.items():
        spk_df = df[df["speaker"] == speaker]
        t1_exemplars = []
        t2_exemplars = {}
        for t1_code, t2_codes in groups.items():
            # Candidates whose T2 code is consistent with the T1 group
            # (guards against noisy GT pairs like T1=O with T2=OQ).
            consistent = spk_df[
                (spk_df["t1_label_GT"] == t1_code)
                & (spk_df["t2_label_GT"].isin(t2_codes))
            ]
            ex = _pick_exemplar(df, consistent, num_context_turns)
            if ex is not None:
                t1_exemplars.append(ex)
            else:
                print(f"WARNING: no exemplar for T1 {speaker}/{t1_code}")

            group_list = []
            for t2_code in t2_codes:
                cand = consistent[consistent["t2_label_GT"] == t2_code]
                ex2 = _pick_exemplar(df, cand, num_context_turns)
                if ex2 is not None:
                    group_list.append(ex2)
                else:
                    print(f"NOTE: no exemplar for T2 {speaker}/{t1_code}/{t2_code} (absent in HLQC)")
            t2_exemplars[t1_code] = group_list
        exemplars[speaker] = {"t1": t1_exemplars, "t2": t2_exemplars}
    return exemplars


def generate_rationales(exemplars: dict, model: str = "gpt-4o", provider: str = "azure") -> None:
    """Generate a frozen 1-2 sentence rationale for every exemplar, conditioned
    on the gold label, using the original (rationale-full) AutoMISC prompts."""
    from components.utils import call_chat_model

    def _rationale(system_prompt: str, ex: dict, label: str) -> str:
        user_prompt = render_user_prompt(
            transcript=ex["transcript"], speaker=ex["speaker"], utterance=ex["utterance"]
        )
        user_prompt += (
            f"\n\n## Gold Label\nThe correct label for this utterance is **{label}**. "
            "Write the 1-2 sentence explanation that justifies this label."
        )
        res = call_chat_model(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            provider=provider,
            response_format=_Rationale,
            temperature=0.0,
        )
        return res["explanation"]

    for speaker, tiers in exemplars.items():
        t1_system = render_prompt(speaker=speaker, structure="t1")
        for ex in tiers["t1"]:
            ex["t1_explanation"] = _rationale(t1_system, ex, ex["t1_label"])
            print(f"rationale t1 {speaker}/{ex['t1_label']}: {ex['t1_explanation'][:80]}...")
        for t1_code, group in tiers["t2"].items():
            t2_system = render_prompt(speaker=speaker, structure="t2", label=t1_code)
            for ex in group:
                ex["t2_explanation"] = _rationale(t2_system, ex, ex["t2_label"])
                print(f"rationale t2 {speaker}/{t1_code}/{ex['t2_label']}: {ex['t2_explanation'][:80]}...")


def load_exemplars(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_fewshot_messages(
    exemplars: dict,
    speaker: str,
    tier: str,
    rationales: bool,
    t1_label: str | None = None,
) -> list[dict]:
    """Build alternating user/assistant few-shot messages for a given prompt.

    Assistant replies are JSON strings matching the structured-output schema of
    the run (with or without the explanation field).
    """
    if tier == "t1":
        pool = exemplars[speaker]["t1"]
        label_key, expl_key = "t1_label", "t1_explanation"
    elif tier == "t2":
        pool = exemplars[speaker]["t2"][t1_label]
        label_key, expl_key = "t2_label", "t2_explanation"
    else:
        raise ValueError(f"Unknown tier: {tier}")

    messages = []
    for ex in pool:
        user_prompt = render_user_prompt(
            transcript=ex["transcript"], speaker=ex["speaker"], utterance=ex["utterance"]
        )
        if rationales:
            reply = {"explanation": ex[expl_key], "label": ex[label_key]}
        else:
            reply = {"label": ex[label_key]}
        messages.append({"role": "user", "content": user_prompt})
        messages.append({"role": "assistant", "content": json.dumps(reply)})
    return messages


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx", type=int, default=DEFAULT_CONTEXT_TURNS,
                        help="number of prior context volleys in exemplar transcripts")
    args = parser.parse_args()

    exemplars = build_exemplars(num_context_turns=args.ctx)
    generate_rationales(exemplars)
    out_path = FEWSHOT_DIR / f"exemplars_ctx{args.ctx}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(exemplars, f, indent=2)
    n_t1 = sum(len(v["t1"]) for v in exemplars.values())
    n_t2 = sum(len(g) for v in exemplars.values() for g in v["t2"].values())
    print(f"Saved {n_t1} T1 + {n_t2} T2 exemplars to {out_path}")


if __name__ == "__main__":
    main()
