"""Tier-1 quality rubric: LLM-judge scoring of generated dialogue windows.

This is the quality half of the measurement framework, and it follows how the
source papers actually evaluate synthetic data -- neither measures diversity
quantitatively, both judge quality:

  * SpeechDialogueFactory (arXiv 2503.23848) scores generated content 0-100 on
    three dimensions -- Consistency (adherence to the scenario/metadata and
    internal consistency), Coherence, and Naturalness (explicitly including
    "utterance length and rhythm" and "naturalness of vocabulary choices") --
    and filters BEFORE spending downstream compute, which "significantly
    improves overall dataset quality while reducing computational waste".
  * SocialDial (SIGIR'23) asks human raters the same naturalness/coherence
    question and reports agreement; its bar is that synthetic is *commensurate
    with real*. `--real-csv` scores real transcript windows on the identical
    rubric so we have that reference column rather than an absolute cut-off.

Usage:
    PYTHONPATH=src python -m synth.judge --raw data/synth/raw/<gen>.csv \
        --model <judge> --provider vllm_server \
        --real-csv data/manual/HLQC_balanced_manual.csv --out outputs/synth/judge.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]

RUBRIC = """You are evaluating an excerpt of a Motivational Interviewing counselling session.
Score it on three dimensions, each 0-100:

- consistency: does the excerpt stay faithful to the stated scenario (setting, topic,
  the client's stage and manner), and is it internally consistent (no contradictions
  about who the client is or what has been said)?
- coherence: does the conversation follow sensibly turn to turn, with each utterance
  responding to what came before?
- naturalness: does this read like a REAL transcribed counselling conversation?
  Judge utterance length and rhythm (real speech is short, often fragmentary),
  contextual appropriateness, and naturalness of vocabulary. Polished, uniformly
  well-formed therapist prose should score LOW on naturalness.

Return JSON: {"consistency": int, "coherence": int, "naturalness": int, "notes": "<one sentence>"}"""


class QualityScores(BaseModel):
    consistency: int
    coherence: int
    naturalness: int
    notes: str


def render_window(win: pd.DataFrame) -> str:
    return "\n".join(f"{r['speaker']}: {r['utt_text']}" for _, r in win.iterrows())


def scenario_of(win: pd.DataFrame) -> str:
    r = win.iloc[0]
    bits = [f"{k}: {r[k]}" for k in ("setting", "session", "domain", "stage", "affect",
                                     "register", "style") if k in win.columns and pd.notna(r.get(k))]
    return "; ".join(bits) if bits else "(not specified -- judge consistency as internal only)"


def judge_window(text: str, scenario: str, model: str, provider: str,
                 base_url: Optional[str]) -> QualityScores:
    from components.utils import call_chat_model

    kw = {"base_url": base_url} if base_url else {}
    user = f"SCENARIO\n{scenario}\n\nEXCERPT\n{text}"
    res = call_chat_model(messages=[{"role": "system", "content": RUBRIC},
                                    {"role": "user", "content": user}],
                          model=model, provider=provider, temperature=0.0,
                          response_format=QualityScores, **kw)
    return res if isinstance(res, QualityScores) else QualityScores(**res)


def real_windows(path: Path, n: int, turns: int) -> List[pd.DataFrame]:
    """Consecutive real-transcript slices, as a same-rubric reference column."""
    df = pd.read_csv(path)
    out = []
    for _, conv in df.groupby("conv_id", sort=False):
        conv = conv.reset_index(drop=True)
        for s in range(0, max(1, len(conv) - turns), turns):
            w = conv.iloc[s:s + turns]
            if len(w) >= turns:
                out.append(w)
            if len(out) >= n:
                return out
    return out


def _summarise(rows: List[dict], label: str) -> None:
    if not rows:
        print(f"  {label}: no scores"); return
    d = pd.DataFrame(rows)
    print(f"  {label:12s} n={len(d):3d}  " + "  ".join(
        f"{c}={d[c].mean():5.1f}±{d[c].std():4.1f}"
        for c in ("consistency", "coherence", "naturalness")))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True, help="generated windows CSV")
    ap.add_argument("--out", default=None, help="per-window scores CSV")
    ap.add_argument("--model", default="Qwen2.5-32B-Instruct-AWQ")
    ap.add_argument("--provider", default="vllm_server")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--limit", type=int, default=40, help="windows to judge")
    ap.add_argument("--real-csv", default=None, help="score real windows as the reference column")
    ap.add_argument("--real-windows", type=int, default=20)
    ap.add_argument("--turns", type=int, default=10)
    args = ap.parse_args(argv)

    raw = pd.read_csv(REPO_ROOT / args.raw if not Path(args.raw).is_absolute() else args.raw)
    wins = [w for _, w in raw.groupby("conv_id", sort=False)][: args.limit]

    rows = []
    for w in tqdm(wins, desc="judging synth", unit="win"):
        try:
            s = judge_window(render_window(w), scenario_of(w), args.model,
                             args.provider, args.base_url)
            rows.append({"conv_id": w.iloc[0]["conv_id"], "kind": "synth", **s.model_dump()})
        except Exception as e:
            print(f"  judge failed: {type(e).__name__}: {str(e)[:100]}")

    real_rows = []
    if args.real_csv:
        p = REPO_ROOT / args.real_csv if not Path(args.real_csv).is_absolute() else Path(args.real_csv)
        for w in tqdm(real_windows(p, args.real_windows, args.turns), desc="judging real", unit="win"):
            try:
                s = judge_window(render_window(w), "(real transcript excerpt)",
                                 args.model, args.provider, args.base_url)
                real_rows.append({"conv_id": str(w.iloc[0]["conv_id"]), "kind": "real",
                                  **s.model_dump()})
            except Exception as e:
                print(f"  judge failed (real): {type(e).__name__}")

    print("\nTier-1 quality rubric (0-100, mean±sd):")
    _summarise(rows, "synthetic")
    _summarise(real_rows, "real(ref)")
    if rows and real_rows:
        d, r = pd.DataFrame(rows), pd.DataFrame(real_rows)
        print("\n  gap vs real (synthetic - real):  " + "  ".join(
            f"{c}={d[c].mean()-r[c].mean():+5.1f}" for c in ("consistency", "coherence", "naturalness")))
        print("  SocialDial's bar is that synthetic is commensurate with real.")

    if args.out and (rows or real_rows):
        out = Path(args.out) if Path(args.out).is_absolute() else REPO_ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows + real_rows).to_csv(out, index=False)
        print(f"\nper-window scores -> {out}")


if __name__ == "__main__":
    main()
