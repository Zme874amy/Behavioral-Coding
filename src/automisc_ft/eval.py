"""Per-speaker, per-tier evaluation for the AutoMISC fine-tuning experiment.

Matches the paper's reporting:
  - Tier 1: accuracy, reported separately for counsellor and client.
  - Tier 2: macro-F1 and accuracy, reported separately for counsellor (19
    codes) and client (17 codes), plus a Change/Sustain Talk breakdown.

All metrics are computed on POOLED out-of-fold predictions (every utterance is
predicted exactly once, by the adapter trained on the folds that exclude it),
so the numbers are directly comparable to a single held-out run over the full
corpus.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from sft.data import (
    CHANGE_TALK_T2,
    CLIENT_T1,
    CLIENT_T2,
    COUNSELLOR_T1,
    COUNSELLOR_T2,
    SUSTAIN_TALK_T2,
)

try:
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    SKLEARN = True
except Exception:  # pragma: no cover
    SKLEARN = False


SPEAKER_T1 = {"counsellor": COUNSELLOR_T1, "client": CLIENT_T1}
SPEAKER_T2 = {"counsellor": COUNSELLOR_T2, "client": CLIENT_T2}


def _valid_rows(df: pd.DataFrame, speaker: str, gold_col: str, allowed: List[str]) -> pd.DataFrame:
    sub = df[df["speaker"] == speaker].copy()
    sub = sub[sub[gold_col].isin(allowed)]
    return sub


def _t1_accuracy(df: pd.DataFrame, pred_col: str) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for speaker, allowed in SPEAKER_T1.items():
        sub = _valid_rows(df, speaker, "t1_label_GT", allowed)
        if len(sub) == 0:
            out[speaker] = {"accuracy": None, "support": 0}
            continue
        acc = accuracy_score(sub["t1_label_GT"], sub[pred_col]) if SKLEARN else None
        out[speaker] = {"accuracy": acc, "support": int(len(sub))}
    return out


def _t2_metrics(df: pd.DataFrame, pred_col: str) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for speaker, allowed in SPEAKER_T2.items():
        sub = _valid_rows(df, speaker, "t2_label_GT", allowed)
        if len(sub) == 0 or not SKLEARN:
            out[speaker] = {"macro_f1": None, "accuracy": None, "support": int(len(sub))}
            continue
        y_true = sub["t2_label_GT"].tolist()
        y_pred = sub[pred_col].tolist()
        out[speaker] = {
            "macro_f1": f1_score(y_true, y_pred, labels=allowed, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, y_pred, labels=allowed, average="weighted", zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred),
            "support": int(len(sub)),
            "per_class": classification_report(
                y_true, y_pred, labels=allowed, output_dict=True, zero_division=0
            ),
        }
    return out


def _change_sustain(df: pd.DataFrame, pred_col: str) -> Dict[str, Dict]:
    """Client-side Change/Sustain Talk recall and F1 (the long-tail focus)."""
    if not SKLEARN:
        return {}
    client = df[df["speaker"] == "client"]
    out: Dict[str, Dict] = {}
    for name, codes in [("change_talk", CHANGE_TALK_T2), ("sustain_talk", SUSTAIN_TALK_T2)]:
        sub = client[client["t2_label_GT"].isin(codes)]
        support = int(len(sub))
        if support == 0:
            out[name] = {"f1": 0.0, "recall": 0.0, "precision": 0.0, "support": 0}
            continue
        y_true = sub["t2_label_GT"].tolist()
        y_pred = sub[pred_col].tolist()
        rep = classification_report(y_true, y_pred, labels=codes, output_dict=True, zero_division=0)
        rows = [rep[c] for c in codes if c in rep]
        tot = sum(r["support"] for r in rows) or 1
        out[name] = {
            "f1": sum(r["f1-score"] * r["support"] for r in rows) / tot,
            "recall": sum(r["recall"] * r["support"] for r in rows) / tot,
            "precision": sum(r["precision"] * r["support"] for r in rows) / tot,
            "support": support,
        }
    return out


def compute_condition_metrics(df: pd.DataFrame, t1_pred_col: str, t2_pred_col: str) -> Dict:
    """All metrics for a single condition (zero-shot or fine-tuned)."""
    return {
        "tier1_accuracy": _t1_accuracy(df, t1_pred_col),
        "tier2": _t2_metrics(df, t2_pred_col),
        "change_sustain": _change_sustain(df, t2_pred_col),
    }


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)


def format_comparison(zs: Dict, ft: Dict) -> str:
    """Human-readable ZS vs FT table with deltas and the paper reference."""
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("AutoMISC pipeline: Zero-Shot vs Fine-Tuned (per-speaker, pooled CV)")
    lines.append("=" * 78)

    # Tier 1 accuracy
    lines.append("\nTier 1 accuracy (paper GPT-4.1: counsellor 0.82, client 0.88)")
    for sp in ("counsellor", "client"):
        z = zs["tier1_accuracy"][sp]["accuracy"]
        f = ft["tier1_accuracy"][sp]["accuracy"]
        sup = zs["tier1_accuracy"][sp]["support"]
        d = (f - z) if isinstance(z, float) and isinstance(f, float) else None
        lines.append(
            f"  {sp:11s}: ZS {_fmt(z)} | FT {_fmt(f)} | Δ {('%+.4f' % d) if d is not None else 'NA'}  (n={sup})"
        )

    # Tier 2 macro-F1 / accuracy
    paper_t2 = {"counsellor": (0.42, 0.68), "client": (0.41, 0.76)}
    lines.append("\nTier 2 macro-F1 / accuracy (paper GPT-4.1 in parentheses)")
    for sp in ("counsellor", "client"):
        z = zs["tier2"][sp]
        f = ft["tier2"][sp]
        pf1, pacc = paper_t2[sp]
        df1 = (f["macro_f1"] - z["macro_f1"]) if isinstance(z["macro_f1"], float) and isinstance(f["macro_f1"], float) else None
        dacc = (f["accuracy"] - z["accuracy"]) if isinstance(z["accuracy"], float) and isinstance(f["accuracy"], float) else None
        lines.append(f"  {sp} (n={z['support']}, paper F1={pf1}, acc={pacc}):")
        lines.append(
            f"      macro-F1 : ZS {_fmt(z['macro_f1'])} | FT {_fmt(f['macro_f1'])} | Δ {('%+.4f' % df1) if df1 is not None else 'NA'}"
        )
        lines.append(
            f"      accuracy : ZS {_fmt(z['accuracy'])} | FT {_fmt(f['accuracy'])} | Δ {('%+.4f' % dacc) if dacc is not None else 'NA'}"
        )

    # Change / Sustain talk
    lines.append("\nClient Change/Sustain Talk (F1, support)")
    for name in ("change_talk", "sustain_talk"):
        z = zs["change_sustain"].get(name, {})
        f = ft["change_sustain"].get(name, {})
        if not z:
            continue
        df1 = (f.get("f1", 0.0) - z.get("f1", 0.0))
        lines.append(
            f"  {name:13s}: ZS {_fmt(z.get('f1'))} | FT {_fmt(f.get('f1'))} | Δ {df1:+.4f}  (support={z.get('support')})"
        )
    lines.append("=" * 78)
    return "\n".join(lines)
