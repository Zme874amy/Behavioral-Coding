"""Standalone macro-F1 scorer for a self-training prediction CSV.

Avoids touching `baseline.eval` (and its EXPECTED_GRID) for a first read: given a
prediction CSV with gold (`t{1,2}_label_GT`) and predicted (`t{1,2}_label_auto`)
columns, report accuracy and macro-F1 per tier x scope, exactly the headline
numbers to compare against the baseline ft_bare row.

`--vs-train` drops T2 codes absent from a training CSV (e.g. TS+, AC-) to also
print the "learnable" macro-F1, the fair arm-to-arm number.

Usage:
    PYTHONPATH=src python -m selftrain.score \
        --pred data/annotated/selftrain/qwen_ft_bare_inf_bare_ctx5.csv \
        --vs-train data/manual/HLQC_balanced_manual.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Set

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def _score(y_true, y_pred, keep: Optional[Set[str]] = None):
    mask = y_true.notna() & (y_true.astype(str) != "")
    yt, yp = y_true[mask].astype(str), y_pred[mask].fillna("UNKNOWN").astype(str)
    labels = sorted(set(yt) if keep is None else (set(yt) & keep))
    acc = accuracy_score(yt, yp)
    f1 = f1_score(yt, yp, labels=labels, average="macro", zero_division=0)
    return acc, f1, len(yt), len(labels)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--vs-train", default=None, help="training CSV; T2 codes absent from it define 'learnable'")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.pred)
    df["speaker"] = df["speaker"].astype(str).str.lower()

    learnable: Optional[Set[str]] = None
    if args.vs_train:
        tr = pd.read_csv(args.vs_train)
        learnable = set(tr["t2_label_GT"].dropna().astype(str))

    print(f"Scoring {args.pred}  (n={len(df)})\n")
    for tier in ("t1", "t2"):
        gt, pred = f"{tier}_label_GT", f"{tier}_label_auto"
        if gt not in df or pred not in df:
            print(f"  [{tier}] missing columns; skipping")
            continue
        print(f"== {tier.upper()} ==")
        for scope in ("all", "counsellor", "client"):
            sub = df if scope == "all" else df[df["speaker"] == scope]
            if sub.empty:
                continue
            acc, f1g, n, k = _score(sub[gt], sub[pred])
            line = f"  {scope:11s} n={n:4d}  acc={acc:.3f}  macroF1(gold)={f1g:.3f} ({k} codes)"
            if tier == "t2" and learnable is not None:
                _, f1l, _, kl = _score(sub[gt], sub[pred], keep=learnable)
                line += f"  macroF1(learnable)={f1l:.3f} ({kl})"
            print(line)
        print()


if __name__ == "__main__":
    main()
