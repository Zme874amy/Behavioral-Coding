"""Paired bootstrap + per-code breakdown for two prediction CSVs.

Answers "is the gap real or noise?" without retraining: given a baseline and a
candidate prediction CSV (both scored on the same fixed gold set), it computes
the macro-F1 delta and a bootstrap confidence interval on that delta, resampling
two ways:
  * utterance-level  -- treats the 821 utterances as independent (optimistic; a
                        lower bound on the true variance);
  * conversation-level -- resamples whole conversations, respecting within-
                        conversation correlation (honest, but wide with only ~10
                        conversations).
P(delta>0) across resamples is how often the candidate wins. A per-code F1
breakdown shows WHERE any gain comes from -- the rare tail we targeted, or not.

This is a within-run uncertainty estimate; it does NOT replace multi-seed runs
(training randomness), which are the definitive check.

Usage:
    PYTHONPATH=src python -m selftrain.bootstrap \
        --base data/annotated/baseline/qwen_ft_bare_inf_bare_ctx5.csv \
        --cand data/annotated/selftrain/qwen_ft_bare_inf_bare_ctx5.csv \
        --tier t2 --scope counsellor
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def macro_f1(gt: pd.Series, pred: pd.Series) -> float:
    labels = sorted(set(gt.dropna().astype(str)))
    return f1_score(gt.astype(str), pred.fillna("UNKNOWN").astype(str),
                    labels=labels, average="macro", zero_division=0)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--tier", default="t2", choices=["t1", "t2"])
    ap.add_argument("--scope", default="counsellor", choices=["all", "counsellor", "client"])
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    gt_col, pr_col = f"{args.tier}_label_GT", f"{args.tier}_label_auto"
    key = "corp_utt_idx"
    b = pd.read_csv(args.base)[[key, "conv_id", "speaker", gt_col, pr_col]].rename(columns={pr_col: "pred_base"})
    c = pd.read_csv(args.cand)[[key, pr_col]].rename(columns={pr_col: "pred_cand"})
    m = b.merge(c, on=key)
    m["speaker"] = m["speaker"].astype(str).str.lower()
    if args.scope != "all":
        m = m[m["speaker"] == args.scope]
    m = m.reset_index(drop=True)

    gt = m[gt_col]
    f1_b, f1_c = macro_f1(gt, m["pred_base"]), macro_f1(gt, m["pred_cand"])
    print(f"{args.scope} {args.tier.upper()} macro-F1: base={f1_b:.3f}  cand={f1_c:.3f}  "
          f"delta={f1_c - f1_b:+.3f}   (n={len(m)})\n")

    # utterance-level paired bootstrap
    idx = np.arange(len(m))
    du = np.empty(args.B)
    for i in range(args.B):
        s = m.iloc[rng.choice(idx, len(idx), replace=True)]
        du[i] = macro_f1(s[gt_col], s["pred_cand"]) - macro_f1(s[gt_col], s["pred_base"])
    print(f"utterance-level bootstrap:    delta 95% CI [{np.percentile(du,2.5):+.3f}, "
          f"{np.percentile(du,97.5):+.3f}]   P(delta>0)={(du>0).mean():.3f}")

    # conversation-level (cluster) bootstrap
    convs = m["conv_id"].unique()
    dc = np.empty(args.B)
    groups = {cv: m[m["conv_id"] == cv] for cv in convs}
    for i in range(args.B):
        s = pd.concat([groups[cv] for cv in rng.choice(convs, len(convs), replace=True)])
        dc[i] = macro_f1(s[gt_col], s["pred_cand"]) - macro_f1(s[gt_col], s["pred_base"])
    print(f"conversation-level bootstrap: delta 95% CI [{np.percentile(dc,2.5):+.3f}, "
          f"{np.percentile(dc,97.5):+.3f}]   P(delta>0)={(dc>0).mean():.3f}   "
          f"({len(convs)} conversations)\n")

    # per-code F1 breakdown (one-vs-rest), sorted by change
    rows = []
    for code in sorted(set(gt.dropna().astype(str))):
        yt = (gt.astype(str) == code)
        fb = f1_score(yt, m["pred_base"].astype(str) == code, zero_division=0)
        fc = f1_score(yt, m["pred_cand"].astype(str) == code, zero_division=0)
        rows.append((code, int(yt.sum()), fb, fc, fc - fb))
    print("per-code F1 (base -> cand, sorted by change):")
    for code, sup, fb, fc, d in sorted(rows, key=lambda r: r[4]):
        print(f"  {code:4s} n={sup:3d}  {fb:.3f} -> {fc:.3f}  ({d:+.3f})")


if __name__ == "__main__":
    main()
