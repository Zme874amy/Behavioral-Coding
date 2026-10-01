"""Calibrate a pseudo-labeller on labelled HLQC data and derive per-code thresholds.

Input: one or more *scored* CSVs -- rows labelled with a full distribution over
codes (`TieredAnnotator.predict_row_scored`, or the teacher's vLLM scorer in
`label_pool --confidence likelihood`) that also carry the gold labels:
  * student: `cv_anchor score-fold` over each held-out HLQC fold (out-of-fold, so
    every row is scored by an adapter that never saw its conversation);
  * teacher: `label_pool --labeler remote --confidence likelihood --keep-gold`
    over all of HLQC (it never trained on HLQC).

Output (`--out` JSON), all measured on labelled data only -- the test set never
chooses anything:
  * the labeller's own T1/T2 accuracy and macro-F1 with conversation-clustered
    CIs -- for the teacher, this is the "is it actually a stronger coder?" check;
  * expected calibration error of the joint confidence p(T1)*p(T2|T1);
  * per (speaker, predicted T2 code) the LOWEST confidence threshold at which the
    labeller's predictions of that code are correct >= `--target-precision` of
    the time over at least `--min-n` rows (FlexMatch/UPS-style class-adaptive
    thresholds). A code that never reaches the target is excluded: its
    pseudo-labels are not trustworthy at any confidence.
  * optionally (`--greedy`) how often the scored argmax equals the greedy decode
    of the same adapter -- a check that the scorer reflects the model's real
    behaviour.

Usage:
    PYTHONPATH=src python -m selftrain.calibrate --labeller ft1mix \
        --scored data/annotated/cv_hlqc/scored_*_fold*.csv \
        --greedy data/annotated/cv_hlqc/qwen_ft1mix_bare_inf_bare_ctx5_fold*.csv \
        --out outputs/selftrain/calibration_ft1mix_ctx5.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from baseline.eval import score

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(patterns: List[str]) -> pd.DataFrame:
    files = sorted({f for p in patterns for f in glob.glob(str(REPO_ROOT / p) if not Path(p).is_absolute() else p)})
    if not files:
        raise SystemExit(f"No files match {patterns}")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return d.drop_duplicates("corp_utt_idx", keep="last")


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: mean |accuracy - confidence| over equal-width bins."""
    edges = np.linspace(0, 1, bins + 1)
    total, err = len(conf), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            err += m.sum() / total * abs(correct[m].mean() - conf[m].mean())
    return float(err)


def thresholds(d: pd.DataFrame, target: float, min_n: int) -> Dict:
    """Lowest confidence per (speaker, predicted T2 code) reaching `target` precision.

    Rows predicting the code are sorted by confidence (descending); walking down,
    precision of the top-k is recomputed, and the lowest confidence at which it
    is still >= target with k >= min_n is kept. Maximises how many pseudo-labels
    of that code pass while holding their measured precision at the target.
    """
    out: Dict = {}
    for (spk, code), g in d.groupby(["speaker", "t2_label_auto"]):
        g = g.sort_values("confidence", ascending=False)
        ok = (g["t2_label_auto"] == g["t2_label_GT"]).to_numpy()
        prec = np.cumsum(ok) / np.arange(1, len(ok) + 1)
        k_ok = [k for k in range(min_n, len(ok) + 1) if prec[k - 1] >= target]
        entry = {"n_pred": int(len(ok)), "precision_all": round(float(ok.mean()), 3)}
        if k_ok:
            k = max(k_ok)
            entry.update(tau=float(g["confidence"].iloc[k - 1]), n_at_tau=int(k),
                         precision_at_tau=round(float(prec[k - 1]), 3))
        else:
            entry.update(tau=None)
        out.setdefault(spk, {})[code] = entry
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", nargs="+", required=True)
    ap.add_argument("--labeller", required=True, help="name recorded in the output, e.g. ft1mix / qwen32b_zs")
    ap.add_argument("--greedy", nargs="*", default=None, help="greedy predictions of the same model, for the argmax check")
    ap.add_argument("--target-precision", type=float, default=0.8)
    ap.add_argument("--min-n", type=int, default=15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    d = _load(args.scored)
    d = d[d["t2_label_GT"].notna() & d["t2_label_auto"].notna()].copy()
    res: Dict = {"labeller": args.labeller, "n_rows": int(len(d)),
                 "target_precision": args.target_precision, "min_n": args.min_n, "scores": []}

    for level in ("t1", "t2"):
        for scope in ("all", "counsellor", "client"):
            x = d if scope == "all" else d[d["speaker"] == scope]
            if x.empty:
                continue
            s = score(x[f"{level}_label_GT"], x[f"{level}_label_auto"], level, conv=x["conv_id"])
            res["scores"].append({"level": level.upper(), "scope": scope,
                                  **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                     for k, v in s.items()}})

    joint_ok = ((d["t1_label_auto"] == d["t1_label_GT"]) & (d["t2_label_auto"] == d["t2_label_GT"])).to_numpy()
    res["ece_joint"] = round(ece(d["confidence"].to_numpy(), joint_ok), 4)
    res["ece_t2"] = round(ece(d["t2_conf"].to_numpy(), (d["t2_label_auto"] == d["t2_label_GT"]).to_numpy()), 4)
    res["thresholds"] = thresholds(d, args.target_precision, args.min_n)

    if args.greedy:
        g = _load(args.greedy)[["corp_utt_idx", "t1_label_auto", "t2_label_auto"]]
        m = d.merge(g, on="corp_utt_idx", suffixes=("", "_greedy"))
        res["argmax_eq_greedy"] = {
            "t1": round(float((m["t1_label_auto"] == m["t1_label_auto_greedy"]).mean()), 4),
            "t2": round(float((m["t2_label_auto"] == m["t2_label_auto_greedy"]).mean()), 4),
            "n": int(len(m)),
        }

    out = Path(args.out) if Path(args.out).is_absolute() else REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=float))

    print(f"{args.labeller}: {len(d)} labelled rows")
    for s in res["scores"]:
        print(f"  {s['level']} {s['scope']:10s} acc {s['accuracy']:.3f} "
              f"[{s.get('accuracy_lo', float('nan')):.3f}-{s.get('accuracy_hi', float('nan')):.3f}]  "
              f"macro-F1 {s['f1_macro_gold']:.3f} "
              f"[{s.get('f1_macro_gold_lo', float('nan')):.3f}-{s.get('f1_macro_gold_hi', float('nan')):.3f}]")
    print(f"  ECE joint={res['ece_joint']}  ECE t2={res['ece_t2']}")
    if "argmax_eq_greedy" in res:
        print(f"  argmax == greedy: {res['argmax_eq_greedy']}")
    kept = {spk: [c for c, e in v.items() if e["tau"] is not None] for spk, v in res["thresholds"].items()}
    drop = {spk: [c for c, e in v.items() if e["tau"] is None] for spk, v in res["thresholds"].items()}
    print(f"  codes reaching {args.target_precision:.0%} precision: {kept}")
    print(f"  excluded (never reach it): {drop}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
