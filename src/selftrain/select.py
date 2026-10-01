"""Select confident pseudo-labels and build an augmented training CSV.

Takes one or more pseudo-labeled pool CSVs (from `label_pool.py`) plus the human
training CSV, and emits an augmented CSV in the manual schema that
`automisc_ft.data.build_tier_examples` consumes unchanged.

Design choices that matter:
  * Whole pool conversations are retained so `build_context_excerpt` still sees
    real neighbours; only ACCEPTED rows carry a `t{1,2}_label_GT`. Non-accepted
    rows keep their text (context) but have null labels, so they never become
    training targets.
  * Acceptance = confidence >= threshold AND hierarchy/speaker validity. Head
    classes are capped per code; rare codes are exempt so the tail is preserved.
  * Pool conv_ids are namespaced (``<tag>:<id>``) so they cannot collide with
    the human conversations when both live in one frame.

Usage:
    PYTHONPATH=src python -m selftrain.select \
        --pseudo data/selftrain/pseudo/annomi_ft_bare_ctx5.csv \
        --train-csv data/manual/HLQC_balanced_manual.csv \
        --out data/selftrain/aug/annomi_ft_bare_ctx5.csv \
        --min-confidence 0.8 --per-class-cap 300
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Optional

import pandas as pd

from automisc_ft.data import t1_codes_for_speaker, t2_codes_for_group
from baseline.local_arm import REPO_ROOT

RARE_DEFAULT = ["SU", "EC", "AF", "GI", "TS+", "AC-"]

MANUAL_COLS = [
    "corp_conv_idx", "conv_id", "speaker", "corp_vol_idx", "conv_vol_idx",
    "vol_text", "corp_utt_idx", "conv_utt_idx", "utt_text",
    "t1_label_GT", "t2_label_GT",
]


def _valid(speaker: str, t1: str, t2: str) -> bool:
    """T1 valid for the speaker and T2 a valid child of that T1 group."""
    if not isinstance(t1, str) or not isinstance(t2, str):
        return False
    if t1 in ("", "UNKNOWN") or t2 in ("", "UNKNOWN"):
        return False
    if t1 not in t1_codes_for_speaker(speaker):
        return False
    return t2 in t2_codes_for_group(speaker, t1)


def _tag_from_path(p: str) -> str:
    return Path(p).stem.split("_")[0].lower()


def select_pool(
    pseudo: pd.DataFrame,
    tag: str,
    min_conf: float,
    per_class_cap: Optional[int],
    rare: List[str],
    rng: random.Random,
) -> pd.DataFrame:
    """Return pool rows (whole conversations) with GT set only on accepted rows."""
    df = pseudo.copy()
    df["conv_id"] = tag + ":" + df["conv_id"].astype(str)

    conf = df.get("confidence")
    valid = df.apply(lambda r: _valid(r["speaker"], r.get("t1_label_auto"), r.get("t2_label_auto")), axis=1)
    accept = valid & (conf.fillna(0.0) >= float(min_conf) if conf is not None else valid)

    # Per-class caps on head codes; rare codes exempt so the tail survives.
    if per_class_cap is not None:
        rare_set = set(rare)
        for (spk, code), idx in df[accept].groupby(["speaker", "t2_label_auto"]).groups.items():
            if code in rare_set:
                continue
            idx = list(idx)
            if len(idx) > per_class_cap:
                drop = rng.sample(idx, len(idx) - per_class_cap)
                accept.loc[drop] = False

    df["t1_label_GT"] = df["t1_label_auto"].where(accept)
    df["t2_label_GT"] = df["t2_label_auto"].where(accept)

    # Keep only conversations that contribute at least one accepted row.
    keep_convs = set(df.loc[accept, "conv_id"].unique())
    df = df[df["conv_id"].isin(keep_convs)].reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# v2 selection: self-ensemble + calibrated per-code thresholds + prior alignment
# -----------------------------------------------------------------------------
KEY = ["conv_id", "corp_utt_idx"]


def merge_ensemble(paths: List[str]) -> pd.DataFrame:
    """Combine labellings of the SAME pool rows by several copies of one model.

    A row is a candidate only if every labelling agrees on its (T1, T2) argmax;
    its confidence is the mean of their joint confidences. Independently trained
    copies make different mistakes, so agreement filters errors that one model's
    own sampled votes (which share its biases) cannot (Co-teaching; SemiEvol).
    A single path is a single labeller (the teacher arm).
    """
    frames = [pd.read_csv(REPO_ROOT / p if not Path(p).is_absolute() else p) for p in paths]
    base = frames[0].copy()
    if len(frames) == 1:
        base["n_agree"] = base["t2_label_auto"].notna().astype(int)
        return base
    lab = [f.set_index(KEY)[["t1_label_auto", "t2_label_auto", "confidence"]] for f in frames]
    joined = base.set_index(KEY)
    agree = pd.Series(True, index=joined.index)
    for f in lab[1:]:
        f = f.reindex(joined.index)
        agree &= (f["t1_label_auto"] == joined["t1_label_auto"]) & (f["t2_label_auto"] == joined["t2_label_auto"])
    conf = pd.concat([f["confidence"].reindex(joined.index) for f in lab], axis=1).mean(axis=1)
    joined["confidence"] = conf.where(agree)
    joined["n_agree"] = agree.astype(int) * len(frames)
    for c in ("t1_label_auto", "t2_label_auto"):
        joined[c] = joined[c].where(agree)
    return joined.reset_index()


def human_prior(human: pd.DataFrame, speaker: str) -> pd.Series:
    return human.loc[human["speaker"] == speaker, "t2_label_GT"].dropna().value_counts(normalize=True)


def select_v2(pool: pd.DataFrame, human: pd.DataFrame, tag: str, thresholds: dict,
              alpha: float, budget_ratio: float, speakers: List[str]) -> pd.DataFrame:
    """Accept pseudo-labels that pass the labeller's own calibrated per-code
    threshold, then fill per-code quotas proportional to the human prior**alpha.

    * Threshold: the lowest confidence at which, on labelled HLQC, this labeller's
      predictions of that code were right >= the target precision
      (`selftrain.calibrate`). Codes that never reach it get no pseudo-labels.
    * Prior alignment (CReST/DARP): quota_c = budget * p_c**alpha / sum p**alpha,
      so the pseudo-label mix follows the human mix, mildly flattened, instead of
      whatever the labeller happens to be confident about (v1 inverted it).
    * Budget: at most `budget_ratio` pseudo-labels per human label, per speaker,
      so human supervision stays dominant (Arazo et al.).
    Highest-confidence rows fill each quota first. Whole conversations are kept
    for context; only accepted rows carry labels.
    """
    df = pool.copy()
    df["conv_id"] = tag + ":" + df["conv_id"].astype(str)
    valid = df.apply(lambda r: _valid(r["speaker"], r.get("t1_label_auto"), r.get("t2_label_auto")), axis=1)
    accept = pd.Series(False, index=df.index)
    for spk in speakers:
        tau = {c: e["tau"] for c, e in thresholds.get(spk, {}).items() if e.get("tau") is not None}
        prior = human_prior(human, spk)
        weights = prior[prior.index.isin(tau)] ** alpha
        if weights.empty:
            continue
        budget = int(budget_ratio * human.loc[human["speaker"] == spk, "t2_label_GT"].notna().sum())
        quota = (weights / weights.sum() * budget).round().astype(int)
        cand = df[valid & (df["speaker"] == spk) & df["t2_label_auto"].isin(tau.keys())]
        cand = cand[cand["confidence"] >= cand["t2_label_auto"].map(tau)]
        for code, q in quota.items():
            top = cand[cand["t2_label_auto"] == code].nlargest(int(q), "confidence")
            accept.loc[top.index] = True
    df["t1_label_GT"] = df["t1_label_auto"].where(accept)
    df["t2_label_GT"] = df["t2_label_auto"].where(accept)
    keep = set(df.loc[accept, "conv_id"])
    return df[df["conv_id"].isin(keep)].reset_index(drop=True)


def quality_report(sel: pd.DataFrame, human: pd.DataFrame, miti_csv: Optional[str]) -> dict:
    """The quality gate, printed before any training.

    MITI agreement is a METER only (independent human labels on the Welivita
    counsellor turns, over codes both schemes share); it never selects.
    """
    import numpy as np

    SHARED = {"GI", "ADW", "CR", "SU", "AF", "CQ", "DI", "SR", "ADP", "OQ", "CO", "EC", "WA"}
    acc = sel[sel["t2_label_GT"].notna()].copy()
    rep: dict = {"n_pseudo": int(len(acc)),
                 "pseudo_by_speaker": acc["speaker"].value_counts().to_dict(),
                 "mean_confidence": round(float(acc["confidence"].mean()), 3) if "confidence" in acc else None}
    n_h = human["t2_label_GT"].notna().groupby(human["speaker"]).sum()
    tot = {s: int(n_h.get(s, 0)) + int(rep["pseudo_by_speaker"].get(s, 0)) for s in n_h.index}
    rep["client_share_targets"] = {"human_only": round(float(n_h.get("client", 0) / n_h.sum()), 3),
                                   "with_pseudo": round(tot.get("client", 0) / sum(tot.values()), 3)}
    for spk in ("counsellor", "client"):
        p = acc.loc[acc["speaker"] == spk, "t2_label_GT"].value_counts(normalize=True)
        if p.empty:
            continue
        h = human_prior(human, spk)
        idx = sorted(set(p.index) | set(h.index))
        pp, hh = p.reindex(idx).fillna(1e-6), h.reindex(idx).fillna(1e-6)
        rep[f"{spk}_kl_to_human"] = round(float((pp * np.log(pp / hh)).sum()), 3)
        rep[f"{spk}_top_codes"] = {c: [round(float(p.get(c, 0)), 3), round(float(h.get(c, 0)), 3)]
                                   for c in p.sort_values(ascending=False).index[:6]}
    if miti_csv:
        w = pd.read_csv(REPO_ROOT / miti_csv, usecols=["conv_id", "corp_utt_idx", "weak_t2"], low_memory=False)
        w["conv_id"] = w["conv_id"].astype(str)
        a = acc[acc["speaker"] == "counsellor"].copy()
        a["orig"] = a["conv_id"].str.split(":", n=1).str[1]
        m = a.merge(w, left_on=["orig", "corp_utt_idx"], right_on=["conv_id", "corp_utt_idx"])
        m = m[m["weak_t2"].isin(SHARED) & m["t2_label_GT"].isin(SHARED)]
        rep["miti_agreement_shared"] = round(float((m["t2_label_GT"] == m["weak_t2"]).mean()), 3) if len(m) else None
        rep["miti_n"] = int(len(m))
    return rep


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pseudo", nargs="+", default=None, help="one or more pseudo-labeled pool CSVs (v1)")
    # v2 (self-training v2 / teacher comparison); v1 flags are ignored when --ensemble is given
    ap.add_argument("--ensemble", nargs="+", default=None,
                    help="labellings of ONE pool by several copies of a model (or one labeller)")
    ap.add_argument("--tag", default=None, help="conv_id namespace for --ensemble (default: from filename)")
    ap.add_argument("--thresholds", default=None, help="calibration JSON from selftrain.calibrate")
    ap.add_argument("--alpha", type=float, default=0.5, help="prior-alignment exponent (0=uniform, 1=human prior)")
    ap.add_argument("--budget-ratio", type=float, default=1.0, help="max pseudo labels per human label, per speaker")
    ap.add_argument("--speakers", nargs="+", default=["counsellor"], help="speakers that may receive pseudo labels")
    ap.add_argument("--report", action="store_true", help="print the quality gate (and write <out>.report.json)")
    ap.add_argument("--miti-csv", default=None, help="Welivita parsed CSV with weak_t2, for the MITI meter")
    ap.add_argument("--train-csv", default="data/manual/HLQC_balanced_manual.csv", help="human training CSV")
    ap.add_argument("--out", required=True, help="destination augmented training CSV")
    ap.add_argument("--min-confidence", type=float, default=0.8)
    ap.add_argument("--per-class-cap", type=int, default=300, help="max accepted rows per head T2 code; -1 disables")
    ap.add_argument("--rare-codes", nargs="*", default=RARE_DEFAULT, help="T2 codes exempt from the cap")
    ap.add_argument("--human-repeat", type=int, default=1, help="duplicate the human rows this many times to keep them dominant")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    cap = None if int(args.per_class_cap) < 0 else int(args.per_class_cap)

    human = pd.read_csv(REPO_ROOT / args.train_csv if not Path(args.train_csv).is_absolute() else args.train_csv)
    human_out = pd.concat([human] * max(1, int(args.human_repeat)), ignore_index=True)

    pool_frames = []
    if args.ensemble:
        import json
        if not args.thresholds:
            raise SystemExit("--ensemble needs --thresholds (from selftrain.calibrate)")
        thr = json.loads((REPO_ROOT / args.thresholds if not Path(args.thresholds).is_absolute()
                          else Path(args.thresholds)).read_text())["thresholds"]
        pool = merge_ensemble(args.ensemble)
        tag = args.tag or _tag_from_path(args.ensemble[0])
        sel = select_v2(pool, human, tag, thr, args.alpha, args.budget_ratio, list(args.speakers))
        print(f"v2 select ({len(args.ensemble)} labelling(s)): "
              f"{int(sel['t2_label_GT'].notna().sum())} accepted rows across "
              f"{sel['conv_id'].nunique()} conversations (tag={tag})")
        pool_frames.append(sel)
    else:
        if not args.pseudo:
            raise SystemExit("give --pseudo (v1) or --ensemble (v2)")
        for p in args.pseudo:
            pseudo = pd.read_csv(REPO_ROOT / p if not Path(p).is_absolute() else p)
            tag = _tag_from_path(p)
            sel = select_pool(pseudo, tag, args.min_confidence, cap, list(args.rare_codes), rng)
            n_acc = int(sel["t2_label_GT"].notna().sum())
            print(f"{p}: {n_acc} accepted rows across {sel['conv_id'].nunique()} conversations (tag={tag})")
            pool_frames.append(sel)

    if args.report:
        import json
        rep = quality_report(pd.concat(pool_frames, ignore_index=True), human, args.miti_csv)
        print("QUALITY GATE:", json.dumps(rep, indent=2))
        out_rep = (Path(args.out) if Path(args.out).is_absolute() else REPO_ROOT / args.out)
        out_rep.parent.mkdir(parents=True, exist_ok=True)
        out_rep.with_suffix(".report.json").write_text(json.dumps(rep, indent=2))

    frames = [human_out] + pool_frames
    cols = list(dict.fromkeys(MANUAL_COLS + [c for f in frames for c in f.columns]))
    aug = pd.concat([f.reindex(columns=cols) for f in frames], ignore_index=True)

    out_path = Path(args.out) if Path(args.out).is_absolute() else REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    aug.to_csv(out_path, index=False)

    n_human = len(human_out)
    n_pool_examples = int(pd.concat(pool_frames)["t2_label_GT"].notna().sum()) if pool_frames else 0
    print(
        f"Wrote augmented train CSV: {len(aug)} rows "
        f"({n_human} human incl. x{args.human_repeat}, {n_pool_examples} accepted pseudo examples) -> {out_path}"
    )


if __name__ == "__main__":
    main()
