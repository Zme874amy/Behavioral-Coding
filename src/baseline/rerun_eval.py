"""Scoring for re-run cells (docs/RERUN_PLAN.md P4).

Scores the POOLED out-of-fold predictions of a cell once (all 821 MIV
utterances for MISC), with:

  primary     T2 macro-F1 per speaker over LEARNABLE codes: a code counts if it
              occurs in the training labels of every fold where it has gold
              support (per-fold, from baseline.fold_stats; replaces eval's
              global HLQC list)
  secondary   accuracy, kappa, macro-F1 over all gold codes, T1 accuracy,
              tail macro-F1 (codes rare in training, fold_stats rule), and the
              never-seen list (gold codes no fold trained on; reported, not scored)
  robust      the same after merging known-noisy distinctions (never replaces
              the primary): FA+FI -> FA ("O-minor"), and SR/CR scored at T1
              (reflection) - DATASETS.md section 4
  session     MI summary scores per session from predicted vs gold codes
              (R:Q, %CR, %OQ, %MIC, %CT), agreement by ICC(2,1) and MAE: the
              clinical use (feedback per session), less sensitive to single labels
  CIs         session-cluster bootstrap (N_BOOT draws)
  paired      arm vs reference over matched seeds: seed-averaged delta, paired
              session bootstrap CI, session-level sign-flip permutation p, and
              the verdict rule: "real" only if the CI excludes 0 AND the sign
              holds in >= 2 of 3 seeds

    PYTHONPATH=src python -m baseline.rerun_eval --stage conf/stages/stage1.yaml
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from schemes.misc import MICO, MIIN

N_BOOT = 5000
N_PERM = 5000
SEED = 0
SPEAKERS = ("counsellor", "client")

ROBUST = {
    "fa_fi_merged": {"FI": "FA"},
    "reflection_t1": {"SR": "REFL", "CR": "REFL"},
}


# ------------------------------------------------------------------ label sets
def learnable_codes(df: pd.DataFrame, fold_learnable: Dict[int, Dict[str, List[str]]], speaker: str) -> List[str]:
    """Gold codes learnable in every fold where they have gold support."""
    d = df[df["speaker"] == speaker]
    out = []
    for c, g in d.groupby("t2_label_GT"):
        if all(c in fold_learnable[int(k)][speaker] for k in g["fold"].unique()):
            out.append(c)
    return sorted(out)


def never_seen(df: pd.DataFrame, fold_learnable, speaker: str) -> List[str]:
    d = df[df["speaker"] == speaker]
    return sorted(c for c, g in d.groupby("t2_label_GT")
                  if not any(c in fold_learnable[int(k)][speaker] for k in g["fold"].unique()))


# ------------------------------------------------------------------ metrics
def _conf(yt, yp, groups, labels):
    K = len(labels)
    idx = {c: i for i, c in enumerate(labels)}
    G = {g: i for i, g in enumerate(pd.unique(groups))}
    conf = np.zeros((len(G), K * K))
    np.add.at(conf, (np.array([G[g] for g in groups]),
                     np.array([idx[c] for c in yt]) * K + np.array([idx[c] for c in yp])), 1.0)
    return conf, idx, len(G)


def _f1_from(C, K):
    C = C.reshape(-1, K, K)
    diag = np.diagonal(C, axis1=1, axis2=2)
    row, col = C.sum(2), C.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(col > 0, diag / col, 0.0)
        r = np.where(row > 0, diag / row, 0.0)
        f1 = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
        acc = diag.sum(1) / np.maximum(C.sum((1, 2)), 1)
    return f1, row, acc


def macro(f1, row, idx_codes):
    if not len(idx_codes):
        return np.full(f1.shape[0], np.nan)
    sup = row[:, idx_codes] > 0
    n = sup.sum(1)
    return np.where(n > 0, (f1[:, idx_codes] * sup).sum(1) / np.maximum(n, 1), np.nan)


def score_block(yt: Sequence[str], yp: Sequence[str], conv: Sequence[str], code_sets: Dict[str, List[str]],
                boot: bool = True) -> dict:
    """Accuracy + macro-F1 over each named code set, with session-bootstrap CIs."""
    yt, yp, conv = np.asarray(yt), np.asarray(yp), np.asarray(conv)
    labels = sorted(set(yt) | set(yp))
    conf, idx, G = _conf(yt, yp, conv, labels)
    K = len(labels)
    total = conf.sum(0, keepdims=True)
    f1, row, acc = _f1_from(total, K)
    out = {"n": int(len(yt)), "accuracy": float(acc[0])}
    sets = {k: np.array([idx[c] for c in v if c in idx], dtype=int) for k, v in code_sets.items()}
    for k, ix in sets.items():
        out[f"f1_{k}"] = float(macro(f1, row, ix)[0])
    if boot and G > 1:
        rng = np.random.default_rng(SEED)
        w = rng.multinomial(G, np.full(G, 1 / G), size=N_BOOT).astype(float)
        f1b, rowb, accb = _f1_from(w @ conf, K)
        out["accuracy_ci"] = tuple(np.nanquantile(accb, [0.025, 0.975]).round(4))
        for k, ix in sets.items():
            out[f"f1_{k}_ci"] = tuple(np.nanquantile(macro(f1b, rowb, ix), [0.025, 0.975]).round(4))
    return out


def score_cell(df: pd.DataFrame, fold_stats: Dict[int, dict]) -> Dict[str, dict]:
    """Every metric for one pooled cell (one seed). df has fold, uid, conv_id, speaker, *_GT, *_auto."""
    learn = {k: v["learnable_t2"] for k, v in fold_stats.items()}
    rare = {k: v["rare_t2"] for k, v in fold_stats.items()}
    res = {}
    for spk in SPEAKERS:
        d = df[(df["speaker"] == spk) & df["t2_label_GT"].notna()].copy()
        d["t2_label_auto"] = d["t2_label_auto"].fillna("UNKNOWN")
        L = learnable_codes(d, learn, spk)
        tail = [c for c in L if any(c in rare[int(k)][spk] for k in d.loc[d.t2_label_GT == c, "fold"].unique())]
        gold = sorted(d["t2_label_GT"].unique())
        r = score_block(d.t2_label_GT, d.t2_label_auto, d.conv_id, {"learnable": L, "gold": gold, "tail": tail})
        r.update({"learnable_codes": L, "tail_codes": tail, "never_seen": never_seen(d, learn, spk)})
        t1 = df[(df["speaker"] == spk) & df["t1_label_GT"].notna()]
        r["t1_accuracy"] = float((t1.t1_label_GT == t1.t1_label_auto).mean())
        for name, m in ROBUST.items():
            if spk == "client" and name != "fa_fi_merged":
                continue
            yt, yp = d.t2_label_GT.replace(m), d.t2_label_auto.replace(m)
            Lr = sorted({m.get(c, c) for c in L})
            r[f"robust_{name}"] = score_block(yt, yp, d.conv_id, {"learnable": Lr}, boot=False)["f1_learnable"]
        res[spk] = r
    res["session"] = session_agreement(df)
    return res


# ------------------------------------------------------------------ session level
def _summary(d: pd.DataFrame, col: str) -> pd.DataFrame:
    def ratio(a, b):
        return a / b if b else np.nan
    rows = {}
    for conv, g in d.groupby("conv_id"):
        c = g[g.speaker == "counsellor"][col].value_counts()
        k = g[g.speaker == "client"][col.replace("t2", "t1")].value_counts()
        refl, q = c.get("SR", 0) + c.get("CR", 0), c.get("OQ", 0) + c.get("CQ", 0)
        mico, miin = sum(c.get(x, 0) for x in MICO), sum(c.get(x, 0) for x in MIIN)
        rows[conv] = {"R:Q": ratio(refl, q), "%CR": ratio(c.get("CR", 0), refl), "%OQ": ratio(c.get("OQ", 0), q),
                      "%MIC": ratio(mico, mico + miin), "%CT": ratio(k.get("C", 0), k.get("C", 0) + k.get("S", 0))}
    return pd.DataFrame(rows).T


def icc_2_1(x: np.ndarray, y: np.ndarray) -> float:
    """Two-way random, absolute agreement, single rater (Shrout & Fleiss ICC(2,1))."""
    m = ~(np.isnan(x) | np.isnan(y))
    X = np.c_[x[m], y[m]]
    n, k = X.shape
    if n < 3:
        return np.nan
    gm = X.mean()
    msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((X.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / den) if den else np.nan


def session_agreement(df: pd.DataFrame) -> Dict[str, dict]:
    gold, pred = _summary(df, "t2_label_GT"), _summary(df, "t2_label_auto")
    out = {}
    for m in gold.columns:
        x, y = gold[m].astype(float).values, pred[m].astype(float).values
        out[m] = {"icc": icc_2_1(x, y), "mae": float(np.nanmean(np.abs(x - y))), "n_sessions": int(len(x))}
    return out


# ------------------------------------------------------------------ paired comparison
def paired(arm: Dict[int, pd.DataFrame], ref: Dict[int, pd.DataFrame], fold_stats_by_seed: Dict[int, dict],
           speaker: str) -> dict:
    """Seed-matched arm - ref on learnable macro-F1 for one speaker."""
    seeds = sorted(set(arm) & set(ref))
    deltas, boots, perm_stats = [], [], []
    for s in seeds:
        a = arm[s][arm[s].speaker == speaker].set_index("uid")
        r = ref[s][ref[s].speaker == speaker].set_index("uid")
        common = a.index.intersection(r.index)
        a, r = a.loc[common], r.loc[common]
        learn = {k: v["learnable_t2"] for k, v in fold_stats_by_seed[s].items()}
        L = learnable_codes(a.reset_index().assign(fold=a["fold"].values), learn, speaker)
        yt = a.t2_label_GT.values
        pa, pr = a.t2_label_auto.fillna("UNKNOWN").values, r.t2_label_auto.fillna("UNKNOWN").values
        conv = a.conv_id.values
        labels = sorted(set(yt) | set(pa) | set(pr))
        ca, idx, G = _conf(yt, pa, conv, labels)
        cr, _, _ = _conf(yt, pr, conv, labels)
        K, ix = len(labels), np.array([idx[c] for c in L], dtype=int)
        point = lambda C: macro(*_f1_from(C, K)[:2], ix)
        deltas.append(float(point(ca.sum(0, keepdims=True))[0] - point(cr.sum(0, keepdims=True))[0]))
        rng = np.random.default_rng(SEED)
        w = rng.multinomial(G, np.full(G, 1 / G), size=N_BOOT).astype(float)
        boots.append(point(w @ ca) - point(w @ cr))
        flip = rng.integers(0, 2, size=(N_PERM, G)).astype(float)      # swap arm/ref per session
        perm_stats.append(point(flip @ ca + (1 - flip) @ cr) - point(flip @ cr + (1 - flip) @ ca))
    if not seeds:
        return {"seeds": 0, "verdict": "pending"}
    d, B, P = float(np.mean(deltas)), np.nanmean(boots, 0), np.nanmean(perm_stats, 0)
    ci = tuple(np.nanquantile(B, [0.025, 0.975]).round(4))
    p = float((np.abs(P) >= abs(d)).mean())
    same_sign = sum(np.sign(x) == np.sign(d) for x in deltas)
    real = (ci[0] > 0 or ci[1] < 0) and same_sign >= min(2, len(seeds)) and len(seeds) >= 2
    verdict = "real" if real else ("single-seed" if len(seeds) == 1 and (ci[0] > 0 or ci[1] < 0) else "ns")
    return {"seeds": len(seeds), "delta": d, "per_seed": deltas, "ci": ci, "perm_p": p, "verdict": verdict}


# ------------------------------------------------------------------ CLI
def _fold_stats(cell_files: Dict[int, pd.DataFrame]) -> Dict[int, dict]:
    import json
    out = {}
    for k, path in cell_files.items():
        out[k] = json.loads(path.with_suffix(".meta.json").read_text())["train_label_stats"]
    return out


def main() -> None:
    import yaml
    from baseline import folds, rerun
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--reference", default="ft1mix_bare")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.stage))
    rows = []
    for cell in cfg["cells"]:
        for pname in cell["protocols"]:
            pr = cfg["protocols"][pname]
            for sname in cell["students"]:
                st = cfg["students"][sname]
                if not st.get("enabled", True):
                    continue
                for seed in cell.get("seeds", cfg["seeds"]):
                    src = {"misc.hlqc.gold": cell["source"]} if cell.get("source") else None
                    nf = folds.n_folds(pr["manifest"], pr["design"])
                    try:
                        df = rerun.pooled(pr["manifest"], pr["design"], rerun.student_slug(st["model"]), cell["arm"],
                                          cell["inf"], cfg["ctx"], cfg["prompt"], seed, nf, src)
                    except FileNotFoundError as e:
                        rows.append({"cell": rerun.arm_label(cell["arm"], src), "protocol": pname,
                                     "student": sname, "seed": seed, "status": f"missing: {e}"})
                        continue
                    paths = {k: rerun.result_path(pr["manifest"], pr["design"], rerun.student_slug(st["model"]),
                                                  cell["arm"], cell["inf"], cfg["ctx"], cfg["prompt"], seed, k, src)
                             for k in range(nf)}
                    res = score_cell(df, _fold_stats(paths))
                    rows.append({"cell": rerun.arm_label(cell["arm"], src), "protocol": pname, "student": sname,
                                 "seed": seed, "status": "ok",
                                 **{f"{s}_{m}": res[s][m] for s in SPEAKERS
                                    for m in ("f1_learnable", "f1_tail", "accuracy", "t1_accuracy")}})
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------ MITI space (CASAA)
def score_miti(df: pd.DataFrame) -> dict:
    """A MISC model's CASAA predictions scored in MITI 4.2.1 space.

    Gold = the CASAA MITI code of single-code counsellor turns (multi-code and SAME
    turns dropped, Seek excluded from the macro: no MISC code maps to it). The
    prediction's MISC T2 code is mapped with schemes.mappings.misc_to_miti; an
    unmappable prediction counts as wrong ("UNMAPPED").
    """
    from schemes.mappings import misc_to_miti
    d = df[(df["speaker"] == "counsellor") & df["miti_codes"].notna()].copy()
    d = d[~d["miti_codes"].astype(str).str.contains("|", regex=False) & (d["miti_codes"] != "SAME")]
    if "hlqc_overlap" in d.columns:
        d = d[d["hlqc_overlap"].isna()]                     # Emmy / Rounder duplicate HLQC sessions
    d["pred_miti"] = d["t2_label_auto"].map(lambda c: misc_to_miti(c) or "UNMAPPED")
    codes = sorted(set(d["miti_codes"]) - {"Seek"})
    r = score_block(d["miti_codes"], d["pred_miti"], d["conv_id"], {"miti": codes})
    r["codes"] = codes
    return r
