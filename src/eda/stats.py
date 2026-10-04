"""EDA summaries over the long format returned by `eda.registry` loaders."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def structure(df: pd.DataFrame) -> Dict[str, float]:
    words = df.text.astype(str).str.split().str.len()
    per_conv = df.groupby("conv_id").size()
    vols = df.groupby("conv_id").vol_idx.nunique()
    return {
        "conversations": df.conv_id.nunique(),
        "rows (units)": len(df),
        "counsellor rows": int((df.speaker == "counsellor").sum()),
        "client rows": int((df.speaker == "client").sum()),
        "units / conv (median)": float(per_conv.median()),
        "units / conv (min-max)": f"{per_conv.min()}-{per_conv.max()}",
        "turns / conv (median)": float(vols.median()),
        "words / unit (median)": float(words.median()),
        "words / unit (p95)": float(words.quantile(0.95)),
        "coded rows": int(df.native.notna().sum()),
        "MISC T2 rows": int(df.t2.notna().sum()),
        "MISC T1 rows": int(df.t1.notna().sum()),
    }


def label_table(df: pd.DataFrame, col: str = "t2") -> pd.DataFrame:
    """Counts and shares per speaker; flags tail codes (<1% of that speaker's coded rows or <20)."""
    rows = []
    for spk, g in df[df[col].notna()].groupby("speaker"):
        vc = g[col].astype(str).value_counts()
        for code, n in vc.items():
            share = n / vc.sum()
            rows.append({"speaker": spk, "code": code, "n": int(n), "share": round(share, 4),
                         "is_tail": bool(share < 0.01 or n < 20)})
    return pd.DataFrame(rows)


def position_profile(df: pd.DataFrame, col: str = "t1", bins: int = 10) -> pd.DataFrame:
    """Share of each label by session-position decile (where in the session codes occur)."""
    d = df[df[col].notna()].copy()
    if d.empty:
        return d
    d["pos"] = d.groupby("conv_id").cumcount() / d.groupby("conv_id").conv_id.transform("size")
    d["decile"] = np.minimum((d.pos * bins).astype(int), bins - 1)
    return pd.crosstab(d.decile, d[col], normalize="index").round(3)


def text_quality(df: pd.DataFrame) -> Dict[str, float]:
    t = df.text.astype(str)
    fillers = t.str.lower().str.contains(r"\b(?:um+|uh+|mm-?hmm|uh-huh|you know)\b")
    return {
        "has uppercase": float(t.str.contains(r"[A-Z]").mean()),
        "has sentence punctuation": float(t.str.contains(r"[.?!]").mean()),
        "contains filler (um/uh/mm-hmm/you know)": float(fillers.mean()),
        "type-token ratio (5k-word sample)": _ttr(t),
    }


def _ttr(t: pd.Series, n: int = 5000, seed: int = 0) -> float:
    toks = " ".join(t.sample(frac=1, random_state=seed)).lower().split()[:n]
    return round(len(set(toks)) / max(1, len(toks)), 3)


def js_divergence(p: pd.Series, q: pd.Series) -> float:
    idx = p.index.union(q.index)
    p = p.reindex(idx, fill_value=0).astype(float) + 1e-9
    q = q.reindex(idx, fill_value=0).astype(float) + 1e-9
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    kl = lambda a, b: float((a * np.log2(a / b)).sum())
    return round(0.5 * kl(p, m) + 0.5 * kl(q, m), 4)


def js_matrix(frames: Dict[str, pd.DataFrame], speaker: str = "counsellor", col: str = "t2") -> pd.DataFrame:
    dists = {k: f[(f.speaker == speaker) & f[col].notna()][col].value_counts()
             for k, f in frames.items() if ((f.speaker == speaker) & f[col].notna()).sum() > 0}
    ks = list(dists)
    return pd.DataFrame([[js_divergence(dists[a], dists[b]) for b in ks] for a in ks], index=ks, columns=ks)
