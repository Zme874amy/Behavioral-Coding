"""Evidence for choosing the MISC train/test protocol (docs/experiments/2026-10-05-split-review.md).

A transparent CPU proxy: one TF-IDF (word 1-2 + char 2-4) logistic-regression
classifier per speaker, trained under each candidate protocol and scored on the
same gold. It does not replace the 7B model; it isolates how much each protocol's
training data matches the test labels (domain, register and coding convention),
which is the question the protocol choice turns on.

    PYTHONPATH=src python -m eda.split_eval
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline, make_union

from eda import quality, registry

warnings.filterwarnings("ignore")
SEED = 42


def _model():
    feats = make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                       TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True))
    return make_pipeline(feats, LogisticRegression(max_iter=2000, C=5, class_weight="balanced"))


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    pred = np.empty(len(test), dtype=object)
    for spk in ("counsellor", "client"):
        tr, m = train[train.speaker == spk], (test.speaker == spk).values
        if m.any():
            pred[m] = _model().fit(tr.text, tr.t2).predict(test.text[m])
    return pred


def _score(test: pd.DataFrame, pred: np.ndarray) -> dict:
    out = {}
    for spk in ("counsellor", "client"):
        m = (test.speaker == spk).values
        y, p = test.t2[m], pred[m]
        out[f"{spk} macro-F1"] = round(f1_score(y, p, average="macro", labels=sorted(y.unique()), zero_division=0), 3)
        out[f"{spk} acc"] = round(accuracy_score(y, p), 3)
    return out


def miv_folds(n_folds: int = 5) -> dict:
    from automisc_ft.data import assign_folds
    return assign_folds(registry.load("misc.miv63a.gold"), n_folds, SEED)


def protocols() -> pd.DataFrame:
    """P0 current (HLQC -> MIV), P1 reversed, P2 MIV-only CV, P3 HLQC + MIV CV; all MIV scores cover all 821."""
    H, M = registry.load("misc.hlqc.gold"), registry.load("misc.miv63a.gold")
    folds = M.conv_id.map(miv_folds())
    res = {"P0  train HLQC -> test MIV (current)": _score(M, _fit_predict(H, M))}
    for name, with_h in (("P2  MIV 5-fold CV, MIV only", False), ("P3  HLQC + other MIV folds -> MIV fold", True)):
        pred = np.empty(len(M), dtype=object)
        for k in sorted(folds.unique()):
            te, tr = M[folds == k], M[folds != k]
            pred[(folds == k).values] = _fit_predict(pd.concat([H, tr]) if with_h else tr, te)
        res[name] = _score(M, pred)
    res["P1  train MIV -> test HLQC (reversed)"] = _score(H, _fit_predict(M, H))
    return pd.DataFrame(res).T


def external_casaa() -> pd.DataFrame:
    """Does adding MIV to training hurt a third domain? Counsellor only, CASAA clean sessions."""
    from automisc_ft.data import COUNSELLOR_GROUPS
    H, M, C = (registry.load(k) for k in ("misc.hlqc.gold", "misc.miv63a.gold", "miti.casaa.gold"))
    keep = set(json.load(open(registry.REPO / "data/splits/casaa_test.json"))["test"])
    c = C[C.conv_id.isin(keep) & (C.speaker == "counsellor")]
    t2, t1 = c[c.t2.notna()], c[c.t1.notna()]
    group = {code: g for g, codes in COUNSELLOR_GROUPS.items() for code in codes}
    rows = {}
    for name, tr in (("train HLQC", H), ("train MIV", M), ("train HLQC + MIV", pd.concat([H, M]))):
        tr = tr[tr.speaker == "counsellor"]
        clf = _model().fit(tr.text, tr.t2)
        p2, p1 = clf.predict(t2.text), pd.Series(clf.predict(t1.text)).map(group)
        rows[name] = {"CASAA T2 macro-F1 (6 exact codes)": round(f1_score(t2.t2, p2, average="macro",
                                                                            labels=sorted(t2.t2.unique()), zero_division=0), 3),
                      "CASAA T2 acc": round(accuracy_score(t2.t2, p2), 3),
                      "CASAA T1 acc": round(accuracy_score(t1.t1, p1), 3)}
    return pd.DataFrame(rows).T


def template_leak() -> dict:
    """Exact-text repeats of longer MIV utterances across CV folds (chatbot templates)."""
    M = registry.load("misc.miv63a.gold").assign(k=lambda d: d.text.map(quality.norm))
    M["fold"] = M.conv_id.map(miv_folds())
    long = M.k.str.split().str.len() >= 6
    rep = [bool((M[M.fold != f].k == k).any()) for k, f in zip(M.k, M.fold)]
    return {"long utterances": int(long.sum()), "repeated in another fold": int((pd.Series(rep) & long.values).sum())}


def test_power(pred_path: str = "data/annotated/baseline/qwen_ft1mix_bare_inf_bare_ctx5.csv", n_boot: int = 1000) -> pd.DataFrame:
    """Session-cluster bootstrap CI of the main model's T2 macro-F1 on the MIV test, and thin codes."""
    p = pd.read_csv(registry.REPO / pred_path)
    p["spk"] = np.where(p.speaker.str.lower().str.startswith("couns"), "counsellor", "client")
    rng = np.random.default_rng(0)
    convs = p.conv_id.unique()
    rows = []
    for spk in ("counsellor", "client"):
        d = p[p.spk == spk]
        f = lambda x: f1_score(x.t2_label_GT, x.t2_label_auto, average="macro", labels=sorted(x.t2_label_GT.unique()), zero_division=0)
        bs = [f(pd.concat([d[d.conv_id == c] for c in rng.choice(convs, len(convs))])) for _ in range(n_boot)]
        loo = [f(d[d.conv_id != c]) for c in convs]
        sup = d.t2_label_GT.value_counts()
        rows.append({"speaker": spk, "macro-F1": round(f(d), 3), "95% CI": f"[{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]",
                     "drop-one-session range": f"{min(loo):.3f}-{max(loo):.3f}", "codes": len(sup),
                     "codes with < 5 test examples": ", ".join(sup[sup < 5].index)})
    return pd.DataFrame(rows).set_index("speaker")


def test_representativeness() -> pd.DataFrame:
    from scipy.stats import mannwhitneyu
    pool = registry.load("pool.miv63a")
    test = set(registry.load("misc.miv63a.gold").conv_id)
    s = pool.groupby("conv_id").agg(utterances=("text", "size"))
    o = registry.miv_outcomes("A").set_index("conv_id")
    s = s.join(o[["delta_confidence", "delta_readiness", "Status"]])
    s["test"] = s.index.isin(test)
    target = s[s.Status == "low-confidence-or-discordant"]          # the thesis's MI target population
    rows = []
    for c in ("utterances", "delta_confidence", "delta_readiness"):
        a, b = target[target.test][c].dropna(), target[~target.test][c].dropna()
        rows.append({"measure": c, "test (10)": round(a.mean(), 2), "rest of target population": round(b.mean(), 2),
                     "Mann-Whitney p": round(mannwhitneyu(a, b).pvalue, 3)})
    out = pd.DataFrame(rows).set_index("measure")
    out.attrs["status of test sessions"] = s[s.test].Status.value_counts().to_dict()
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(protocols(), "\n"); print(external_casaa(), "\n"); print(template_leak(), "\n")
    print(test_power(), "\n"); r = test_representativeness(); print(r); print(r.attrs)


# ------------------------------------------------- other schemes: leakage checks
def annomi_grouping() -> pd.DataFrame:
    """Does splitting AnnoMI by transcript (vs by video series, vs unseen annotator) inflate scores?"""
    from sklearn.model_selection import GroupKFold
    from eda.splits import annomi_series
    an = registry.load("annomi.gold").copy()
    an["t2"] = an["native"]
    an["series"] = an.conv_id.map(annomi_series(an))
    dom = an.groupby("conv_id").annotator_id.agg(lambda s: s[s >= 0].mode().iloc[0] if (s >= 0).any() else -1)
    an["ann"] = an.conv_id.map(dom).astype(str)
    rows = {}
    for name, groups, drop_series in (("by transcript (old split)", an.conv_id, False),
                                      ("by video series (new split)", an.series, False),
                                      ("by unseen annotator + series", an.ann, True)):
        pred = np.empty(len(an), dtype=object)
        for tr_i, te_i in GroupKFold(5).split(an, groups=groups):
            tr, te = an.iloc[tr_i], an.iloc[te_i]
            if drop_series:
                tr = tr[~tr.series.isin(set(te.series))]
            pred[te_i] = _fit_predict(tr, te)
        rows[name] = _score(an, pred)
    return pd.DataFrame(rows).T


def welivita_grouping() -> pd.DataFrame:
    """Same-opening-post leakage and source shift (CounselChat vs Reddit) on the agreed labels."""
    from sklearn.model_selection import GroupKFold
    w = registry.load("welivita.gold")
    L = w[(w.speaker == "counsellor") & w.stage1_agreed].reset_index(drop=True)
    man = json.load(open(registry.REPO / "data/splits/welivita_own.json"))
    cl = {m: r for r, ms in man["duplicate_clusters"].items() for m in ms}
    L["cluster"] = L.conv_id.map(lambda c: cl.get(c, c))
    rows = {}
    for name, groups in (("by dialogue (same post may cross)", L.conv_id), ("by same-post cluster (split used)", L.cluster)):
        pred = np.empty(len(L), dtype=object)
        for tr_i, te_i in GroupKFold(5).split(L, groups=groups):
            pred[te_i] = _model().fit(L.text.iloc[tr_i], L.native.iloc[tr_i]).predict(L.text.iloc[te_i])
        rows[name] = {"macro-F1": round(f1_score(L.native, pred, average="macro"), 3), "acc": round(accuracy_score(L.native, pred), 3)}
    for a, b in (("counsel_chat", "RED"), ("RED", "counsel_chat")):
        tr, te = L[L.source == a], L[L.source == b]
        p = _model().fit(tr.text, tr.native).predict(te.text)
        rows[f"train {a} -> test {b}"] = {"macro-F1": round(f1_score(te.native, p, average="macro", labels=sorted(te.native.unique()),
                                                                       zero_division=0), 3), "acc": round(accuracy_score(te.native, p), 3)}
    return pd.DataFrame(rows).T


def pooled_cv_leakage() -> dict:
    """Leakage channels specific to MIV pooled CV."""
    import difflib
    M = registry.load("misc.miv63a.gold").assign(k=lambda d: d.text.map(quality.norm))
    M["fold"] = M.conv_id.map(miv_folds())
    c = M[(M.speaker == "counsellor") & (M.k.str.split().str.len() >= 6)]
    near = sum(any(difflib.SequenceMatcher(None, t, x).ratio() >= 0.9 for x in c[c.fold != f].k)
               for t, f in zip(c.k, c.fold))
    cov = M.groupby("t2").fold.nunique()
    o = registry.miv_outcomes("A")
    return {"distinct participants in the 10 sessions": int(M.conv_id.nunique()),
            "duplicate participant ids among 173": int(o.conv_id.duplicated().sum()),
            "counsellor utterances (>=6 words) with a >=0.9-similar one in another fold": f"{near} of {len(c)}",
            "codes present in only one fold": cov[cov == 1].index.tolist(),
            "test utterances per fold": M.groupby("fold").size().to_dict()}
