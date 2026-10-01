"""Classic (non-LLM) MISC classifier: sklearn on n-gram or Qwen-embedding features.

Trains per-speaker T1 and T2 classifiers on HLQC, predicts MIV6.3A with the T2
scores masked to the predicted-T1's children (hierarchy-consistent), writes a CSV
to data/annotated/classic/, and prints a self-contained report incl. a
complementarity line vs the LLM `ft_bare` arm.

Runs CPU-only for --features ngram (no GPU, no cluster). --features qwen loads the
cached 7B once to embed the texts.

Usage:
    PYTHONPATH=src python -m classic.sklearn_run --features ngram --clf logreg --balanced --ctx 5
    PYTHONPATH=src python -m classic.sklearn_run --features ngram --clf svm --ctx 5 --context
    PYTHONPATH=src python -m classic.sklearn_run --features qwen  --clf logreg --ctx 5
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from automisc_ft.data import load_manual, t2_codes_for_group
from baseline.local_arm import REPO_ROOT, load_config
from classic.features import NgramFeaturizer, embed_texts
from components.context import build_context_excerpt

TAIL = ["SU", "EC", "AF", "GI"]
OUT_DIR = REPO_ROOT / "data" / "annotated" / "classic"
SPEAKERS = ("counsellor", "client")


def make_clf(kind: str, balanced: bool):
    cw = "balanced" if balanced else None
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=2000, class_weight=cw, C=1.0)
    if kind == "svm":
        from sklearn.svm import LinearSVC
        return LinearSVC(class_weight=cw)
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, class_weight=cw, n_jobs=-1,
                                      random_state=0)
    raise ValueError(kind)


def _scores(clf, X) -> np.ndarray:
    """(n_samples, n_classes) score matrix aligned to clf.classes_."""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)
    d = clf.decision_function(X)
    return np.vstack([-d, d]).T if d.ndim == 1 else d


def _texts(df: pd.DataFrame, use_context: bool, context_mode: str, ctx: int) -> List[str]:
    if not use_context:
        return df["utt_text"].fillna("").astype(str).tolist()
    out = []
    for pos in range(len(df)):
        exc = build_context_excerpt(df, pos, context_mode, ctx)
        out.append(f"{exc}\n{df.iloc[pos]['utt_text']}")
    return out


def _featurize(kind, train_texts, eval_texts, cfg):
    if kind == "ngram":
        f = NgramFeaturizer().fit(train_texts)
        return f.transform(train_texts), f.transform(eval_texts)
    # qwen embeddings: load the cached model once
    from components.hf_load import load_model_and_tokenizer
    model, tok, _ = load_model_and_tokenizer(
        cfg.model.base_model, for_training=False,
        force_cpu=bool(cfg.inference.force_cpu),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)))
    model.eval()
    print(f"Embedding {len(train_texts)}+{len(eval_texts)} texts with {cfg.model.base_model}...")
    return embed_texts(model, tok, train_texts), embed_texts(model, tok, eval_texts)


def run(args) -> None:
    cfg = load_config(args.overrides)
    ctx = args.ctx
    context_mode = cfg.annotator.context_mode

    train = load_manual(REPO_ROOT / cfg.dataset.train_csv)
    ev = load_manual(REPO_ROOT / cfg.dataset.eval_csv)
    ev = ev.drop(columns=[c for c in ev.columns if c.endswith("_auto")]).reset_index(drop=True)
    if args.limit:
        ev = ev.iloc[: int(args.limit)].copy()

    ev["t1_label_auto"] = "UNKNOWN"
    ev["t2_label_auto"] = "UNKNOWN"

    for spk in SPEAKERS:
        tr = train[train["speaker"] == spk].reset_index(drop=True)
        ev_idx = ev.index[ev["speaker"] == spk].to_numpy()
        if len(tr) == 0 or len(ev_idx) == 0:
            continue
        ev_spk = ev.loc[ev_idx].reset_index(drop=True)

        Xtr, Xev = _featurize(
            args.features,
            _texts(tr, args.context, context_mode, ctx),
            _texts(ev_spk, args.context, context_mode, ctx), cfg)

        clf1 = make_clf(args.clf, args.balanced).fit(Xtr, tr["t1_label_GT"])
        clf2 = make_clf(args.clf, args.balanced).fit(Xtr, tr["t2_label_GT"])

        pred_t1 = clf1.predict(Xev)
        S = _scores(clf2, Xev)
        col = {c: i for i, c in enumerate(clf2.classes_)}
        pred_t2 = []
        for j, t1 in enumerate(pred_t1):
            kids = [c for c in t2_codes_for_group(spk, t1) if c in col]
            if kids:
                pred_t2.append(kids[int(np.argmax([S[j, col[c]] for c in kids]))])
            else:
                pred_t2.append(clf2.classes_[int(np.argmax(S[j]))])

        ev.loc[ev_idx, "t1_label_auto"] = pred_t1
        ev.loc[ev_idx, "t2_label_auto"] = np.asarray(pred_t2, dtype=object)

    tag = f"{args.features}_{args.clf}" + ("_bal" if args.balanced else "") + \
          ("_ctx" if args.context else "")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"classic_{tag}_ctx{ctx}.csv"
    ev.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    _report(ev, tag)


def _report(ev: pd.DataFrame, tag: str) -> None:
    yt1, yp1 = ev["t1_label_GT"], ev["t1_label_auto"]
    yt2, yp2 = ev["t2_label_GT"], ev["t2_label_auto"]
    labs2 = sorted(yt2.dropna().unique())
    macro = f1_score(yt2, yp2, labels=labs2, average="macro", zero_division=0)
    print(f"\n=== classic:{tag} on {len(ev)} rows ===")
    print(f"T1 acc={accuracy_score(yt1, yp1):.3f}  T2 acc={accuracy_score(yt2, yp2):.3f}  "
          f"T2 macro-F1(gold)={macro:.3f}")
    for c in TAIL:
        f1 = f1_score(yt2, yp2, labels=[c], average="macro", zero_division=0)
        sup = int((yt2 == c).sum())
        print(f"  F1[{c}]={f1:.3f} (support {sup})")

    # Complementarity vs the LLM ft_bare arm, if present.
    ftb = REPO_ROOT / "data" / "annotated" / "baseline" / "qwen_ft_bare_inf_cot_ctx5.csv"
    if ftb.exists():
        f = pd.read_csv(ftb).drop_duplicates("corp_utt_idx")[["corp_utt_idx", "t2_label_auto"]]
        m = ev.merge(f, on="corp_utt_idx", suffixes=("", "_ftb"))
        cls_ok = m["t2_label_auto"] == m["t2_label_GT"]
        ftb_ok = m["t2_label_auto_ftb"] == m["t2_label_GT"]
        tail = m["t2_label_GT"].isin(TAIL)
        agree = float((m["t2_label_auto"] == m["t2_label_auto_ftb"]).mean())
        print(f"\n  vs ft_bare: label agreement={agree:.3f}  "
              f"classic-right/ft_bare-wrong={int((cls_ok & ~ftb_ok).sum())}  "
              f"ft_bare-right/classic-wrong={int((~cls_ok & ftb_ok).sum())}")
        print(f"  on TAIL rows only: classic-right/ft_bare-wrong="
              f"{int((cls_ok & ~ftb_ok & tail).sum())}  "
              f"ft_bare-right/classic-wrong={int((~cls_ok & ftb_ok & tail).sum())}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", choices=("ngram", "qwen"), default="ngram")
    p.add_argument("--clf", choices=("logreg", "svm", "rf"), default="logreg")
    p.add_argument("--balanced", action="store_true", help="class_weight=balanced")
    p.add_argument("--context", action="store_true", help="prepend context excerpt to text")
    p.add_argument("--ctx", type=int, default=5)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")
    run(p.parse_args())


if __name__ == "__main__":
    main()
