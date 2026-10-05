"""Manifest-driven train/test folds: the one way the re-run picks its data.

Every train/predict entry point takes (manifest, design, fold, seed) and calls
`load_fold`, so no script hard-codes "train HLQC, test MIV" any more
(docs/RERUN_PLAN.md P3). The manifests are `data/splits/*.json`, written by
`python -m eda.splits`; see docs/DATASETS.md section 6 for why each split is
what it is.

    manifest         design             fold   seed  train                       test
    misc_main        cold_start         0      -     misc.hlqc.gold              misc.miv63a.gold (+ CASAA external)
    misc_pooled_cv   loso               0..9   -     HLQC + 9 MIV sessions       1 MIV session
    misc_pooled_cv   5fold_seed_tied    0..4   s     HLQC + 8 MIV sessions       2 MIV sessions (fold draw s)
    annomi_own       5fold              0..4   -     other AnnoMI series         fold k
    welivita_own     5fold              0..4   -     other Welivita clusters     fold k

Frames come back in the manual-CSV schema the training code already reads
(`automisc_ft.data.load_manual`: conv_id, speaker, ..., utt_text, t1_label_GT,
t2_label_GT), plus two columns:
    dataset   the canonical dataset id
    uid       "<dataset>:<corp_utt_idx>", unique when corpora are pooled
              (corp_utt_idx alone collides between HLQC and MIV).
Key every store (predictions, OOF T1, rationales, resume) by `uid`.

`training_source` lets an ablation swap the training corpus file for a derived
copy (e.g. misc.hlqc.gold -> misc.hlqc.gold.cleaned) without changing sessions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS = REPO_ROOT / "data" / "splits"

# Datasets that already exist as manual-schema CSVs.
MANUAL_CSV = {
    "misc.hlqc.gold": REPO_ROOT / "data" / "manual" / "HLQC_balanced_manual.csv",
    "misc.miv63a.gold": REPO_ROOT / "data" / "manual" / "MIV6.3A_manual.csv",
    "misc.hlqc.gold.cleaned": REPO_ROOT / "data" / "clean" / "misc.hlqc.gold.cleaned.csv",
}
DESIGNS = {
    "misc_main": ("cold_start",),
    "misc_pooled_cv": ("loso", "5fold_seed_tied"),
    "annomi_own": ("5fold",),
    "welivita_own": ("5fold",),
}


def manifest(name: str) -> dict:
    return json.loads((SPLITS / f"{name}.json").read_text())


def _test_key(k: str) -> str:
    """Manifest keys carry a note after the id ('miti.casaa.gold (MISC-mapped)')."""
    return k.split(" ")[0]


def fold_ids(name: str, design: str, fold: int = 0, seed: Optional[int] = None) -> Dict[str, Dict[str, List[str]]]:
    """{'train': {dataset: [conv_ids]}, 'test': {dataset: [conv_ids]}} for one fold."""
    if design not in DESIGNS.get(name, ()):
        raise ValueError(f"{name} has designs {DESIGNS.get(name)}, not {design!r}")
    m = manifest(name)
    if name == "misc_main":
        if fold != 0:
            raise ValueError("misc_main (cold start) has a single fold, 0")
        return {"train": {k: list(v) for k, v in m["train"].items()},
                "test": {_test_key(k): list(v) for k, v in m["test"].items()}}
    if name == "misc_pooled_cv":
        block = m["designs"][design]
        if design == "5fold_seed_tied":
            if seed is None:
                raise ValueError("5fold_seed_tied needs the training seed (fold draw s for seed s)")
            block = block[str(seed)]
        f = block[str(fold)]
        return {"train": {k: list(v) for k, v in f["train"].items()},
                "test": {k: list(v) for k, v in f["test"].items()}}
    ds = "annomi.gold" if name == "annomi_own" else "welivita.gold"
    fold_of = m["folds"]
    if fold not in set(fold_of.values()):
        raise ValueError(f"{name} has folds {sorted(set(fold_of.values()))}")
    return {"train": {ds: sorted(c for c, k in fold_of.items() if k != fold)},
            "test": {ds: sorted(c for c, k in fold_of.items() if k == fold)}}


def n_folds(name: str, design: str) -> int:
    if name == "misc_main":
        return 1
    m = manifest(name)
    if name == "misc_pooled_cv":
        block = m["designs"][design]
        return len(next(iter(block.values()))) if design == "5fold_seed_tied" else len(block)
    return len(set(m["folds"].values()))


def _read(dataset: str) -> pd.DataFrame:
    if dataset not in MANUAL_CSV:
        raise NotImplementedError(f"{dataset} has no manual-schema CSV yet (MISC datasets only for now)")
    from automisc_ft.data import load_manual
    df = load_manual(MANUAL_CSV[dataset])
    df["dataset"] = dataset
    df["uid"] = dataset + ":" + df["corp_utt_idx"].astype(str)
    return df


def _frame(parts: Dict[str, List[str]], sources: Dict[str, str]) -> pd.DataFrame:
    frames = []
    for ds, ids in parts.items():
        src = sources.get(ds, ds)
        d = _read(src)
        d = d[d["conv_id"].isin(set(map(str, ids)))]
        if src != ds:
            # same sessions and rows under the original id, so uids pair with the default arm
            d = d.assign(dataset=ds, uid=ds + ":" + d["corp_utt_idx"].astype(str), source_file=src)
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    # Positional order must follow conversation order (build_context_excerpt uses iloc).
    out = pd.concat(frames, ignore_index=True)
    sort_cols = [c for c in ("corp_utt_idx", "conv_vol_idx", "conv_utt_idx") if c in out.columns]
    out = out.sort_values(["dataset", "conv_id"] + sort_cols, kind="stable").reset_index(drop=True)
    assert out["uid"].is_unique, "uid collision"
    return out


def load_fold(name: str, design: str, fold: int = 0, seed: Optional[int] = None,
              training_source: Optional[Dict[str, str]] = None,
              test_datasets: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(train_df, test_df, meta) for one fold.

    training_source maps a training dataset id to the derived copy to read instead
    (cleaned-HLQC ablation). test_datasets restricts the test side (misc_main also
    lists CASAA, which is scored through its own MITI/MISC route, not this frame).
    """
    ids = fold_ids(name, design, fold, seed)
    test_parts = {k: v for k, v in ids["test"].items() if k in MANUAL_CSV
                  and (test_datasets is None or k in test_datasets)}
    train = _frame(ids["train"], training_source or {})
    test = _frame(test_parts, {})
    overlap = set(zip(train["dataset"], train["conv_id"])) & set(zip(test["dataset"], test["conv_id"]))
    assert not overlap, f"train/test session overlap: {sorted(overlap)[:3]}"
    meta = {"manifest": name, "design": design, "fold": fold, "seed": seed,
            "training_source": training_source or {},
            "train_sessions": {k: len(v) for k, v in ids["train"].items()},
            "test_sessions": {k: len(v) for k, v in test_parts.items()},
            "n_train": len(train), "n_test": len(test)}
    return train, test, meta
