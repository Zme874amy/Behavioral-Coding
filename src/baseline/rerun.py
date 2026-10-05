"""Paths, provenance and data for re-run cells (docs/RERUN_PLAN.md P3).

One cell = (manifest, design, student, arm, inference style, ctx, prompt
version, training source, seed, fold). Its predictions live at

    data/annotated/rerun/<manifest>/<design>/<student>/<arm>[+<source>]_inf_<style>_ctx<C>_<pv>/seed<S>/fold<k>.csv

with a `.meta.json` beside it, and its adapter (if any) at the same relative
path under data/fine_tuning/rerun/ (deleted after prediction to save quota,
except when KEEP_ADAPTERS=1). Every component of the key is in the path, so no
two cells can overwrite each other: the overwrite bugs in sc_arm, grpo_tc,
the agentic stems and the retrain seed came from keys missing in the path.

`pooled(...)` concatenates a cell's fold files into the out-of-fold prediction
set that is scored once (821 utterances for MISC).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "data" / "annotated" / "rerun"
ADAPTERS = REPO_ROOT / "data" / "fine_tuning" / "rerun"
SEEDS = (42, 1, 2)

SOURCE_TAGS = {"misc.hlqc.gold.cleaned": "cleaned"}


def student_slug(base_model: str) -> str:
    return re.sub(r"[^A-Za-z0-9.\-]+", "-", str(base_model).rstrip("/").split("/")[-1])


def arm_label(arm: str, training_source: Optional[Dict[str, str]] = None) -> str:
    tags = [SOURCE_TAGS.get(v, v) for v in (training_source or {}).values()]
    return arm + "".join(f"+{t}" for t in tags)


def cell_dir(root: Path, manifest: str, design: str, student: str, arm: str, style: Optional[str],
             ctx: int, prompt_version: str, training_source=None) -> Path:
    leaf = arm_label(arm, training_source) + (f"_inf_{style}" if style else "") + f"_ctx{ctx}_{prompt_version}"
    return root / manifest / design / student / leaf


def result_path(manifest, design, student, arm, style, ctx, prompt_version, seed, fold,
                training_source=None) -> Path:
    return (cell_dir(RESULTS, manifest, design, student, arm, style, ctx, prompt_version, training_source)
            / f"seed{seed}" / f"fold{fold}.csv")


def adapter_path(manifest, design, student, arm, ctx, prompt_version, seed, fold, training_source=None) -> Path:
    return (cell_dir(ADAPTERS, manifest, design, student, arm, None, ctx, prompt_version, training_source)
            / f"seed{seed}" / f"fold{fold}")


def git_state() -> Dict[str, Optional[str]]:
    """Commit and whether the tree had uncommitted changes. Runs must come from a
    clean, tagged tree (the old scp workflow stamped every run with one stale sha)."""
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:
            return None
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"git_sha": run("rev-parse", "--short", "HEAD"),
            "git_tag": run("describe", "--tags", "--exact-match"),
            "git_dirty": None if status is None else bool(status)}


def write_meta(csv_path: Path, meta: dict) -> Path:
    p = csv_path.with_suffix(".meta.json")
    p.write_text(json.dumps({**meta, **git_state(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")},
                            indent=2, default=str))
    return p


def pooled(manifest, design, student, arm, style, ctx, prompt_version, seed, n_folds: int,
           training_source=None) -> pd.DataFrame:
    """Out-of-fold predictions of every fold of one cell; raises if a fold is missing."""
    frames, missing = [], []
    for k in range(n_folds):
        p = result_path(manifest, design, student, arm, style, ctx, prompt_version, seed, k, training_source)
        (frames.append(pd.read_csv(p).assign(fold=k)) if p.exists() else missing.append(k))
    if missing:
        raise FileNotFoundError(f"folds {missing} missing for {arm} seed {seed} ({manifest}/{design})")
    out = pd.concat(frames, ignore_index=True)
    dup = out["uid"].duplicated()
    if dup.any():
        raise ValueError(f"{int(dup.sum())} utterances predicted by more than one fold")
    return out
