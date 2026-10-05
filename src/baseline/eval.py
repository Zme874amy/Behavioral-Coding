"""Score the Model Scale x Adaptation x Rationale Alignment grid.

Scans data/annotated/baseline/ for result files named

    <tier>_<arm>_inf_<style>_ctx<N>.csv

where tier is the model scale (`gpt4o`, `qwen`), arm is the adaptation method
(`zs`, `fs`, `ft_bare`, `ft_rat`) and style is the inference prompt (`bare`,
`cot`). Result files written before this scheme existed are mapped through
LEGACY_STEMS, and an un-suffixed file counts as ctx5.

Three things are reported per cell:

    performance     accuracy and Cohen's kappa
    class coverage  macro-F1, which is where long-tail codes show up
    compliance      did the model actually follow the inference instruction

Compliance matters because half the fine-tuning cells exist to test it: does a
label-trained model start explaining when asked, does a rationale-trained model
go quiet when told to. It is only meaningful where decoding was unconstrained.
The gpt-4o rows were produced with a pydantic `response_format`, so their output
shape was forced by the schema and their compliance figures are marked as such.

Outputs:
    outputs/baseline_eval/comparison.csv
    outputs/baseline_eval/compliance.csv
    outputs/baseline_eval/<condition>_ctx<N>_<t1|t2>_report.csv  (per-code P/R/F1)
    docs/BASELINE_RESULTS.md

Usage:
    PYTHONPATH=src .venv/bin/python -m baseline.eval
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from automisc_ft.data import t2_codes_for_group
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
)

# -- Uncertainty reporting -------------------------------------------------
# Error bars come from a CLUSTERED bootstrap: the evaluation utterances are
# nested in a handful of conversations (MIV6.3A is 821 utterances from only 10),
# and utterances inside one conversation are correlated, so the independent unit
# is the conversation, not the row. Resampling rows would report intervals far
# too narrow. We resample conversations with replacement instead.
N_BOOT = 10_000
CI_LEVEL = 0.95
BOOT_SEED = 0

# Training-seed noise. Where seed replicates exist, the paired comparison derives
# its floor from them (`_seed_floor`): the 95% band for the difference of two
# single-seed runs, 1.96 * sqrt(sd_ref^2 + sd_arm^2). An arm without replicates
# borrows the reference's sd. This constant is only the FALLBACK for cells where
# no seed spread has been measured at all. 0.08 is the worst case seen on this
# task -- the staged teacher-forcing arms on cell B, where changing only the seed
# moved T2 by up to 0.08 (docs/experiments/2026-08-19-teacher-forcing.md). The
# plain 1-adapter baseline is far steadier: over seeds 42/1/2 its T2/all accuracy
# spans 0.010 and Macro-F1 (gold) 0.034. So the fallback is deliberately
# conservative.
SEED_NOISE_FLOOR = 0.08

REPO_ROOT = Path(__file__).resolve().parents[2]
ANNOTATED_DIR = REPO_ROOT / "data" / "annotated" / "baseline"
OUT_DIR = REPO_ROOT / "outputs" / "baseline_eval"
DOCS_PATH = REPO_ROOT / "docs" / "BASELINE_RESULTS.md"
# The rationale-swap probe is produced by a separate job and read back in here,
# so the report carries it instead of leaving it stranded as raw JSON.
FAITHFULNESS_DIR = REPO_ROOT / "outputs" / "grpo"

TRAIN_CSV = REPO_ROOT / "data" / "manual" / "HLQC_balanced_manual.csv"

# `qwen32b` is the Qwen2.5-32B-Instruct-AWQ teacher, evaluated report-only (it
# chooses nothing) so the self-training comparison can show how strong a
# labeller it actually is next to the frontier and student rows.
TIER_ORDER = ["gpt4o", "qwen32b", "qwen"]
# The `sc_` arms use the single-call format (one generation for both tiers,
# src/baseline/single_call.py) and are only comparable to each other. They are
# listed after the two-call arms so the two ladders read as separate blocks.
TWO_CALL_ARMS = [
    "zs", "fs", "ft_bare", "ft_rat",
    "ft1mix_bare", "ft1mix_rat", "ft1seq_bare", "ft1seq_rat",
    # Teacher-forcing variants of ft_bare: identical structure and prompts,
    # differing only in which T1 label conditioned T2 during training.
    "ft_bare_tffull", "ft_bare_tfdyn", "ft_bare_tfzero",
    # The same axis on cell B: 1 adapter, 2 calls.
    "ft1mix_bare_tffull", "ft1mix_bare_tfdyn", "ft1mix_bare_tfzero",
    # Smart-retrieval few-shot: like `fs`, but exemplars are retrieved per
    # utterance (src/agentic/) rather than loaded from a frozen file.
    "ag",
]
SINGLE_CALL_ARMS = [
    "sc_zs", "sc_fs", "sc_ft_bare", "sc_ft_rat",
    "sc_grpo", "sc_grpo_unw", "sc_grpo_cold",
]
ARM_ORDER = TWO_CALL_ARMS + SINGLE_CALL_ARMS
STYLE_ORDER = ["bare", "cot"]

# Where each arm sits in the (adapters x calls) matrix. Comparing across call
# counts is only identifiable when adapter count is held fixed, so the report is
# grouped by this rather than by arm alone: `ft_bare` against `ft1mix_bare`
# varies adapters at 2 calls, and `ft1mix_bare` against `sc_ft_bare` varies calls
# at 1 adapter.
ARM_STRUCTURE = {
    "zs": "2call_0ad",
    "fs": "2call_0ad",
    "ag": "2call_0ad",
    "ft_bare": "2call_2ad",
    "ft_rat": "2call_2ad",
    "ft_bare_tffull": "2call_2ad",
    "ft_bare_tfdyn": "2call_2ad",
    "ft_bare_tfzero": "2call_2ad",
    "ft1mix_bare_tffull": "2call_1ad",
    "ft1mix_bare_tfdyn": "2call_1ad",
    "ft1mix_bare_tfzero": "2call_1ad",
    "ft1mix_bare": "2call_1ad",
    "ft1mix_rat": "2call_1ad",
    "ft1seq_bare": "2call_1ad",
    "ft1seq_rat": "2call_1ad",
    "sc_zs": "1call_0ad",
    "sc_fs": "1call_0ad",
    "sc_ft_bare": "1call_1ad",
    "sc_ft_rat": "1call_1ad",
    "sc_grpo": "1call_1ad",
    "sc_grpo_unw": "1call_1ad",
    "sc_grpo_cold": "1call_1ad",
}
STRUCTURE_ORDER = ["2call_0ad", "2call_2ad", "2call_1ad", "1call_0ad", "1call_1ad"]
STRUCTURE_LABELS = {
    "2call_0ad": "Two-call, no adapter (in-context)",
    "2call_2ad": "Two-call, 2 adapters (one per tier)",
    "2call_1ad": "Two-call, 1 adapter (shared across tiers)",
    "1call_0ad": "Single-call, no adapter (in-context)",
    "1call_1ad": "Single-call, 1 adapter",
}

TIER_LABELS = {
    "gpt4o": "GPT-4o (frontier)",
    "qwen32b": "Qwen2.5-32B-Instruct-AWQ (teacher, report-only)",
    "qwen": "Qwen2.5-7B-Instruct (SLM)",
}

# (arm, style) -> display name, using the shorthand from the experiment design.
CELL_LABELS = {
    ("zs", "bare"): "ZS-Bare",
    ("zs", "cot"): "ZS-CoT",
    ("fs", "bare"): "FS-Bare",
    ("fs", "cot"): "FS-CoT",
    ("ag", "bare"): "AG-Retrieval-Bare",
    ("ag", "cot"): "AG-Retrieval-CoT",
    ("ft_bare", "bare"): "FT-Bare_Inf-Bare",
    ("ft_bare", "cot"): "FT-Bare_Inf-CoT",
    ("ft_rat", "bare"): "FT-Rat_Inf-Bare",
    ("ft_rat", "cot"): "FT-Rat_Inf-CoT",
    # Teacher-forcing axis: p_gold is the share of T2 training examples
    # conditioned on the gold T1 rather than an out-of-fold predicted one.
    ("ft_bare_tffull", "bare"): "FT-Bare TF-Full (p_gold=1)",
    ("ft_bare_tfdyn", "bare"): "FT-Bare TF-Dyn (p_gold 1→0)",
    ("ft_bare_tfzero", "bare"): "FT-Bare TF-Zero (p_gold=0)",
    ("ft1mix_bare_tffull", "bare"): "FT1-Mix TF-Full (p_gold=1)",
    ("ft1mix_bare_tfdyn", "bare"): "FT1-Mix TF-Dyn (p_gold 1\u21920)",
    ("ft1mix_bare_tfzero", "bare"): "FT1-Mix TF-Zero (p_gold=0)",
    # 1-adapter / 2-call arms. `Mix` trains one adapter on the shuffled union of
    # both tiers; `Seq` trains on T1 then continues the same adapter on T2.
    ("ft1mix_bare", "bare"): "FT1-Mix-Bare_Inf-Bare",
    ("ft1mix_bare", "cot"): "FT1-Mix-Bare_Inf-CoT",
    ("ft1mix_rat", "bare"): "FT1-Mix-Rat_Inf-Bare",
    ("ft1mix_rat", "cot"): "FT1-Mix-Rat_Inf-CoT",
    ("ft1seq_bare", "bare"): "FT1-Seq-Bare_Inf-Bare",
    ("ft1seq_bare", "cot"): "FT1-Seq-Bare_Inf-CoT",
    ("ft1seq_rat", "bare"): "FT1-Seq-Rat_Inf-Bare",
    ("ft1seq_rat", "cot"): "FT1-Seq-Rat_Inf-CoT",
    # Single-call ladder. Each arm is evaluated in the style it was trained in,
    # so there is one cell per arm rather than the two-call grid's bare/cot pair.
    ("sc_zs", "cot"): "SC ZS-CoT",
    ("sc_fs", "cot"): "SC FS-CoT",
    ("sc_ft_bare", "bare"): "SC FT-Bare",
    ("sc_ft_rat", "cot"): "SC FT-Rat",
    ("sc_grpo", "cot"): "SC GRPO",
    ("sc_grpo_unw", "cot"): "SC GRPO (no rare-class weighting)",
    ("sc_grpo_cold", "cot"): "SC GRPO (cold start)",
}

# Result files produced before the tier/arm/style naming existed.
LEGACY_STEMS = {
    "zeroshot_rationales": ("gpt4o", "zs", "cot"),
    "zeroshot_bare": ("gpt4o", "zs", "bare"),
    "fewshot_rationales": ("gpt4o", "fs", "cot"),
    "fewshot_bare": ("gpt4o", "fs", "bare"),
    # The original Azure fine-tune was label-only, evaluated label-only.
    "finetuned": ("gpt4o", "ft_bare", "bare"),
}

# The grid this project intends to fill, reported against in the Coverage
# section so a cell that was never run is visible rather than merely absent.
# Two-call arms are crossed with both inference styles; each single-call arm is
# evaluated only in the style it was trained in, which is why its tuple is one
# entry long.
EXPECTED_CTXS = (3, 5)
EXPECTED_GRID = {
    "gpt4o": {
        "zs": ("bare", "cot"),
        "fs": ("bare", "cot"),
        "ft_bare": ("bare", "cot"),
        "ft_rat": ("bare", "cot"),
    },
    "qwen": {
        "zs": ("bare", "cot"),
        "fs": ("bare", "cot"),
        "ft_bare": ("bare", "cot"),
        "ft_rat": ("bare", "cot"),
        "ft1mix_bare": ("bare", "cot"),
        "ft1mix_rat": ("bare", "cot"),
        "ft1seq_bare": ("bare", "cot"),
        "ft1seq_rat": ("bare", "cot"),
        # Smart-retrieval few-shot, evaluated CoT, ctx5 (see ARM_CTXS).
        "ag": ("cot",),
        # Evaluated Inf-Bare only. Inf-CoT measures instruction compliance,
        # which is a different question and already answered by ft_bare.
        "ft_bare_tffull": ("bare",),
        "ft_bare_tfdyn": ("bare",),
        "ft_bare_tfzero": ("bare",),
        "ft1mix_bare_tffull": ("bare",),
        "ft1mix_bare_tfdyn": ("bare",),
        "ft1mix_bare_tfzero": ("bare",),
        "sc_zs": ("cot",),
        "sc_fs": ("cot",),
        "sc_ft_bare": ("bare",),
        "sc_ft_rat": ("cot",),
        "sc_grpo": ("cot",),
        "sc_grpo_unw": ("cot",),
        "sc_grpo_cold": ("cot",),
    },
}
# Arms deliberately scoped to one context length. The two Phase-2 GRPO ablations
# only ever answer a ctx5 question, so listing them as missing at ctx3 would be
# noise rather than a gap.
# The teacher-forcing axis is scoped to ctx5. ctx3 was attempted and abandoned
# (docs/experiments/2026-08-19-teacher-forcing.md), so listing those cells as
# missing would report an open gap where there is a closed decision.
ARM_CTXS = {
    "sc_grpo_unw": (5,), "sc_grpo_cold": (5,), "ag": (5,),
    **{f"{h}_tf{m}": (5,)
       for h in ("ft_bare", "ft1mix_bare")
       for m in ("full", "dyn", "zero")},
}

# -- Two-call GRPO arms (cells A and B under RL; baseline.grpo_tc). Registered
# programmatically so the 4 arms x {weighted, _unw, _cold} cross does not bloat
# every dict above. `pair` is 2 adapters (cell A), `mix` is 1 (cell B); all are
# cot (they emit a rationale), run at 3 seeds, and aggregated like `sc_grpo`. The
# two ablations, like cell F's, only answer a ctx5 question.
_TC_GRPO = {
    "grpo_pair_dec": ("2call_2ad", "TC GRPO Pair-Dec"),
    "grpo_pair_joint": ("2call_2ad", "TC GRPO Pair-Joint"),
    "grpo_mix_dec": ("2call_1ad", "TC GRPO Mix-Dec"),
    "grpo_mix_joint": ("2call_1ad", "TC GRPO Mix-Joint"),
}
_TC_VARIANTS = {"": "", "_unw": " (no rare-class weighting)", "_cold": " (cold start)"}
for _base, (_struct, _label) in _TC_GRPO.items():
    for _suf in ("", "_unw", "_cold"):
        _arm = _base + _suf
        ARM_STRUCTURE[_arm] = _struct
        CELL_LABELS[(_arm, "cot")] = _label + _TC_VARIANTS[_suf]
        EXPECTED_GRID["qwen"][_arm] = ("cot",)
        TWO_CALL_ARMS.append(_arm)
        if _suf:
            ARM_CTXS[_arm] = (5,)
# Rebuilt so the new arms sit in the two-call block, ahead of the single-call one.
ARM_ORDER = TWO_CALL_ARMS + SINGLE_CALL_ARMS

# Rows in the evaluation set. Prediction checkpoints every 25 utterances so a
# requeued job can resume, which means a running cell has a valid but SHORT
# result file on disk. Without this, a half-finished cell would be scored and
# reported as though it were a result.
EXPECTED_EVAL_N = 821

# Paper reference: GPT-4.1, hierarchical prompts, 3 context volleys,
# MIV6.3A clean set (docs/AUTOMISC_FT.md).
PAPER_REFERENCE = [
    ("T1", "counsellor", "accuracy", 0.82),
    ("T1", "client", "accuracy", 0.88),
    ("T2", "counsellor", "accuracy", 0.68),
    ("T2", "counsellor", "macro-F1", 0.42),
    ("T2", "client", "accuracy", 0.76),
    ("T2", "client", "macro-F1", 0.41),
]

# Primary evaluation corpus; its result files carry no `_ds` token so every
# result produced before cross-dataset eval existed still parses to it.
DEFAULT_DATASET = "miv63a"

_STEM_RE = re.compile(
    r"(?P<head>.+?)_inf_(?P<style>bare|cot)(?:_ctx(?P<ctx>\d+))?"
    r"(?:_ds(?P<ds>[a-z0-9]+))?"
    r"(?:_seed(?P<seed>\d+))?$"
)
_LEGACY_RE = re.compile(r"(?P<stem>[a-z_]+?)(?:_ctx(?P<ctx>\d+))?$")


def parse_stem(stem: str) -> tuple[str, str, str, int, int | None, str] | None:
    """Map a filename stem to ``(tier, arm, style, ctx, seed, dataset)``, or None.

    `seed` is None for the deterministic arms and an integer for the GRPO runs,
    which are repeated across seeds and aggregated in the reported table.
    `dataset` is the evaluation corpus slug (`miv63a` when no `_ds` token is
    present), which keeps other corpora's results from colliding with the
    primary grid. Placed last so callers that slice ``rec[:5]`` are unaffected.
    """
    m = _STEM_RE.fullmatch(stem)
    if m:
        head, style = m.group("head"), m.group("style")
        ctx = int(m.group("ctx")) if m.group("ctx") else 5
        seed = int(m.group("seed")) if m.group("seed") else None
        dataset = m.group("ds") or DEFAULT_DATASET
        # Longest arm first so `ft_bare` is not shadowed by a shorter match.
        for arm in sorted(ARM_ORDER, key=len, reverse=True):
            if head.endswith(f"_{arm}"):
                tier = head[: -(len(arm) + 1)]
                if tier:
                    return tier, arm, style, ctx, seed, dataset
        return None

    m = _LEGACY_RE.fullmatch(stem)
    if m and m.group("stem") in LEGACY_STEMS:
        tier, arm, style = LEGACY_STEMS[m.group("stem")]
        ctx = int(m.group("ctx")) if m.group("ctx") else 5
        return tier, arm, style, ctx, None, DEFAULT_DATASET
    return None


def condition_label(tier: str, arm: str, style: str) -> str:
    cell = CELL_LABELS.get((arm, style), f"{arm}/{style}")
    return f"{TIER_LABELS.get(tier, tier)} — {cell}"


def _sort_key(rec: tuple) -> tuple:
    tier, arm, style, ctx, seed = rec[:5]
    return (
        TIER_ORDER.index(tier) if tier in TIER_ORDER else len(TIER_ORDER),
        ARM_ORDER.index(arm) if arm in ARM_ORDER else len(ARM_ORDER),
        STYLE_ORDER.index(style) if style in STYLE_ORDER else len(STYLE_ORDER),
        ctx,
        -1 if seed is None else seed,
    )


def _precedence(stem: str) -> tuple[int, int]:
    """Rank candidates for the same cell: new naming beats legacy, explicit ctx
    beats the un-suffixed ctx5 fallback."""
    is_new = _STEM_RE.fullmatch(stem) is not None
    return (int(is_new), int("_ctx" in stem))


def discover_result_files() -> list[tuple[str, str, str, int, int | None, Path]]:
    """Return (tier, arm, style, ctx, seed, path) for every recognised result file.

    Several filenames can describe the same cell (a legacy name and its new-scheme
    equivalent), so keep the highest-precedence one and say which were ignored
    rather than silently picking by sort order.
    """
    found: dict[tuple, Path] = {}
    for path in sorted(ANNOTATED_DIR.glob("*.csv")):
        key = parse_stem(path.stem)
        if key is None:
            continue
        current = found.get(key)
        if current is None or _precedence(path.stem) > _precedence(current.stem):
            if current is not None:
                print(f"note: {path.name} supersedes {current.name} for the same cell")
            found[key] = path
        else:
            print(f"note: ignoring {path.name}; {current.name} covers the same cell")
    return sorted([(*k, v) for k, v in found.items()], key=_sort_key)


META_SUFFIX = ".meta.json"


def load_run_dates() -> dict[tuple, str]:
    """Map ``(tier, arm, style, ctx, seed)`` to the date its cell finished.

    Read from the `.meta.json` sidecar `local_arm` writes beside each result.
    Cells produced before sidecars existed have none, and are reported as
    unknown rather than guessed from a file mtime -- after a bulk rescore every
    mtime is the rescore's, not the run's, so inferring one would put a
    confident wrong date next to a real result.
    """
    dates: dict[tuple, str] = {}
    for path in sorted(ANNOTATED_DIR.glob(f"*{META_SUFFIX}")):
        key = parse_stem(path.name[: -len(META_SUFFIX)])
        if key is None or key[5] != DEFAULT_DATASET:
            # The run-date table lives in the primary-grid Coverage section;
            # cross-dataset cells are dated in their own section.
            continue
        try:
            meta = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            print(f"note: could not read {path.name}")
            continue
        finished = meta.get("finished_utc")
        if finished:
            dates[key[:5]] = str(finished)[:10]
    return dates


def _trainable_codes() -> dict[str, set[str]]:
    """Codes that occur in the training corpus, per MISC tier.

    A code that never appears in HLQC cannot be learned by any arm that trains
    on it -- `TS+` (2 occurrences in MIV6.3A) and `AC-` (1) are the cases here.
    Both still enter Macro-F1 (gold) as a guaranteed zero, so they depress the
    reported figure by construction and by an amount that depends only on how
    many distinct codes the evaluation set happens to contain. Reporting
    Macro-F1 (learnable) alongside separates "the model missed the long tail"
    from "the long tail was never in the training data".
    """
    try:
        train = pd.read_csv(TRAIN_CSV)
    except FileNotFoundError:
        return {}
    return {
        level: set(train[f"{level}_label_GT"].dropna().astype(str).unique())
        for level in ("t1", "t2")
        if f"{level}_label_GT" in train.columns
    }


TRAINABLE_CODES = _trainable_codes()


def _metric_values(
    y_true: np.ndarray, y_pred: np.ndarray, level: str, gold_classes: list
) -> dict:
    """The five reported metrics for one array of predictions.

    ``gold_classes`` are the codes with gold support in this sample, so
    Macro-F1 (gold) averages over the codes that actually occur. The bootstrap
    in ``_clustered_bootstrap_ci`` applies the same rule per resample.
    """
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro_gold": f1_score(
            y_true, y_pred, labels=gold_classes, average="macro", zero_division=0
        ),
    }
    trainable = TRAINABLE_CODES.get(level)
    if trainable:
        learnable = [c for c in gold_classes if c in trainable]
        out["f1_macro_learnable"] = f1_score(
            y_true, y_pred, labels=learnable, average="macro", zero_division=0
        )
    else:
        out["f1_macro_learnable"] = None
    return out


def _macro_over_supported(f1: np.ndarray, rowsum: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Per-draw macro-F1 over the codes in ``idx`` that have gold support in that draw.

    A resample that leaves out every conversation containing a rare code has no
    gold instance of it, so that code is dropped from the average for that draw,
    exactly as the point estimate only averages codes that occur. Scoring the
    absent code as F1 = 0 instead would add a zero that no model could avoid,
    pulling the lower bound down even for a perfect predictor.
    """
    supported = rowsum[:, idx] > 0                         # (B, len(idx))
    n = supported.sum(1)
    return np.where(n > 0, (f1[:, idx] * supported).sum(1) / np.maximum(n, 1), 0.0)


def _clustered_bootstrap_ci(
    y_true: pd.Series, y_pred: pd.Series, conv: pd.Series, level: str
) -> dict:
    """95% CI per metric from resampling CONVERSATIONS with replacement.

    The unit of resampling is ``conv`` (conversation id), not the row: utterances
    within a conversation share speaker, topic and an overlapping context window,
    so they are not independent. Resampling whole conversations propagates that
    clustering into the interval; a row-level bootstrap would report intervals
    several times too narrow. Returns ``{metric: (lo, hi)}``; a cell with fewer
    than two conversations has no meaningful spread and returns empty.

    Implemented on per-conversation confusion matrices: each resample is a
    weighted sum of those matrices, and all five metrics are read off the summed
    matrix in vectorised numpy. This gives the same values as ``_metric_values``
    (accuracy, kappa, and the macro-F1 variants all being functions of the
    confusion counts) while running the full ``N_BOOT`` draws without a Python
    loop over predictions.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    conv_arr = np.asarray(conv)
    groups = pd.unique(conv_arr)
    if len(groups) < 2:
        return {}

    labels = sorted(pd.unique(np.concatenate([yt, yp])))
    lab_idx = {c: i for i, c in enumerate(labels)}
    K = len(labels)
    gold_classes = sorted(pd.unique(yt))
    gold_idx = np.array([lab_idx[c] for c in gold_classes], dtype=int)
    trainable = TRAINABLE_CODES.get(level)
    learn_idx = (
        np.array([lab_idx[c] for c in gold_classes if c in trainable], dtype=int)
        if trainable else None
    )

    # Per-conversation confusion matrix, flattened: conf[g] is (K*K,).
    g_of = {g: i for i, g in enumerate(groups)}
    conf = np.zeros((len(groups), K * K), dtype=np.float64)
    ti = np.array([lab_idx[c] for c in yt], dtype=int)
    pi = np.array([lab_idx[c] for c in yp], dtype=int)
    gi = np.array([g_of[g] for g in conv_arr], dtype=int)
    np.add.at(conf, (gi, ti * K + pi), 1.0)

    # Resample: counts[b, g] = how many times conversation g was drawn in draw b
    # (multinomial over uniform conversations). Summed confusion per draw is then
    # counts @ conf.
    rng = np.random.default_rng(BOOT_SEED)
    counts = rng.multinomial(len(groups), np.full(len(groups), 1.0 / len(groups)),
                             size=N_BOOT).astype(np.float64)
    C = (counts @ conf).reshape(N_BOOT, K, K)              # (B, K, K)

    total = C.sum((1, 2))
    diag = np.diagonal(C, axis1=1, axis2=2)                # (B, K)
    rowsum = C.sum(2)                                       # true counts
    colsum = C.sum(1)                                       # pred counts
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.where(total > 0, diag.sum(1) / total, 0.0)
        prec = np.where(colsum > 0, diag / colsum, 0.0)
        rec = np.where(rowsum > 0, diag / rowsum, 0.0)
        denom = prec + rec
        f1 = np.where(denom > 0, 2 * prec * rec / denom, 0.0)
        pe = (rowsum * colsum).sum(1) / np.where(total > 0, total ** 2, 1.0)
        kappa = np.where(pe < 1.0, (acc - pe) / (1.0 - pe), 0.0)
        present = (rowsum + colsum) > 0                    # labels in true or pred
        n_present = present.sum(1)
        f1_all = np.where(n_present > 0, (f1 * present).sum(1) / n_present, 0.0)

    series = {
        "accuracy": acc,
        "kappa": kappa,
        "f1_macro": f1_all,
        "f1_macro_gold": (
            _macro_over_supported(f1, rowsum, gold_idx) if len(gold_idx) else None
        ),
        "f1_macro_learnable": (
            _macro_over_supported(f1, rowsum, learn_idx)
            if learn_idx is not None and len(learn_idx) else None
        ),
    }
    alpha = (1.0 - CI_LEVEL) / 2.0
    ci = {}
    for m in AGG_METRICS:
        vs = series.get(m)
        if vs is None:
            continue
        lo, hi = np.quantile(vs, [alpha, 1.0 - alpha])
        ci[m] = (float(lo), float(hi))
    return ci


# The main setting: 1 shared adapter, 2 calls (cell B). Moved from `ft_bare`
# (2 adapters) on 2026-09-28 -- it is the stronger backbone and the one the
# data-expansion arms are trained on.
PAIRED_REFERENCE = "ft1mix_bare"


def _mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on the discordant counts.

    ``b`` and ``c`` are the utterances where exactly one of the two arms is
    correct. Under the null they split 50/50, so this is a two-sided binomial
    test. Exact (not the chi-square approximation) because the discordant count
    is often small once the two arms agree on most utterances.
    """
    from scipy.stats import binomtest

    n = b + c
    if n == 0:
        return 1.0
    return float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)


def _paired_delta_ci(
    conv: np.ndarray, ref_ok: np.ndarray, arm_ok: np.ndarray
) -> tuple[float, float]:
    """Conversation-clustered 95% CI for the accuracy DELTA (arm − reference).

    Paired: both arms are scored on the same resampled conversations each draw,
    so the shared item difficulty cancels and the interval is about the gap
    itself. Resampling unit is the conversation, as everywhere else here.
    """
    groups = pd.unique(conv)
    if len(groups) < 2:
        return (float("nan"), float("nan"))
    g_of = {g: i for i, g in enumerate(groups)}
    gi = np.array([g_of[g] for g in conv], dtype=int)
    n_g = np.bincount(gi, minlength=len(groups)).astype(np.float64)
    ref_g = np.bincount(gi, weights=ref_ok.astype(float), minlength=len(groups))
    arm_g = np.bincount(gi, weights=arm_ok.astype(float), minlength=len(groups))
    rng = np.random.default_rng(BOOT_SEED)
    counts = rng.multinomial(len(groups), np.full(len(groups), 1.0 / len(groups)),
                             size=N_BOOT).astype(np.float64)
    tot = counts @ n_g
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(tot > 0, (counts @ arm_g) / tot - (counts @ ref_g) / tot, 0.0)
    alpha = (1.0 - CI_LEVEL) / 2.0
    lo, hi = np.quantile(delta, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def score(
    y_true: pd.Series,
    y_pred: pd.Series,
    level: str = "",
    conv: pd.Series | None = None,
) -> dict:
    gold_classes = sorted(y_true.unique())
    out = {
        "n": len(y_true),
        **_metric_values(np.asarray(y_true), np.asarray(y_pred), level, gold_classes),
    }
    trainable = TRAINABLE_CODES.get(level)
    out["n_gold_codes"] = len(gold_classes)
    out["n_unlearnable_codes"] = (
        len(gold_classes) - len([c for c in gold_classes if c in trainable])
        if trainable else None
    )
    # Conversation-clustered test-sampling CI for each metric, flattened into
    # `<metric>_lo` / `<metric>_hi` columns so they ride through aggregation and
    # rendering next to the point estimate.
    if conv is not None:
        for metric, (lo, hi) in _clustered_bootstrap_ci(
            y_true, y_pred, conv, level
        ).items():
            out[f"{metric}_lo"] = lo
            out[f"{metric}_hi"] = hi
    return out


def evaluate_file(
    tier: str, arm: str, style: str, ctx: int, seed: int | None, df: pd.DataFrame
) -> list[dict]:
    """Score one result file. `level` is the MISC tier (t1/t2), not the model tier."""
    rows = []
    suffix = f"_seed{seed}" if seed is not None else ""
    for level in ("t1", "t2"):
        gt, pred = df[f"{level}_label_GT"], df[f"{level}_label_auto"]
        valid = gt.notna() & pred.notna()
        conv = df["conv_id"] if "conv_id" in df.columns else None
        for scope in ("all", "counsellor", "client"):
            mask = valid if scope == "all" else valid & (df["speaker"] == scope)
            if mask.sum() == 0:
                continue
            rows.append({
                "tier": tier, "arm": arm, "style": style, "ctx": ctx, "seed": seed,
                "level": level.upper(), "scope": scope,
                **score(gt[mask], pred[mask], level,
                        conv=conv[mask] if conv is not None else None),
            })
        report = classification_report(
            gt[valid], pred[valid], output_dict=True, zero_division=0
        )
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        name = f"{tier}_{arm}_inf_{style}_ctx{ctx}{suffix}_{level}_report.csv"
        pd.DataFrame(report).transpose().to_csv(OUT_DIR / name)
    return rows


def exposure_file(
    tier: str, arm: str, style: str, ctx: int, seed: int | None, df: pd.DataFrame
) -> list[dict]:
    """Split T2 accuracy by whether the T1 it was conditioned on was correct.

    The two-call arms commit to a T1 label and inject it into the T2 prompt, so
    every T2 call is answered under either correct or incorrect conditioning.
    Marginal T2 accuracy averages the two and can hide a change entirely: an arm
    can get better on the wrong-T1 rows and worse on the right-T1 rows and move
    barely at all overall. That trade is exactly what the teacher-forcing axis
    is expected to make, which is why it needs its own table.

    Single-call arms emit both tiers in one generation with no committed T1, so
    they have no conditioning to split on and are skipped.

    Note the T1 groups PARTITION the T2 vocabulary and the T2 prompt lists only
    the predicted group's codes, so `T2 acc | T1 wrong` can only be non-zero
    when the model emits a code its own prompt did not offer. That is measured
    directly as `t2_out_of_group_rate`.
    """
    if arm not in TWO_CALL_ARMS:
        return []
    needed = ["t1_label_GT", "t1_label_auto", "t2_label_GT", "t2_label_auto"]
    if not set(needed).issubset(df.columns):
        return []
    d = df[df[needed].notna().all(axis=1)]
    if d.empty:
        return []

    t1_ok = d["t1_label_auto"] == d["t1_label_GT"]
    t2_ok = d["t2_label_auto"] == d["t2_label_GT"]
    # A wrong T1 and an UNPARSEABLE T1 are different conditions and must not be
    # pooled. When T1 comes back UNKNOWN the pipeline falls back to the
    # speaker's first code (infer.py:256), so T2 is conditioned on an arbitrary
    # group that is sometimes right by luck -- which shows up as "recovery" that
    # has nothing to do with the model overriding its spec.
    t1_unk = d["t1_label_auto"] == "UNKNOWN"
    t1_bad = (~t1_ok) & (~t1_unk)          # wrong, but a real named group
    parseable = ~t1_unk
    # Only meaningful where T1 named a group: t2_codes_for_group falls back to
    # the full speaker vocabulary for UNKNOWN, so nothing there can be "out of
    # group" and including those rows silently dilutes the rate.
    out_of_group = ~pd.Series(
        [
            r.t2_label_auto in t2_codes_for_group(r.speaker, r.t1_label_auto)
            for r in d[parseable].itertuples()
        ],
        index=d[parseable].index,
    )
    right, wrong, unk = t1_ok.sum(), t1_bad.sum(), t1_unk.sum()
    acc_right = float(t2_ok[t1_ok].mean()) if right else None
    acc_wrong = float(t2_ok[t1_bad].mean()) if wrong else None
    acc_unk = float(t2_ok[t1_unk].mean()) if unk else None
    return [{
        "tier": tier, "arm": arm, "style": style, "ctx": ctx, "seed": seed,
        "n": int(len(d)),
        "t1_err_rate": float(1.0 - t1_ok.mean()),
        "n_t1_correct": int(right),
        "t2_acc_given_t1_correct": acc_right,
        "n_t1_wrong": int(wrong),
        "t2_acc_given_t1_wrong": acc_wrong,
        "n_t1_unparseable": int(unk),
        "t2_acc_given_t1_unparseable": acc_unk,
        "gap": (None if acc_right is None or acc_wrong is None
                else acc_right - acc_wrong),
        "t2_acc_marginal": float(t2_ok.mean()),
        # How often the model emitted a T2 code outside the group its prompt
        # offered. The T1 groups partition the T2 vocabulary and the T2 prompt
        # lists only the predicted group, so a wrong T1 makes a correct T2
        # impossible UNLESS the model overrides the spec. This rate is whether
        # it ever does -- and the mechanism the teacher-forcing axis has to move.
        "t2_out_of_group_rate": (
            float(out_of_group.mean()) if len(out_of_group) else None
        ),
    }]


def collect_paired(store: dict, tier, arm, style, ctx, seed, df: pd.DataFrame) -> None:
    """Stash per-utterance correctness so arms can be compared as PAIRED samples.

    Every arm is scored on the same evaluation utterances, so a comparison
    against the reference is a paired test — far more powerful than treating the
    two arms' accuracies as independent. Keyed by ``(tier, style, ctx, level,
    scope)`` so only like-for-like cells are ever paired; the per-arm value is
    aligned on ``corp_utt_idx`` at table-build time.
    """
    if "corp_utt_idx" not in df.columns or "conv_id" not in df.columns:
        return
    for level in ("t1", "t2"):
        gt, pred = df.get(f"{level}_label_GT"), df.get(f"{level}_label_auto")
        if gt is None or pred is None:
            continue
        valid = gt.notna() & pred.notna()
        sub = df[valid]
        rec = pd.DataFrame({
            "corp_utt_idx": sub["corp_utt_idx"].to_numpy(),
            "conv_id": sub["conv_id"].to_numpy(),
            "correct": (gt[valid].to_numpy() == pred[valid].to_numpy()),
        })
        key = (tier, style, ctx, level.upper(), "all")
        store.setdefault(key, {})[(arm, seed)] = rec


def _primary_seed(seeds) -> object:
    """The replicate a paired test uses: the original run (seed None, i.e. 42)
    when present, else the lowest numbered seed. Using the same rule for the
    reference and every arm keeps the comparison seed-matched where it can be."""
    seeds = list(seeds)
    return None if None in seeds else min(seeds)


def _seed_sd(recs: dict) -> float | None:
    """Across-seed sample SD of accuracy, from ``{seed: rec}``; None if < 2 seeds."""
    if len(recs) < 2:
        return None
    return float(np.std([r["correct"].mean() for r in recs.values()], ddof=1))


def _seed_floor(sd_ref: float | None, sd_arm: float | None) -> float:
    """Smallest accuracy gap not explained by training-seed noise.

    The 95% band for the difference of two independent single-seed runs,
    1.96 * sqrt(sd_ref^2 + sd_arm^2). An arm with no replicates borrows the
    reference's SD. With no measured SD on either side, the conservative
    SEED_NOISE_FLOOR fallback applies.
    """
    if sd_ref is None and sd_arm is None:
        return SEED_NOISE_FLOOR
    a = sd_ref if sd_ref is not None else sd_arm
    b = sd_arm if sd_arm is not None else sd_ref
    return float(1.96 * np.sqrt(a ** 2 + b ** 2))


def _paired_tables(store: dict) -> list[str]:
    """Each arm against the PAIRED_REFERENCE arm, as a paired test on shared items.

    For every cell we report the accuracy of both arms, their difference, a
    conversation-clustered 95% CI for that difference, a two-sided exact
    McNemar p-value, and the seed floor that difference has to clear. A gap
    whose CI crosses zero is `ns` (not distinguishable from test-sampling
    noise); a gap smaller than the seed floor is `<floor` (not distinguishable
    from training-seed noise) even if its test CI excludes zero. Only a gap that
    clears both is called a real difference.
    """
    if not store:
        return []
    ref_label = CELL_LABELS.get((PAIRED_REFERENCE, "bare"), PAIRED_REFERENCE)
    lines = [
        f"## Paired comparison against {ref_label.split('_Inf')[0]}",
        "",
        f"Every arm scored against **{ref_label}** "
        "(same inference style) as a **paired** test: both arms are scored on the "
        "same utterances, so the comparison removes item difficulty and is far "
        "more powerful than comparing two separate accuracies. `Δacc` is arm minus "
        "reference; the bracket is its conversation-clustered 95% CI; `McNemar p` "
        "is the two-sided exact test on the utterances where the two arms disagree.",
        "",
        "`Floor` is the gap training-seed noise alone could produce: the 95% band "
        "for the difference of two single-seed runs, from the measured across-seed "
        "SD of the reference and of the arm (an arm without replicates borrows the "
        f"reference's; with neither measured, a conservative {SEED_NOISE_FLOOR:.2f}). "
        "It is larger for arms whose own replicates disagree — the staged "
        "teacher-forcing arms, for instance — and a staged arm without its own "
        f"replicates uses the {SEED_NOISE_FLOOR:.2f} fallback rather than borrowing "
        "the steadier plain reference's spread.",
        "",
        "`Verdict`: **ns** = the Δ CI crosses zero (within test-sampling noise); "
        "**<floor** = |Δ| below `Floor` (within training-seed noise); **real** = "
        "clears both. The test pairs the original run of each arm (seed 42) where "
        "both have one, else the lowest seed.",
        "",
    ]
    for level in ("T1", "T2"):
        header_done = False
        for tier in TIER_ORDER:
            for ctx in sorted({k[2] for k in store}):
                for style in STYLE_ORDER:
                    key = (tier, style, ctx, level, "all")
                    cell = store.get(key)
                    if not cell:
                        continue
                    by_arm: dict = {}
                    for (arm, seed), rec in cell.items():
                        by_arm.setdefault(arm, {})[seed] = rec
                    ref_recs = by_arm.pop(PAIRED_REFERENCE, None)
                    if not ref_recs or not by_arm:
                        continue
                    ref = ref_recs[_primary_seed(ref_recs)]
                    sd_ref = _seed_sd(ref_recs)
                    if not header_done:
                        lines += [f"### {level}", ""]
                        header_done = True
                    lines += [
                        f"#### {TIER_LABELS.get(tier, tier)} — Inf-{style} (ctx {ctx})",
                        "",
                        "| Arm | n | Seeds | Ref acc | Arm acc | Δacc | 95% CI "
                        "| McNemar p | Floor | Verdict |",
                        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
                    ]
                    for arm in ARM_ORDER:
                        if arm not in by_arm:
                            continue
                        recs = by_arm[arm]
                        rec = recs[_primary_seed(recs)]
                        floor = _seed_floor(sd_ref, _seed_sd(recs))
                        # Staged (teacher-forcing) training is measurably noisier
                        # than the plain recipe (cell B replicates moved up to
                        # 0.08), so a staged arm WITHOUT its own replicates must
                        # not borrow the plain reference's small SD.
                        if len(recs) < 2 and "_tf" in arm:
                            floor = max(floor, SEED_NOISE_FLOOR)
                        m = ref.merge(rec, on="corp_utt_idx", suffixes=("_ref", "_arm"))
                        if m.empty:
                            continue
                        r_ok = m["correct_ref"].to_numpy()
                        a_ok = m["correct_arm"].to_numpy()
                        ref_acc, arm_acc = r_ok.mean(), a_ok.mean()
                        delta = arm_acc - ref_acc
                        b = int(((~a_ok) & r_ok).sum())   # arm wrong, ref right
                        c = int((a_ok & (~r_ok)).sum())   # arm right, ref wrong
                        p = _mcnemar_p(b, c)
                        lo, hi = _paired_delta_ci(
                            m["conv_id_ref"].to_numpy(), r_ok, a_ok
                        )
                        crosses_zero = pd.isna(lo) or (lo <= 0.0 <= hi)
                        if crosses_zero:
                            verdict = "ns"
                        elif abs(delta) < floor:
                            verdict = "&lt;floor"
                        else:
                            verdict = "**real**"
                        label = CELL_LABELS.get((arm, style), f"{arm}/{style}")
                        lines.append(
                            f"| {label} | {len(m)} | {len(recs)} | {_fmt(ref_acc)} "
                            f"| {_fmt(arm_acc)} | {_fmt(delta, '+.3f')} "
                            f"| [{_fmt(lo, '+.3f')}, {_fmt(hi, '+.3f')}] "
                            f"| {_fmt(p)} | {_fmt(floor)} | {verdict} |"
                        )
                    lines.append("")
    return lines


def _fmt(value, spec: str = ".3f") -> str:
    return "—" if value is None or pd.isna(value) else format(value, spec)


def _exposure_tables(exposure: pd.DataFrame) -> list[str]:
    if exposure.empty:
        return []
    lines = [
        "## T2 accuracy by Tier-1 correctness",
        "",
        "Every two-call arm trains its T2 adapter conditioned on the **gold** T1 "
        "label but runs it conditioned on the **predicted** one, so a fifth of "
        "T2 calls meet a group spec the model never saw paired with that "
        "utterance in training. This table separates the two populations.",
        "",
        "`T2 acc | T1 ok` is accuracy on the rows where the conditioning was "
        "right; `T2 acc | T1 wrong` where it was wrong. **Gap** is the "
        "difference — how much of T2's performance depends on T1 having been "
        "correct. Training on predicted rather than gold T1 should shrink the "
        "gap; whether it also moves the marginal figure is a separate question, "
        "and the two can move in opposite directions.",
        "",
        "**Wrong and unparseable T1 are separated on purpose.** When T1 comes "
        "back `UNKNOWN` the pipeline falls back to the speaker's first code, so "
        "T2 is conditioned on an arbitrary group that is sometimes right by "
        "luck. Pooling those rows with genuinely wrong-but-named T1 makes an "
        "arm look like it recovers when it is only benefiting from a lucky "
        "fallback — `ft1seq_bare` is the case in point, where half the T1 "
        "errors are unparseable. `Out-of-group` is measured over parseable rows "
        "only, since the group is undefined for the rest.",
        "",
        "**Read the wrong-T1 column against the last one.** The T1 groups "
        "partition the T2 vocabulary, and the T2 prompt lists only the predicted "
        "group's codes. So when T1 is wrong the gold T2 sits in a group the "
        "prompt never showed, and T2 can only be right if the model emits a code "
        "it was not offered. `Out-of-group` is how often it does. Decoding is "
        "*not* constrained (`restrict_t2_to_group: false`) — the model restricts "
        "itself — so this is a behavioural ceiling, not an arithmetic one, and "
        "it is the mechanism the teacher-forcing axis exists to move.",
        "",
    ]
    group_cols = ["tier", "arm", "style", "ctx"]
    agg = exposure.groupby(group_cols, dropna=False).mean(numeric_only=True).reset_index()

    for tier in TIER_ORDER:
        sub_t = agg[agg["tier"] == tier]
        if sub_t.empty:
            continue
        lines += [
            f"### {TIER_LABELS.get(tier, tier)}",
            "",
            "| Condition | ctx | n | T1 err | n (T1 ok) | T2 acc \\| T1 ok "
            "| n (T1 wrong) | T2 acc \\| T1 wrong | n (T1 unparseable) "
            "| T2 acc \\| unparseable | Gap | T2 acc | Out-of-group |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        sub_t = sub_t.assign(
            _arm=sub_t["arm"].map({a: i for i, a in enumerate(ARM_ORDER)})
        ).sort_values(["_arm", "ctx", "style"])
        for _, r in sub_t.iterrows():
            label = CELL_LABELS.get((r["arm"], r["style"]), f"{r['arm']}/{r['style']}")
            lines.append(
                f"| {label} | {int(r['ctx'])} | {int(r['n'])} "
                f"| {_fmt(r['t1_err_rate'], '.1%')} "
                f"| {int(r['n_t1_correct'])} | {_fmt(r['t2_acc_given_t1_correct'])} "
                f"| {int(r['n_t1_wrong'])} | {_fmt(r['t2_acc_given_t1_wrong'])} "
                f"| {int(r['n_t1_unparseable'])} "
                f"| {_fmt(r['t2_acc_given_t1_unparseable'])} "
                f"| {_fmt(r['gap'])} | {_fmt(r['t2_acc_marginal'])} "
                f"| {_fmt(r['t2_out_of_group_rate'], '.1%')} |"
            )
        lines.append("")
    return lines


def compliance_file(
    tier: str, arm: str, style: str, ctx: int, seed: int | None, df: pd.DataFrame
) -> list[dict]:
    """Measure whether each cell followed its inference instruction.

    Free-generation runs (the local tier) carry an explicit
    `<level>_emitted_rationale` flag. The gpt-4o runs went through a pydantic
    `response_format`, so an explanation was present exactly when the schema had
    the field; that is recorded as `constrained` because the model had no
    opportunity to disobey.
    """
    rows = []
    for level in ("t1", "t2"):
        pred = df.get(f"{level}_label_auto")
        if pred is None:
            continue
        n = int(len(df))
        unknown = int((pred == "UNKNOWN").sum())

        flag_col = f"{level}_emitted_rationale"
        expl_col = f"{level}_expl_auto"
        if flag_col in df.columns:
            emitted = int(df[flag_col].fillna(False).astype(bool).sum())
            constrained = False
        elif expl_col in df.columns:
            expl = df[expl_col].fillna("").astype(str).str.strip()
            emitted = int((expl != "").sum())
            constrained = True
        else:
            emitted, constrained = None, None

        rec = {
            "tier": tier, "arm": arm, "style": style, "ctx": ctx, "seed": seed,
            "level": level.upper(), "n": n,
            "unparseable": unknown,
            "unparseable_rate": unknown / n if n else 0.0,
            "emitted_rationale": emitted,
            "emitted_rationale_rate": (emitted / n) if (emitted is not None and n) else None,
            "expected_rationale": style == "cot",
            "constrained_decoding": constrained,
        }
        if emitted is not None and n:
            # 1.0 = every utterance did what the prompt asked.
            rate = emitted / n
            rec["instruction_followed_rate"] = rate if style == "cot" else 1.0 - rate
        else:
            rec["instruction_followed_rate"] = None
        rows.append(rec)
    return rows


def _fmt(v, spec: str = ".3f") -> str:
    return "—" if v is None or pd.isna(v) else format(v, spec)


# Metrics averaged across seeds. Everything else in a cell (n, the code counts)
# is a property of the evaluation set and is identical for every seed.
AGG_METRICS = ("accuracy", "kappa", "f1_macro_gold", "f1_macro_learnable", "f1_macro")
CELL_KEYS = ["tier", "arm", "style", "ctx", "level", "scope"]


def aggregate_seeds(results: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-seed rows into one row per cell, with a spread.

    The GRPO arms are run at several seeds and a single seed's macro-F1 is a
    noisy thing to draw a conclusion from, so the reported figure is the mean
    over seeds and `<metric>_std` its sample standard deviation. Deterministic
    arms have one seed, no spread, and pass through unchanged.
    """
    rows = []
    for key, grp in results.groupby(CELL_KEYS, dropna=False, sort=False):
        rec = dict(zip(CELL_KEYS, key))
        rec["n_seeds"] = len(grp)
        rec["n"] = int(grp["n"].iloc[0])
        for col in ("n_gold_codes", "n_unlearnable_codes"):
            if col in grp.columns:
                rec[col] = grp[col].iloc[0]
        for metric in AGG_METRICS:
            if metric not in grp.columns:
                continue
            vals = grp[metric].dropna()
            rec[metric] = vals.mean() if len(vals) else None
            rec[f"{metric}_std"] = vals.std(ddof=1) if len(vals) > 1 else None
            # Clustered test-sampling CI. Single-seed cells pass their interval
            # through unchanged; multi-seed cells average the endpoints across
            # seeds (test-sampling uncertainty is reported per seed, the seed
            # spread separately via `_std`), so the column stays a per-seed
            # test-sampling band rather than being conflated with seed variance.
            for bound in ("lo", "hi"):
                col = f"{metric}_{bound}"
                if col in grp.columns:
                    bvals = grp[col].dropna()
                    rec[col] = bvals.mean() if len(bvals) else None
        rows.append(rec)
    return pd.DataFrame(rows)


def _fmt_agg(row, metric: str) -> str:
    """A metric as `mean [lo–hi]`, with `± std` added when several seeds exist.

    The bracket is the 95% conversation-clustered test-sampling CI; `± std` is
    the across-seed spread (only present for the repeated arms). The two are
    distinct kinds of uncertainty and are shown side by side, never merged.
    """
    mean = _fmt(row.get(metric))
    if mean == "—":
        return mean
    std = row.get(f"{metric}_std")
    lo, hi = row.get(f"{metric}_lo"), row.get(f"{metric}_hi")
    out = mean
    if std is not None and not pd.isna(std):
        out = f"{out} ± {format(std, '.3f')}"
    if lo is not None and hi is not None and not pd.isna(lo) and not pd.isna(hi):
        out = f"{out} [{format(lo, '.3f')}–{format(hi, '.3f')}]"
    return out


def coverage_frame(results: pd.DataFrame) -> pd.DataFrame:
    """One row per intended cell, marked done or missing, plus anything extra.

    Built from the T1/all rows because every cell has exactly one of those; the
    seed count comes from how many per-seed rows collapsed into it.
    """
    have: dict[tuple, tuple[int, int]] = {}
    if not results.empty:
        sub = results[(results["level"] == "T1") & (results["scope"] == "all")]
        for key, grp in sub.groupby(["tier", "arm", "style", "ctx"], sort=False):
            tier, arm, style, ctx = key
            have[(tier, arm, style, int(ctx))] = (int(grp["n"].iloc[0]), len(grp))

    rows, seen = [], set()
    for tier, arms in EXPECTED_GRID.items():
        for arm, styles in arms.items():
            for ctx in ARM_CTXS.get(arm, EXPECTED_CTXS):
                for style in styles:
                    key = (tier, arm, style, int(ctx))
                    seen.add(key)
                    n, seeds = have.get(key, (0, 0))
                    if not seeds:
                        status = "missing"
                    elif n < EXPECTED_EVAL_N:
                        status = "partial"
                    else:
                        status = "done"
                    rows.append({
                        "tier": tier,
                        "structure": ARM_STRUCTURE.get(arm, "?"),
                        "arm": arm,
                        "style": style,
                        "ctx": int(ctx),
                        "status": status,
                        "n": n,
                        "n_seeds": seeds,
                    })
    # A result that exists but was never part of the intended grid still has to
    # show up, or the coverage table would quietly under-report what was run.
    for key, (n, seeds) in have.items():
        if key in seen:
            continue
        tier, arm, style, ctx = key
        rows.append({
            "tier": tier, "structure": ARM_STRUCTURE.get(arm, "?"), "arm": arm,
            "style": style, "ctx": ctx, "status": "extra", "n": n, "n_seeds": seeds,
        })

    cov = pd.DataFrame(rows)
    return cov.sort_values(
        by=["tier", "structure", "arm", "ctx", "style"],
        key=lambda col: col.map(
            {v: i for i, v in enumerate(STRUCTURE_ORDER)}
        ) if col.name == "structure" else col,
    ).reset_index(drop=True)


def _coverage_tables(cov: pd.DataFrame, dates: dict[tuple, str]) -> list[str]:
    lines = [
        "## Coverage",
        "",
        "Which cells of the intended grid exist. `done` counts cells scored over "
        f"all {EXPECTED_EVAL_N} evaluation utterances; a cell run at several "
        "seeds still counts once. `Partial` cells are mid-run: prediction "
        "checkpoints every 25 utterances, so a job still working has a valid but "
        "short file on disk. **Do not read a partial cell as a result.**",
        "",
        "| Tier | Structure | ctx | Done | Partial | Expected |",
        "|---|---|---:|---:|---:|---:|",
    ]
    planned = cov[cov["status"] != "extra"]
    for tier in TIER_ORDER:
        for structure in STRUCTURE_ORDER:
            for ctx in sorted(planned["ctx"].unique()):
                sub = planned[
                    (planned["tier"] == tier)
                    & (planned["structure"] == structure)
                    & (planned["ctx"] == ctx)
                ]
                if sub.empty:
                    continue
                done = int((sub["status"] == "done").sum())
                part = int((sub["status"] == "partial").sum())
                lines.append(
                    f"| {TIER_LABELS.get(tier, tier)} "
                    f"| {STRUCTURE_LABELS.get(structure, structure)} "
                    f"| {ctx} | {done} | {part or '—'} | {len(sub)} |"
                )
    lines.append("")

    partial = planned[planned["status"] == "partial"]
    if not partial.empty:
        lines += [
            f"Mid-run, not yet complete ({len(partial)} cells):",
            "",
        ]
        for _, r in partial.iterrows():
            lines.append(
                f"- `{r['tier']}_{r['arm']}_inf_{r['style']}_ctx{r['ctx']}` "
                f"— {r['n']}/{EXPECTED_EVAL_N} utterances"
            )
        lines.append("")

    missing = planned[planned["status"] == "missing"]
    if not missing.empty:
        lines += [
            f"Still missing ({len(missing)} cells):",
            "",
        ]
        for _, r in missing.iterrows():
            lines.append(
                f"- `{r['tier']}_{r['arm']}_inf_{r['style']}_ctx{r['ctx']}`"
            )
        lines.append("")

    extra = cov[cov["status"] == "extra"]
    if not extra.empty:
        lines += [
            "Scored but outside the intended grid:",
            "",
        ]
        for _, r in extra.iterrows():
            lines.append(
                f"- `{r['tier']}_{r['arm']}_inf_{r['style']}_ctx{r['ctx']}`"
            )
        lines.append("")

    lines += _run_date_table(cov, dates)
    return lines


def _run_date_table(cov: pd.DataFrame, dates: dict[tuple, str]) -> list[str]:
    """When each cell was produced, for cells that recorded it.

    Dates come from the `.meta.json` sidecar written when a prediction run
    finishes. Cells that predate sidecars have no recorded date and are counted
    rather than dated: file mtimes here are all the timestamp of the last bulk
    rescore, so inferring from them would attach a confident wrong date to a
    real result. Campaign-level dates for those runs are in
    `docs/EXPERIMENT_LOG.md`, sourced from the commit history.
    """
    # A cell can carry several seeds; report the most recent.
    by_cell: dict[tuple, str] = {}
    for (tier, arm, style, ctx, _seed), date in dates.items():
        key = (tier, arm, style, ctx)
        by_cell[key] = max(date, by_cell.get(key, ""))

    scored = cov[cov["status"].isin(["done", "partial", "extra"])]
    dated = [
        (r["tier"], r["arm"], r["style"], int(r["ctx"]))
        for _, r in scored.iterrows()
        if (r["tier"], r["arm"], r["style"], int(r["ctx"])) in by_cell
    ]
    if not dated:
        return [
            f"No cell carries a recorded run date yet ({len(scored)} scored). "
            "Dates are stamped by prediction runs from this point on; see "
            "`docs/EXPERIMENT_LOG.md` for campaign-level history.",
            "",
        ]

    lines = [
        "### Run dates",
        "",
        f"{len(dated)} of {len(scored)} scored cells carry a recorded finish "
        "date. The rest predate run stamping and are dated only at campaign "
        "level in `docs/EXPERIMENT_LOG.md` — file timestamps here all reflect "
        "the last rescore rather than the run, so they are not used.",
        "",
        "| Cell | ctx | Date (UTC) |",
        "|---|---:|---|",
    ]
    for key in sorted(dated, key=lambda k: (by_cell[k], k), reverse=True):
        tier, arm, style, ctx = key
        lines.append(f"| `{tier}_{arm}_inf_{style}` | {ctx} | {by_cell[key]} |")
    lines.append("")
    return lines


def _performance_tables(results: pd.DataFrame) -> list[str]:
    agg = aggregate_seeds(results)
    lines = []
    for level in ("T1", "T2"):
        for ctx in sorted(agg["ctx"].unique()):
            sub = agg[(agg["level"] == level) & (agg["ctx"] == ctx)]
            if sub.empty:
                continue
            lines += [f"## {level} results — context = {ctx} volleys", ""]
            for tier in TIER_ORDER:
                tier_sub = sub[sub["tier"] == tier]
                if tier_sub.empty:
                    continue
                lines += [f"### {TIER_LABELS.get(tier, tier)}", ""]
                for structure in STRUCTURE_ORDER:
                    arms = [a for a in ARM_ORDER if ARM_STRUCTURE.get(a) == structure]
                    struct_sub = tier_sub[tier_sub["arm"].isin(arms)]
                    if struct_sub.empty:
                        continue
                    # Only the GRPO arms are repeated, so the column would be a
                    # column of 1s in every table that has no RL arm in it.
                    show_seeds = int(struct_sub["n_seeds"].max()) > 1
                    head = ["Condition", "Scope", "n"]
                    align = ["---", "---", "---:"]
                    if show_seeds:
                        head.append("Seeds")
                        align.append("---:")
                    head += [
                        "Accuracy", "Cohen's kappa", "Macro-F1 (gold)",
                        "Macro-F1 (learnable)", "Macro-F1 (all)",
                    ]
                    align += ["---:"] * 5
                    lines += [
                        f"#### {STRUCTURE_LABELS.get(structure, structure)}",
                        "",
                        "| " + " | ".join(head) + " |",
                        "|" + "|".join(align) + "|",
                    ]
                    for arm in arms:
                        for style in STYLE_ORDER:
                            for scope in ("all", "counsellor", "client"):
                                r = struct_sub[
                                    (struct_sub["arm"] == arm)
                                    & (struct_sub["style"] == style)
                                    & (struct_sub["scope"] == scope)
                                ]
                                if r.empty:
                                    continue
                                r = r.iloc[0]
                                cell = CELL_LABELS.get((arm, style), f"{arm}/{style}")
                                vals = [cell, scope, str(int(r["n"]))]
                                if show_seeds:
                                    vals.append(str(int(r["n_seeds"])))
                                vals += [
                                    _fmt_agg(r, "accuracy"),
                                    _fmt_agg(r, "kappa"),
                                    _fmt_agg(r, "f1_macro_gold"),
                                    _fmt_agg(r, "f1_macro_learnable"),
                                    _fmt_agg(r, "f1_macro"),
                                ]
                                lines.append("| " + " | ".join(vals) + " |")
                    lines.append("")
    return lines


# The structural contrast, arm by arm at a matched training target. Each row is
# one cell of the adapters x calls matrix; reading down a column varies exactly
# one structural axis.
CONTRAST_ROWS = [
    ("bare", "ft_bare", "A", "2 adapters, 2 calls"),
    ("bare", "ft1mix_bare", "B", "1 adapter, 2 calls (mixed)"),
    ("bare", "ft1seq_bare", "B", "1 adapter, 2 calls (sequential)"),
    ("bare", "sc_ft_bare", "F", "1 adapter, 1 call"),
    ("rat", "ft_rat", "A", "2 adapters, 2 calls"),
    ("rat", "ft1mix_rat", "B", "1 adapter, 2 calls (mixed)"),
    ("rat", "ft1seq_rat", "B", "1 adapter, 2 calls (sequential)"),
    ("rat", "sc_ft_rat", "F", "1 adapter, 1 call"),
]
TARGET_LABELS = {"bare": "bare-label training", "rat": "rationale + label training"}


def _contrast_tables(results: pd.DataFrame) -> list[str]:
    """Line up cells A, B and F so one structural axis moves at a time.

    The per-block tables above are grouped by structure, which makes each block
    internally readable but leaves the cross-structure comparison spread over
    three tables. This puts them in one place at a matched target and context.
    """
    if results.empty:
        return []
    agg = aggregate_seeds(results)
    sub = agg[(agg["tier"] == "qwen") & (agg["scope"] == "all")]
    if sub.empty:
        return []

    lines = [
        "## Structural contrast (adapters x calls)",
        "",
        "The same numbers as above, arranged so that **one structural axis moves "
        "at a time**. Within a target and context: A against B varies adapter "
        "count with calls fixed at 2; B against F varies call count with adapters "
        "fixed at 1. A against F moves both and cannot separate them.",
        "",
        "Each single-call arm is only evaluated in the style it was trained in, "
        "so its row is filled under whichever style that is and blank under the "
        "other. Compare like with like: read a column, not a row.",
        "",
    ]
    for ctx in sorted(sub["ctx"].unique()):
        for target in ("bare", "rat"):
            wanted = [r for r in CONTRAST_ROWS if r[0] == target]
            present = sub[(sub["ctx"] == ctx) & (sub["arm"].isin(a for _, a, _, _ in wanted))]
            if present.empty:
                continue
            lines += [
                f"### ctx = {ctx}, {TARGET_LABELS[target]}",
                "",
                "| Cell | Structure | Inf | n | T1 acc | T2 acc | "
                "T2 Macro-F1 (learnable) |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
            for _, arm, cell, label in wanted:
                for style in STYLE_ORDER:
                    rows = present[
                        (present["arm"] == arm) & (present["style"] == style)
                    ]
                    if rows.empty:
                        continue
                    t1 = rows[rows["level"] == "T1"]
                    t2 = rows[rows["level"] == "T2"]
                    if t1.empty or t2.empty:
                        continue
                    t1, t2 = t1.iloc[0], t2.iloc[0]
                    lines.append(
                        f"| {cell} | {label} | {style} | {int(t1['n'])} "
                        f"| {_fmt_agg(t1, 'accuracy')} "
                        f"| {_fmt_agg(t2, 'accuracy')} "
                        f"| {_fmt_agg(t2, 'f1_macro_learnable')} |"
                    )
            lines.append("")
    return lines


def _faithfulness_tables() -> list[str]:
    """Fold the rationale-swap probe into the report.

    The probe is produced by a separate job (`baseline.faithfulness`) and lands
    as JSON, so it is read from disk here rather than recomputed. Without this
    the only record of it would be the raw files and a hand-maintained table in
    docs/GRPO.md.
    """
    probes = []
    for path in sorted(FAITHFULNESS_DIR.glob("faithfulness_*.json")):
        try:
            probes.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    if not probes:
        return []

    lines = [
        "## Faithfulness (rationale-swap probe)",
        "",
        "Accuracy cannot say whether an emitted rationale had anything to do "
        "with the label: a model can write fluent MI-flavoured prose and predict "
        "the code from the utterance alone, scoring the same either way. The "
        "probe forces in a rationale the model generated for a DIFFERENT "
        "utterance with a different gold code and lets it continue.",
        "",
        "Forcing a prefix is itself an intervention, so it runs against a control "
        "that forces the model's own rationale back in; the reported effect is "
        "`net = swap - control`. High net flip means the rationale was "
        "load-bearing, near zero means it is post-hoc narration. `Donor-match` "
        "separates steering from disturbance: drifting to a random third code "
        "means the swap merely confused the model, landing on the donor's own "
        "code means it followed it.",
        "",
    ]
    for level in ("T1", "T2"):
        p = level.lower()
        lines += [
            f"### {level}",
            "",
            "| Arm | ctx | Seed | n | Control flip | Swap flip | Net flip | "
            "Donor-match |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for d in sorted(probes, key=lambda x: (x.get("ctx", 0), str(x.get("arm")))):
            seed = d.get("seed")
            donor = (
                _fmt(d.get(f"{p}_donor_match_rate"), ".3f")
                if f"{p}_donor_match_rate" in d else "—"
            )
            lines.append(
                f"| `{d.get('arm')}` | {d.get('ctx')} "
                f"| {'—' if seed is None else seed} | {d.get('n')} "
                f"| {_fmt(d.get(f'{p}_control_flip_rate'))} "
                f"| {_fmt(d.get(f'{p}_swap_flip_rate'))} "
                f"| **{_fmt(d.get(f'{p}_net_flip_rate'), '+.3f')}** "
                f"| {donor} |"
            )
        lines.append("")

    lines += [
        "Read `sc_zs` first. Taken alone a large net flip on a trained arm looks "
        "like a finding, but a model with no rationale training at all scores "
        "highest on every column, so a forced prefix steering the label is what "
        "an autoregressive model does rather than evidence about training. What "
        "survives is the comparison between arms, not the absolute magnitude. "
        "Discussion: `docs/GRPO.md`.",
        "",
    ]
    return lines


def _aggregate_compliance(comp: pd.DataFrame) -> pd.DataFrame:
    """Average the compliance counts and rates over seeds, as `aggregate_seeds`
    does for performance. Counts become means and are rounded for display."""
    keys = ["tier", "arm", "style", "ctx", "level"]
    numeric = [
        "n", "unparseable", "unparseable_rate", "emitted_rationale",
        "emitted_rationale_rate", "instruction_followed_rate",
    ]
    rows = []
    for key, grp in comp.groupby(keys, dropna=False, sort=False):
        rec = dict(zip(keys, key))
        rec["n_seeds"] = len(grp)
        for col in numeric:
            vals = grp[col].dropna() if col in grp.columns else pd.Series(dtype=float)
            rec[col] = vals.mean() if len(vals) else None
        for col in ("expected_rationale", "constrained_decoding"):
            rec[col] = grp[col].iloc[0] if col in grp.columns else None
        rows.append(rec)
    return pd.DataFrame(rows)


def _compliance_tables(comp: pd.DataFrame) -> list[str]:
    comp = _aggregate_compliance(comp)
    lines = [
        "## Instruction compliance",
        "",
        "Whether each cell obeyed its inference instruction: `Inf-CoT` should "
        "produce a rationale, `Inf-Bare` should not. `Followed` is the share of "
        "utterances that did the requested thing.",
        "",
        "Rows marked `constrained` were decoded through a JSON schema that "
        "forced the output shape, so the model had no opportunity to disobey and "
        "the figure is structural rather than behavioural.",
        "",
    ]
    for ctx in sorted(comp["ctx"].unique()):
        sub = comp[comp["ctx"] == ctx]
        if sub.empty:
            continue
        lines += [
            f"### Context = {ctx} volleys",
            "",
            "| Tier | Condition | Level | n | Rationale expected | "
            "Rationale emitted | Followed | Unparseable | Decoding |",
            "|---|---|---|---:|---|---:|---:|---:|---|",
        ]
        for tier in TIER_ORDER:
            for arm in ARM_ORDER:
                for style in STYLE_ORDER:
                    for level in ("T1", "T2"):
                        r = sub[
                            (sub["tier"] == tier) & (sub["arm"] == arm)
                            & (sub["style"] == style) & (sub["level"] == level)
                        ]
                        if r.empty:
                            continue
                        r = r.iloc[0]
                        cell = CELL_LABELS.get((arm, style), f"{arm}/{style}")
                        # Values arrive as numpy scalars, so identity checks
                        # against None/False do not hold; test for nullness.
                        # Counts are means over seeds for the repeated arms, so
                        # round rather than truncate.
                        emitted = (
                            "—" if pd.isna(r["emitted_rationale"])
                            else f"{round(r['emitted_rationale'])} "
                                 f"({_fmt(r['emitted_rationale_rate'], '.1%')})"
                        )
                        cd = r["constrained_decoding"]
                        decoding = (
                            "—" if pd.isna(cd)
                            else "constrained" if bool(cd) else "free"
                        )
                        lines.append(
                            f"| {TIER_LABELS.get(tier, tier)} | {cell} | {level} "
                            f"| {round(r['n'])} | {'yes' if r['expected_rationale'] else 'no'} "
                            f"| {emitted} | {_fmt(r['instruction_followed_rate'], '.1%')} "
                            f"| {round(r['unparseable'])} "
                            f"({_fmt(r['unparseable_rate'], '.1%')}) | {decoding} |"
                        )
        lines.append("")
    return lines


# -- Cross-scheme generalization datasets ---------------------------------
# Corpora coded in a DIFFERENT MI scheme, scored only on the vocabulary they
# share with MISC (see docs; the crosswalk is 1:1 on these codes). `t1`/`t2`
# map a MISC tier to {speaker: [shared codes]}. A model prediction outside the
# shared set counts as wrong (it is a real error against that gold), and is also
# tracked as the out-of-vocab rate — part of "seeing the pattern".
CROSS_SCHEME = {
    "annomi": {
        "label": "AnnoMI — cross-scheme (gold)",
        "note": "AnnoMI's core codes overlap MISC: client change/sustain/neutral "
                "talk on T1, therapist question/reflection/information on T2. "
                "The overlap is approximate, not identical: AnnoMI 'open' is not "
                "MISC OQ (68/114 disagree) and its SR/CR line differs "
                "(docs/DATASETS.md). Out-of-scheme AnnoMI codes and multi-label "
                "volleys are excluded upstream (see `prep_crossscheme.py`). "
                "**Optimistic:** AnnoMI transcripts 15, 21, 44 and 53 are HLQC "
                "training sessions and are still included here; the re-run "
                "excludes them.",
        "levels": {
            "t1": {"client": ["C", "S", "N"]},
            "t2": {"counsellor": ["OQ", "CQ", "SR", "CR", "GI"]},
        },
    },
    "welivita": {
        "label": "Welivita / MITI — cross-scheme (weak-gold)",
        "note": "A MITI variant (labels adapted from MITI 2.0/4.2.1; maps to MITI "
                "via `schemes.mappings.WELIVITA_TO_MITI`), scored here after the "
                "corpus's own mapping to MISC T2. Counsellor-only, and the labels "
                "are crowd labels (kappa 0.34), not consensus gold, so read this "
                "as a coarse generalization signal. The self-training arm is "
                "excluded (it trained on this corpus).",
        "levels": {
            "t2": {"counsellor": ["GI", "ADW", "CR", "SU", "AF", "CQ", "DI",
                                  "SR", "ADP", "OQ", "CO", "EC", "WA"]},
        },
    },
}


def _crossscheme_metrics(
    gt: pd.Series, pred: pd.Series, conv: pd.Series, allowed: list
) -> dict:
    """Accuracy / kappa / macro-F1 over the shared-code set, with clustered CI.

    Macro-F1 (shared) averages over the shared codes that have gold support —
    in the point estimate and, per resample, in the conversation-clustered
    bootstrap (`_macro_over_supported`). A shared code with no gold instance
    contributes no information about the model and is left out rather than
    scored as a zero.
    """
    yt, yp = np.asarray(gt), np.asarray(pred)
    labels = sorted(set(allowed) | set(pd.unique(yt)) | set(pd.unique(yp)))
    lab_idx = {c: i for i, c in enumerate(labels)}
    K = len(labels)
    allowed_idx = np.array([lab_idx[c] for c in allowed], dtype=int)
    supported = [c for c in allowed if c in set(yt)]
    out = {
        "n": int(len(yt)),
        "accuracy": accuracy_score(yt, yp),
        "kappa": cohen_kappa_score(yt, yp),
        "f1_macro_shared": f1_score(
            yt, yp, labels=supported, average="macro", zero_division=0
        ) if supported else 0.0,
        "oov_pred_rate": float(np.mean([p not in allowed for p in yp])),
    }
    conv_arr = np.asarray(conv)
    groups = pd.unique(conv_arr)
    if len(groups) >= 2:
        g_of = {g: i for i, g in enumerate(groups)}
        conf = np.zeros((len(groups), K * K))
        ti = np.array([lab_idx[c] for c in yt])
        pi = np.array([lab_idx[c] for c in yp])
        gi = np.array([g_of[g] for g in conv_arr])
        np.add.at(conf, (gi, ti * K + pi), 1.0)
        rng = np.random.default_rng(BOOT_SEED)
        counts = rng.multinomial(
            len(groups), np.full(len(groups), 1.0 / len(groups)), size=N_BOOT
        ).astype(float)
        C = (counts @ conf).reshape(N_BOOT, K, K)
        total = C.sum((1, 2))
        diag = np.diagonal(C, axis1=1, axis2=2)
        rowsum, colsum = C.sum(2), C.sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            acc = np.where(total > 0, diag.sum(1) / total, 0.0)
            prec = np.where(colsum > 0, diag / colsum, 0.0)
            rec = np.where(rowsum > 0, diag / rowsum, 0.0)
            den = prec + rec
            f1 = np.where(den > 0, 2 * prec * rec / den, 0.0)
        f1_shared = (_macro_over_supported(f1, rowsum, allowed_idx)
                     if len(allowed_idx) else np.zeros(N_BOOT))
        a = (1.0 - CI_LEVEL) / 2.0
        for name, s in (("accuracy", acc), ("f1_macro_shared", f1_shared)):
            lo, hi = np.quantile(s, [a, 1.0 - a])
            out[f"{name}_lo"], out[f"{name}_hi"] = float(lo), float(hi)
    return out


def _crossscheme_tables(cross_files: list) -> list[str]:
    """Score every non-default-dataset result on its shared vocabulary.

    `cross_files` is a list of ``(dataset, tier, arm, style, ctx, seed, df)``.
    One block per dataset, one sub-table per (tier, level, speaker), one row per
    arm: accuracy and macro-F1 on the shared codes (with clustered CIs), coverage
    (how much of the corpus is in the shared vocabulary), and the out-of-vocab
    prediction rate. This is the generalization number plus the "pattern" of how
    the model behaves on a corpus in a different coding scheme.
    """
    if not cross_files:
        return []
    lines = [
        "# Cross-scheme generalization",
        "",
        "The model, trained on MISC-coded HLQC, run on **other MI corpora** and "
        "scored only on the vocabulary those corpora share with MISC. A result "
        "that holds here — not just on MIV6.3A — is generalization evidence that "
        "no amount of seed-tuning on one dataset can give. Metrics carry the same "
        "conversation-clustered 95% CI as the primary tables.",
        "",
        "`Coverage` = share of that speaker's gold labels that fall in the shared "
        "vocabulary (the rest cannot be scored cross-scheme). `OOV pred` = share "
        "of the model's predictions that fall OUTSIDE the shared set — high OOV "
        "means the model reaches for codes the other scheme does not use.",
        "",
    ]
    by_ds: dict = {}
    for rec in cross_files:
        by_ds.setdefault(rec[0], []).append(rec)

    for ds, spec in CROSS_SCHEME.items():
        recs = by_ds.get(ds)
        if not recs:
            continue
        lines += [f"## {spec['label']}", "", spec["note"], ""]
        for level, speakers in spec["levels"].items():
            for speaker, allowed in speakers.items():
                lines += [
                    f"### {level.upper()} — {speaker} "
                    f"(shared codes: {', '.join(allowed)})",
                    "",
                    "| Arm | n | Coverage | Accuracy | Macro-F1 (shared) | OOV pred |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
                rows = []
                for _ds, tier, arm, style, ctx, seed, df in recs:
                    gt_col, pred_col = f"{level}_label_GT", f"{level}_label_auto"
                    if gt_col not in df or pred_col not in df:
                        continue
                    valid = df[gt_col].notna() & df[pred_col].notna() \
                        & (df["speaker"] == speaker)
                    if valid.sum() == 0:
                        continue
                    gold_all = df[valid]
                    in_vocab = gold_all[gt_col].isin(allowed)
                    coverage = float(in_vocab.mean())
                    sub = gold_all[in_vocab]
                    if sub.empty:
                        continue
                    m = _crossscheme_metrics(
                        sub[gt_col], sub[pred_col],
                        sub["conv_id"] if "conv_id" in sub else pd.Series(range(len(sub))),
                        allowed,
                    )
                    label = CELL_LABELS.get((arm, style), f"{arm}/{style}")
                    acc = _fmt(m["accuracy"])
                    if "accuracy_lo" in m:
                        acc += f" [{_fmt(m['accuracy_lo'])}–{_fmt(m['accuracy_hi'])}]"
                    f1s = _fmt(m["f1_macro_shared"])
                    if "f1_macro_shared_lo" in m:
                        f1s += f" [{_fmt(m['f1_macro_shared_lo'])}–{_fmt(m['f1_macro_shared_hi'])}]"
                    rows.append((
                        (TIER_ORDER.index(tier) if tier in TIER_ORDER else 9,
                         ARM_ORDER.index(arm) if arm in ARM_ORDER else 99),
                        f"| {label} | {m['n']} | {coverage:.1%} | {acc} | {f1s} "
                        f"| {m['oov_pred_rate']:.1%} |",
                    ))
                for _, row in sorted(rows, key=lambda r: r[0]):
                    lines.append(row)
                lines.append("")
    return lines


def main():
    files = discover_result_files()
    if not files:
        print(f"no result files found in {ANNOTATED_DIR}")
        return

    all_rows, comp_rows, exp_rows = [], [], []
    paired_store: dict = {}
    cross_files: list = []
    for tier, arm, style, ctx, seed, dataset, path in files:
        df = pd.read_csv(path)
        tag = "" if seed is None else f" seed={seed}"
        ds_tag = "" if dataset == DEFAULT_DATASET else f" ds={dataset}"
        print(
            f"evaluating {tier}/{arm}/inf_{style} ctx={ctx}{tag}{ds_tag}: "
            f"{len(df)} rows ({path.name})"
        )
        if dataset != DEFAULT_DATASET:
            # Different coding scheme: scored on shared vocabulary in its own
            # section, never pooled with the primary grid.
            cross_files.append((dataset, tier, arm, style, ctx, seed, df))
            continue
        all_rows.extend(evaluate_file(tier, arm, style, ctx, seed, df))
        comp_rows.extend(compliance_file(tier, arm, style, ctx, seed, df))
        exp_rows.extend(exposure_file(tier, arm, style, ctx, seed, df))
        collect_paired(paired_store, tier, arm, style, ctx, seed, df)

    results = pd.DataFrame(all_rows)
    comp = pd.DataFrame(comp_rows)
    exposure = pd.DataFrame(exp_rows)
    cov = coverage_frame(results)
    dates = load_run_dates()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_DIR / "comparison.csv", index=False)
    comp.to_csv(OUT_DIR / "compliance.csv", index=False)
    cov.to_csv(OUT_DIR / "coverage.csv", index=False)
    if not exposure.empty:
        exposure.to_csv(OUT_DIR / "exposure.csv", index=False)
    print(results.to_string(index=False))

    counts = cov["status"].value_counts()
    print(
        f"\ncoverage: {counts.get('done', 0)} cells done, "
        f"{counts.get('partial', 0)} partial (mid-run), "
        f"{counts.get('missing', 0)} missing"
    )

    from datetime import datetime, timezone

    lines = [
        "# Model Scale x Adaptation x Rationale Alignment — Results",
        "",
        f"*Generated {datetime.now(timezone.utc):%Y-%m-%d} by `baseline.eval`. "
        "Per-cell run dates are in the Coverage section; campaign history is in "
        "[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).*",
        "",
        "Grid: 2 model tiers x adaptation arm x 2 inference styles "
        "(`Inf-Bare`, `Inf-CoT`), read within structural blocks.",
        "",
        "Results are grouped by where an arm sits in the **adapters x calls** "
        "matrix, because a comparison across one axis is only interpretable "
        "with the other held fixed:",
        "",
        "| | 2 adapters | 1 adapter |",
        "|---|---|---|",
        "| **2-call** | `ft_bare`, `ft_rat` | `ft1mix_*`, `ft1seq_*` |",
        "| **1-call** | structurally impossible | `sc_ft_bare`, `sc_ft_rat`, "
        "`sc_grpo*` |",
        "",
        "`ft_*` against `ft1*` varies adapter count at a fixed 2 calls; `ft1*` "
        "against `sc_*` varies call count at a fixed 1 adapter. Comparing "
        "`ft_bare` directly against `sc_ft_bare` moves both at once and cannot "
        "separate the two effects.",
        "",
        "The two 1-adapter regimes differ only in how that single adapter was "
        "trained: `Mix` on the shuffled union of the T1 and T2 examples, `Seq` "
        "on T1 first and then continued on T2. Both see the same number of "
        "example-passes, so any gap between them is ordering rather than "
        "budget; a large T1 drop on `Seq` is the T2 stage overwriting the T1 "
        "stage.",
        "",
        "Evaluation set: `data/manual/MIV6.3A_manual.csv` (821 human-consensus "
        "utterances). Few-shot exemplars and fine-tuning data both come from "
        "`data/manual/HLQC_balanced_manual.csv`, held out from evaluation. "
        "Exemplars, distilled rationales, and fine-tuning prompts are all frozen "
        "per context length so training and evaluation contexts always match.",
        "",
        "`FT-Bare` trains on bare-label targets; `FT-Rat` trains on gpt-4o "
        "distilled rationale + label targets. Those rationales are post-hoc, "
        "generated conditioned on the gold label, so they may not be faithful to "
        "any reasoning that would independently produce the label.",
        "",
        "Arms prefixed `SC` use the single-call format: one generation emits the "
        "rationale and both tier labels together, instead of a Tier-1 call "
        "followed by a Tier-2 call conditioned on its answer. Compare them to the "
        "1-adapter two-call block above, not to the 2-adapter block, or the "
        "adapter count moves with the call count. `SC GRPO` is initialised from "
        "`SC FT-Bare` and trained with a reward on label correctness, hierarchy "
        "consistency and output format; its two Phase-2 variants isolate "
        "rare-class weighting and the initialisation.",
        "",
        "Arm definitions, frozen artifacts, hyperparameters and how to reproduce "
        "any cell: `docs/EXPERIMENTS.md`. This file is generated by "
        "`baseline.eval` — edit that, not this.",
        "",
        "`Macro-F1 (gold)` averages F1 only over codes that occur in the gold "
        "labels — the number comparable to the paper. `Macro-F1 (all)` also counts "
        "codes the model predicted but that never occur in gold (each "
        "contributing 0). `Macro-F1 (learnable)` drops the codes that never occur "
        "in the training corpus at all (`TS+`, `AC-`), which no arm trained on "
        "HLQC can predict and which therefore enter Macro-F1 (gold) as guaranteed "
        "zeros.",
        "",
        "Where an arm was run at several seeds, the figure is the mean over seeds "
        "and the `± std` is the sample standard deviation across seeds; a `Seeds` "
        "column appears in those tables. Per-seed rows are in "
        "`outputs/baseline_eval/comparison.csv`.",
        "",
        f"**Error bars.** The bracket after each metric is a {int(CI_LEVEL * 100)}% "
        "confidence interval from a bootstrap that resamples whole **conversations** "
        "(not utterances), because the evaluation utterances are nested in only a "
        "handful of conversations and rows within one are correlated — a row-level "
        "bootstrap would report intervals several times too narrow. This is "
        "test-sampling uncertainty (how much the number depends on which "
        "conversations were scored) and is separate from `± std` (how much it "
        "depends on the training seed). A difference between two arms is only "
        "meaningful if it clears **both**.",
        "",
        "**Noise floor.** How much a result moves when only the training seed "
        "changes depends on the arm. The plain 1-adapter baseline is steady: over "
        "three seeds its T2 accuracy spans 0.010. The staged teacher-forcing arms "
        "have moved T2 by up to 0.08. The paired comparison therefore derives each "
        "gap's floor from the measured seed spread of the two arms involved (the "
        "`Floor` column), and falls back to a conservative "
        f"{SEED_NOISE_FLOOR:.2f} only where no spread has been measured. A gap "
        "below its floor is **within noise** — not a result — until it is repeated "
        "across seeds.",
        "",
        "## Paper reference (GPT-4.1, hierarchical, 3 context volleys, clean set)",
        "",
        "| Tier | Scope | Metric | Paper |",
        "|---|---|---|---:|",
    ]
    for level, scope, metric, value in PAPER_REFERENCE:
        lines.append(f"| {level} | {scope} | {metric} | {value:.2f} |")
    lines.append("")

    lines += _coverage_tables(cov, dates)
    lines += _paired_tables(paired_store)
    lines += _contrast_tables(results)
    lines += _performance_tables(results)
    lines += _exposure_tables(exposure)
    if not comp.empty:
        lines += _compliance_tables(comp)
    lines += _faithfulness_tables()
    lines += _crossscheme_tables(cross_files)

    lines += [
        "Per-code precision/recall/F1 reports: "
        "`outputs/baseline_eval/<tier>_<arm>_inf_<style>_ctx<N>_<t1|t2>_report.csv`. "
        "Arm definitions and the artifacts each one needs: `docs/EXPERIMENTS.md`.",
        "",
    ]
    DOCS_PATH.write_text("\n".join(lines))
    print(
        f"\nwrote {OUT_DIR / 'comparison.csv'}, {OUT_DIR / 'compliance.csv'}, "
        f"{OUT_DIR / 'coverage.csv'} and {DOCS_PATH}"
    )


if __name__ == "__main__":
    main()
