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

import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ANNOTATED_DIR = REPO_ROOT / "data" / "annotated" / "baseline"
OUT_DIR = REPO_ROOT / "outputs" / "baseline_eval"
DOCS_PATH = REPO_ROOT / "docs" / "BASELINE_RESULTS.md"

TRAIN_CSV = REPO_ROOT / "data" / "manual" / "HLQC_balanced_manual.csv"

TIER_ORDER = ["gpt4o", "qwen"]
# The `sc_` arms use the single-call format (one generation for both tiers,
# src/baseline/single_call.py) and are only comparable to each other. They are
# listed after the two-call arms so the two ladders read as separate blocks.
TWO_CALL_ARMS = ["zs", "fs", "ft_bare", "ft_rat"]
SINGLE_CALL_ARMS = [
    "sc_zs", "sc_fs", "sc_ft_bare", "sc_ft_rat",
    "sc_grpo", "sc_grpo_unw", "sc_grpo_cold",
]
ARM_ORDER = TWO_CALL_ARMS + SINGLE_CALL_ARMS
STYLE_ORDER = ["bare", "cot"]

TIER_LABELS = {
    "gpt4o": "GPT-4o (frontier)",
    "qwen": "Qwen2.5-7B-Instruct (SLM)",
}

# (arm, style) -> display name, using the shorthand from the experiment design.
CELL_LABELS = {
    ("zs", "bare"): "ZS-Bare",
    ("zs", "cot"): "ZS-CoT",
    ("fs", "bare"): "FS-Bare",
    ("fs", "cot"): "FS-CoT",
    ("ft_bare", "bare"): "FT-Bare_Inf-Bare",
    ("ft_bare", "cot"): "FT-Bare_Inf-CoT",
    ("ft_rat", "bare"): "FT-Rat_Inf-Bare",
    ("ft_rat", "cot"): "FT-Rat_Inf-CoT",
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

_STEM_RE = re.compile(
    r"(?P<head>.+?)_inf_(?P<style>bare|cot)(?:_ctx(?P<ctx>\d+))?"
    r"(?:_seed(?P<seed>\d+))?$"
)
_LEGACY_RE = re.compile(r"(?P<stem>[a-z_]+?)(?:_ctx(?P<ctx>\d+))?$")


def parse_stem(stem: str) -> tuple[str, str, str, int, int | None] | None:
    """Map a filename stem to ``(tier, arm, style, ctx, seed)``, or None.

    `seed` is None for the deterministic arms and an integer for the GRPO runs,
    which are repeated across seeds and aggregated in the reported table.
    """
    m = _STEM_RE.fullmatch(stem)
    if m:
        head, style = m.group("head"), m.group("style")
        ctx = int(m.group("ctx")) if m.group("ctx") else 5
        seed = int(m.group("seed")) if m.group("seed") else None
        # Longest arm first so `ft_bare` is not shadowed by a shorter match.
        for arm in sorted(ARM_ORDER, key=len, reverse=True):
            if head.endswith(f"_{arm}"):
                tier = head[: -(len(arm) + 1)]
                if tier:
                    return tier, arm, style, ctx, seed
        return None

    m = _LEGACY_RE.fullmatch(stem)
    if m and m.group("stem") in LEGACY_STEMS:
        tier, arm, style = LEGACY_STEMS[m.group("stem")]
        ctx = int(m.group("ctx")) if m.group("ctx") else 5
        return tier, arm, style, ctx, None
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


def score(y_true: pd.Series, y_pred: pd.Series, level: str = "") -> dict:
    gold_classes = sorted(y_true.unique())
    out = {
        "n": len(y_true),
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
        out["n_gold_codes"] = len(gold_classes)
        out["n_unlearnable_codes"] = len(gold_classes) - len(learnable)
    else:
        out["f1_macro_learnable"] = None
        out["n_gold_codes"] = len(gold_classes)
        out["n_unlearnable_codes"] = None
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
        for scope in ("all", "counsellor", "client"):
            mask = valid if scope == "all" else valid & (df["speaker"] == scope)
            if mask.sum() == 0:
                continue
            rows.append({
                "tier": tier, "arm": arm, "style": style, "ctx": ctx, "seed": seed,
                "level": level.upper(), "scope": scope,
                **score(gt[mask], pred[mask], level),
            })
        report = classification_report(
            gt[valid], pred[valid], output_dict=True, zero_division=0
        )
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        name = f"{tier}_{arm}_inf_{style}_ctx{ctx}{suffix}_{level}_report.csv"
        pd.DataFrame(report).transpose().to_csv(OUT_DIR / name)
    return rows


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
        rows.append(rec)
    return pd.DataFrame(rows)


def _fmt_agg(row, metric: str) -> str:
    """A metric as `mean ± std`, or bare when there is only one seed."""
    mean = _fmt(row.get(metric))
    std = row.get(f"{metric}_std")
    if std is None or pd.isna(std):
        return mean
    return f"{mean} ± {format(std, '.3f')}"


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
                # Only the GRPO arms are repeated, so the column would be a
                # column of 1s in every table that has no RL arm in it.
                show_seeds = int(tier_sub["n_seeds"].max()) > 1
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
                    f"### {TIER_LABELS.get(tier, tier)}",
                    "",
                    "| " + " | ".join(head) + " |",
                    "|" + "|".join(align) + "|",
                ]
                for arm in ARM_ORDER:
                    for style in STYLE_ORDER:
                        for scope in ("all", "counsellor", "client"):
                            r = tier_sub[
                                (tier_sub["arm"] == arm)
                                & (tier_sub["style"] == style)
                                & (tier_sub["scope"] == scope)
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


def main():
    files = discover_result_files()
    if not files:
        print(f"no result files found in {ANNOTATED_DIR}")
        return

    all_rows, comp_rows = [], []
    for tier, arm, style, ctx, seed, path in files:
        df = pd.read_csv(path)
        tag = "" if seed is None else f" seed={seed}"
        print(
            f"evaluating {tier}/{arm}/inf_{style} ctx={ctx}{tag}: "
            f"{len(df)} rows ({path.name})"
        )
        all_rows.extend(evaluate_file(tier, arm, style, ctx, seed, df))
        comp_rows.extend(compliance_file(tier, arm, style, ctx, seed, df))

    results = pd.DataFrame(all_rows)
    comp = pd.DataFrame(comp_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_DIR / "comparison.csv", index=False)
    comp.to_csv(OUT_DIR / "compliance.csv", index=False)
    print(results.to_string(index=False))

    lines = [
        "# Model Scale x Adaptation x Rationale Alignment — Results",
        "",
        "Grid: 2 model tiers x 4 adaptation arms (`ZS`, `FS`, `FT-Bare`, `FT-Rat`) "
        "x 2 inference styles (`Inf-Bare`, `Inf-CoT`).",
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
        "followed by a Tier-2 call conditioned on its answer. The two formats are "
        "each internally comparable but not comparable to each other, so read the "
        "`SC` block as its own ladder. `SC GRPO` is initialised from `SC FT-Bare` "
        "and trained with a reward on label correctness, hierarchy consistency "
        "and output format; its two Phase-2 variants isolate rare-class weighting "
        "and the initialisation.",
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
        "and the spread is the sample standard deviation; a `Seeds` column appears "
        "in those tables. Per-seed rows are in "
        "`outputs/baseline_eval/comparison.csv`.",
        "",
        "## Paper reference (GPT-4.1, hierarchical, 3 context volleys, clean set)",
        "",
        "| Tier | Scope | Metric | Paper |",
        "|---|---|---|---:|",
    ]
    for level, scope, metric, value in PAPER_REFERENCE:
        lines.append(f"| {level} | {scope} | {metric} | {value:.2f} |")
    lines.append("")

    lines += _performance_tables(results)
    if not comp.empty:
        lines += _compliance_tables(comp)

    lines += [
        "Per-code precision/recall/F1 reports: "
        "`outputs/baseline_eval/<tier>_<arm>_inf_<style>_ctx<N>_<t1|t2>_report.csv`.",
        "",
    ]
    DOCS_PATH.write_text("\n".join(lines))
    print(
        f"\nwrote {OUT_DIR / 'comparison.csv'}, {OUT_DIR / 'compliance.csv'} "
        f"and {DOCS_PATH}"
    )


if __name__ == "__main__":
    main()
