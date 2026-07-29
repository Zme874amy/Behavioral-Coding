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

TIER_ORDER = ["gpt4o", "qwen"]
ARM_ORDER = ["zs", "fs", "ft_bare", "ft_rat"]
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

_STEM_RE = re.compile(r"(?P<head>.+?)_inf_(?P<style>bare|cot)(?:_ctx(?P<ctx>\d+))?$")
_LEGACY_RE = re.compile(r"(?P<stem>[a-z_]+?)(?:_ctx(?P<ctx>\d+))?$")


def parse_stem(stem: str) -> tuple[str, str, str, int] | None:
    """Map a filename stem to ``(tier, arm, style, ctx)``, or None if unknown."""
    m = _STEM_RE.fullmatch(stem)
    if m:
        head, style = m.group("head"), m.group("style")
        ctx = int(m.group("ctx")) if m.group("ctx") else 5
        # Longest arm first so `ft_bare` is not shadowed by a shorter match.
        for arm in sorted(ARM_ORDER, key=len, reverse=True):
            if head.endswith(f"_{arm}"):
                tier = head[: -(len(arm) + 1)]
                if tier:
                    return tier, arm, style, ctx
        return None

    m = _LEGACY_RE.fullmatch(stem)
    if m and m.group("stem") in LEGACY_STEMS:
        tier, arm, style = LEGACY_STEMS[m.group("stem")]
        ctx = int(m.group("ctx")) if m.group("ctx") else 5
        return tier, arm, style, ctx
    return None


def condition_label(tier: str, arm: str, style: str) -> str:
    cell = CELL_LABELS.get((arm, style), f"{arm}/{style}")
    return f"{TIER_LABELS.get(tier, tier)} — {cell}"


def _sort_key(rec: tuple) -> tuple:
    tier, arm, style, ctx = rec[:4]
    return (
        TIER_ORDER.index(tier) if tier in TIER_ORDER else len(TIER_ORDER),
        ARM_ORDER.index(arm) if arm in ARM_ORDER else len(ARM_ORDER),
        STYLE_ORDER.index(style) if style in STYLE_ORDER else len(STYLE_ORDER),
        ctx,
    )


def _precedence(stem: str) -> tuple[int, int]:
    """Rank candidates for the same cell: new naming beats legacy, explicit ctx
    beats the un-suffixed ctx5 fallback."""
    is_new = _STEM_RE.fullmatch(stem) is not None
    return (int(is_new), int("_ctx" in stem))


def discover_result_files() -> list[tuple[str, str, str, int, Path]]:
    """Return (tier, arm, style, ctx, path) for every recognised result file.

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


def score(y_true: pd.Series, y_pred: pd.Series) -> dict:
    gold_classes = sorted(y_true.unique())
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro_gold": f1_score(
            y_true, y_pred, labels=gold_classes, average="macro", zero_division=0
        ),
    }


def evaluate_file(tier: str, arm: str, style: str, ctx: int, df: pd.DataFrame) -> list[dict]:
    """Score one result file. `level` is the MISC tier (t1/t2), not the model tier."""
    rows = []
    for level in ("t1", "t2"):
        gt, pred = df[f"{level}_label_GT"], df[f"{level}_label_auto"]
        valid = gt.notna() & pred.notna()
        for scope in ("all", "counsellor", "client"):
            mask = valid if scope == "all" else valid & (df["speaker"] == scope)
            if mask.sum() == 0:
                continue
            rows.append({
                "tier": tier, "arm": arm, "style": style, "ctx": ctx,
                "level": level.upper(), "scope": scope,
                **score(gt[mask], pred[mask]),
            })
        report = classification_report(
            gt[valid], pred[valid], output_dict=True, zero_division=0
        )
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        name = f"{tier}_{arm}_inf_{style}_ctx{ctx}_{level}_report.csv"
        pd.DataFrame(report).transpose().to_csv(OUT_DIR / name)
    return rows


def compliance_file(tier: str, arm: str, style: str, ctx: int, df: pd.DataFrame) -> list[dict]:
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
            "tier": tier, "arm": arm, "style": style, "ctx": ctx,
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


def _performance_tables(results: pd.DataFrame) -> list[str]:
    lines = []
    for level in ("T1", "T2"):
        for ctx in sorted(results["ctx"].unique()):
            sub = results[(results["level"] == level) & (results["ctx"] == ctx)]
            if sub.empty:
                continue
            lines += [f"## {level} results — context = {ctx} volleys", ""]
            for tier in TIER_ORDER:
                tier_sub = sub[sub["tier"] == tier]
                if tier_sub.empty:
                    continue
                lines += [
                    f"### {TIER_LABELS.get(tier, tier)}",
                    "",
                    "| Condition | Scope | n | Accuracy | Cohen's kappa "
                    "| Macro-F1 (gold) | Macro-F1 (all) |",
                    "|---|---|---:|---:|---:|---:|---:|",
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
                            lines.append(
                                f"| {cell} | {scope} | {int(r['n'])} "
                                f"| {_fmt(r['accuracy'])} | {_fmt(r['kappa'])} "
                                f"| {_fmt(r['f1_macro_gold'])} | {_fmt(r['f1_macro'])} |"
                            )
                lines.append("")
    return lines


def _compliance_tables(comp: pd.DataFrame) -> list[str]:
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
                        emitted = (
                            "—" if pd.isna(r["emitted_rationale"])
                            else f"{int(r['emitted_rationale'])} "
                                 f"({_fmt(r['emitted_rationale_rate'], '.1%')})"
                        )
                        cd = r["constrained_decoding"]
                        decoding = (
                            "—" if pd.isna(cd)
                            else "constrained" if bool(cd) else "free"
                        )
                        lines.append(
                            f"| {TIER_LABELS.get(tier, tier)} | {cell} | {level} "
                            f"| {int(r['n'])} | {'yes' if r['expected_rationale'] else 'no'} "
                            f"| {emitted} | {_fmt(r['instruction_followed_rate'], '.1%')} "
                            f"| {int(r['unparseable'])} "
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
    for tier, arm, style, ctx, path in files:
        df = pd.read_csv(path)
        print(
            f"evaluating {tier}/{arm}/inf_{style} ctx={ctx}: "
            f"{len(df)} rows ({path.name})"
        )
        all_rows.extend(evaluate_file(tier, arm, style, ctx, df))
        comp_rows.extend(compliance_file(tier, arm, style, ctx, df))

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
        "`Macro-F1 (gold)` averages F1 only over codes that occur in the gold "
        "labels — the number comparable to the paper. `Macro-F1 (all)` also counts "
        "codes the model predicted but that never occur in gold (each "
        "contributing 0).",
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
