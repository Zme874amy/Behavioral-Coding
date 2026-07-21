"""Evaluate the AutoMISC baseline conditions against human ground truth.

Scans data/annotated/baseline/ for per-condition, per-context result files
(<condition>_ctx<N>.csv; a legacy un-suffixed <condition>.csv is treated as
ctx5), scores T1 and T2 predictions against t1_label_GT / t2_label_GT
(overall and per speaker), and writes a combined comparison table:

    outputs/baseline_eval/comparison.csv
    outputs/baseline_eval/<condition>_ctx<N>_<tier>_report.csv  (per-code P/R/F1)
    docs/BASELINE_RESULTS.md

Metrics:
    accuracy, Cohen's kappa
    f1_macro       macro-F1 over all classes sklearn sees (gold plus predicted)
    f1_macro_gold  macro-F1 over classes that occur in the gold labels only —
                   the comparable number to the AutoMISC paper, since codes the
                   model predicts but that never occur in gold otherwise
                   contribute an automatic 0 to the average.

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

CONDITION_ORDER = [
    "zeroshot_rationales",
    "zeroshot_bare",
    "fewshot_rationales",
    "fewshot_bare",
    "finetuned",
]

CONDITION_LABELS = {
    "zeroshot_rationales": "Zero-shot + rationales (original AutoMISC)",
    "zeroshot_bare": "Zero-shot, no rationales",
    "fewshot_rationales": "Few-shot + rationales",
    "fewshot_bare": "Few-shot, no rationales",
    "finetuned": "Fine-tuned gpt-4o (label-only, zero-shot)",
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


def discover_result_files() -> list[tuple[str, int, Path]]:
    """Return (condition, ctx, path) for every result file, ordered by
    condition then context. Legacy un-suffixed files count as ctx5."""
    found = []
    for path in ANNOTATED_DIR.glob("*.csv"):
        m = re.fullmatch(r"(?P<cond>[a-z_]+?)(?:_ctx(?P<ctx>\d+))?", path.stem)
        if not m or m.group("cond") not in CONDITION_ORDER:
            continue
        ctx = int(m.group("ctx")) if m.group("ctx") else 5
        found.append((m.group("cond"), ctx, path))
    # Legacy + suffixed duplicates for the same (cond, ctx): prefer suffixed.
    dedup = {}
    for cond, ctx, path in found:
        key = (cond, ctx)
        if key not in dedup or "_ctx" in path.stem:
            dedup[key] = path
    return sorted(
        [(c, x, p) for (c, x), p in dedup.items()],
        key=lambda t: (CONDITION_ORDER.index(t[0]), t[1]),
    )


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


def evaluate_file(condition: str, ctx: int, df: pd.DataFrame) -> list[dict]:
    rows = []
    for tier in ("t1", "t2"):
        gt, pred = df[f"{tier}_label_GT"], df[f"{tier}_label_auto"]
        valid = gt.notna() & pred.notna()
        for scope in ("all", "counsellor", "client"):
            mask = valid if scope == "all" else valid & (df["speaker"] == scope)
            if mask.sum() == 0:
                continue
            rows.append({
                "condition": condition,
                "ctx": ctx,
                "tier": tier.upper(),
                "scope": scope,
                **score(gt[mask], pred[mask]),
            })
        report = classification_report(
            gt[valid], pred[valid], output_dict=True, zero_division=0
        )
        report_df = pd.DataFrame(report).transpose()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(OUT_DIR / f"{condition}_ctx{ctx}_{tier}_report.csv")
    return rows


def main():
    files = discover_result_files()
    if not files:
        print(f"no result files found in {ANNOTATED_DIR}")
        return

    all_rows = []
    for condition, ctx, path in files:
        df = pd.read_csv(path)
        print(f"evaluating {condition} ctx={ctx}: {len(df)} rows ({path.name})")
        all_rows.extend(evaluate_file(condition, ctx, df))

    results = pd.DataFrame(all_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_DIR / "comparison.csv", index=False)
    print(results.to_string(index=False))

    contexts = sorted(results["ctx"].unique())

    lines = [
        "# AutoMISC Baseline Reproduction — Results",
        "",
        "Model: Azure OpenAI `gpt-4o`, temperature 0, hierarchical T1→T2 prompts, "
        "interval context. Context length (number of prior volleys) is compared "
        "across runs.",
        "Evaluation set: `data/manual/MIV6.3A_manual.csv` (human consensus labels).",
        "Few-shot exemplars and fine-tuning data: `data/manual/HLQC_balanced_manual.csv` "
        "(held out, no leakage). Exemplars and fine-tuning prompts are rebuilt per "
        "context length so training/eval contexts always match.",
        "",
        "`Macro-F1 (gold)` averages F1 only over codes that occur in the gold labels "
        "— the number comparable to the paper. `Macro-F1 (all)` also counts codes "
        "the model predicted but that never occur in gold (each contributing 0).",
        "",
        "## Paper reference (GPT-4.1, hierarchical, 3 context volleys, clean set)",
        "",
        "| Tier | Scope | Metric | Paper |",
        "|---|---|---|---:|",
    ]
    for tier, scope, metric, value in PAPER_REFERENCE:
        lines.append(f"| {tier} | {scope} | {metric} | {value:.2f} |")
    lines.append("")

    for tier in ("T1", "T2"):
        for ctx in contexts:
            sub = results[(results["tier"] == tier) & (results["ctx"] == ctx)]
            if sub.empty:
                continue
            lines.append(f"## {tier} results — context = {ctx} volleys")
            lines.append("")
            lines.append("| Condition | Scope | n | Accuracy | Cohen's kappa "
                         "| Macro-F1 (gold) | Macro-F1 (all) |")
            lines.append("|---|---|---:|---:|---:|---:|---:|")
            for name in CONDITION_ORDER:
                for scope in ("all", "counsellor", "client"):
                    r = sub[(sub["condition"] == name) & (sub["scope"] == scope)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    lines.append(
                        f"| {CONDITION_LABELS[name]} | {scope} | {int(r['n'])} "
                        f"| {r['accuracy']:.3f} | {r['kappa']:.3f} "
                        f"| {r['f1_macro_gold']:.3f} | {r['f1_macro']:.3f} |"
                    )
            lines.append("")
    lines.append("Per-code precision/recall/F1 reports: "
                 "`outputs/baseline_eval/<condition>_ctx<N>_<tier>_report.csv`.")
    lines.append("")
    DOCS_PATH.write_text("\n".join(lines))
    print(f"\nwrote {OUT_DIR / 'comparison.csv'} and {DOCS_PATH}")


if __name__ == "__main__":
    main()
