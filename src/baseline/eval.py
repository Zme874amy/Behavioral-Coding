"""Evaluate the AutoMISC baseline conditions against human ground truth.

Reads every per-condition CSV in data/annotated/baseline/, scores T1 and T2
predictions against t1_label_GT / t2_label_GT (overall and per speaker), and
writes a combined comparison table:

    outputs/baseline_eval/comparison.csv
    outputs/baseline_eval/<condition>_<tier>_report.csv   (per-code P/R/F1)
    docs/BASELINE_RESULTS.md

Usage:
    PYTHONPATH=src .venv/bin/python -m baseline.eval
"""
from __future__ import annotations

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


def score(y_true: pd.Series, y_pred: pd.Series) -> dict:
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def evaluate_condition(name: str, df: pd.DataFrame) -> list[dict]:
    rows = []
    for tier in ("t1", "t2"):
        gt, pred = df[f"{tier}_label_GT"], df[f"{tier}_label_auto"]
        valid = gt.notna() & pred.notna()
        for scope in ("all", "counsellor", "client"):
            mask = valid if scope == "all" else valid & (df["speaker"] == scope)
            if mask.sum() == 0:
                continue
            rows.append({
                "condition": name,
                "tier": tier.upper(),
                "scope": scope,
                **score(gt[mask], pred[mask]),
            })
        # Per-code report on all valid rows
        report = classification_report(
            gt[valid], pred[valid], output_dict=True, zero_division=0
        )
        report_df = pd.DataFrame(report).transpose()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(OUT_DIR / f"{name}_{tier}_report.csv")
    return rows


def main():
    all_rows = []
    found = []
    for name in CONDITION_ORDER:
        path = ANNOTATED_DIR / f"{name}.csv"
        if not path.exists():
            print(f"skipping {name}: {path} not found")
            continue
        df = pd.read_csv(path)
        found.append((name, len(df)))
        all_rows.extend(evaluate_condition(name, df))

    results = pd.DataFrame(all_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_DIR / "comparison.csv", index=False)
    print(results.to_string(index=False))

    # Markdown summary
    lines = [
        "# AutoMISC Baseline Reproduction — Results",
        "",
        "Model: Azure OpenAI `gpt-4o`, temperature 0, hierarchical T1→T2 prompts, "
        "interval context of 5 turns.",
        "Evaluation set: `data/manual/MIV6.3A_manual.csv` (human consensus labels).",
        "Few-shot exemplars and fine-tuning data: `data/manual/HLQC_balanced_manual.csv` "
        "(held out, no leakage).",
        "",
    ]
    for tier in ("T1", "T2"):
        lines.append(f"## {tier} results")
        lines.append("")
        lines.append("| Condition | Scope | n | Accuracy | Cohen's kappa | Macro-F1 |")
        lines.append("|---|---|---:|---:|---:|---:|")
        sub = results[results["tier"] == tier]
        for name in CONDITION_ORDER:
            for scope in ("all", "counsellor", "client"):
                r = sub[(sub["condition"] == name) & (sub["scope"] == scope)]
                if r.empty:
                    continue
                r = r.iloc[0]
                lines.append(
                    f"| {CONDITION_LABELS[name]} | {scope} | {int(r['n'])} "
                    f"| {r['accuracy']:.3f} | {r['kappa']:.3f} | {r['f1_macro']:.3f} |"
                )
        lines.append("")
    lines.append("Per-code precision/recall/F1 reports: `outputs/baseline_eval/<condition>_<tier>_report.csv`.")
    lines.append("")
    DOCS_PATH.write_text("\n".join(lines))
    print(f"\nwrote {OUT_DIR / 'comparison.csv'} and {DOCS_PATH}")
    print(f"conditions evaluated: {found}")


if __name__ == "__main__":
    main()
