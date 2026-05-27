"""Re-evaluate the zero-shot baseline using the **production rich prompt**
from `src/components/prompts/templates/{speaker}/flat.j2` so we have a fair
comparison against the fine-tuned LoRA model.

Run AFTER `src/sft/main.py` (or `src/sft/eval.py`) has produced
`predictions.csv`. This script reuses the exact same 300 test rows and only
overwrites the zero-shot column.

Usage:
  python src/sft/rich_zero_shot.py \\
    --predictions outputs/sft_runs/qwen05b_full/eval/sft_evaluation/predictions.csv \\
    --base-model Qwen/Qwen2.5-0.5B-Instruct \\
    --out outputs/sft_runs/qwen05b_full/eval/sft_evaluation/predictions_rich.csv
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from components.prompts.loader import render_prompt  # noqa: E402  (after sys.path)
from sft.data import (  # noqa: E402
    CHANGE_TALK_T2,
    SUSTAIN_TALK_T2,
    COUNSELLOR_T2,
    CLIENT_T2,
    vocab,
)

# Production flat.j2 templates use slightly longer abbreviations than the
# label vocabulary the dataset uses. Normalise the model's output so both
# sets resolve to the canonical short form.
LABEL_ALIASES = {
    "ADWP": "ADW",
    "RCWP": "RCW",
    "CON": "CO",
    "DIR": "DI",
}


def build_rich_prompt(speaker: str, utt_text: str, class_structure: str) -> str:
    """Render the production flat.j2 system prompt and append the target utterance.

    The production prompt was originally designed for a *structured* (JSON)
    response with both `explanation` and `label`. We append an explicit
    instruction to emit *only* the abbreviated label so a plain HF generate()
    call returns something we can parse.
    """
    speaker = "counsellor" if speaker == "counsellor" else "client"
    system_prompt = render_prompt(speaker=speaker, structure="flat")
    # Strip the "Output Format" block (it asks for `explanation` + `label`)
    # and replace with a short, machine-friendly instruction.
    cut_idx = system_prompt.find("## **Output Format**")
    if cut_idx != -1:
        system_prompt = system_prompt[:cut_idx].rstrip() + "\n\n"
    instructions = (
        "## **Output Format**\n"
        "Reply with exactly one abbreviated label (e.g. `SR`, `OQ`, `R+`, `N`) "
        "and nothing else. Do not include an explanation.\n\n"
        "---\n\n"
    )
    return (
        f"{system_prompt}{instructions}"
        f"## **Target Utterance**\n"
        f"Speaker: {speaker}\n"
        f"Utterance: {utt_text}\n\n"
        f"Label:"
    )


def parse_label(generated: str, allowed: List[str]) -> str:
    if not generated:
        return "UNKNOWN"
    text = generated.strip()
    # Pull tokens out of the first ~30 chars (the answer should be at the front).
    head = re.split(r"[\s,;.\n]+", text)[:6]
    for tok in head:
        tok = tok.strip("`*_:()[]").strip()
        if not tok:
            continue
        if tok in allowed:
            return tok
        if tok in LABEL_ALIASES and LABEL_ALIASES[tok] in allowed:
            return LABEL_ALIASES[tok]
    # Fallback: scan the full string
    for tok in re.split(r"[\s,;.\n]+", text):
        tok = tok.strip("`*_:()[]").strip()
        if tok in allowed:
            return tok
        if tok in LABEL_ALIASES and LABEL_ALIASES[tok] in allowed:
            return LABEL_ALIASES[tok]
    return "UNKNOWN"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, required=True,
                   help="Existing predictions.csv produced by sft/eval.py")
    p.add_argument("--base-model", type=str, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--max-input-len", type=int, default=768)
    p.add_argument("--class-structure", type=str, default="t2")
    args = p.parse_args()

    if not args.predictions.exists():
        raise FileNotFoundError(args.predictions)

    df = pd.read_csv(args.predictions)
    n = len(df)
    print(f"Loaded {n} predictions from {args.predictions}")

    allowed = vocab(args.class_structure)
    print(f"Allowed labels ({len(allowed)}): {allowed}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cpu"  # MPS has a 2**32-byte NDArray cap that LM generate() can blow past.
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"Loading model {args.base_model} on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    rich_preds: List[str] = []
    raw_outputs: List[str] = []
    for _, row in tqdm(list(df.iterrows()), desc="rich zero-shot", unit="utt"):
        prompt = build_rich_prompt(row["speaker"], row["utt_text"], args.class_structure)
        inputs = tok(
            prompt, return_tensors="pt", truncation=True, max_length=args.max_input_len
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        gen = tok.decode(
            out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        raw_outputs.append(gen)
        rich_preds.append(parse_label(gen, allowed))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    df["zero_shot_rich_pred"] = rich_preds
    df["zero_shot_rich_raw"] = raw_outputs
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out} with new column 'zero_shot_rich_pred'.")

    # Quick metrics ----------------------------------------------------------
    from sklearn.metrics import accuracy_score, f1_score

    y_true = df["human_label"].tolist()
    for col in ["zero_shot_pred", "zero_shot_rich_pred", "fine_tuned_pred"]:
        if col not in df.columns:
            continue
        y_pred = df[col].tolist()
        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, labels=allowed, average="macro", zero_division=0)
        f1w = f1_score(y_true, y_pred, labels=allowed, average="weighted", zero_division=0)
        print(f"\n--- {col} ---")
        print(f"  accuracy:    {acc:.4f}")
        print(f"  f1_macro:    {f1m:.4f}")
        print(f"  f1_weighted: {f1w:.4f}")
        # Change Talk / Sustain Talk
        for name, group in [("Change Talk", CHANGE_TALK_T2), ("Sustain Talk", SUSTAIN_TALK_T2)]:
            mask = df["human_label"].isin(group)
            sup = int(mask.sum())
            if sup == 0:
                print(f"  {name:13s}: n=0 (no support)")
                continue
            n_correct = int(((df["human_label"] == df[col]) & mask).sum())
            n_pred_grp = int(df[col].isin(group).sum())
            rec = n_correct / sup
            prec = n_correct / n_pred_grp if n_pred_grp else 0.0
            print(
                f"  {name:13s}: support={sup}  pred_in_group={n_pred_grp}  "
                f"correct={n_correct}  recall={rec:.2%}  prec={prec:.2%}"
            )
        # Top predicted labels
        top = Counter(y_pred).most_common(8)
        print(f"  top predictions: {top}")


if __name__ == "__main__":
    main()
