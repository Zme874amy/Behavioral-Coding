"""Preflight checks before a long SFT run (HF auth, load, token length, generate).

Usage:
  PYTHONPATH=src python src/sft/preflight_hf.py --model Qwen/Qwen2.5-7B-Instruct
  PYTHONPATH=src python src/sft/preflight_hf.py --model google/gemma-3n-E4B-it
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from components.hf_load import load_model_and_tokenizer, resolve_hf_token
from sft.data import build_lodo, build_prompt


def main() -> int:
    parser = argparse.ArgumentParser(description="HF SFT preflight")
    parser.add_argument("--model", required=True, help="Hugging Face model id")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-length", type=int, default=832)
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    print(f"Preflight: {args.model}")
    print("=" * 60)

    token = resolve_hf_token()
    if token:
        print("  HF token: found")
    else:
        print("  HF token: NOT FOUND (gated models will fail)")
        if "gemma" in args.model.lower():
            print(
                "  Action: run `huggingface-cli login` and accept the license at "
                f"https://huggingface.co/{args.model}"
            )
            return 1

    print("\n1. Data files...")
    manual = ROOT / "data" / "manual" / "MIV6.3A_manual.csv"
    if not manual.exists():
        print(f"   ERROR: missing {manual}")
        print("   Manual CSVs are not cloned if *.csv was gitignored.")
        print("   Fix: git pull (after manual CSVs are pushed), or from Mac:")
        print("        bash scripts/sync_data_to_mlerp.sh")
        return 1
    print(f"   OK: {manual}")

    print("\n2. Rich prompt token length (MIV6.3A sample)...")
    train, _, summary = build_lodo(
        train_datasets=["miv63a"],
        test_dataset="hlqc",
        class_structure="t2",
        prompt_style="rich",
    )
    print(f"   LODO summary: {summary}")
    if not train:
        print("ERROR: no training examples")
        return 1

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        args.model, token=token, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    sample = train[0]
    n_tok = len(tok(sample["prompt"])["input_ids"])
    print(f"   first prompt tokens: {n_tok} (max_length={args.max_length})")
    if n_tok > args.max_length:
        print("   WARNING: prompt exceeds max_length; training will truncate")

    import torch

    force_cpu = args.force_cpu
    if not force_cpu and not torch.cuda.is_available():
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            print("   (no CUDA; using CPU for generate to avoid MPS 4GiB NDArray cap)")
            force_cpu = True

    print("\n3. Model load + single generate()...")
    model, tokenizer, device = load_model_and_tokenizer(
        args.model,
        for_training=False,
        force_cpu=force_cpu,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"   device: {device}")

    inputs = tokenizer(
        sample["prompt"],
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=6,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    gen = tokenizer.decode(
        out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    print(f"   generated: {gen!r}")

    print("\nPreflight PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
