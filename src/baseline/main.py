"""Baseline runner for the AutoMISC reproduction study.

Runs the hierarchical T1 -> T2 annotation loop over the already-parsed,
human-labelled MIV6.3A_manual.csv for one prompting condition:

    zeroshot_rationales   original AutoMISC setting (explanation + label)
    zeroshot_bare         label-only output
    fewshot_rationales    HLQC exemplars with frozen rationales
    fewshot_bare          HLQC exemplars, label-only

Usage:
    PYTHONPATH=src .venv/bin/python -m baseline.main condition=zeroshot_rationales
    PYTHONPATH=src .venv/bin/python -m baseline.main condition=fewshot_bare limit=10
    PYTHONPATH=src .venv/bin/python -m baseline.main condition=zeroshot_rationales num_context_turns=3

Writes data/annotated/baseline/<condition>_ctx<N>.csv (N = num_context_turns)
with checkpoint/resume on corp_utt_idx, mirroring the production annotator.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from components.context import build_context_excerpt
from components.prompts.loader import render_prompt, render_user_prompt
from components.prompts import response_formats as rf
from components.utils import call_chat_model
from baseline.fewshot import exemplars_path, load_exemplars, build_fewshot_messages

REPO_ROOT = Path(__file__).resolve().parents[2]

RESPONSE_FORMATS = {
    # (speaker, tier, rationales) -> pydantic model
    ("counsellor", "t1", True): rf.CounsellorUtterance_t1,
    ("counsellor", "t2", True): rf.CounsellorUtterance_t2,
    ("counsellor", "t1", False): rf.CounsellorUtterance_t1_bare,
    ("counsellor", "t2", False): rf.CounsellorUtterance_t2_bare,
    ("client", "t1", True): rf.ClientUtterance_t1,
    ("client", "t2", True): rf.ClientUtterance_t2,
    ("client", "t1", False): rf.ClientUtterance_t1_bare,
    ("client", "t2", False): rf.ClientUtterance_t2_bare,
}


def call_with_retry(cfg, messages, response_format):
    delay = 2.0
    for attempt in range(int(cfg.max_retries)):
        try:
            return call_chat_model(
                messages=messages,
                model=cfg.model,
                provider=cfg.provider,
                temperature=cfg.temperature,
                response_format=response_format,
            )
        except Exception as e:
            if attempt == int(cfg.max_retries) - 1:
                raise
            print(f"  retry {attempt + 1} after error: {e}", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 60)


def classify_utterance(cfg, rationales, fewshot, exemplars, speaker, user_prompt):
    """Run T1 then T2 for one utterance; returns dict of labels/explanations."""
    structure_suffix = "" if rationales else "_bare"

    t1_system = render_prompt(speaker=speaker, structure=f"t1{structure_suffix}")
    t1_messages = [{"role": "system", "content": t1_system}]
    if fewshot:
        t1_messages += build_fewshot_messages(exemplars, speaker, "t1", rationales)
    t1_messages.append({"role": "user", "content": user_prompt})
    t1 = call_with_retry(cfg, t1_messages, RESPONSE_FORMATS[(speaker, "t1", rationales)])

    t2_system = render_prompt(speaker=speaker, structure=f"t2{structure_suffix}", label=t1["label"])
    t2_messages = [{"role": "system", "content": t2_system}]
    if fewshot:
        t2_messages += build_fewshot_messages(
            exemplars, speaker, "t2", rationales, t1_label=t1["label"]
        )
    t2_messages.append({"role": "user", "content": user_prompt})
    t2 = call_with_retry(cfg, t2_messages, RESPONSE_FORMATS[(speaker, "t2", rationales)])

    return {
        "t1_label_auto": t1["label"],
        "t1_expl_auto": t1.get("explanation", ""),
        "t2_label_auto": t2["label"],
        "t2_expl_auto": t2.get("explanation", ""),
    }


def main():
    cfg = OmegaConf.load(REPO_ROOT / "conf" / "baseline_config.yaml")
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(sys.argv[1:]))

    condition = cfg.condition
    cond_cfg = cfg.conditions[condition]
    rationales = bool(cond_cfg.rationales)
    fewshot = bool(cond_cfg.fewshot)

    ctx = int(cfg.num_context_turns)
    exemplars = None
    if fewshot:
        ex_path = exemplars_path(ctx)
        if not ex_path.exists():
            raise FileNotFoundError(
                f"No few-shot exemplars for ctx={ctx} at {ex_path}. "
                f"Build them first: python -m baseline.fewshot --ctx {ctx}"
            )
        exemplars = load_exemplars(ex_path)
        print(f"Loaded few-shot exemplars from {ex_path}")

    df = pd.read_csv(REPO_ROOT / cfg.eval_csv).reset_index(drop=True)
    # Drop any pre-existing auto columns from the source file; we produce our own.
    df = df.drop(columns=[c for c in df.columns if c.endswith("_auto")])

    limit = cfg.limit
    n_total = len(df) if limit in (None, "null") else min(int(limit), len(df))

    out_dir = REPO_ROOT / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"{condition}_ctx{ctx}.csv"
    # Results produced before the ctx suffix existed were all run at ctx=5.
    legacy_path = out_dir / f"{condition}.csv"
    if ctx == 5 and not save_path.exists() and legacy_path.exists():
        legacy_path.rename(save_path)
        print(f"Renamed legacy {legacy_path.name} -> {save_path.name}")

    utt_checkpoint = -1
    existing_df = None
    if save_path.exists():
        existing_df = pd.read_csv(save_path)
        if not existing_df.empty:
            utt_checkpoint = existing_df["corp_utt_idx"].max()
            print(f"Resuming {condition}: {len(existing_df)} rows done (up to corp_utt_idx {utt_checkpoint})")

    output_rows = []

    def save():
        nonlocal existing_df, output_rows
        if not output_rows:
            return
        out = pd.DataFrame(output_rows)
        if existing_df is not None:
            out = pd.concat([existing_df, out], ignore_index=True)
        out.to_csv(save_path, index=False)
        existing_df = out
        output_rows = []

    print(f"Condition={condition} (rationales={rationales}, fewshot={fewshot}, ctx={ctx}) "
          f"model={cfg.model} n={n_total} -> {save_path}")

    try:
        for i, row in tqdm(list(df.iloc[:n_total].iterrows()), desc=condition, unit="utt"):
            if row["corp_utt_idx"] <= utt_checkpoint:
                continue
            speaker = row["speaker"]
            context = build_context_excerpt(df, i, cfg.context_mode, cfg.num_context_turns)
            user_prompt = render_user_prompt(
                transcript=context, speaker=speaker, utterance=row["utt_text"]
            )
            result = classify_utterance(cfg, rationales, fewshot, exemplars, speaker, user_prompt)
            output_rows.append({**row.to_dict(), **result})
            if len(output_rows) >= int(cfg.checkpoint_every):
                save()
    finally:
        save()

    done = len(existing_df) if existing_df is not None else 0
    print(f"{condition}: {done}/{n_total} utterances annotated -> {save_path}")


if __name__ == "__main__":
    main()
