"""Qwen tier of the Model Scale x Adaptation x Rationale Alignment grid.

Eight conditions, from crossing four adaptation arms with two inference styles:

    arm      how the model is adapted
      zs       none (plain base model)
      fs       none; stratified HLQC exemplars in context
      ft_bare  LoRA trained on HLQC with bare-label targets
      ft_rat   LoRA trained on HLQC with distilled rationale + label targets

    inf      how it is prompted at evaluation time
      bare     label-only prompt (t1_bare / t2_bare)
      cot      rationale-first prompt (t1 / t2)

The four fine-tuning cells are the point of the design: `ft_bare` under `inf_cot`
tests whether label-only training overrides an inference-time CoT instruction,
and `ft_rat` under `inf_bare` tests whether a rationale-trained model can
suppress its rationale on demand. Because generation is unconstrained, either
instruction can genuinely be ignored, so every row keeps the raw generation and a
rationale-emission flag; compliance is not recoverable from a parsed label.

Results land in data/annotated/baseline/qwen_<arm>_inf_<style>_ctx<N>.csv, the
layout src/baseline/eval.py reads, so the Qwen rows are scored alongside gpt-4o.

Usage:
    PYTHONPATH=src python -m baseline.local_arm train   --target bare --ctx 5
    PYTHONPATH=src python -m baseline.local_arm train   --target rat  --ctx 5
    PYTHONPATH=src python -m baseline.local_arm predict --arm zs      --inf cot --ctx 5
    PYTHONPATH=src python -m baseline.local_arm predict --arm ft_rat  --inf bare --ctx 5

Smoke test on CPU with a tiny model:
    PYTHONPATH=src python -m baseline.local_arm predict --arm zs --inf bare --ctx 3 \
        --limit 4 model.base_model=Qwen/Qwen2.5-0.5B-Instruct inference.force_cpu=true
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "conf" / "baseline_ft_local_config.yaml"

ARMS = ("zs", "fs", "ft_bare", "ft_rat")
STYLES = ("bare", "cot")
# Which frozen adapter pair each arm loads; None means no adapters.
ARM_TARGET = {"zs": None, "fs": None, "ft_bare": "bare", "ft_rat": "rat"}
TRAIN_TARGETS = ("bare", "rat")


def load_config(overrides: Optional[List[str]] = None) -> DictConfig:
    cfg = OmegaConf.load(CONFIG_PATH)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def condition_name(tier: str, arm: str, style: str) -> str:
    return f"{tier}_{arm}_inf_{style}"


def result_path(cfg: DictConfig, arm: str, style: str, ctx: int) -> Path:
    name = condition_name(cfg.tier, arm, style)
    return REPO_ROOT / cfg.paths.output_dir / f"{name}_ctx{ctx}.csv"


def adapter_root(cfg: DictConfig, target: str, ctx: int) -> Path:
    return REPO_ROOT / cfg.paths.adapter_dir / f"ctx{ctx}" / target


def adapter_dirs(cfg: DictConfig, target: str, ctx: int) -> tuple[Path, Path]:
    """Resolve the saved T1/T2 adapter directories for a trained target.

    `run_local_fine_tuning` saves into a `local_finetuned_model` subdirectory,
    so that suffix is part of the path.
    """
    root = adapter_root(cfg, target, ctx)
    return root / "t1" / "local_finetuned_model", root / "t2" / "local_finetuned_model"


def structure_suffix_for(style: str) -> str:
    """`inf_bare` uses the label-only templates, `inf_cot` the originals."""
    return "_bare" if style == "bare" else ""


# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
def cmd_train(args) -> None:
    from automisc_ft.data import load_manual
    from automisc_ft.train import train_adapter_pair
    from baseline.rationalize import load_rationales

    cfg = load_config(args.overrides)
    ctx = args.ctx
    target = args.target

    df = load_manual(REPO_ROOT / cfg.dataset.train_csv)
    positions = list(range(len(df)))
    if args.limit:
        positions = positions[: int(args.limit)]

    # ft_bare trains on the label-only prompts it will be evaluated under;
    # ft_rat trains on the rationale-first prompts, matching its target shape.
    structure_suffix = "_bare" if target == "bare" else ""
    rationales = None
    if target == "rat":
        rationales = load_rationales(ctx)
        if not rationales:
            raise SystemExit(
                f"No frozen rationales for ctx={ctx}. Generate them first:\n"
                f"  PYTHONPATH=src python -m baseline.rationalize --ctx {ctx}"
            )
        print(f"Loaded {len(rationales)} frozen rationale pairs for ctx={ctx}")

    out_dir = adapter_root(cfg, target, ctx)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Training target={target} ctx={ctx} on {len(positions)} rows of "
        f"{cfg.dataset.train_csv} -> {out_dir}"
    )

    t1_dir, t2_dir = train_adapter_pair(
        cfg, df, positions, out_dir, structure_suffix, rationales
    )

    meta = {
        "tier": cfg.tier,
        "target": target,
        "num_context_turns": ctx,
        "context_mode": cfg.annotator.context_mode,
        "structure_suffix": structure_suffix,
        "base_model": cfg.model.base_model,
        "train_csv": cfg.dataset.train_csv,
        "n_rows": len(positions),
        "n_epochs": cfg.training.num_train_epochs,
        "learning_rate": cfg.training.learning_rate,
        "max_length": cfg.training.max_length,
        "max_target_length": cfg.training.max_target_length,
        "lora": {
            "r": cfg.model.peft_r,
            "alpha": cfg.model.peft_alpha,
            "dropout": cfg.model.peft_dropout,
            "target_modules": list(cfg.model.target_modules),
        },
        "t1_adapter": str(t1_dir),
        "t2_adapter": str(t2_dir),
    }
    meta_path = out_dir / "train_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"Adapters saved. Metadata -> {meta_path}")


# -----------------------------------------------------------------------------
# predict
# -----------------------------------------------------------------------------
def _make_fewshot_provider(exemplars: dict, rationales: bool):
    """Wrap `build_fewshot_messages` in the (speaker, tier, t1_label) signature
    `TieredAnnotator` expects, tolerating T1 groups that have no exemplars."""
    from baseline.fewshot import build_fewshot_messages

    def provider(speaker: str, tier: str, t1_label: Optional[str]) -> List[Dict[str, str]]:
        if tier == "t2" and (t1_label is None or t1_label not in exemplars[speaker]["t2"]):
            return []
        return build_fewshot_messages(
            exemplars, speaker, tier, rationales, t1_label=t1_label
        )

    return provider


def cmd_predict(args) -> None:
    from automisc_ft.data import load_manual
    from automisc_ft.infer import TieredAnnotator
    from baseline.fewshot import exemplars_path, load_exemplars

    cfg = load_config(args.overrides)
    ctx = args.ctx
    arm, style = args.arm, args.inf
    suffix = structure_suffix_for(style)

    df = load_manual(REPO_ROOT / cfg.dataset.eval_csv)
    # The evaluation CSV ships predictions from an earlier run; we produce our own.
    df = df.drop(columns=[c for c in df.columns if c.endswith("_auto")])

    limit = args.limit if args.limit is not None else cfg.limit
    n_total = len(df) if limit in (None, "null") else min(int(limit), len(df))

    save_path = result_path(cfg, arm, style, ctx)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    utt_checkpoint = -1
    existing_df = None
    if save_path.exists():
        existing_df = pd.read_csv(save_path)
        if not existing_df.empty:
            utt_checkpoint = existing_df["corp_utt_idx"].max()
            print(
                f"Resuming: {len(existing_df)} rows done "
                f"(up to corp_utt_idx {utt_checkpoint})"
            )

    positions = [
        i for i in range(n_total) if df.iloc[i]["corp_utt_idx"] > utt_checkpoint
    ]
    if not positions:
        print(f"Nothing to do; {save_path} is already complete.")
        return

    # Adapters, only for the two fine-tuning arms.
    target = ARM_TARGET[arm]
    t1_adapter = t2_adapter = None
    if target is not None:
        t1_adapter, t2_adapter = adapter_dirs(cfg, target, ctx)
        for p in (t1_adapter, t2_adapter):
            if not p.exists():
                raise SystemExit(
                    f"Missing adapter {p}. Train it first:\n"
                    f"  PYTHONPATH=src python -m baseline.local_arm train "
                    f"--target {target} --ctx {ctx}"
                )

    # Few-shot exemplars, only for the fs arm. `inf_cot` uses the exemplars'
    # frozen rationales; `inf_bare` shows label-only replies.
    fewshot_provider = None
    if arm == "fs":
        ex_path = exemplars_path(ctx)
        if not ex_path.exists():
            raise SystemExit(
                f"No few-shot exemplars for ctx={ctx} at {ex_path}. Build them:\n"
                f"  PYTHONPATH=src python -m baseline.fewshot --ctx {ctx}"
            )
        fewshot_provider = _make_fewshot_provider(
            load_exemplars(ex_path), rationales=(style == "cot")
        )
        print(f"Loaded few-shot exemplars from {ex_path}")

    max_input_len = int(cfg.inference.max_input_len[arm])
    cond = condition_name(cfg.tier, arm, style)
    print(
        f"Condition={cond} ctx={ctx} model={cfg.model.base_model} "
        f"max_input_len={max_input_len} max_new_tokens={cfg.inference.max_new_tokens} "
        f"n={len(positions)} -> {save_path}"
    )

    annotator = TieredAnnotator(
        base_model=cfg.model.base_model,
        t1_adapter_dir=str(t1_adapter) if t1_adapter else None,
        t2_adapter_dir=str(t2_adapter) if t2_adapter else None,
        force_cpu=bool(cfg.inference.force_cpu),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        max_new_tokens=int(cfg.inference.max_new_tokens),
        max_input_len=max_input_len,
        structure_suffix=suffix,
        fewshot_provider=fewshot_provider,
    )

    context_mode = cfg.annotator.context_mode
    restrict_t2 = bool(cfg.annotator.get("restrict_t2_to_group", False))
    output_rows: List[Dict] = []

    def save() -> None:
        nonlocal existing_df, output_rows
        if not output_rows:
            return
        out = pd.DataFrame(output_rows)
        if existing_df is not None:
            out = pd.concat([existing_df, out], ignore_index=True)
        out.to_csv(save_path, index=False)
        existing_df = out
        output_rows = []

    try:
        for pos in tqdm(positions, desc=cond, unit="utt"):
            pred = annotator.predict_row(df, pos, context_mode, ctx, restrict_t2)
            row = df.iloc[pos].to_dict()
            output_rows.append({
                **row,
                "t1_label_auto": pred["t1_pred"],
                "t2_label_auto": pred["t2_pred"],
                "t1_raw": pred["t1_raw"],
                "t2_raw": pred["t2_raw"],
                "t1_emitted_rationale": pred["t1_emitted_rationale"],
                "t2_emitted_rationale": pred["t2_emitted_rationale"],
                "t1_n_prompt_tokens": pred["t1_n_prompt_tokens"],
                "t2_n_prompt_tokens": pred["t2_n_prompt_tokens"],
                "t1_n_gen_tokens": pred["t1_n_gen_tokens"],
                "t2_n_gen_tokens": pred["t2_n_gen_tokens"],
            })
            if len(output_rows) >= int(cfg.checkpoint_every):
                save()
    finally:
        save()
        annotator.close()

    _report(existing_df, cond, save_path, int(cfg.inference.max_new_tokens), max_input_len)


def _report(df: Optional[pd.DataFrame], cond: str, save_path: Path,
            max_new_tokens: int, max_input_len: int) -> None:
    """Print the diagnostics that decide whether the run is trustworthy."""
    if df is None or df.empty:
        print("No rows written.")
        return
    n = len(df)
    print(f"\n{cond}: {n} utterances -> {save_path}")
    for tier in ("t1", "t2"):
        unknown = int((df[f"{tier}_label_auto"] == "UNKNOWN").sum())
        rationale = int(df[f"{tier}_emitted_rationale"].fillna(False).astype(bool).sum())
        clipped = int((df[f"{tier}_n_gen_tokens"] >= max_new_tokens).sum())
        truncated = int((df[f"{tier}_n_prompt_tokens"] >= max_input_len).sum())
        print(
            f"  {tier.upper()}: unparseable={unknown} ({unknown / n:.1%})  "
            f"emitted_rationale={rationale} ({rationale / n:.1%})  "
            f"hit_token_cap={clipped}  prompt_truncated={truncated}"
        )
    if any(
        (df[f"{t}_n_prompt_tokens"] >= max_input_len).any() for t in ("t1", "t2")
    ):
        print(
            "  WARNING: some prompts hit max_input_len, so the target utterance "
            "may have been cut off. Raise inference.max_input_len for this arm."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="train one LoRA adapter pair")
    p_train.add_argument("--target", choices=TRAIN_TARGETS, required=True,
                         help="bare = label-only targets; rat = rationale + label targets")

    p_pred = sub.add_parser("predict", help="annotate the evaluation set")
    p_pred.add_argument("--arm", choices=ARMS, required=True)
    p_pred.add_argument("--inf", choices=STYLES, required=True,
                        help="inference prompt style")

    for p in (p_train, p_pred):
        p.add_argument("--ctx", type=int, default=None,
                       help="prior context volleys (default: config)")
        p.add_argument("--limit", type=int, default=None,
                       help="cap rows processed (smoke tests)")
        p.add_argument("overrides", nargs="*",
                       help="OmegaConf dotlist overrides, e.g. inference.force_cpu=true")

    args = parser.parse_args()

    if args.ctx is None:
        args.ctx = int(load_config(args.overrides).annotator.num_context_turns)
    else:
        args.overrides = list(args.overrides) + [f"annotator.num_context_turns={args.ctx}"]

    {"train": cmd_train, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()
