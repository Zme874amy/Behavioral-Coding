"""Train and evaluate the single-call arms of the GRPO ladder.

Mirrors `baseline.local_arm` (the two-call Qwen tier) but for the joint
T1+T2 format in `baseline.single_call`:

    arm          how the model is adapted
      sc_zs        none (plain base model)
      sc_fs        none; HLQC exemplars in the prompt, one per T1 group
      sc_ft_bare   LoRA on HLQC, bare-label targets
      sc_ft_rat    LoRA on HLQC, distilled gpt-4o rationale + label targets
      sc_grpo      GRPO from sc_ft_bare (trained by `baseline.grpo`)

The supervised arms reuse `components.fine_tuning.local_trainer` with the
hyperparameters copied verbatim from `conf/baseline_ft_local_config.yaml`. That
is deliberate: it makes "did switching to a single call change the result" a
question the numbers can answer, because the trainer and its settings are the
same and only the prompt/target format moved.

Results land in data/annotated/baseline/qwen_<arm>_inf_<style>_ctx<N>.csv, the
layout `baseline.eval` already scans, so the single-call rows sit in the same
comparison table as the two-call and gpt-4o rows.

Usage:
    PYTHONPATH=src python -m baseline.sc_arm train   --target bare --ctx 5
    PYTHONPATH=src python -m baseline.sc_arm train   --target rat  --ctx 5
    PYTHONPATH=src python -m baseline.sc_arm predict --arm sc_ft_bare --ctx 5
    PYTHONPATH=src python -m baseline.sc_arm predict --arm sc_grpo --seed 0 --ctx 5

Smoke test on CPU with a tiny model:
    PYTHONPATH=src python -m baseline.sc_arm predict --arm sc_zs --ctx 3 \
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
CONFIG_PATH = REPO_ROOT / "conf" / "grpo_config.yaml"

# The three GRPO runs differ only in training configuration, so they must not
# share a result filename or an adapter directory: Phase 2 would silently
# overwrite Phase 1 and the ablation would end up compared against itself.
GRPO_VARIANTS = {
    "weighted": "sc_grpo",        # Phase 1 headline
    "unweighted": "sc_grpo_unw",  # Phase 2: no rare-class emphasis
    "coldstart": "sc_grpo_cold",  # Phase 2: init from the base model
}
GRPO_ARMS = tuple(GRPO_VARIANTS.values())
ARM_TO_VARIANT = {v: k for k, v in GRPO_VARIANTS.items()}

ARMS = ("sc_zs", "sc_fs", "sc_ft_bare", "sc_ft_rat", *GRPO_ARMS)
TRAIN_TARGETS = ("bare", "rat")
# Arms that put HLQC exemplars in the prompt instead of in a LoRA adapter.
FEWSHOT_ARMS = ("sc_fs",)
# Which arms carry a rationale at inference. The two supervised arms are
# evaluated in the style they were trained in; comparing a rationale-trained
# model under a bare prompt is the two-call experiment's question, already
# answered, and re-asking it here would double the run count for no new claim.
ARM_EMITS_RATIONALE = {
    "sc_zs": True,
    "sc_fs": True,
    "sc_ft_bare": False,
    "sc_ft_rat": True,
    **{arm: True for arm in GRPO_ARMS},
}


def load_config(overrides: Optional[List[str]] = None) -> DictConfig:
    cfg = OmegaConf.load(CONFIG_PATH)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def condition_name(tier: str, arm: str, emit_rationale: bool) -> str:
    style = "cot" if emit_rationale else "bare"
    return f"{tier}_{arm}_inf_{style}"


def result_path(cfg: DictConfig, arm: str, ctx: int, seed: Optional[int]) -> Path:
    name = condition_name(cfg.tier, arm, ARM_EMITS_RATIONALE[arm])
    suffix = f"_seed{seed}" if seed is not None else ""
    return REPO_ROOT / cfg.paths.output_dir / f"{name}_ctx{ctx}{suffix}.csv"


def sft_adapter_dir(cfg: DictConfig, target: str, ctx: int) -> Path:
    return (
        REPO_ROOT
        / cfg.paths.sft_adapter_dir
        / f"ctx{ctx}"
        / target
        / "local_finetuned_model"
    )


def grpo_adapter_dir(cfg: DictConfig, ctx: int, seed: int, variant: str = "weighted") -> Path:
    if variant not in GRPO_VARIANTS:
        raise ValueError(f"unknown GRPO variant {variant!r}")
    return REPO_ROOT / cfg.paths.grpo_adapter_dir / f"ctx{ctx}" / variant / f"seed{seed}"


def adapter_for_arm(
    cfg: DictConfig, arm: str, ctx: int, seed: Optional[int]
) -> Optional[Path]:
    if arm in ("sc_zs", *FEWSHOT_ARMS):
        return None
    if arm == "sc_ft_bare":
        return sft_adapter_dir(cfg, "bare", ctx)
    if arm == "sc_ft_rat":
        return sft_adapter_dir(cfg, "rat", ctx)
    if arm in ARM_TO_VARIANT:
        if seed is None:
            raise SystemExit(f"--seed is required for the {arm} arm")
        return grpo_adapter_dir(cfg, ctx, seed, ARM_TO_VARIANT[arm])
    raise ValueError(f"unknown arm {arm!r}")


# -----------------------------------------------------------------------------
# train (supervised arms)
# -----------------------------------------------------------------------------
def cmd_train(args) -> None:
    from automisc_ft.data import load_manual
    from automisc_ft.train import _load_tokenizer, _local_trainer_config, _save_jsonl
    from baseline.rationalize import load_rationales
    from baseline.single_call import build_messages, build_target
    from components.fine_tuning.local_trainer import run_local_fine_tuning

    cfg = load_config(args.overrides)
    ctx, target = args.ctx, args.target
    emit_rationale = target == "rat"

    df = load_manual(REPO_ROOT / cfg.dataset.train_csv)
    positions = list(range(len(df)))
    if args.limit:
        positions = positions[: int(args.limit)]

    rationales = None
    if emit_rationale:
        rationales = load_rationales(ctx)
        if not rationales:
            raise SystemExit(
                f"No frozen rationales for ctx={ctx}. Generate them first:\n"
                f"  PYTHONPATH=src python -m baseline.rationalize --ctx {ctx}"
            )
        print(f"Loaded {len(rationales)} frozen rationale pairs for ctx={ctx}")

    tokenizer = _load_tokenizer(cfg)
    rows: List[Dict[str, str]] = []
    n_skipped = 0
    for pos in positions:
        row = df.iloc[pos]
        speaker = row["speaker"]
        if speaker not in {"counsellor", "client"}:
            continue
        t1, t2 = row.get("t1_label_GT"), row.get("t2_label_GT")
        if not isinstance(t1, str) or not isinstance(t2, str):
            continue

        rationale = None
        if emit_rationale:
            entry = rationales.get(str(row.get("corp_utt_idx"))) or {}
            # The joint call makes one argument, so the T1 rationale (which
            # carries the permission chain of thought) is the one that belongs
            # in front of both labels; fall back to T2's if T1 has none.
            rationale = entry.get("t1_explanation") or entry.get("t2_explanation")
            if not rationale:
                n_skipped += 1
                continue

        messages = build_messages(df, pos, cfg.annotator.context_mode, ctx, emit_rationale)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        rows.append(
            {
                "prompt": prompt,
                "completion": " " + build_target(t1, t2, rationale) + "\n",
                "conv_id": str(row["conv_id"]),
                "speaker": speaker,
                "t1": t1,
                "t2": t2,
            }
        )

    if n_skipped:
        print(f"WARNING: skipped {n_skipped} rows with no frozen rationale on file")
    if not rows:
        raise RuntimeError("No training examples were built.")

    out_dir = REPO_ROOT / cfg.paths.sft_adapter_dir / f"ctx{ctx}" / target
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_jsonl(rows, out_dir / "sft_artifacts" / "train.jsonl")

    print(
        f"Training single-call target={target} ctx={ctx} on {len(rows)} examples "
        f"from {cfg.dataset.train_csv} -> {out_dir}"
    )
    model_dir = run_local_fine_tuning(_local_trainer_config(cfg, out_dir), rows, None)

    meta = {
        "format": "single_call",
        "tier": cfg.tier,
        "target": target,
        "emit_rationale": emit_rationale,
        "num_context_turns": ctx,
        "context_mode": cfg.annotator.context_mode,
        "base_model": cfg.model.base_model,
        "train_csv": cfg.dataset.train_csv,
        "n_examples": len(rows),
        "n_skipped_no_rationale": n_skipped,
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
        "adapter": str(model_dir),
    }
    (out_dir / "train_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"Adapter saved to {model_dir}")


# -----------------------------------------------------------------------------
# predict
# -----------------------------------------------------------------------------
def cmd_predict(args) -> None:
    from automisc_ft.data import load_manual
    from baseline.sc_infer import SingleCallAnnotator

    cfg = load_config(args.overrides)
    ctx, arm = args.ctx, args.arm
    emit_rationale = ARM_EMITS_RATIONALE[arm]

    df = load_manual(REPO_ROOT / cfg.dataset.eval_csv)
    df = df.drop(columns=[c for c in df.columns if c.endswith("_auto")])

    limit = args.limit if args.limit is not None else cfg.limit
    n_total = len(df) if limit in (None, "null") else min(int(limit), len(df))

    save_path = result_path(cfg, arm, ctx, args.seed)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    utt_checkpoint = -1
    existing_df = None
    if save_path.exists():
        existing_df = pd.read_csv(save_path)
        if not existing_df.empty:
            utt_checkpoint = existing_df["corp_utt_idx"].max()
            print(f"Resuming: {len(existing_df)} rows done (to corp_utt_idx {utt_checkpoint})")

    positions = [i for i in range(n_total) if df.iloc[i]["corp_utt_idx"] > utt_checkpoint]
    if not positions:
        print(f"Nothing to do; {save_path} is already complete.")
        return

    adapter = adapter_for_arm(cfg, arm, ctx, args.seed)
    if adapter is not None and not adapter.exists():
        raise SystemExit(f"Missing adapter {adapter}. Train it first.")

    exemplars = None
    if arm in FEWSHOT_ARMS:
        from baseline.fewshot import exemplars_path, load_exemplars

        ex_path = exemplars_path(ctx)
        if not ex_path.exists():
            raise SystemExit(
                f"Missing exemplars {ex_path}. Build them first (needs Azure, not a GPU):\n"
                f"  PYTHONPATH=src python -m baseline.fewshot --ctx {ctx}"
            )
        exemplars = load_exemplars(ex_path)

    cond = condition_name(cfg.tier, arm, emit_rationale)
    print(
        f"Condition={cond} ctx={ctx} seed={args.seed} model={cfg.model.base_model} "
        f"adapter={adapter} n={len(positions)} -> {save_path}"
    )

    annotator = SingleCallAnnotator(
        base_model=cfg.model.base_model,
        adapter_dir=str(adapter) if adapter else None,
        emit_rationale=emit_rationale,
        max_new_tokens=int(cfg.inference.max_new_tokens),
        max_input_len=int(cfg.inference.max_input_len),
        force_cpu=bool(cfg.inference.force_cpu),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        fewshot=exemplars,
    )

    context_mode = cfg.annotator.context_mode
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
            pred = annotator.predict_row(df, pos, context_mode, ctx)
            output_rows.append({**df.iloc[pos].to_dict(), **pred})
            if len(output_rows) >= int(cfg.checkpoint_every):
                save()
    finally:
        save()
        n_truncated = annotator.n_truncated
        annotator.close()

    _report(existing_df, cond, save_path, int(cfg.inference.max_new_tokens))
    if n_truncated:
        print(
            f"  WARNING: {n_truncated} prompts hit max_input_len="
            f"{cfg.inference.max_input_len} and were truncated from the right, "
            "which removes the utterance being coded. Raise max_input_len or "
            "shorten the exemplars before trusting these rows."
        )


def _report(df: Optional[pd.DataFrame], cond: str, save_path: Path, max_new_tokens: int) -> None:
    """Print the diagnostics that decide whether the run is trustworthy."""
    if df is None or df.empty:
        print("No rows written.")
        return
    n = len(df)
    print(f"\n{cond}: {n} utterances -> {save_path}")
    for tier in ("t1", "t2"):
        unknown = int((df[f"{tier}_label_auto"] == "UNKNOWN").sum())
        print(f"  {tier.upper()}: unparseable={unknown} ({unknown / n:.1%})")
    bad_json = int((~df["json_ok"].fillna(False).astype(bool)).sum())
    emitted = int(df["emitted_rationale"].fillna(False).astype(bool).sum())
    clipped = int((df["n_gen_tokens"] >= max_new_tokens).sum())
    print(
        f"  malformed_json={bad_json} ({bad_json / n:.1%})  "
        f"emitted_rationale={emitted} ({emitted / n:.1%})  hit_token_cap={clipped}"
    )
    if "hierarchy_ok" in df.columns:
        hier = int(df["hierarchy_ok"].fillna(False).astype(bool).sum())
        print(f"  t2_consistent_with_t1={hier} ({hier / n:.1%})")
    if clipped:
        print(
            "  WARNING: some generations hit max_new_tokens, so a rationale may "
            "have been cut off before the labels were emitted."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="train one single-call LoRA adapter")
    p_train.add_argument(
        "--target",
        choices=TRAIN_TARGETS,
        required=True,
        help="bare = label-only targets; rat = rationale + label targets",
    )

    p_pred = sub.add_parser("predict", help="annotate the evaluation set")
    p_pred.add_argument("--arm", choices=ARMS, required=True)
    p_pred.add_argument(
        "--seed", type=int, default=None, help="GRPO seed (required for the GRPO arms)"
    )

    for p in (p_train, p_pred):
        p.add_argument("--ctx", type=int, default=None, help="prior context volleys")
        p.add_argument("--limit", type=int, default=None, help="cap rows (smoke tests)")
        p.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")

    args = parser.parse_args()
    if args.ctx is None:
        args.ctx = int(load_config(args.overrides).annotator.num_context_turns)
    else:
        args.overrides = list(args.overrides) + [f"annotator.num_context_turns={args.ctx}"]

    {"train": cmd_train, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()
