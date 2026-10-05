"""Qwen tier of the Model Scale x Adaptation x Rationale Alignment grid.

Every arm here runs the same two-call flow: a Tier-1 call, then a Tier-2 call
conditioned on the predicted T1. Arms differ in how the model is adapted, and
in particular in HOW MANY adapters carry that adaptation:

    arm           adapters  how the model is adapted
      zs             0      none (plain base model)
      fs             0      none; stratified HLQC exemplars in context
      ft_bare        2      a T1 adapter and a T2 adapter, bare-label targets
      ft_rat         2      same, distilled rationale + label targets
      ft1mix_bare    1      one adapter on the shuffled union of both tiers
      ft1mix_rat     1      same, rationale + label targets
      ft1seq_bare    1      one adapter trained on T1, then continued on T2
      ft1seq_rat     1      same, rationale + label targets

    inf      how it is prompted at evaluation time
      bare     label-only prompt (t1_bare / t2_bare)
      cot      rationale-first prompt (t1 / t2)

The single-adapter arms exist to make the format comparison identifiable. Against
`ft_*` they hold the call count fixed and vary adapter count; against the `sc_*`
single-call ladder in `baseline.sc_arm` they hold adapter count fixed and vary
call count. Without them, single-call and two-call differ on two axes at once.

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
    PYTHONPATH=src python -m baseline.local_arm train   --target bare --regime mixed --ctx 5
    PYTHONPATH=src python -m baseline.local_arm train   --target bare --regime sequential --ctx 5
    PYTHONPATH=src python -m baseline.local_arm predict --arm zs          --inf cot  --ctx 5
    PYTHONPATH=src python -m baseline.local_arm predict --arm ft_rat      --inf bare --ctx 5
    PYTHONPATH=src python -m baseline.local_arm predict --arm ft1mix_bare --inf bare --ctx 5

Smoke test on CPU with a tiny model:
    PYTHONPATH=src python -m baseline.local_arm predict --arm zs --inf bare --ctx 3 \
        --limit 4 model.base_model=Qwen/Qwen2.5-0.5B-Instruct inference.force_cpu=true
"""
from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "conf" / "baseline_ft_local_config.yaml"

STYLES = ("bare", "cot")
TRAIN_TARGETS = ("bare", "rat")

# Adapter layout. `pair` is the original two-adapter arm (a T1 adapter and a T2
# adapter, switched per call); `mixed` and `sequential` are the two single-adapter
# regimes, where ONE LoRA serves both calls. All three run the same two-call
# inference flow, so `pair` vs the others isolates adapter count, and the
# single-adapter arms against the `sc_*` ladder isolates call count.
REGIMES = ("pair", "mixed", "sequential")

# Which frozen adapters each arm loads: None for the in-context arms, otherwise
# (regime, target).
ARM_ADAPTER = {
    "zs": None,
    "fs": None,
    "ft_bare": ("pair", "bare"),
    "ft_rat": ("pair", "rat"),
    "ft1mix_bare": ("mixed", "bare"),
    "ft1mix_rat": ("mixed", "rat"),
    "ft1seq_bare": ("sequential", "bare"),
    "ft1seq_rat": ("sequential", "rat"),
    # Teacher-forcing variants of ft_bare: same two-adapter, two-call structure,
    # differing only in which T1 label conditioned T2 during training.
    "ft_bare_tffull": ("pair", "bare"),
    "ft_bare_tfdyn": ("pair", "bare"),
    "ft_bare_tfzero": ("pair", "bare"),
    # The same axis on cell B (1 adapter, 2 calls). Inference is the identical
    # two-call flow, so the exposure is identical; what differs is that ONE
    # adapter carries both tiers, so re-conditioning T2 also retrains T1.
    "ft1mix_bare_tffull": ("mixed", "bare"),
    "ft1mix_bare_tfdyn": ("mixed", "bare"),
    "ft1mix_bare_tfzero": ("mixed", "bare"),
}
ARMS = tuple(ARM_ADAPTER)

# How much of the T1 conditioning was gold while T2 trained. `full` is the
# control: it runs the identical staged procedure at p_gold=1, so the three
# differ on the axis and nothing else. The published `ft_bare` numbers are
# untouched; `ft_bare` vs `ft_bare_tffull` prices the staging itself.
TF_MODES = ("full", "dyn", "zero")
# The axis is hosted on ft_bare (cell A, 2 adapters) and ft1mix_bare (cell B,
# 1 adapter). Not on ft1seq_bare: its T2 stage overwrites the T1 stage badly
# enough that 39.8% of its T1 predictions are wrong and half of those are
# unparseable, which would confound the axis with catastrophic forgetting.
TF_HOSTS = ("ft_bare", "ft1mix_bare")
ARM_TF = {
    f"{host}_tf{mode}": mode for host in TF_HOSTS for mode in TF_MODES
}


def load_config(overrides: Optional[List[str]] = None) -> DictConfig:
    cfg = OmegaConf.load(CONFIG_PATH)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def condition_name(tier: str, arm: str, style: str) -> str:
    return f"{tier}_{arm}_inf_{style}"


DEFAULT_EVAL_NAME = "miv63a"


def eval_name(cfg: DictConfig) -> str:
    """Short slug for the evaluation corpus, from `cfg.dataset.eval_name`.

    Defaults to `miv63a`, the primary human-consensus set, for which the result
    filename carries NO dataset token — so every result produced before
    cross-dataset evaluation existed keeps its name and `baseline.eval` scores it
    unchanged. Any other corpus (annomi, welivita, ...) gets a `_ds{slug}` token
    so its results live beside the MIV6.3A ones without colliding.
    """
    return str(cfg.dataset.get("eval_name", DEFAULT_EVAL_NAME))


def result_path(
    cfg: DictConfig, arm: str, style: str, ctx: int, seed: Optional[int] = None
) -> Path:
    """Result CSV for one cell. `_seed{N}` mirrors the single-call convention in
    `sc_arm.result_path`, and `baseline.eval` already parses and aggregates it.
    `_ds{slug}` namespaces a non-default evaluation corpus (see `eval_name`)."""
    name = condition_name(cfg.tier, arm, style)
    ds = eval_name(cfg)
    ds_suffix = f"_ds{ds}" if ds != DEFAULT_EVAL_NAME else ""
    suffix = f"_seed{seed}" if seed is not None else ""
    return REPO_ROOT / cfg.paths.output_dir / f"{name}_ctx{ctx}{ds_suffix}{suffix}.csv"


def adapter_root(cfg: DictConfig, target: str, ctx: int, regime: str = "pair") -> Path:
    """Where one trained arm's adapters live.

    The two layouts are kept apart so a single-adapter run can never overwrite
    the two-adapter arm it is being compared against.
    """
    if regime == "pair":
        return REPO_ROOT / cfg.paths.adapter_dir / f"ctx{ctx}" / target
    return REPO_ROOT / cfg.paths.adapter1_dir / f"ctx{ctx}" / regime / target


def adapter_dirs(cfg: DictConfig, target: str, ctx: int) -> tuple[Path, Path]:
    """Resolve the saved T1/T2 adapter directories for a trained target.

    `run_local_fine_tuning` saves into a `local_finetuned_model` subdirectory,
    so that suffix is part of the path.
    """
    root = adapter_root(cfg, target, ctx)
    return root / "t1" / "local_finetuned_model", root / "t2" / "local_finetuned_model"


def shared_adapter_dir(cfg: DictConfig, target: str, ctx: int, regime: str) -> Path:
    """The single adapter covering both tiers, for the 1-adapter arms."""
    return adapter_root(cfg, target, ctx, regime) / "local_finetuned_model"


def tf_adapter_root(
    cfg: DictConfig, target: str, ctx: int, tf_mode: str, regime: str = "pair",
    seed: Optional[int] = None,
) -> Path:
    """Where one teacher-forcing variant's adapter(s) live.

    A sibling of the base tree in both layouts, so a variant can never overwrite
    the adapter of the arm it is being compared against.
    """
    # A seeded replicate gets its own tree so it cannot overwrite the original.
    leaf = f"{target}_tf{tf_mode}" + (f"_seed{seed}" if seed is not None else "")
    if regime == "pair":
        return REPO_ROOT / cfg.paths.adapter_dir / f"ctx{ctx}" / leaf
    return REPO_ROOT / cfg.paths.adapter1_dir / f"ctx{ctx}" / regime / leaf


def resolve_adapters(
    cfg: DictConfig, arm: str, ctx: int, seed: Optional[int] = None
) -> tuple[Optional[Path], Optional[Path], Optional[Path], Optional[str]]:
    """Return ``(t1_dir, t2_dir, shared_dir, retrain_command)`` for one arm.

    The teacher-forcing arms resolve their T1 from the BASE tree and only their
    T2 from the per-mode tree. That makes "all three share one T1 adapter" a
    property of the paths rather than a claim in a document.
    """
    spec = ARM_ADAPTER[arm]
    if spec is None:
        return None, None, None, None
    regime, target = spec
    base = "PYTHONPATH=src python -m baseline.local_arm train"

    if arm in ARM_TF:
        mode = ARM_TF[arm]
        cmd = (f"{base} --target {target} --regime {regime} --tf {mode} "
               f"--ctx {ctx}" + (f" --seed {seed}" if seed is not None else ""))
        if regime == "pair":
            # Cell A: T1 resolves from the BASE tree and only T2 from the
            # per-mode tree, so "all variants share one T1 adapter" is a
            # property of the paths rather than a claim in a document.
            return (
                adapter_root(cfg, target, ctx) / "t1" / "local_finetuned_model",
                tf_adapter_root(cfg, target, ctx, mode, seed=seed)
                / "t2" / "local_finetuned_model",
                None,
                cmd,
            )
        # Cell B: one adapter serves both calls, so there is no T1 to hold
        # fixed -- the whole adapter is retrained under the schedule.
        return (
            None, None,
            tf_adapter_root(cfg, target, ctx, mode, regime, seed)
            / "local_finetuned_model",
            cmd,
        )

    cmd = f"{base} --target {target} --regime {regime} --ctx {ctx}"
    if regime == "pair":
        t1, t2 = adapter_dirs(cfg, target, ctx)
        return t1, t2, None, cmd
    return None, None, shared_adapter_dir(cfg, target, ctx, regime), cmd


# -----------------------------------------------------------------------------
# Provenance. Nothing in this repo recorded WHEN a result was produced, so every
# finished train or predict now stamps itself. Dates are only ever recorded, not
# inferred: a cell with no stamp reads as unknown rather than guessing a mtime.
# -----------------------------------------------------------------------------
def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_sha() -> Optional[str]:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _provenance(started_utc: str) -> Dict[str, Optional[str]]:
    import os

    return {
        "started_utc": started_utc,
        "finished_utc": _utc_now(),
        "git_sha": _git_sha(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def structure_suffix_for(style: str) -> str:
    """`inf_bare` uses the label-only templates, `inf_cot` the originals."""
    return "_bare" if style == "bare" else ""


# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
def _train_tf_variant(cfg, df, positions, target, ctx, tf_mode, regime,
                      structure_suffix, rationales, seed=None):
    """Train one teacher-forcing variant. Returns ``(adapter_meta, schedule)``.

    The two structures need different things:

    * **pair (cell A)** -- two adapters, so only T2 is retrained and the base
      arm's T1 adapter is loaded unchanged. The axis moves exactly one thing,
      and a T1 regression is impossible by construction.
    * **mixed (cell B)** -- one adapter serves both calls, so there is nothing
      to hold fixed: the whole adapter is retrained with the T2 half
      re-conditioned. Whether that damages T1 is the question this arm adds.
    """
    from automisc_ft.train import (
        gold_prob_schedule, train_single_staged, train_t2_staged,
    )
    from baseline.oof_t1 import load_oof_t1, oof_t1_path

    t1_dir = None
    if regime == "pair":
        t1_dir = adapter_root(cfg, target, ctx) / "t1" / "local_finetuned_model"
        if not t1_dir.exists():
            raise SystemExit(
                f"The cell-A teacher-forcing variants share the base T1 adapter, "
                f"which is missing at {t1_dir}. Train the base arm first:\n"
                f"  PYTHONPATH=src python -m baseline.local_arm train "
                f"--target {target} --regime pair --ctx {ctx}"
            )

    schedule = gold_prob_schedule(
        cfg.training.tf_modes[tf_mode], int(cfg.training.num_train_epochs)
    )

    predicted_t1 = None
    if any(p < 1.0 for p in schedule):
        predicted_t1 = load_oof_t1(ctx)
        if not predicted_t1:
            raise SystemExit(
                f"No out-of-fold predicted T1 for ctx={ctx} at {oof_t1_path(ctx)}. "
                f"Generate it first (one job per fold):\n"
                f"  PYTHONPATH=src python -m baseline.oof_t1 train-fold   --ctx {ctx} --fold K\n"
                f"  PYTHONPATH=src python -m baseline.oof_t1 predict-fold --ctx {ctx} --fold K"
            )
        # Same reasoning as the rationale gate: an incomplete store would let the
        # uncovered rows fall back to gold, which is an adapter quietly trained
        # at a higher p_gold than the schedule claims.
        covered = sum(
            1 for p in positions
            if str(df.iloc[p].get("corp_utt_idx")) in predicted_t1
        )
        if covered < len(positions):
            raise SystemExit(
                f"Out-of-fold predictions for ctx={ctx} cover only {covered}/"
                f"{len(positions)} training rows, so the rest would silently fall "
                f"back to gold conditioning. Finish every fold, then check:\n"
                f"  PYTHONPATH=src python -m baseline.oof_t1 report --ctx {ctx}"
            )
        print(f"Loaded {len(predicted_t1)} out-of-fold T1 predictions for ctx={ctx}")

    root = tf_adapter_root(cfg, target, ctx, tf_mode, regime, seed)
    if regime == "pair":
        out_dir = root / "t2"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"tf={tf_mode} (cell A): p_gold {schedule}, T1 shared from {t1_dir}")
        t2_dir = train_t2_staged(
            cfg, df, positions, out_dir, schedule,
            predicted_t1=predicted_t1,
            structure_suffix=structure_suffix,
            rationales=rationales,
        )
        meta = {
            "t1_adapter": str(t1_dir),
            "t2_adapter": str(t2_dir),
            "tf_t1_shared_from": str(adapter_root(cfg, target, ctx)),
        }
    else:
        root.mkdir(parents=True, exist_ok=True)
        print(
            f"tf={tf_mode} (cell B): p_gold {schedule}, ONE adapter for both "
            f"tiers -- T1 is retrained too, not held fixed"
        )
        shared = train_single_staged(
            cfg, df, positions, root, schedule,
            predicted_t1=predicted_t1,
            structure_suffix=structure_suffix,
            rationales=rationales,
        )
        meta = {"shared_adapter": str(shared), "tf_t1_shared_from": None}

    meta.update({
        "tf_mode": tf_mode,
        "tf_p_gold_schedule": schedule,
        "tf_staged": True,
        "tf_regime": regime,
    })
    return meta, schedule


def cmd_train(args) -> None:
    from automisc_ft.data import load_manual
    from automisc_ft.train import train_adapter_pair, train_single_adapter
    from baseline.rationalize import load_rationales

    started = _utc_now()
    cfg = load_config(args.overrides)
    if getattr(args, "manifest", None):
        return _train_manifest(args, cfg, started)
    ctx = args.ctx
    target = args.target
    regime = args.regime
    tf_mode = getattr(args, "tf", None)
    seed = getattr(args, "seed", None)
    if seed is not None and not tf_mode:
        raise SystemExit(
            "--seed is only wired for the teacher-forcing arms (--tf). Other arms "
            "write to unseeded adapter paths, so a replicate would overwrite the "
            "original rather than sit beside it."
        )
    if tf_mode and (regime not in ("pair", "mixed") or target != "bare"):
        raise SystemExit(
            "--tf is defined for `--regime pair` (cell A) and `--regime mixed` "
            "(cell B), with `--target bare`. Both run the two-call flow, so both "
            "condition T2 on a PREDICTED T1. `sequential` is excluded: its T2 "
            "stage overwrites the T1 stage badly enough that the axis would be "
            "confounded with catastrophic forgetting."
        )

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
        # An interrupted generation run leaves a valid but partial file, and
        # rows without a rationale are silently dropped when the examples are
        # built. Without this check that becomes an adapter quietly trained on a
        # fraction of the corpus, which is far worse than a failed job.
        covered = sum(
            1 for p in positions
            if str(df.iloc[p].get("corp_utt_idx")) in rationales
        )
        if covered < len(positions):
            raise SystemExit(
                f"Frozen rationales for ctx={ctx} cover only {covered}/"
                f"{len(positions)} training rows, so the rest would be dropped. "
                f"Finish the generation (it resumes from the checkpoint):\n"
                f"  PYTHONPATH=src python -m baseline.rationalize --ctx {ctx}"
            )
        print(f"Loaded {len(rationales)} frozen rationale pairs for ctx={ctx}")

    out_dir = (
        tf_adapter_root(cfg, target, ctx, tf_mode, regime, seed) if tf_mode
        else adapter_root(cfg, target, ctx, regime)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Training target={target} regime={regime}"
        f"{f' tf={tf_mode}' if tf_mode else ''} ctx={ctx} on "
        f"{len(positions)} rows of {cfg.dataset.train_csv} -> {out_dir}"
    )

    if tf_mode:
        adapter_meta, _schedule = _train_tf_variant(
            cfg, df, positions, target, ctx, tf_mode, regime,
            structure_suffix, rationales, seed,
        )
        adapter_meta["seed"] = seed if seed is not None else int(cfg.training.seed)
    elif regime == "pair":
        t1_dir, t2_dir = train_adapter_pair(
            cfg, df, positions, out_dir, structure_suffix, rationales
        )
        adapter_meta = {"t1_adapter": str(t1_dir), "t2_adapter": str(t2_dir)}
    else:
        shared_dir = train_single_adapter(
            cfg, df, positions, out_dir, structure_suffix, rationales, regime
        )
        adapter_meta = {"shared_adapter": str(shared_dir)}

    meta = {
        "tier": cfg.tier,
        "target": target,
        "regime": regime,
        "n_adapters": 2 if regime == "pair" else 1,
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
        **adapter_meta,
        **_provenance(started),
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

    started = _utc_now()
    cfg = load_config(args.overrides)
    if getattr(args, "manifest", None):
        return _predict_manifest(args, cfg, started)
    ctx = args.ctx
    arm, style = args.arm, args.inf
    suffix = structure_suffix_for(style)

    df = load_manual(REPO_ROOT / cfg.dataset.eval_csv)
    # The evaluation CSV ships predictions from an earlier run; we produce our own.
    df = df.drop(columns=[c for c in df.columns if c.endswith("_auto")])

    limit = args.limit if args.limit is not None else cfg.limit
    n_total = len(df) if limit in (None, "null") else min(int(limit), len(df))

    seed = getattr(args, "seed", None)
    save_path = result_path(cfg, arm, style, ctx, seed)
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

    # Adapters, only for the fine-tuning arms.
    t1_adapter, t2_adapter, shared_adapter, retrain_cmd = resolve_adapters(
        cfg, arm, ctx, seed
    )
    for path in (t1_adapter, t2_adapter, shared_adapter):
        if path is not None and not path.exists():
            raise SystemExit(f"Missing adapter {path}. Train it first:\n  {retrain_cmd}")

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
        shared_adapter_dir=str(shared_adapter) if shared_adapter else None,
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
    _write_run_meta(save_path, arm, style, ctx, existing_df, started, seed,
                    dataset=eval_name(cfg))


def _write_run_meta(save_path: Path, arm: str, style: str, ctx: int,
                    df: Optional[pd.DataFrame], started: str,
                    seed: Optional[int] = None,
                    dataset: str = DEFAULT_EVAL_NAME) -> None:
    """Stamp the finished cell with when it ran, beside the CSV it describes.

    A sidecar rather than a shared ledger: colocated so it cannot drift from its
    artifact, and one writer per file so concurrent SLURM jobs cannot race. The
    `.json` extension is not gitignored here, so it is tracked automatically.
    """
    meta = {
        "arm": arm,
        "style": style,
        "ctx": int(ctx),
        "seed": seed,
        "dataset": dataset,
        "n_rows": int(len(df)) if df is not None else 0,
        "artifact": save_path.name,
        **_provenance(started),
    }
    meta_path = save_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"Run metadata -> {meta_path}")


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


# -----------------------------------------------------------------------------
# Re-run (manifest) mode. Same training/inference code as above; only where the
# data comes from and where the outputs go differ (baseline.folds, baseline.rerun).
# -----------------------------------------------------------------------------
REGIME_PREFIX = {"pair": "ft", "mixed": "ft1mix", "sequential": "ft1seq"}


def _manifest_context(args, cfg):
    from baseline import folds, rerun
    source = {"misc.hlqc.gold": args.training_source} if args.training_source else None
    train_df, test_df, fold_meta = folds.load_fold(
        args.manifest, args.design, args.fold, args.seed if args.design == "5fold_seed_tied" else None,
        training_source=source, test_datasets=["misc.miv63a.gold", "misc.hlqc.gold"])
    key = dict(manifest=args.manifest, design=args.design, student=rerun.student_slug(cfg.model.base_model),
               ctx=args.ctx, prompt_version=os.environ["PROMPT_VERSION"], seed=args.seed, fold=args.fold,
               training_source=source)
    return train_df, test_df, fold_meta, key


def _cell_meta(args, cfg, fold_meta, train_df, started) -> dict:
    from baseline import fold_stats
    from components.prompts.loader import prompt_fingerprint
    return {"fold": fold_meta, "train_label_stats": fold_stats.summary(train_df),
            "prompts": prompt_fingerprint(), "base_model": cfg.model.base_model,
            "training_seed": args.seed, "config": CONFIG_PATH.name,
            "overrides": list(args.overrides), "started_utc": started, "finished_utc": _utc_now()}


def _train_manifest(args, cfg, started) -> None:
    from automisc_ft.train import train_adapter_pair, train_single_adapter
    from baseline import rerun
    arm = args.arm
    if arm is None or ARM_ADAPTER.get(arm) is None or arm in ARM_TF:
        raise SystemExit("manifest mode trains --arm ft_bare / ft1mix_bare / ft1seq_bare "
                         "(rationale and teacher-forcing arms join in Stages 2-3)")
    regime, target = ARM_ADAPTER[arm]
    if target != "bare":
        raise SystemExit("rationale targets need per-fold rationales (Stage 2); not wired yet")
    train_df, _, fold_meta, key = _manifest_context(args, cfg)
    positions = list(range(len(train_df)))[: args.limit or None]
    k = {x: key[x] for x in ("manifest", "design", "student", "ctx", "prompt_version", "seed", "fold",
                             "training_source")}
    out_dir = rerun.adapter_path(arm=arm, **k)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[rerun] train {arm} {args.manifest}/{args.design} fold {args.fold} seed {args.seed} "
          f"prompts {key['prompt_version']}: {len(positions)} rows -> {out_dir}")
    if regime == "pair":
        t1, t2 = train_adapter_pair(cfg, train_df, positions, out_dir, "_bare", None)
        adapters = {"t1_adapter": str(t1), "t2_adapter": str(t2)}
    else:
        shared = train_single_adapter(cfg, train_df, positions, out_dir, "_bare", None, regime)
        adapters = {"shared_adapter": str(shared)}
    meta = {"arm": arm, "regime": regime, "target": target, "n_rows": len(positions),
            "n_epochs": cfg.training.num_train_epochs, "learning_rate": cfg.training.learning_rate,
            "lora": {"r": cfg.model.peft_r, "alpha": cfg.model.peft_alpha}, **adapters,
            **_cell_meta(args, cfg, fold_meta, train_df, started)}
    rerun.write_meta(out_dir / "train_metadata.csv", meta)


def _predict_manifest(args, cfg, started) -> None:
    import shutil
    from automisc_ft.infer import TieredAnnotator
    from baseline import rerun
    from baseline.fewshot import build_exemplars

    arm, style = args.arm, args.inf
    train_df, test_df, fold_meta, key = _manifest_context(args, cfg)
    test_df = test_df.drop(columns=[c for c in test_df.columns if c.endswith("_auto")])
    k = {x: key[x] for x in ("manifest", "design", "student", "ctx", "prompt_version", "seed", "fold",
                             "training_source")}
    save_path = rerun.result_path(arm=arm, style=style, **k)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    existing_df = None
    if save_path.exists():
        existing_df = pd.read_csv(save_path)
        done = set(existing_df["uid"])
    positions = [i for i in range(len(test_df)) if test_df.iloc[i]["uid"] not in done][: args.limit or None]

    t1 = t2 = shared = None
    spec = ARM_ADAPTER[arm]
    if spec is not None:
        if arm in ARM_TF or spec[1] != "bare":
            raise SystemExit("manifest mode predicts zs / fs / *_bare fine-tuned arms for now")
        root = rerun.adapter_path(arm=arm, **k)
        if spec[0] == "pair":
            t1, t2 = root / "t1" / "local_finetuned_model", root / "t2" / "local_finetuned_model"
        else:
            shared = root / "local_finetuned_model"
        for p in (t1, t2, shared):
            if p is not None and not p.exists():
                raise SystemExit(f"Missing adapter {p}; train it with the same --manifest/--design/--fold/--seed")

    fewshot_provider, exemplar_info = None, None
    if arm == "fs":
        if style != "bare":
            raise SystemExit("few-shot CoT needs per-fold exemplar rationales (Stage 2)")
        ex_seed = args.seed
        exemplars = build_exemplars(args.ctx, seed=ex_seed, source=train_df)
        fewshot_provider = _make_fewshot_provider(exemplars, rationales=False)
        exemplar_info = {"seed": ex_seed, "source": "fold training side",
                         "sessions": sorted({e["conv_id"] for v in exemplars.values()
                                             for e in v["t1"] + [x for g in v["t2"].values() for x in g]})}

    if positions:
        max_input_len = int(cfg.inference.max_input_len[arm])
        annotator = TieredAnnotator(
            base_model=cfg.model.base_model,
            t1_adapter_dir=str(t1) if t1 else None, t2_adapter_dir=str(t2) if t2 else None,
            shared_adapter_dir=str(shared) if shared else None,
            force_cpu=bool(cfg.inference.force_cpu),
            trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
            max_new_tokens=int(cfg.inference.max_new_tokens), max_input_len=max_input_len,
            structure_suffix=structure_suffix_for(style), fewshot_provider=fewshot_provider)
        rows: List[Dict] = []

        def save() -> None:
            nonlocal existing_df, rows
            if rows:
                out = pd.DataFrame(rows)
                existing_df = out if existing_df is None else pd.concat([existing_df, out], ignore_index=True)
                existing_df.to_csv(save_path, index=False)
                rows = []
        try:
            for pos in tqdm(positions, desc=f"{arm}_{style} fold{args.fold}", unit="utt"):
                pred = annotator.predict_row(test_df, pos, cfg.annotator.context_mode, args.ctx,
                                             bool(cfg.annotator.get("restrict_t2_to_group", False)))
                rows.append({**test_df.iloc[pos].to_dict(),
                             "t1_label_auto": pred["t1_pred"], "t2_label_auto": pred["t2_pred"],
                             "t1_raw": pred["t1_raw"], "t2_raw": pred["t2_raw"],
                             "t1_emitted_rationale": pred["t1_emitted_rationale"],
                             "t2_emitted_rationale": pred["t2_emitted_rationale"],
                             "t1_n_prompt_tokens": pred["t1_n_prompt_tokens"],
                             "t2_n_prompt_tokens": pred["t2_n_prompt_tokens"],
                             "t1_n_gen_tokens": pred["t1_n_gen_tokens"],
                             "t2_n_gen_tokens": pred["t2_n_gen_tokens"]})
                if len(rows) >= int(cfg.checkpoint_every):
                    save()
        finally:
            save()
            annotator.close()
        _report(existing_df, f"{arm}_inf_{style}", save_path, int(cfg.inference.max_new_tokens), max_input_len)

    n = 0 if existing_df is None else len(existing_df)
    complete = n == len(test_df)
    rerun.write_meta(save_path, {"arm": arm, "style": style, "n_rows": n, "n_expected": len(test_df),
                                 "complete": complete, "exemplars": exemplar_info,
                                 "adapter": str(shared or t1 or "") or None,
                                 **_cell_meta(args, cfg, fold_meta, train_df, started)})
    print(f"[rerun] {n}/{len(test_df)} rows -> {save_path}")
    if complete and spec is not None and os.environ.get("KEEP_ADAPTERS") != "1" and args.drop_adapter:
        shutil.rmtree(rerun.adapter_path(arm=arm, **k), ignore_errors=True)
        print("[rerun] fold adapter deleted after a complete prediction (--drop-adapter)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="train the LoRA adapter(s) for one arm")
    p_train.add_argument("--target", choices=TRAIN_TARGETS, default=None,
                         help="bare = label-only targets; rat = rationale + label targets")
    p_train.add_argument("--regime", choices=REGIMES, default="pair",
                         help="pair = one adapter per tier (2 adapters); "
                              "mixed = one adapter on the shuffled union of both "
                              "tiers; sequential = one adapter trained on T1 then "
                              "continued on T2")
    p_train.add_argument("--tf", choices=TF_MODES, default=None,
                         help="teacher forcing for the T2 stage: which T1 label "
                              "conditions it. full = always gold (the control), "
                              "zero = always the out-of-fold prediction, dyn = "
                              "decayed across epochs. Trains T2 only; T1 is "
                              "shared from the base arm. Requires --regime pair "
                              "--target bare.")

    p_pred = sub.add_parser("predict", help="annotate the evaluation set")
    p_pred.add_argument("--arm", choices=ARMS, required=True)
    p_pred.add_argument("--inf", choices=STYLES, required=True,
                        help="inference prompt style")
    p_pred.add_argument("--drop-adapter", action="store_true",
                        help="manifest mode: delete this fold's adapter once its prediction file is "
                             "complete (disk quota); KEEP_ADAPTERS=1 overrides")

    p_train.add_argument("--arm", choices=ARMS, default=None,
                         help="manifest mode: the fine-tuning arm to train (sets --regime/--target)")
    for p in (p_train, p_pred):
        p.add_argument("--manifest", default=None,
                       help="re-run mode: data/splits manifest (misc_main, misc_pooled_cv, ...); "
                            "train/test come from baseline.folds, results from baseline.rerun paths")
        p.add_argument("--design", default=None, help="manifest design (cold_start, loso, 5fold_seed_tied)")
        p.add_argument("--fold", type=int, default=0)
        p.add_argument("--training-source", default=None,
                       help="swap the HLQC training file, e.g. misc.hlqc.gold.cleaned (ablation)")
        p.add_argument("--prompt-version", default=None,
                       help="prompt set (default v2 in manifest mode, v1 otherwise)")
        p.add_argument("--seed", type=int, default=None,
                       help="replicate index for the teacher-forcing arms. Gives "
                            "the run its own adapter tree and a _seed{N} result "
                            "file, so replicates sit beside the original instead "
                            "of overwriting it.")
        p.add_argument("--ctx", type=int, default=None,
                       help="prior context volleys (default: config)")
        p.add_argument("--limit", type=int, default=None,
                       help="cap rows processed (smoke tests)")
        p.add_argument("overrides", nargs="*",
                       help="OmegaConf dotlist overrides, e.g. inference.force_cpu=true")

    args = parser.parse_args()

    if args.cmd == "train" and not args.manifest and args.target is None:
        parser.error("--target is required (manifest mode uses --arm instead)")
    if args.manifest:
        if args.seed is None:
            parser.error("manifest mode needs --seed (42/1/2)")
        os.environ["PROMPT_VERSION"] = args.prompt_version or "v2"
    elif args.prompt_version:
        os.environ["PROMPT_VERSION"] = args.prompt_version

    if args.ctx is None:
        args.ctx = int(load_config(args.overrides).annotator.num_context_turns)
    else:
        args.overrides = list(args.overrides) + [f"annotator.num_context_turns={args.ctx}"]

    if getattr(args, "seed", None) is not None:
        args.overrides = list(args.overrides) + [f"training.seed={args.seed}"]

    {"train": cmd_train, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()
