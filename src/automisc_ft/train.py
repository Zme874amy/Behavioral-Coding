"""Train the two per-fold LoRA adapters (T1 and T2) for the AutoMISC
fine-tuning experiment.

Each adapter is trained independently with the existing local trainer
(`components.fine_tuning.local_trainer.run_local_fine_tuning`) on the SAME
prompts the inference pipeline uses (original AutoMISC templates rendered
through the model's chat template).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from hydra.utils import log
from omegaconf import DictConfig

from automisc_ft.data import (
    TierExample,
    build_tier_examples,
    examples_to_prompt_completion,
)
from components.fine_tuning.local_trainer import (
    LocalTrainerConfig,
    run_local_fine_tuning,
)


def _local_trainer_config(
    cfg: DictConfig, output_dir: Path, init_adapter_dir: Optional[Path] = None
) -> LocalTrainerConfig:
    m = cfg.model
    t = cfg.training
    return LocalTrainerConfig(
        init_adapter_dir=str(init_adapter_dir) if init_adapter_dir else None,
        base_model=m.base_model,
        use_peft=bool(m.use_peft),
        peft_r=int(m.peft_r),
        peft_alpha=int(m.peft_alpha),
        peft_dropout=float(m.peft_dropout),
        target_modules=list(m.target_modules),
        fp16=bool(m.fp16),
        bf16=bool(m.get("bf16", False)),
        trust_remote_code=bool(m.get("trust_remote_code", False)),
        attn_implementation=m.get("attn_implementation", None),
        per_device_train_batch_size=int(t.per_device_train_batch_size),
        per_device_eval_batch_size=int(t.per_device_eval_batch_size),
        gradient_accumulation_steps=int(t.gradient_accumulation_steps),
        num_train_epochs=float(t.num_train_epochs),
        learning_rate=float(t.learning_rate),
        max_length=int(t.max_length),
        max_target_length=int(t.max_target_length),
        logging_steps=int(t.logging_steps),
        save_steps=int(t.save_steps),
        save_total_limit=int(t.save_total_limit),
        # OmegaConf reads a bare `no` (e.g. `training.save_strategy=no`) as YAML
        # boolean False, so map it back to HF's "no".
        save_strategy=("no" if t.get("save_strategy", "steps") is False
                       else str(t.get("save_strategy", "steps"))),
        output_dir=str(output_dir),
        show_tqdm=bool(t.show_tqdm),
        max_grad_norm=float(t.get("max_grad_norm", 1.0)),
        warmup_ratio=float(t.get("warmup_ratio", 0.05)),
        weight_decay=float(t.get("weight_decay", 0.0)),
        lr_scheduler_type=str(t.get("lr_scheduler_type", "cosine")),
        seed=int(t.get("seed", 42)),
        gradient_checkpointing=bool(t.get("gradient_checkpointing", True)),
    )


def _load_tokenizer(cfg: DictConfig):
    """Tokenizer used only to render the chat template into prompt strings."""
    from transformers import AutoTokenizer

    from components.hf_load import resolve_hf_token

    token = resolve_hf_token()
    tok = AutoTokenizer.from_pretrained(
        cfg.model.base_model,
        token=token,
        use_fast=True,
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _save_jsonl(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def train_adapter_pair(
    cfg: DictConfig,
    df,
    train_positions: List[int],
    out_dir: Path,
    structure_suffix: str = "",
    rationales: Optional[Dict[str, dict]] = None,
) -> Tuple[Path, Path]:
    """Train one T1 + T2 LoRA adapter pair over the given rows.

    Args:
        structure_suffix: prompt variant to train on ("" rationale-first,
            "_bare" label-only).
        rationales: frozen distilled-rationale store. When given, completion
            targets become rationale + label instead of the bare label (the
            `ft_rat` arm).

    Returns ``(t1_adapter_dir, t2_adapter_dir)`` pointing at the saved adapters.
    """
    context_mode = cfg.annotator.context_mode
    num_context_turns = int(cfg.annotator.num_context_turns)

    tokenizer = _load_tokenizer(cfg)

    t1_examples: List[TierExample] = build_tier_examples(
        df, train_positions, "t1", context_mode, num_context_turns,
        structure_suffix, rationales,
    )
    t2_examples: List[TierExample] = build_tier_examples(
        df, train_positions, "t2", context_mode, num_context_turns,
        structure_suffix, rationales,
    )
    log.info(
        "Training examples: T1=%d  T2=%d (from %d rows)",
        len(t1_examples), len(t2_examples), len(train_positions),
    )
    if not t1_examples or not t2_examples:
        raise RuntimeError(
            "No training examples were built. If this is the ft_rat arm, the "
            "frozen rationale file for this context length is missing or empty."
        )

    t1_rows = examples_to_prompt_completion(t1_examples, tokenizer)
    t2_rows = examples_to_prompt_completion(t2_examples, tokenizer)

    artifacts = out_dir / "sft_artifacts"
    _save_jsonl(t1_rows, artifacts / "t1_train.jsonl")
    _save_jsonl(t2_rows, artifacts / "t2_train.jsonl")

    # T1 adapter
    log.info("Training T1 adapter -> %s", out_dir / "t1")
    t1_cfg = _local_trainer_config(cfg, out_dir / "t1")
    t1_model_dir = run_local_fine_tuning(t1_cfg, t1_rows, None)

    # T2 adapter
    log.info("Training T2 adapter -> %s", out_dir / "t2")
    t2_cfg = _local_trainer_config(cfg, out_dir / "t2")
    t2_model_dir = run_local_fine_tuning(t2_cfg, t2_rows, None)

    return Path(t1_model_dir), Path(t2_model_dir)


# -----------------------------------------------------------------------------
# Teacher-forcing axis: which T1 label conditions T2 at training time.
#
# Every arm here is fully TOKEN-level teacher forced, including `tf=zero`. What
# varies is one field in the prompt -- the injected T1 group spec -- so this is
# the stage-level analogue: training the downstream stage on predicted rather
# than gold upstream inputs, the standard fix for pipeline error propagation.
# -----------------------------------------------------------------------------
def _train_tier(
    cfg: DictConfig,
    df,
    train_positions: List[int],
    out_dir: Path,
    tier: str,
    structure_suffix: str = "",
    rationales: Optional[Dict[str, dict]] = None,
    predicted_t1: Optional[Dict[str, dict]] = None,
    gold_prob: float = 1.0,
    rng: Optional[random.Random] = None,
    init_adapter_dir: Optional[Path] = None,
    tokenizer=None,
    **trainer_overrides,
) -> Path:
    """Train one tier's LoRA adapter and return the directory it was saved to.

    The shared primitive behind the out-of-fold T1 adapters and the staged T2
    trainer. `train_adapter_pair` deliberately does NOT route through this: all
    eight existing arms depend on it, so it is left exactly as it was.

    `trainer_overrides` are applied to the `LocalTrainerConfig` after it is
    built from the config, which is how the staged trainer forces
    `num_train_epochs=1` and a constant LR without disturbing the config.
    """
    context_mode = cfg.annotator.context_mode
    num_context_turns = int(cfg.annotator.num_context_turns)
    if tokenizer is None:
        tokenizer = _load_tokenizer(cfg)

    examples: List[TierExample] = build_tier_examples(
        df, train_positions, tier, context_mode, num_context_turns,
        structure_suffix, rationales,
        predicted_t1=predicted_t1, gold_prob=gold_prob, rng=rng,
    )
    if not examples:
        raise RuntimeError(
            f"No {tier} training examples were built for {out_dir}. If this is a "
            "rationale arm, the frozen rationale file for this context length is "
            "missing or empty."
        )

    rows = examples_to_prompt_completion(examples, tokenizer)
    _save_jsonl(rows, out_dir / "sft_artifacts" / f"{tier}_train.jsonl")

    trainer_cfg = _local_trainer_config(cfg, out_dir, init_adapter_dir)
    for key, value in trainer_overrides.items():
        if not hasattr(trainer_cfg, key):
            raise ValueError(f"unknown trainer override {key!r}")
        setattr(trainer_cfg, key, value)

    log.info("Training %s adapter on %d examples -> %s", tier, len(rows), out_dir)
    return Path(run_local_fine_tuning(trainer_cfg, rows, None))


def gold_prob_schedule(spec, n_epochs: int) -> List[float]:
    """Materialise the per-epoch p_gold schedule from a config spec.

    `p_gold` is the probability that a T2 training example is conditioned on the
    GOLD T1 rather than the predicted one, so 1.0 is full teacher forcing and
    0.0 is none.

        constant   eps(e) = value
        linear     eps(e) = start + (end - start) * e / (E - 1)

    Only these two are implemented, and that is deliberate: at E=3 every decay
    function yields three points, so the choice of shape buys almost nothing.
    `exponential` (start * k**e) and inverse sigmoid are one branch each if a
    longer run ever makes them worth distinguishing.
    """
    E = int(n_epochs)
    if E < 1:
        raise ValueError(f"n_epochs must be >= 1, got {n_epochs}")

    kind = str(spec.get("kind", "constant"))
    if kind == "constant":
        schedule = [float(spec.get("value", 1.0))] * E
    elif kind == "linear":
        start = float(spec.get("start", 1.0))
        end = float(spec.get("end", 0.0))
        schedule = (
            [start] if E == 1
            else [start + (end - start) * e / (E - 1) for e in range(E)]
        )
    else:
        raise ValueError(
            f"unknown tf schedule kind {kind!r}; implemented: constant, linear"
        )

    for p in schedule:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p_gold outside [0, 1] in schedule {schedule}")
    return schedule


def train_t2_staged(
    cfg: DictConfig,
    df,
    train_positions: List[int],
    out_dir: Path,
    schedule: List[float],
    predicted_t1: Optional[Dict[str, dict]] = None,
    structure_suffix: str = "",
    rationales: Optional[Dict[str, dict]] = None,
) -> Path:
    """Train the T2 adapter as one 1-epoch stage per entry in `schedule`.

    The trainer pre-tokenises a fixed example list once, so an example's T1
    conditioning is frozen for the whole run -- a `p_gold` that CHANGES across
    epochs is impossible without rebuilding the dataset between them. Chaining
    1-epoch runs through `init_adapter_dir` is how it gets rebuilt, and that
    mechanism already carries the `sequential` regime.

    Staging costs the LR schedule: chained runs under cosine + warmup give one
    cosine cycle and one warmup PER STAGE rather than one across the whole run.
    So every staged arm -- including the p_gold=1 control -- runs under a
    constant LR with no warmup. That keeps the staged arms comparable to each
    other, at the price of not being directly comparable to the unstaged
    `ft_bare`; comparing `ft_bare` against the p_gold=1 staged arm is what
    prices that difference.

    The optimizer state also resets at each stage boundary. Unavoidable with
    this trainer, and the constant LR keeps it benign.
    """
    tokenizer = _load_tokenizer(cfg)
    n_stages = len(schedule)
    # `print`, not `log.info`: this module logs through hydra's logger, which is
    # unconfigured when `local_arm` drives it, so log records never reach the
    # SLURM log. Stage progress is the one thing worth watching during a
    # multi-stage job, so it goes to stdout like the rest of the CLI output.
    print(f"Staged T2 training: {n_stages} stages, p_gold schedule {schedule}",
          flush=True)

    final: Optional[Path] = None
    for stage_idx, gold_prob in enumerate(schedule):
        is_last = stage_idx == n_stages - 1
        # Intermediate stages are kept for inspection; the final adapter lands
        # at out_dir so it resolves like every other T2 adapter.
        stage_dir = out_dir if is_last else out_dir / f"stage{stage_idx}"
        print(
            f"T2 stage {stage_idx + 1}/{n_stages}: p_gold={gold_prob:.3f}, "
            f"resuming from {final or '(fresh LoRA)'} -> {stage_dir}",
            flush=True,
        )
        final = _train_tier(
            cfg, df, train_positions, stage_dir, "t2",
            structure_suffix=structure_suffix,
            rationales=rationales,
            predicted_t1=predicted_t1,
            gold_prob=gold_prob,
            # Per-stage seed: the stages must draw differently, but the whole
            # run still has to reproduce.
            rng=random.Random(int(cfg.training.seed) + stage_idx),
            init_adapter_dir=final,
            tokenizer=tokenizer,
            num_train_epochs=1.0,
            lr_scheduler_type="constant",
            warmup_ratio=0.0,
        )
    return final



def train_single_staged(
    cfg: DictConfig,
    df,
    train_positions: List[int],
    out_dir: Path,
    schedule: List[float],
    predicted_t1: Optional[Dict[str, dict]] = None,
    structure_suffix: str = "",
    rationales: Optional[Dict[str, dict]] = None,
) -> Path:
    """Staged `mixed` regime -- the teacher-forcing axis on cell B.

    Cell B trains ONE adapter on the shuffled union of the T1 and T2 examples,
    so unlike cell A there is no separate T2 adapter to teacher-force in
    isolation. Re-conditioning the T2 examples necessarily retrains T1 behaviour
    through the same weights. That coupling is not a flaw in the design, it is
    the question this arm exists to answer: in cell A the T1 adapter is provably
    untouched, so a T1 regression is impossible; here it is not.

    Each stage rebuilds BOTH tiers -- T1 unchanged, T2 at that stage's p_gold --
    reshuffles the union, and continues from the previous stage's adapter. See
    `train_t2_staged` for why staging is forced and what it costs.
    """
    tokenizer = _load_tokenizer(cfg)
    context_mode = cfg.annotator.context_mode
    num_context_turns = int(cfg.annotator.num_context_turns)
    seed = int(cfg.training.seed)
    n_stages = len(schedule)
    # `print`, not `log.info`: this module logs through hydra's logger, which is
    # not configured when run from the local_arm CLI, so log lines are swallowed.
    print(f"Staged single-adapter training: {n_stages} stages, p_gold {schedule}")

    final: Optional[Path] = None
    for stage_idx, gold_prob in enumerate(schedule):
        is_last = stage_idx == n_stages - 1
        stage_dir = out_dir if is_last else out_dir / f"stage{stage_idx}"

        t1_examples = build_tier_examples(
            df, train_positions, "t1", context_mode, num_context_turns,
            structure_suffix, rationales,
        )
        t2_examples = build_tier_examples(
            df, train_positions, "t2", context_mode, num_context_turns,
            structure_suffix, rationales,
            predicted_t1=predicted_t1, gold_prob=gold_prob,
            # Per-stage seed so the stages draw differently, but the whole run
            # still reproduces.
            rng=random.Random(seed + stage_idx),
        )
        if not t1_examples or not t2_examples:
            raise RuntimeError(
                f"No training examples were built for stage {stage_idx} of {out_dir}."
            )

        rows = (
            examples_to_prompt_completion(t1_examples, tokenizer)
            + examples_to_prompt_completion(t2_examples, tokenizer)
        )
        # Same shuffle discipline as the unstaged `mixed` regime, so the only
        # difference between them is the T2 conditioning and the LR schedule.
        random.Random(seed).shuffle(rows)
        _save_jsonl(rows, stage_dir / "sft_artifacts" / "mixed_train.jsonl")

        trainer_cfg = _local_trainer_config(cfg, stage_dir, final)
        trainer_cfg.num_train_epochs = 1.0
        trainer_cfg.lr_scheduler_type = "constant"
        trainer_cfg.warmup_ratio = 0.0

        print(
            f"Mixed stage {stage_idx + 1}/{n_stages}: p_gold={gold_prob:.3f}, "
            f"{len(rows)} examples, resuming from {final or '(fresh LoRA)'} "
            f"-> {stage_dir}"
        )
        final = Path(run_local_fine_tuning(trainer_cfg, rows, None))
    return final


REGIMES = ("mixed", "sequential")


def train_single_adapter(
    cfg: DictConfig,
    df,
    train_positions: List[int],
    out_dir: Path,
    structure_suffix: str = "",
    rationales: Optional[Dict[str, dict]] = None,
    regime: str = "mixed",
) -> Path:
    """Train ONE LoRA covering both tiers, for the 1-adapter / 2-call arms.

    The adapter is used for both calls at inference, so this differs from
    `train_adapter_pair` in adapter count alone -- the prompts, targets and
    trainer settings are identical.

    Two regimes, deliberately matched on total example-passes so any difference
    between them is ordering rather than budget. With N rows per tier and E
    epochs, both see ``2 * N * E``:

        mixed       one pass over the shuffled union of T1 and T2 rows, E epochs
        sequential  E epochs on T1, then E epochs on T2 resuming from that
                    adapter. Watch T1 accuracy afterwards: the T2 stage can
                    overwrite what the T1 stage learned, and that forgetting is
                    the thing this regime exists to measure.

    Returns the directory of the final adapter.
    """
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")

    context_mode = cfg.annotator.context_mode
    num_context_turns = int(cfg.annotator.num_context_turns)

    tokenizer = _load_tokenizer(cfg)

    t1_examples: List[TierExample] = build_tier_examples(
        df, train_positions, "t1", context_mode, num_context_turns,
        structure_suffix, rationales,
    )
    t2_examples: List[TierExample] = build_tier_examples(
        df, train_positions, "t2", context_mode, num_context_turns,
        structure_suffix, rationales,
    )
    log.info(
        "Training examples: T1=%d  T2=%d (from %d rows), regime=%s",
        len(t1_examples), len(t2_examples), len(train_positions), regime,
    )
    if not t1_examples or not t2_examples:
        raise RuntimeError(
            "No training examples were built. If this is a rationale arm, the "
            "frozen rationale file for this context length is missing or empty."
        )

    t1_rows = examples_to_prompt_completion(t1_examples, tokenizer)
    t2_rows = examples_to_prompt_completion(t2_examples, tokenizer)

    artifacts = out_dir / "sft_artifacts"
    _save_jsonl(t1_rows, artifacts / "t1_train.jsonl")
    _save_jsonl(t2_rows, artifacts / "t2_train.jsonl")

    if regime == "mixed":
        rows = t1_rows + t2_rows
        # Shuffled so the two tiers interleave; a fixed seed keeps the order
        # reproducible across reruns of the same cell.
        random.Random(int(cfg.training.seed)).shuffle(rows)
        _save_jsonl(rows, artifacts / "mixed_train.jsonl")
        log.info("Training mixed single adapter on %d examples -> %s", len(rows), out_dir)
        model_dir = run_local_fine_tuning(_local_trainer_config(cfg, out_dir), rows, None)
        return Path(model_dir)

    # sequential: T1 stage, then T2 stage continuing from it.
    stage_dir = out_dir / "stage_t1"
    log.info("Sequential stage 1/2: T1 on %d examples -> %s", len(t1_rows), stage_dir)
    t1_model_dir = run_local_fine_tuning(
        _local_trainer_config(cfg, stage_dir), t1_rows, None
    )

    log.info(
        "Sequential stage 2/2: T2 on %d examples, continuing from %s -> %s",
        len(t2_rows), t1_model_dir, out_dir,
    )
    model_dir = run_local_fine_tuning(
        _local_trainer_config(cfg, out_dir, init_adapter_dir=Path(t1_model_dir)),
        t2_rows,
        None,
    )
    return Path(model_dir)


def train_fold_adapters(
    cfg: DictConfig,
    df,
    train_positions: List[int],
    fold_dir: Path,
) -> Tuple[Path, Path]:
    """Train the T1 and T2 LoRA adapters for one cross-validation fold."""
    return train_adapter_pair(cfg, df, train_positions, fold_dir)
