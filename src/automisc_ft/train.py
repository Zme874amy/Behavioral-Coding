"""Train the two per-fold LoRA adapters (T1 and T2) for the AutoMISC
fine-tuning experiment.

Each adapter is trained independently with the existing local trainer
(`components.fine_tuning.local_trainer.run_local_fine_tuning`) on the SAME
prompts the inference pipeline uses (original AutoMISC templates rendered
through the model's chat template).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

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


def _local_trainer_config(cfg: DictConfig, output_dir: Path) -> LocalTrainerConfig:
    m = cfg.model
    t = cfg.training
    return LocalTrainerConfig(
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
        output_dir=str(output_dir),
        show_tqdm=bool(t.show_tqdm),
        max_grad_norm=float(t.get("max_grad_norm", 1.0)),
        warmup_ratio=float(t.get("warmup_ratio", 0.05)),
        weight_decay=float(t.get("weight_decay", 0.0)),
        lr_scheduler_type=str(t.get("lr_scheduler_type", "cosine")),
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


def train_fold_adapters(
    cfg: DictConfig,
    df,
    train_positions: List[int],
    fold_dir: Path,
) -> Tuple[Path, Path]:
    """Train the T1 and T2 LoRA adapters for one fold.

    Returns ``(t1_adapter_dir, t2_adapter_dir)`` pointing at the saved adapters.
    """
    context_mode = cfg.annotator.context_mode
    num_context_turns = int(cfg.annotator.num_context_turns)

    tokenizer = _load_tokenizer(cfg)

    t1_examples: List[TierExample] = build_tier_examples(
        df, train_positions, "t1", context_mode, num_context_turns
    )
    t2_examples: List[TierExample] = build_tier_examples(
        df, train_positions, "t2", context_mode, num_context_turns
    )
    log.info(
        "Fold training examples: T1=%d  T2=%d (from %d rows)",
        len(t1_examples), len(t2_examples), len(train_positions),
    )

    t1_rows = examples_to_prompt_completion(t1_examples, tokenizer)
    t2_rows = examples_to_prompt_completion(t2_examples, tokenizer)

    artifacts = fold_dir / "sft_artifacts"
    _save_jsonl(t1_rows, artifacts / "t1_train.jsonl")
    _save_jsonl(t2_rows, artifacts / "t2_train.jsonl")

    # T1 adapter
    log.info("Training T1 adapter -> %s", fold_dir / "t1")
    t1_cfg = _local_trainer_config(cfg, fold_dir / "t1")
    t1_model_dir = run_local_fine_tuning(t1_cfg, t1_rows, None)

    # T2 adapter
    log.info("Training T2 adapter -> %s", fold_dir / "t2")
    t2_cfg = _local_trainer_config(cfg, fold_dir / "t2")
    t2_model_dir = run_local_fine_tuning(t2_cfg, t2_rows, None)

    return Path(t1_model_dir), Path(t2_model_dir)
