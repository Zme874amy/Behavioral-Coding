"""Train a LoRA adapter on HUMAN-labeled MI utterances (no machine labels)."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
from hydra.utils import log
from omegaconf import DictConfig, OmegaConf

from components.fine_tuning.local_trainer import LocalTrainerConfig, run_local_fine_tuning
from sft.data import build_lodo


def _split_train_valid(
    examples: List[Dict[str, str]], validation_split: float, seed: int
) -> Tuple[List[Dict[str, str]], Optional[List[Dict[str, str]]]]:
    if validation_split <= 0.0:
        return examples, None
    rng = random.Random(seed)
    examples = list(examples)
    rng.shuffle(examples)
    n_valid = int(len(examples) * validation_split)
    if n_valid < 1:
        log.warning("Validation split too small for the available data; using all for training.")
        return examples, None
    return examples[n_valid:], examples[:n_valid]


def _save_jsonl(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def train_lora(cfg: DictConfig) -> Path:
    """Build LODO split from human labels, train LoRA, return model dir."""
    class_structure = cfg.class_structure
    prompt_style = str(cfg.get("prompt_style", "bare"))
    train_examples, test_examples, summary = build_lodo(
        train_datasets=list(cfg.train_datasets),
        test_dataset=cfg.test_dataset,
        class_structure=class_structure,
        prompt_style=prompt_style,
    )
    log.info("LODO summary: %s | prompt_style=%s", summary, prompt_style)
    if not train_examples:
        raise RuntimeError(
            "No training examples found. Check `train_datasets` and `class_structure` in conf/sft_config.yaml."
        )

    # Optional cap on the number of training examples (for CPU/MPS time budgets).
    train_subset = cfg.training.get("train_subset_size", None)
    if train_subset is not None and int(train_subset) > 0 and int(train_subset) < len(train_examples):
        rng = random.Random(int(cfg.training.seed))
        rng.shuffle(train_examples)
        train_examples = train_examples[: int(train_subset)]
        log.info("Capped training set to %d examples (seed=%d)", len(train_examples), int(cfg.training.seed))

    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    artifacts_dir = run_dir / "sft_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Persist the exact training and held-out test splits so eval is reproducible.
    _save_jsonl(train_examples, artifacts_dir / "train.jsonl")
    _save_jsonl(test_examples, artifacts_dir / "test.jsonl")
    (artifacts_dir / "lodo_summary.json").write_text(json.dumps(summary, indent=2))

    train_split, valid_split = _split_train_valid(
        train_examples,
        validation_split=float(cfg.training.validation_split),
        seed=int(cfg.training.seed),
    )

    model_root = run_dir / "lora_model"
    local_cfg = LocalTrainerConfig(
        base_model=cfg.model.base_model,
        use_peft=bool(cfg.model.use_peft),
        peft_r=int(cfg.model.peft_r),
        peft_alpha=int(cfg.model.peft_alpha),
        peft_dropout=float(cfg.model.peft_dropout),
        target_modules=list(cfg.model.target_modules),
        fp16=bool(cfg.model.fp16),
        bf16=bool(cfg.model.get("bf16", False)),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        attn_implementation=cfg.model.get("attn_implementation", None),
        per_device_train_batch_size=int(cfg.training.per_device_train_batch_size),
        per_device_eval_batch_size=int(cfg.training.per_device_eval_batch_size),
        gradient_accumulation_steps=int(cfg.training.gradient_accumulation_steps),
        num_train_epochs=float(cfg.training.num_train_epochs),
        learning_rate=float(cfg.training.learning_rate),
        max_length=int(cfg.training.max_length),
        max_target_length=int(cfg.training.max_target_length),
        logging_steps=int(cfg.training.logging_steps),
        save_steps=int(cfg.training.save_steps),
        save_total_limit=int(cfg.training.save_total_limit),
        output_dir=str(model_root),
        show_tqdm=bool(cfg.training.show_tqdm),
        max_grad_norm=float(cfg.training.get("max_grad_norm", 1.0)),
        warmup_ratio=float(cfg.training.get("warmup_ratio", 0.1)),
        weight_decay=float(cfg.training.get("weight_decay", 0.0)),
        lr_scheduler_type=str(cfg.training.get("lr_scheduler_type", "cosine")),
        gradient_checkpointing=bool(cfg.training.get("gradient_checkpointing", False)),
    )

    log.info(
        "Starting SFT: base=%s peft=%s train=%d valid=%d",
        local_cfg.base_model,
        local_cfg.use_peft,
        len(train_split),
        0 if valid_split is None else len(valid_split),
    )
    model_dir = run_local_fine_tuning(local_cfg, train_split, valid_split)

    metadata = {
        "provider": "local",
        "class_structure": class_structure,
        "prompt_style": prompt_style,
        "base_model": local_cfg.base_model,
        "use_peft": local_cfg.use_peft,
        "model_dir": str(model_dir),
        "train_datasets": list(cfg.train_datasets),
        "test_dataset": cfg.test_dataset,
        "num_train_examples": len(train_split),
        "num_valid_examples": 0 if valid_split is None else len(valid_split),
        "num_test_examples": len(test_examples),
        "lodo_summary": summary,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    (artifacts_dir / "sft_metadata.json").write_text(json.dumps(metadata, indent=2))
    log.info("Saved SFT metadata to %s", artifacts_dir / "sft_metadata.json")
    return Path(model_dir)


@hydra.main(config_path="../../conf", config_name="sft_config", version_base=None)
def main(cfg: DictConfig) -> None:
    train_lora(cfg)


if __name__ == "__main__":
    main()
