"""Local fine-tuning using Hugging Face Transformers + optional PEFT (LoRA)."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from hydra.utils import log


@dataclass
class LocalTrainerConfig:
    """Configuration for local fine-tuning."""
    base_model: str
    use_peft: bool = False
    peft_r: int = 8
    peft_alpha: int = 16
    peft_dropout: float = 0.1
    target_modules: Optional[List[str]] = None
    fp16: bool = False
    bf16: bool = False
    trust_remote_code: bool = False
    attn_implementation: Optional[str] = None
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    num_train_epochs: float = 3.0
    learning_rate: float = 2e-4
    max_length: int = 512
    max_target_length: int = 64
    logging_steps: int = 20
    save_steps: int = 200
    save_total_limit: int = 2
    output_dir: str = "data/fine_tuning"
    show_tqdm: bool = True
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"


def tokenize_prompt_completion(
    example: Dict[str, str], tokenizer, cfg: LocalTrainerConfig
) -> Dict[str, List[int]]:
    """Tokenize prompt-completion pair."""
    prompt = example['prompt']
    completion = example['completion']

    tokenized_prompt = tokenizer(
        prompt,
        truncation=True,
        max_length=cfg.max_length,
        add_special_tokens=False,
    )
    tokenized_completion = tokenizer(
        completion,
        truncation=True,
        max_length=cfg.max_target_length,
        add_special_tokens=False,
    )

    input_ids = tokenized_prompt['input_ids'] + tokenized_completion['input_ids']
    labels = [-100] * len(tokenized_prompt['input_ids']) + tokenized_completion['input_ids']

    if tokenizer.eos_token_id is not None:
        input_ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id)

    return {
        'input_ids': input_ids,
        'labels': labels,
    }


def run_local_fine_tuning(
    cfg_ft, train_examples: List[Dict[str, str]], valid_examples: Optional[List[Dict[str, str]]]
) -> Path:
    """
    Perform local fine-tuning on annotated data.
    
    If use_peft=True, uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
    If use_peft=False, performs full fine-tuning (all model parameters updated).
    
    Args:
        cfg_ft: Fine-tuning config
        train_examples: Training examples (prompt-completion pairs)
        valid_examples: Optional validation examples
        
    Returns:
        Path to saved model directory
    """
    try:
        import torch
        from datasets import Dataset
        from transformers import Trainer, TrainingArguments, TrainerCallback
    except ImportError as exc:
        raise ImportError(
            "Local fine-tuning requires: torch, transformers, datasets, accelerate. "
            "Install with: pip install torch transformers datasets accelerate"
        ) from exc

    use_peft = bool(cfg_ft.use_peft)
    if use_peft:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "LoRA fine-tuning requires: peft. Install with: pip install peft"
            ) from exc
        log.info("Using LoRA (parameter-efficient fine-tuning)")
    else:
        log.info("Using full fine-tuning (all parameters will be updated)")

    from components.hf_load import load_model_and_tokenizer

    model, tokenizer, train_device = load_model_and_tokenizer(
        cfg_ft.base_model,
        for_training=True,
        fp16=bool(cfg_ft.fp16),
        bf16=bool(getattr(cfg_ft, "bf16", False)),
        trust_remote_code=bool(getattr(cfg_ft, "trust_remote_code", False)),
        attn_implementation=getattr(cfg_ft, "attn_implementation", None),
    )
    log.info("Training base model loaded (device=%s)", train_device)

    # Apply LoRA if requested
    if use_peft:
        if cfg_ft.target_modules is None:
            cfg_ft.target_modules = ['q_proj', 'v_proj']
        
        peft_config = LoraConfig(
            r=cfg_ft.peft_r,
            lora_alpha=cfg_ft.peft_alpha,
            target_modules=cfg_ft.target_modules,
            lora_dropout=cfg_ft.peft_dropout,
            bias='none',
            task_type='CAUSAL_LM',
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Tokenize examples (and attach attention_mask for the padding collator).
    def _tok(ex):
        rec = tokenize_prompt_completion(ex, tokenizer, cfg_ft)
        rec["attention_mask"] = [1] * len(rec["input_ids"])
        return rec

    tokenized_train = [_tok(ex) for ex in train_examples]
    train_dataset = Dataset.from_list(tokenized_train)

    eval_dataset = None
    if valid_examples:
        tokenized_valid = [_tok(ex) for ex in valid_examples]
        eval_dataset = Dataset.from_list(tokenized_valid)

    # Pad-to-longest-in-batch collator that also right-pads label ids with -100.
    pad_id = tokenizer.pad_token_id

    def collator(features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch_input_ids = []
        batch_attn = []
        batch_labels = []
        for f in features:
            pad = max_len - len(f["input_ids"])
            batch_input_ids.append(f["input_ids"] + [pad_id] * pad)
            batch_attn.append(f["attention_mask"] + [0] * pad)
            batch_labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attn, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }

    # Setup output directory
    output_dir = Path(cfg_ft.output_dir)
    local_model_dir = output_dir / 'local_finetuned_model'
    local_model_dir.mkdir(parents=True, exist_ok=True)

    # Training arguments — compatible with transformers >= 4.46 (eval_strategy)
    # and falling back to legacy `evaluation_strategy` for older releases.
    import inspect

    eval_steps = cfg_ft.logging_steps if eval_dataset is not None else None
    ta_kwargs = dict(
        output_dir=str(local_model_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=cfg_ft.per_device_train_batch_size,
        per_device_eval_batch_size=cfg_ft.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg_ft.gradient_accumulation_steps,
        num_train_epochs=cfg_ft.num_train_epochs,
        learning_rate=cfg_ft.learning_rate,
        fp16=cfg_ft.fp16 and torch.cuda.is_available(),
        bf16=getattr(cfg_ft, "bf16", False) and torch.cuda.is_available(),
        eval_steps=eval_steps,
        save_steps=cfg_ft.save_steps,
        logging_steps=cfg_ft.logging_steps,
        save_total_limit=cfg_ft.save_total_limit,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        disable_tqdm=not cfg_ft.show_tqdm,
        report_to=[],
        max_grad_norm=getattr(cfg_ft, "max_grad_norm", 1.0),
        warmup_ratio=getattr(cfg_ft, "warmup_ratio", 0.1),
        weight_decay=getattr(cfg_ft, "weight_decay", 0.0),
        lr_scheduler_type=getattr(cfg_ft, "lr_scheduler_type", "cosine"),
        dataloader_pin_memory=False,
    )
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_strategy = "steps" if eval_dataset is not None else "no"
    if "eval_strategy" in ta_params:
        ta_kwargs["eval_strategy"] = eval_strategy
    else:
        ta_kwargs["evaluation_strategy"] = eval_strategy

    # On Apple Silicon prefer MPS for speed, but disable grad-clipping there to
    # avoid the known PyTorch MPS NaN bug in clip_grad_norm_.
    on_mps = (
        not torch.cuda.is_available()
        and getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    )
    if on_mps:
        ta_kwargs["max_grad_norm"] = 0.0  # disables clipping (NaN-safe)

    training_args = TrainingArguments(**ta_kwargs)
    log.info(
        "Training device: %s",
        "cuda" if torch.cuda.is_available() else ("mps" if on_mps else "cpu"),
    )

    # Trainer: HF >= 4.41 prefers `processing_class`; older releases want `tokenizer`.
    # On MPS, clear the cache every 25 steps to keep step-time stable.
    callbacks = []
    if on_mps:
        class _MpsCacheClear(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step % 25 == 0:
                    import gc
                    gc.collect()
                    torch.mps.empty_cache()
        callbacks.append(_MpsCacheClear())

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    log.info("Starting local fine-tuning...")
    trainer.train()
    trainer.save_model(local_model_dir)
    log.info(f"Saved fine-tuned model to {local_model_dir}")

    return local_model_dir
