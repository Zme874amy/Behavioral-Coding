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
    # "steps" keeps HF's behaviour, which also writes a final checkpoint-<N>/ (a
    # second copy of the adapter plus ~0.3 GB of optimizer state) at the end of
    # training. "no" skips that: the adapter is still saved by save_model(), but
    # the run cannot be resumed mid-training. Use "no" for short retrains that
    # would simply restart; on MLeRP those checkpoints filled the user quota.
    save_strategy: str = "steps"
    output_dir: str = "data/fine_tuning"
    show_tqdm: bool = True
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = False
    # Continue training an existing LoRA instead of starting a fresh one. Used by
    # the sequential single-adapter regime, where the T2 stage resumes from the
    # adapter the T1 stage produced. Ignored when use_peft is False.
    init_adapter_dir: Optional[str] = None
    # Governs LoRA init and the data sampler. HF's own default is 42, so leaving
    # this at 42 reproduces every run made before it was threaded through.
    seed: int = 42


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
    init_adapter_dir = getattr(cfg_ft, "init_adapter_dir", None)
    if use_peft:
        try:
            from peft import LoraConfig, PeftModel, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "LoRA fine-tuning requires: peft. Install with: pip install peft"
            ) from exc
        if init_adapter_dir:
            log.info("Using LoRA, continuing from adapter %s", init_adapter_dir)
        else:
            log.info("Using LoRA (parameter-efficient fine-tuning)")
    else:
        if init_adapter_dir:
            raise ValueError("init_adapter_dir requires use_peft=True")
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
        if init_adapter_dir:
            # The saved adapter_config.json carries the LoRA geometry, so the
            # r/alpha/target_modules above are deliberately not reapplied here.
            # `is_trainable=True` matters: without it the adapter loads frozen
            # and the second stage would train nothing.
            if not (Path(init_adapter_dir) / "adapter_config.json").exists():
                raise FileNotFoundError(
                    f"No adapter to continue from at {init_adapter_dir}"
                )
            model = PeftModel.from_pretrained(
                model, str(init_adapter_dir), is_trainable=True
            )
        else:
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

    if getattr(cfg_ft, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        log.info("Gradient checkpointing enabled")

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
        save_strategy=getattr(cfg_ft, "save_strategy", "steps"),
        logging_steps=cfg_ft.logging_steps,
        save_total_limit=cfg_ft.save_total_limit,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        disable_tqdm=not cfg_ft.show_tqdm,
        report_to=[],
        max_grad_norm=getattr(cfg_ft, "max_grad_norm", 1.0),
        warmup_ratio=getattr(cfg_ft, "warmup_ratio", 0.1),
        seed=int(getattr(cfg_ft, "seed", 42)),
        weight_decay=getattr(cfg_ft, "weight_decay", 0.0),
        lr_scheduler_type=getattr(cfg_ft, "lr_scheduler_type", "cosine"),
        dataloader_pin_memory=False,
        gradient_checkpointing=getattr(cfg_ft, "gradient_checkpointing", False),
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

    # Drop any kwargs the installed transformers version does not accept. Some
    # arguments were removed across major releases (e.g. `overwrite_output_dir`
    # and `evaluation_strategy` are gone in transformers 5.x), so filter to the
    # current signature to stay compatible across versions.
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in ta_params.values()
    )
    if not accepts_var_kwargs:
        dropped = [k for k in ta_kwargs if k not in ta_params]
        if dropped:
            log.info("Dropping unsupported TrainingArguments kwargs: %s", dropped)
        ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in ta_params}

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
    _write_train_log(trainer, model, local_model_dir)

    return local_model_dir


def _write_train_log(trainer, model, out_dir) -> None:
    """Loss curve, peak GPU memory and LoRA placement, next to the adapter.

    save_strategy=no leaves no trainer_state.json, so without this a run keeps
    no record of whether training converged. `lora_modules_outside_lm` counts
    LoRA layers placed on a vision/audio tower of a multimodal checkpoint
    (wasted parameters for our text-only task).
    """
    import json

    import torch

    names = [n for n, _ in model.named_modules() if n.endswith(".lora_A")]
    off_lm = [n for n in names if any(k in n for k in ("vision", "visual", "audio", "image", "projector"))]
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    rec = {
        "log_history": trainer.state.log_history,
        "peak_gpu_mem_gb": (round(torch.cuda.max_memory_allocated() / 2**30, 2)
                            if torch.cuda.is_available() else None),
        "trainable_params": int(trainable), "total_params": int(total),
        "n_lora_modules": len(names), "lora_modules_outside_lm": len(off_lm),
        "lora_outside_lm_examples": off_lm[:5],
    }
    Path(out_dir, "train_log.json").write_text(json.dumps(rec, indent=2, default=str))

