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
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            default_data_collator,
        )
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

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(cfg_ft.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token or tokenizer.bos_token or 
            tokenizer.cls_token or tokenizer.sep_token
        )

    device_map = 'auto' if torch.cuda.is_available() else None
    torch_dtype = torch.float16 if cfg_ft.fp16 and torch.cuda.is_available() else torch.float32
    
    log.info(f"Loading model: {cfg_ft.base_model} (dtype={torch_dtype}, device_map={device_map})")
    model = AutoModelForCausalLM.from_pretrained(
        cfg_ft.base_model,
        device_map=device_map,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

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

    # Tokenize examples
    tokenized_train = [tokenize_prompt_completion(ex, tokenizer, cfg_ft) for ex in train_examples]
    train_dataset = Dataset.from_dict(tokenized_train)

    eval_dataset = None
    if valid_examples is not None:
        tokenized_valid = [tokenize_prompt_completion(ex, tokenizer, cfg_ft) for ex in valid_examples]
        eval_dataset = Dataset.from_dict(tokenized_valid)

    # Setup output directory
    output_dir = Path(cfg_ft.output_dir)
    local_model_dir = output_dir / 'local_finetuned_model'
    local_model_dir.mkdir(parents=True, exist_ok=True)

    # Training arguments
    eval_steps = cfg_ft.logging_steps if eval_dataset is not None else None
    training_args = TrainingArguments(
        output_dir=str(local_model_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=cfg_ft.per_device_train_batch_size,
        per_device_eval_batch_size=cfg_ft.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg_ft.gradient_accumulation_steps,
        num_train_epochs=cfg_ft.num_train_epochs,
        learning_rate=cfg_ft.learning_rate,
        fp16=cfg_ft.fp16 and torch.cuda.is_available(),
        evaluation_strategy='steps' if eval_dataset is not None else 'no',
        eval_steps=eval_steps,
        save_steps=cfg_ft.save_steps,
        logging_steps=cfg_ft.logging_steps,
        save_total_limit=cfg_ft.save_total_limit,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model='eval_loss',
        disable_tqdm=not cfg_ft.show_tqdm,
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    log.info("Starting local fine-tuning...")
    trainer.train()
    trainer.save_model(local_model_dir)
    log.info(f"Saved fine-tuned model to {local_model_dir}")

    return local_model_dir
