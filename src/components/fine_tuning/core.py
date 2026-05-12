"""Core fine-tuning orchestration."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hydra.utils import log

from .dataset_builder import (
    build_examples,
    load_annotated_data,
    split_examples,
    write_jsonl,
)
from .local_trainer import LocalTrainerConfig, run_local_fine_tuning
from .openai_trainer import run_openai_fine_tuning


@dataclass
class FineTuneOutputPaths:
    """Paths to fine-tuning outputs."""
    job_metadata: Optional[Path] = None
    model_dir: Optional[Path] = None


def run_fine_tuning(cfg) -> FineTuneOutputPaths:
    """
    Orchestrate fine-tuning workflow.
    
    Supports two providers:
    - 'local': Fine-tune locally using Hugging Face Transformers (optionally with LoRA)
    - 'openai': Submit job to OpenAI fine-tuning API
    """
    # Load and prepare data
    log.info("Loading annotated data for fine-tuning...")
    df = load_annotated_data(cfg)
    examples = build_examples(cfg, df)
    train_examples, valid_examples = split_examples(cfg, examples)

    if cfg.fine_tuning.provider == 'local':
        # Local fine-tuning
        log.info("Starting local fine-tuning workflow...")
        
        local_cfg = LocalTrainerConfig(
            base_model=cfg.fine_tuning.base_model,
            use_peft=cfg.fine_tuning.use_peft,
            peft_r=cfg.fine_tuning.peft_r,
            peft_alpha=cfg.fine_tuning.peft_alpha,
            peft_dropout=cfg.fine_tuning.peft_dropout,
            target_modules=cfg.fine_tuning.target_modules,
            fp16=cfg.fine_tuning.fp16,
            per_device_train_batch_size=cfg.fine_tuning.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.fine_tuning.per_device_eval_batch_size,
            gradient_accumulation_steps=cfg.fine_tuning.gradient_accumulation_steps,
            num_train_epochs=cfg.fine_tuning.num_train_epochs,
            learning_rate=cfg.fine_tuning.learning_rate,
            max_length=cfg.fine_tuning.max_length,
            max_target_length=cfg.fine_tuning.max_target_length,
            logging_steps=cfg.fine_tuning.logging_steps,
            save_steps=cfg.fine_tuning.save_steps,
            save_total_limit=cfg.fine_tuning.save_total_limit,
            output_dir=cfg.fine_tuning.output_dir,
            show_tqdm=getattr(cfg.fine_tuning, 'show_tqdm', True),
        )
        
        model_dir = run_local_fine_tuning(local_cfg, train_examples, valid_examples)
        
        # Save metadata
        import json
        metadata_path = Path(cfg.fine_tuning.output_dir) / 'fine_tuning_local_metadata.json'
        metadata = {
            'provider': 'local',
            'base_model': cfg.fine_tuning.base_model,
            'use_peft': cfg.fine_tuning.use_peft,
            'model_dir': str(model_dir),
            'num_train_examples': len(train_examples),
            'num_valid_examples': len(valid_examples) if valid_examples else 0,
        }
        with metadata_path.open('w', encoding='utf-8') as fh:
            json.dump(metadata, fh, indent=2)
        
        log.info(f"Fine-tuning metadata saved to {metadata_path}")
        
        return FineTuneOutputPaths(job_metadata=metadata_path, model_dir=model_dir)

    elif cfg.fine_tuning.provider == 'openai':
        # OpenAI fine-tuning
        log.info("Starting OpenAI fine-tuning workflow...")
        
        metadata_path = run_openai_fine_tuning(cfg, train_examples, valid_examples, write_jsonl)
        
        return FineTuneOutputPaths(job_metadata=metadata_path)

    else:
        raise ValueError(
            f"Unknown fine_tuning.provider: {cfg.fine_tuning.provider}. "
            f"Use 'local' or 'openai'."
        )
