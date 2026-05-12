"""OpenAI fine-tuning integration."""
import json
from pathlib import Path
from typing import Dict, List, Optional

from hydra.utils import log

try:
    import openai
except ImportError:
    openai = None


def upload_file_to_openai(file_path: Path) -> str:
    """Upload a file to OpenAI for fine-tuning."""
    if openai is None:
        raise ImportError(
            "OpenAI SDK required. Install with: pip install openai"
        )

    with file_path.open('rb') as file_obj:
        upload = openai.File.create(file=file_obj, purpose='fine-tune')
    
    file_id = upload.id
    log.info(f"Uploaded {file_path.name} to OpenAI as ID: {file_id}")
    return file_id


def create_fine_tune_job(
    training_file_id: str, cfg, validation_file_id: Optional[str] = None
) -> dict:
    """Create an OpenAI fine-tune job."""
    if openai is None:
        raise ImportError(
            "OpenAI SDK required. Install with: pip install openai"
        )

    params = {
        'training_file': training_file_id,
        'model': cfg.fine_tuning.base_model,
        'n_epochs': int(cfg.fine_tuning.max_epochs),
    }

    if cfg.fine_tuning.batch_size and int(cfg.fine_tuning.batch_size) > 0:
        params['batch_size'] = int(cfg.fine_tuning.batch_size)

    if validation_file_id is not None:
        params['validation_file'] = validation_file_id

    if cfg.fine_tuning.suffix:
        params['suffix'] = cfg.fine_tuning.suffix

    job = openai.FineTune.create(**params)
    log.info(f"Created OpenAI fine-tune job: {job.id}")
    return job


def run_openai_fine_tuning(
    cfg, train_examples: List[Dict[str, str]], valid_examples: Optional[List[Dict[str, str]]], 
    write_jsonl_fn
) -> Path:
    """
    Submit fine-tuning job to OpenAI.
    
    Args:
        cfg: Full config
        train_examples: Training examples
        valid_examples: Optional validation examples
        write_jsonl_fn: Function to write JSONL files
        
    Returns:
        Path to job metadata
    """
    import os
    
    if 'OPENAI_API_KEY' not in os.environ:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is required for OpenAI fine-tuning."
        )

    output_dir = Path(cfg.fine_tuning.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write JSONL files
    annotated_stem = "automisc_training"
    training_path = output_dir / f"{annotated_stem}_train.jsonl"
    write_jsonl_fn(train_examples, training_path)

    # Upload training file
    train_file_id = upload_file_to_openai(training_path)

    # Handle validation file if provided
    validation_file_id = None
    if valid_examples is not None:
        validation_path = output_dir / f"{annotated_stem}_valid.jsonl"
        write_jsonl_fn(valid_examples, validation_path)
        validation_file_id = upload_file_to_openai(validation_path)

    # Create job
    job = create_fine_tune_job(train_file_id, cfg, validation_file_id=validation_file_id)
    
    # Save metadata
    metadata_path = output_dir / f"fine_tune_job_{job.id}.json"
    with metadata_path.open('w', encoding='utf-8') as handle:
        json.dump(job, handle, indent=2, default=str)

    log.info(f"Fine-tune metadata saved to {metadata_path}")
    return metadata_path
