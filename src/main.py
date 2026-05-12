"""Entrypoint for AutoMISC automated behavioural code classifier."""
import json
from pathlib import Path
from itertools import product
from typing import Dict, List, Any
import hashlib

import hydra
from hydra.utils import log
from omegaconf import OmegaConf, DictConfig
from datatypes.corpus import Corpus
from components.parser import Parser
from components.annotator import Annotator
from components.fine_tuning import run_fine_tuning
from unified_eval import unified_eval
import os
import openai
import logging
import lmstudio as lms


def normalize_config(cfg: DictConfig) -> DictConfig:
    """Convert legacy single-model configs to multi-model format for backward compatibility."""
    # Normalize parser config
    if hasattr(cfg.parser, 'model') and not hasattr(cfg.parser, 'models'):
        # Convert single model to list
        cfg.parser.models = [OmegaConf.create({
            'provider': cfg.parser.provider,
            'model': cfg.parser.model,
            'temperature': getattr(cfg.parser, 'temperature', 0.0)
        })]
    
    # Normalize annotator config
    if hasattr(cfg.annotator, 'model') and not hasattr(cfg.annotator, 'models'):
        # Convert single model to list
        cfg.annotator.models = [OmegaConf.create({
            'provider': cfg.annotator.provider,
            'model': cfg.annotator.model,
            'context_mode': getattr(cfg.annotator, 'context_mode', 'interval'),
            'num_context_turns': getattr(cfg.annotator, 'num_context_turns', 5),
            'class_structure': getattr(cfg.annotator, 'class_structure', 'tiered'),
            'temperature': getattr(cfg.annotator, 'temperature', 0.0)
        })]
    
    # Normalize fine_tuning config
    if hasattr(cfg.fine_tuning, 'provider') and not hasattr(cfg.fine_tuning, 'configs'):
        # Convert single config to list
        ft_config = OmegaConf.create({
            'provider': cfg.fine_tuning.provider,
            'base_model': getattr(cfg.fine_tuning, 'base_model', 'gpt2'),
            'use_peft': getattr(cfg.fine_tuning, 'use_peft', False),
            'peft_r': getattr(cfg.fine_tuning, 'peft_r', 8),
            'peft_alpha': getattr(cfg.fine_tuning, 'peft_alpha', 16),
            'peft_dropout': getattr(cfg.fine_tuning, 'peft_dropout', 0.1),
            'target_modules': getattr(cfg.fine_tuning, 'target_modules', ['q_proj', 'v_proj']),
            'fp16': getattr(cfg.fine_tuning, 'fp16', False),
            'per_device_train_batch_size': getattr(cfg.fine_tuning, 'per_device_train_batch_size', 4),
            'per_device_eval_batch_size': getattr(cfg.fine_tuning, 'per_device_eval_batch_size', 4),
            'gradient_accumulation_steps': getattr(cfg.fine_tuning, 'gradient_accumulation_steps', 1),
            'num_train_epochs': getattr(cfg.fine_tuning, 'num_train_epochs', 3),
            'learning_rate': getattr(cfg.fine_tuning, 'learning_rate', 2e-4),
            'max_length': getattr(cfg.fine_tuning, 'max_length', 512),
            'max_target_length': getattr(cfg.fine_tuning, 'max_target_length', 64),
            'validation_split': getattr(cfg.fine_tuning, 'validation_split', 0.1),
            'label_field': getattr(cfg.fine_tuning, 'label_field', 'auto'),
            'output_dir': getattr(cfg.fine_tuning, 'output_dir', 'data/fine_tuning'),
            'logging_steps': getattr(cfg.fine_tuning, 'logging_steps', 20),
            'save_steps': getattr(cfg.fine_tuning, 'save_steps', 200),
            'save_total_limit': getattr(cfg.fine_tuning, 'save_total_limit', 2),
            'show_tqdm': getattr(cfg.fine_tuning, 'show_tqdm', True),
            'seed': getattr(cfg.fine_tuning, 'seed', 42),
        })
        cfg.fine_tuning.configs = [ft_config]
    
    # Ensure experiment section exists
    if not hasattr(cfg, 'experiment'):
        cfg.experiment = OmegaConf.create({
            'checkpoint_dir': 'outputs/checkpoints',
            'resume_from_checkpoint': False,
            'checkpoint_frequency': 1,
            'run_evaluation': True,
            'evaluation_methods': ['IRR', 'relate_outcomes', 'fine_tuning_comparison'],
            'max_parallel_jobs': 1
        })
    
    return cfg


def generate_model_combinations(cfg: DictConfig) -> List[Dict[str, Any]]:
    """Generate all combinations of parser, annotator, and fine-tuning configs."""
    parser_models = cfg.parser.models
    annotator_models = cfg.annotator.models
    ft_configs = cfg.fine_tuning.configs if cfg.fine_tuning.enabled else [None]
    
    combinations = []
    for parser, annotator, ft in product(parser_models, annotator_models, ft_configs):
        combo = {
            'parser': parser,
            'annotator': annotator,
            'fine_tuning': ft,
            'experiment_id': generate_experiment_id(parser, annotator, ft)
        }
        combinations.append(combo)
    
    return combinations


def generate_experiment_id(parser: DictConfig, annotator: DictConfig, ft: DictConfig) -> str:
    """Generate a unique ID for this model combination."""
    combo_str = f"{parser.model}_{annotator.model}"
    if ft:
        combo_str += f"_{ft.provider}_{ft.base_model}"
    return hashlib.md5(combo_str.encode()).hexdigest()[:8]


def save_checkpoint(checkpoint_dir: Path, completed_experiments: List[str], current_experiment: str = None):
    """Save progress checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "experiment_progress.json"
    
    checkpoint_data = {
        'completed_experiments': completed_experiments,
        'current_experiment': current_experiment,
        'timestamp': str(Path.cwd().stat().st_mtime)  # Simple timestamp
    }
    
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)


def load_checkpoint(checkpoint_dir: Path) -> Dict[str, Any]:
    """Load progress checkpoint if it exists."""
    checkpoint_path = checkpoint_dir / "experiment_progress.json"
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {'completed_experiments': [], 'current_experiment': None}


def validate_config(cfg) -> None:
    """Validate configuration and model availability."""
    # Collect all models to validate
    all_models = set()
    
    # From parser models
    for model_config in cfg.parser.models:
        all_models.add(model_config.model)
    
    # From annotator models  
    for model_config in cfg.annotator.models:
        all_models.add(model_config.model)
    
    # Validate each model
    for model_name in all_models:
        try:
            lms_models = {m.model_key for m in lms.list_downloaded_models("llm")}
        except Exception as e:
            raise RuntimeError(f"Failed to fetch LM Studio models: {e}")

        openai_models = set()
        if "OPENAI_API_KEY" in os.environ:
            try:
                openai.api_key = os.environ["OPENAI_API_KEY"]
                openai_models = {m.id for m in openai.models.list().data}
            except Exception as e:
                raise RuntimeError(f"Failed to fetch OpenAI models: {e}")
        else:
            log.warning("OPENAI_API_KEY not set; skipping OpenAI model validation")

        if model_name not in openai_models and model_name not in lms_models:
            log.info(f"Available LM Studio models: {lms_models}")
            log.info(f"Available OpenAI models: {openai_models}")
            raise ValueError(f"Model '{model_name}' not found in OpenAI or LM Studio models.")

    # Check that input file exists
    input_path = Path('data') / f'{cfg.input_dataset.name}.csv'
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset file not found: {input_path}")

    log.info("Configuration successfully validated.")


def run_single_experiment(cfg: DictConfig, experiment_config: Dict[str, Any]) -> bool:
    """Run a single model combination experiment."""
    exp_id = experiment_config['experiment_id']
    parser_cfg = experiment_config['parser']
    annotator_cfg = experiment_config['annotator']
    ft_cfg = experiment_config['fine_tuning']
    
    log.info(f"Starting experiment {exp_id}")
    log.info(f"Parser: {parser_cfg.model} ({parser_cfg.provider})")
    log.info(f"Annotator: {annotator_cfg.model} ({annotator_cfg.provider})")
    if ft_cfg:
        log.info(f"Fine-tuning: {ft_cfg.provider} on {ft_cfg.base_model}")
    
    try:
        # Create experiment-specific config
        exp_cfg = OmegaConf.create({
            'input_dataset': cfg.input_dataset,
            'n_conversations': cfg.n_conversations,
            'parser': parser_cfg,
            'annotator': annotator_cfg,
            'fine_tuning': ft_cfg or OmegaConf.create({'enabled': False}),
            'experiment': cfg.experiment
        })
        
        # Create experiment output directory
        exp_output_dir = Path('outputs') / exp_id
        exp_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save experiment metadata
        metadata = {
            'experiment_id': exp_id,
            'parser_model': parser_cfg.model,
            'parser_provider': parser_cfg.provider,
            'annotator_model': annotator_cfg.model,
            'annotator_provider': annotator_cfg.provider,
            'annotator_context_mode': getattr(annotator_cfg, 'context_mode', 'interval'),
            'annotator_num_context_turns': getattr(annotator_cfg, 'num_context_turns', 5),
            'annotator_class_structure': getattr(annotator_cfg, 'class_structure', 'tiered'),
            'fine_tuning_enabled': ft_cfg is not None,
            'timestamp': str(Path.cwd().stat().st_mtime)
        }
        
        if ft_cfg:
            metadata.update({
                'fine_tuning_provider': ft_cfg.provider,
                'fine_tuning_base_model': ft_cfg.base_model,
                'fine_tuning_use_peft': getattr(ft_cfg, 'use_peft', False)
            })
        
        metadata_path = exp_output_dir / "experiment_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 1) Load volley-level dataset
        corpus = Corpus(exp_cfg)
        log.info(f"Dataset loaded for experiment {exp_id}")
        
        # 2) Parse each conversation into utterances
        parser = Parser(exp_cfg)
        parsed = parser.parse_corpus(corpus)
        log.info(f"Parsing complete for experiment {exp_id}")
        parsed.save_to_csv()
        
        # 3) Annotate utterances
        annotator = Annotator(exp_cfg)
        annotator.annotate_corpus(parsed)
        log.info(f"Annotation complete for experiment {exp_id}")
        
        # 4) Optional fine-tuning
        if ft_cfg and ft_cfg.enabled:
            log.info(f"Fine-tuning enabled for experiment {exp_id}")
            run_fine_tuning(exp_cfg)
        
        log.info(f"Experiment {exp_id} completed successfully")
        return True
        
    except Exception as e:
        log.error(f"Experiment {exp_id} failed: {e}")
        return False


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg) -> None:
    # Normalize config for backward compatibility
    cfg = normalize_config(cfg)
    
    validate_config(cfg)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    log.info("Starting AutoMISC multi-model experiment with configuration:\n%s", OmegaConf.to_yaml(cfg))
    
    # Generate all model combinations
    combinations = generate_model_combinations(cfg)
    log.info(f"Generated {len(combinations)} model combinations to test")
    
    # Setup checkpointing
    checkpoint_dir = Path(cfg.experiment.checkpoint_dir)
    checkpoint_data = load_checkpoint(checkpoint_dir)
    completed_experiments = set(checkpoint_data.get('completed_experiments', []))
    
    if cfg.experiment.resume_from_checkpoint and completed_experiments:
        log.info(f"Resuming from checkpoint. {len(completed_experiments)} experiments already completed.")
    else:
        log.info("Starting fresh experiment run.")
        completed_experiments = set()
    
    # Run experiments
    successful_experiments = []
    failed_experiments = []
    
    for i, combo in enumerate(combinations):
        exp_id = combo['experiment_id']
        
        if exp_id in completed_experiments:
            log.info(f"Skipping completed experiment {exp_id} ({i+1}/{len(combinations)})")
            continue
        
        log.info(f"Running experiment {exp_id} ({i+1}/{len(combinations)})")
        
        # Save checkpoint before starting
        save_checkpoint(checkpoint_dir, list(completed_experiments), exp_id)
        
        # Run the experiment
        success = run_single_experiment(cfg, combo)
        
        if success:
            completed_experiments.add(exp_id)
            successful_experiments.append(exp_id)
        else:
            failed_experiments.append(exp_id)
        
        # Save checkpoint after completion
        save_checkpoint(checkpoint_dir, list(completed_experiments), None)
        
        # Periodic checkpoint save
        if (i + 1) % cfg.experiment.checkpoint_frequency == 0:
            log.info(f"Checkpoint saved after {i+1} experiments")
    
    # Final summary
    log.info("="*50)
    log.info("EXPERIMENT SUMMARY")
    log.info("="*50)
    log.info(f"Total combinations: {len(combinations)}")
    log.info(f"Successful: {len(successful_experiments)}")
    log.info(f"Failed: {len(failed_experiments)}")
    log.info(f"Skipped (completed): {len(completed_experiments) - len(successful_experiments)}")
    
    if successful_experiments:
        log.info(f"Successful experiments: {successful_experiments}")
    
    if failed_experiments:
        log.info(f"Failed experiments: {failed_experiments}")
    
    # Run unified evaluation if requested
    if cfg.experiment.run_evaluation:
        log.info("Running unified evaluation across all experiments...")
        try:
            unified_eval(cfg)
        except Exception as e:
            log.error(f"Unified evaluation failed: {e}")
    
    log.info("All experiments completed!")
    return


if __name__ == "__main__":
    main()