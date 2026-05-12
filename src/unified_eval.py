"""Unified evaluation system for AutoMISC experiments."""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import hashlib

import hydra
import pandas as pd
from hydra.utils import log
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

try:
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    log.warning("sklearn not available. Install with: pip install scikit-learn")

from validation.IRR import IRR
from validation.relate_outcomes import relate_outcomes
from components.utils import call_chat_model, get_provider


def run_irr_evaluation(cfg: DictConfig, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
    """Run Inter-Rater Reliability evaluation."""
    log.info("Running IRR evaluation...")
    try:
        # IRR expects specific config structure, create a temporary config
        irr_cfg = OmegaConf.create({
            'input_dataset': cfg.input_dataset,
            'annotator': cfg.annotator,
            'experiment': cfg.experiment
        })
        irr_results = IRR(irr_cfg)
        return {'success': True, 'results': irr_results}
    except Exception as e:
        log.error(f"IRR evaluation failed: {e}")
        return {'success': False, 'error': str(e)}


def run_relate_outcomes_evaluation(cfg: DictConfig, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
    """Run relate outcomes evaluation."""
    log.info("Running relate outcomes evaluation...")
    try:
        # relate_outcomes expects specific config structure
        ro_cfg = OmegaConf.create({
            'input_dataset': cfg.input_dataset,
            'annotator': cfg.annotator,
            'experiment': cfg.experiment
        })
        ro_results = relate_outcomes(ro_cfg)
        return {'success': True, 'results': ro_results}
    except Exception as e:
        log.error(f"Relate outcomes evaluation failed: {e}")
        return {'success': False, 'error': str(e)}


def load_fine_tuned_model(cfg, experiment_id: str) -> Optional[Dict]:
    """Load fine-tuned model metadata for a specific experiment."""
    if cfg.fine_tuning.provider == 'local':
        metadata_path = Path(cfg.fine_tuning.output_dir) / f"{experiment_id}_fine_tuning_local_metadata.json"
        if metadata_path.exists():
            with metadata_path.open('r') as f:
                return json.load(f)
    elif cfg.fine_tuning.provider == 'openai':
        # Look for the most recent fine-tune job metadata for this experiment
        output_dir = Path(cfg.fine_tuning.output_dir)
        if output_dir.exists():
            job_files = list(output_dir.glob(f'{experiment_id}_fine_tune_job_*.json'))
            if job_files:
                latest_job = max(job_files, key=lambda x: x.stat().st_mtime)
                with latest_job.open('r') as f:
                    return json.load(f)
    return None


def get_model_for_inference(cfg, experiment_id: str, fine_tuned_metadata: Optional[Dict] = None) -> str:
    """Get the model identifier to use for inference."""
    if fine_tuned_metadata and cfg.fine_tuning.provider == 'local':
        # For local models, we need to load from disk
        return fine_tuned_metadata.get('model_dir', cfg.fine_tuning.base_model)
    elif fine_tuned_metadata and cfg.fine_tuning.provider == 'openai':
        # Use the fine-tuned OpenAI model ID
        return fine_tuned_metadata.get('fine_tuned_model', cfg.fine_tuning.base_model)
    else:
        # Use the base model (baseline)
        return cfg.fine_tuning.base_model


def run_inference_on_utterances(
    utterances: List[str], speakers: List[str], model: str, provider: str, cfg
) -> List[str]:
    """Run inference on a list of utterances."""
    predictions = []

    for utterance, speaker in tqdm(zip(utterances, speakers), desc="Running inference"):
        prompt_parts = [f"Speaker: {speaker}", f"Utterance: {utterance}"]
        prompt = "\n\n".join(prompt_parts) + "\n\nLabel:"

        messages = [{'role': 'user', 'content': prompt}]

        try:
            result = call_chat_model(
                messages=messages,
                model=model,
                provider=provider,
                temperature=0.0,  # Deterministic for evaluation
                response_format=None,  # Let it return raw text
            )

            # Extract the label from the response
            prediction = result.strip() if isinstance(result, str) else str(result)
            predictions.append(prediction)

        except Exception as e:
            log.warning(f"Inference failed for utterance: {utterance[:50]}... Error: {e}")
            predictions.append("UNKNOWN")

    return predictions


def load_test_data_for_experiment(cfg, experiment_id: str) -> Tuple[List[str], List[str], List[str]]:
    """Load test utterances, speakers, and ground truth labels for a specific experiment."""
    # Look for annotated data from this experiment
    annotated_path = None
    
    # Try different possible paths for annotated data
    possible_paths = [
        Path('data/annotated') / f"{experiment_id}_annotated.csv",
        Path('outputs') / experiment_id / "annotated.csv",
        # Fallback to any annotated file (for backward compatibility)
        Path('data/annotated') / f"{cfg.input_dataset.name}_{cfg.input_dataset.subset}_{cfg.annotator.class_structure}_{cfg.annotator.model.rsplit('/', 1)[-1]}_{cfg.annotator.context_mode}_{cfg.annotator.num_context_turns if cfg.annotator.context_mode == 'interval' else ''}_annotated.csv"
    ]
    
    for path in possible_paths:
        if path.exists():
            annotated_path = path
            break
    
    if not annotated_path or not annotated_path.exists():
        raise FileNotFoundError(f"Annotated data not found for experiment {experiment_id}. Searched paths: {possible_paths}")

    df = pd.read_csv(annotated_path)

    # Sample a subset for evaluation (e.g., 100 utterances)
    eval_sample = df.sample(n=min(100, len(df)), random_state=42)

    utterances = eval_sample['utt_text'].tolist()
    speakers = eval_sample['speaker'].tolist()

    # Use the auto-annotated labels as "ground truth" for evaluation
    # In practice, you'd want human-annotated ground truth
    label_field = 't2_label_auto' if cfg.annotator.class_structure == 'tiered' else 'label_auto'
    if label_field not in df.columns:
        label_field = 'label_auto'  # fallback
    
    ground_truth = eval_sample[label_field].tolist()

    log.info(f"Loaded {len(utterances)} test utterances for experiment {experiment_id}")
    return utterances, speakers, ground_truth


def compute_metrics(predictions: List[str], ground_truth: List[str]) -> Dict:
    """Compute evaluation metrics."""
    if not SKLEARN_AVAILABLE:
        return {
            'error': 'sklearn not available for metrics computation',
            'predictions': predictions,
            'ground_truth': ground_truth,
        }
    
    # Clean predictions (remove extra whitespace, normalize)
    clean_predictions = [pred.strip().upper() for pred in predictions]
    clean_ground_truth = [gt.strip().upper() for gt in ground_truth]

    # Overall metrics
    accuracy = accuracy_score(clean_ground_truth, clean_predictions)
    f1_macro = f1_score(clean_ground_truth, clean_predictions, average='macro')
    f1_weighted = f1_score(clean_ground_truth, clean_predictions, average='weighted')

    # Detailed classification report
    report = classification_report(
        clean_ground_truth,
        clean_predictions,
        output_dict=True,
        zero_division=0
    )

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'classification_report': report,
        'predictions': predictions,
        'ground_truth': ground_truth,
    }


def run_fine_tuning_comparison(cfg: DictConfig, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
    """Run fine-tuning comparison evaluation across experiments."""
    log.info("Running fine-tuning comparison evaluation...")
    
    comparison_results = {}
    
    # Group experiments by model combination (excluding fine-tuning)
    model_groups = {}
    for exp_id, exp_data in experiment_results.items():
        if exp_data.get('fine_tuning_enabled', False):
            base_key = f"{exp_data['parser_model']}_{exp_data['annotator_model']}"
            if base_key not in model_groups:
                model_groups[base_key] = {}
            model_groups[base_key][exp_id] = exp_data
    
    # For each model combination, compare baseline vs fine-tuned
    for base_key, experiments in model_groups.items():
        if len(experiments) < 2:
            log.warning(f"Skipping {base_key}: need at least baseline + fine-tuned results")
            continue
        
        # Find baseline and fine-tuned experiments
        baseline_exp = None
        ft_experiments = []
        
        for exp_id, exp_data in experiments.items():
            if not exp_data.get('fine_tuning_enabled', False):
                baseline_exp = (exp_id, exp_data)
            else:
                ft_experiments.append((exp_id, exp_data))
        
        if not baseline_exp:
            log.warning(f"No baseline found for {base_key}")
            continue
        
        baseline_id, baseline_data = baseline_exp
        
        for ft_id, ft_data in ft_experiments:
            try:
                # Load test data for this experiment combination
                utterances, speakers, ground_truth = load_test_data_for_experiment(cfg, ft_id)
                
                # Evaluate baseline
                baseline_model = baseline_data['parser_model']  # Use parser model as baseline
                baseline_provider = get_provider(baseline_model)
                baseline_predictions = run_inference_on_utterances(
                    utterances, speakers, baseline_model, baseline_provider, cfg
                )
                baseline_metrics = compute_metrics(baseline_predictions, ground_truth)
                
                # Evaluate fine-tuned
                ft_metadata = load_fine_tuned_model(cfg, ft_id)
                if ft_metadata:
                    ft_model = get_model_for_inference(cfg, ft_id, ft_metadata)
                    if cfg.fine_tuning.provider == 'local':
                        ft_predictions = run_inference_on_utterances(
                            utterances, speakers, ft_model, 'local', cfg
                        )
                    else:
                        ft_provider = get_provider(ft_model)
                        ft_predictions = run_inference_on_utterances(
                            utterances, speakers, ft_model, ft_provider, cfg
                        )
                    ft_metrics = compute_metrics(ft_predictions, ground_truth)
                    
                    # Compute improvement
                    improvement = {
                        'accuracy_delta': ft_metrics['accuracy'] - baseline_metrics['accuracy'],
                        'f1_macro_delta': ft_metrics['f1_macro'] - baseline_metrics['f1_macro'],
                        'f1_weighted_delta': ft_metrics['f1_weighted'] - baseline_metrics['f1_weighted'],
                    }
                    
                    comparison_results[f"{base_key}_{ft_id}"] = {
                        'baseline': baseline_metrics,
                        'fine_tuned': ft_metrics,
                        'improvement': improvement,
                        'baseline_experiment': baseline_id,
                        'fine_tuned_experiment': ft_id
                    }
                else:
                    log.warning(f"No fine-tuned model found for experiment {ft_id}")
                    
            except Exception as e:
                log.error(f"Fine-tuning comparison failed for {ft_id}: {e}")
    
    return {'success': True, 'results': comparison_results}


def collect_experiment_results(cfg: DictConfig) -> Dict[str, Any]:
    """Collect results from all completed experiments."""
    experiment_results = {}
    
    # Look for experiment results in outputs directory
    outputs_dir = Path('outputs')
    if outputs_dir.exists():
        for exp_dir in outputs_dir.iterdir():
            if exp_dir.is_dir() and exp_dir.name.startswith(('202', 'exp_')):  # Date or exp_ prefix
                exp_id = exp_dir.name.split('_')[0] if '_' in exp_dir.name else exp_dir.name
                
                # Try to load experiment metadata
                metadata_file = exp_dir / "experiment_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        exp_data = json.load(f)
                        experiment_results[exp_id] = exp_data
    
    log.info(f"Collected results from {len(experiment_results)} experiments")
    return experiment_results


def save_unified_evaluation_results(results: Dict[str, Any], cfg):
    """Save unified evaluation results to disk."""
    exp_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    eval_dir = exp_output_dir / "unified_evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Save full results
    results_path = eval_dir / "evaluation_results.json"
    with results_path.open('w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save summary
    summary = {
        'evaluation_methods': list(results.keys()),
        'total_experiments_evaluated': len(results.get('experiment_results', {})),
        'timestamp': str(Path.cwd().stat().st_mtime)
    }
    
    # Add method-specific summaries
    if 'irr' in results and results['irr'].get('success'):
        summary['irr_completed'] = True
    
    if 'relate_outcomes' in results and results['relate_outcomes'].get('success'):
        summary['relate_outcomes_completed'] = True
        
    if 'fine_tuning_comparison' in results and results['fine_tuning_comparison'].get('success'):
        ft_results = results['fine_tuning_comparison']['results']
        summary['fine_tuning_comparisons'] = len(ft_results)
        if ft_results:
            # Calculate average improvements
            improvements = [comp['improvement'] for comp in ft_results.values()]
            summary['avg_accuracy_improvement'] = sum(i['accuracy_delta'] for i in improvements) / len(improvements)
            summary['avg_f1_macro_improvement'] = sum(i['f1_macro_delta'] for i in improvements) / len(improvements)

    summary_path = eval_dir / "evaluation_summary.json"
    with summary_path.open('w') as f:
        json.dump(summary, f, indent=2)

    log.info(f"Unified evaluation results saved to {eval_dir}")
    return eval_dir


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def unified_eval(cfg: DictConfig) -> None:
    """Main unified evaluation function."""
    log.info("Starting unified AutoMISC evaluation with configuration:\n%s", OmegaConf.to_yaml(cfg))
    
    # Collect results from all experiments
    experiment_results = collect_experiment_results(cfg)
    
    if not experiment_results:
        log.warning("No experiment results found. Run main.py first to generate experiments.")
        return
    
    results = {
        'experiment_results': experiment_results,
        'evaluation_config': OmegaConf.to_container(cfg)
    }
    
    # Run requested evaluation methods
    evaluation_methods = cfg.experiment.evaluation_methods
    
    if 'IRR' in evaluation_methods:
        results['irr'] = run_irr_evaluation(cfg, experiment_results)
    
    if 'relate_outcomes' in evaluation_methods:
        results['relate_outcomes'] = run_relate_outcomes_evaluation(cfg, experiment_results)
    
    if 'fine_tuning_comparison' in evaluation_methods:
        results['fine_tuning_comparison'] = run_fine_tuning_comparison(cfg, experiment_results)
    
    # Save results
    results_dir = save_unified_evaluation_results(results, cfg)
    
    # Print summary
    print("\n" + "="*60)
    print("UNIFIED EVALUATION RESULTS")
    print("="*60)
    
    print(f"Experiments evaluated: {len(experiment_results)}")
    print(f"Evaluation methods: {evaluation_methods}")
    
    if 'irr' in results and results['irr'].get('success'):
        print("✓ IRR evaluation completed")
    
    if 'relate_outcomes' in results and results['relate_outcomes'].get('success'):
        print("✓ Relate outcomes evaluation completed")
        
    if 'fine_tuning_comparison' in results and results['fine_tuning_comparison'].get('success'):
        ft_results = results['fine_tuning_comparison']['results']
        print(f"✓ Fine-tuning comparison completed ({len(ft_results)} comparisons)")
        if ft_results:
            improvements = [comp['improvement']['accuracy_delta'] for comp in ft_results.values()]
            avg_improvement = sum(improvements) / len(improvements)
            print(".3f")
    
    print(f"\nDetailed results saved to: {results_dir}")
    
    log.info("Unified evaluation completed!")


if __name__ == "__main__":
    unified_eval()