"""Evaluate fine-tuned models and compare to baseline."""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

from components.utils import call_chat_model, get_provider


def load_fine_tuned_model(cfg) -> Optional[Dict]:
    """Load fine-tuned model metadata if available."""
    if cfg.fine_tuning.provider == 'local':
        metadata_path = Path(cfg.fine_tuning.output_dir) / 'fine_tuning_local_metadata.json'
        if metadata_path.exists():
            with metadata_path.open('r') as f:
                return json.load(f)
    elif cfg.fine_tuning.provider == 'openai':
        # Look for the most recent fine-tune job metadata
        output_dir = Path(cfg.fine_tuning.output_dir)
        if output_dir.exists():
            job_files = list(output_dir.glob('fine_tune_job_*.json'))
            if job_files:
                latest_job = max(job_files, key=lambda x: x.stat().st_mtime)
                with latest_job.open('r') as f:
                    return json.load(f)
    return None


def get_model_for_inference(cfg, fine_tuned_metadata: Optional[Dict] = None) -> str:
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


def load_test_data(cfg) -> Tuple[List[str], List[str], List[str]]:
    """Load test utterances, speakers, and ground truth labels."""
    # For now, we'll use a subset of the annotated data as "test data"
    # In practice, you'd want a separate held-out test set
    annotated_path = Path('data/annotated') / (
        f"{cfg.input_dataset.name}_"
        f"{cfg.input_dataset.subset}_"
        f"{cfg.annotator.class_structure}_"
        f"{cfg.annotator.model.rsplit('/', 1)[-1]}_"
        f"{cfg.annotator.context_mode}_"
        f"{cfg.annotator.num_context_turns if cfg.annotator.context_mode == 'interval' else ''}"
        f"_annotated.csv"
    )

    if not annotated_path.exists():
        raise FileNotFoundError(f"Annotated data not found: {annotated_path}")

    df = pd.read_csv(annotated_path)

    # Sample a subset for evaluation (e.g., 100 utterances)
    eval_sample = df.sample(n=min(100, len(df)), random_state=42)

    utterances = eval_sample['utt_text'].tolist()
    speakers = eval_sample['speaker'].tolist()

    # Use the auto-annotated labels as "ground truth" for evaluation
    # In practice, you'd want human-annotated ground truth
    label_field = 't2_label_auto' if cfg.annotator.class_structure == 'tiered' else 'label_auto'
    ground_truth = eval_sample[label_field].tolist()

    log.info(f"Loaded {len(utterances)} test utterances for evaluation")
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


def evaluate_fine_tuned_model(cfg) -> Dict:
    """Evaluate a fine-tuned model and compare to baseline."""
    log.info("Starting fine-tuned model evaluation...")

    # Load fine-tuned model metadata
    fine_tuned_metadata = load_fine_tuned_model(cfg)
    if fine_tuned_metadata:
        log.info(f"Found fine-tuned model: {fine_tuned_metadata}")
    else:
        log.warning("No fine-tuned model found. Evaluating baseline only.")

    # Load test data
    utterances, speakers, ground_truth = load_test_data(cfg)

    results = {}

    # Evaluate baseline (original model)
    log.info("Evaluating baseline model...")
    baseline_model = cfg.fine_tuning.base_model
    baseline_provider = get_provider(baseline_model)

    baseline_predictions = run_inference_on_utterances(
        utterances, speakers, baseline_model, baseline_provider, cfg
    )
    baseline_metrics = compute_metrics(baseline_predictions, ground_truth)
    results['baseline'] = baseline_metrics

    # Evaluate fine-tuned model if available
    if fine_tuned_metadata:
        log.info("Evaluating fine-tuned model...")
        ft_model = get_model_for_inference(cfg, fine_tuned_metadata)

        if cfg.fine_tuning.provider == 'local':
            # For local models, we need to use the transformers pipeline
            ft_predictions = run_inference_on_utterances(
                utterances, speakers, ft_model, 'local', cfg
            )
        else:
            # For OpenAI, use the fine-tuned model ID
            ft_provider = get_provider(ft_model)
            ft_predictions = run_inference_on_utterances(
                utterances, speakers, ft_model, ft_provider, cfg
            )

        ft_metrics = compute_metrics(ft_predictions, ground_truth)
        results['fine_tuned'] = ft_metrics

        # Compute improvement
        results['improvement'] = {
            'accuracy_delta': ft_metrics['accuracy'] - baseline_metrics['accuracy'],
            'f1_macro_delta': ft_metrics['f1_macro'] - baseline_metrics['f1_macro'],
            'f1_weighted_delta': ft_metrics['f1_weighted'] - baseline_metrics['f1_weighted'],
        }

    return results


def save_evaluation_results(results: Dict, cfg):
    """Save evaluation results to disk."""
    exp_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    results_dir = exp_output_dir / "fine_tuning_evaluation"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save full results
    results_path = results_dir / "evaluation_results.json"
    with results_path.open('w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save summary
    summary = {
        'baseline_accuracy': results.get('baseline', {}).get('accuracy'),
        'fine_tuned_accuracy': results.get('fine_tuned', {}).get('accuracy'),
        'improvement_accuracy': results.get('improvement', {}).get('accuracy_delta'),
        'baseline_f1_macro': results.get('baseline', {}).get('f1_macro'),
        'fine_tuned_f1_macro': results.get('fine_tuned', {}).get('f1_macro'),
        'improvement_f1_macro': results.get('improvement', {}).get('f1_macro_delta'),
    }

    summary_path = results_dir / "evaluation_summary.json"
    with summary_path.open('w') as f:
        json.dump(summary, f, indent=2)

    log.info(f"Evaluation results saved to {results_dir}")
    return results_dir


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def evaluate_fine_tuning(cfg: DictConfig) -> None:
    """Main evaluation function."""
    log.info("Fine-tuning evaluation configuration:\n%s", OmegaConf.to_yaml(cfg.fine_tuning))

    try:
        results = evaluate_fine_tuned_model(cfg)
        results_dir = save_evaluation_results(results, cfg)

        # Print summary
        print("\n" + "="*50)
        print("FINE-TUNING EVALUATION RESULTS")
        print("="*50)

        if 'baseline' in results:
            baseline = results['baseline']
            print(".3f")
            print(".3f")
            print(".3f")

        if 'fine_tuned' in results:
            ft = results['fine_tuned']
            improvement = results['improvement']
            print(".3f")
            print(".3f")
            print(".3f")
            print(".3f")
            print(".3f")
            print(".3f")

        print(f"\nDetailed results saved to: {results_dir}")

    except Exception as e:
        log.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    evaluate_fine_tuning()
