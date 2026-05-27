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
except (ImportError, ModuleNotFoundError):
    SKLEARN_AVAILABLE = False

from components.utils import call_chat_model, get_provider


def _eval_cfg_with_annotated(cfg: DictConfig) -> DictConfig:
    """Build a config slice with `annotated` fields expected by validation.IRR / relate_outcomes."""
    if hasattr(cfg, "annotated") and cfg.annotated is not None:
        return cfg
    if hasattr(cfg.parser, "models") and cfg.parser.models:
        m0p = cfg.parser.models[0]
        parser_model = m0p.model
    else:
        parser_model = cfg.parser.model

    if hasattr(cfg.annotator, "models") and cfg.annotator.models:
        m0 = cfg.annotator.models[0]
        annotated = OmegaConf.create(
            {
                "model": m0.model,
                "context_mode": m0.context_mode,
                "num_context_turns": m0.num_context_turns,
                "class_structure": m0.class_structure,
                "temperature": getattr(m0, "temperature", 0.0),
                "parser_model": parser_model,
            }
        )
    else:
        annotated = OmegaConf.create(
            {
                "model": cfg.annotator.model,
                "context_mode": cfg.annotator.context_mode,
                "num_context_turns": cfg.annotator.num_context_turns,
                "class_structure": cfg.annotator.class_structure,
                "temperature": getattr(cfg.annotator, "temperature", 0.0),
                "parser_model": parser_model,
            }
        )
    return OmegaConf.create(
        {
            "input_dataset": cfg.input_dataset,
            "annotated": annotated,
            "experiment": cfg.experiment,
        }
    )


def run_irr_evaluation(cfg: DictConfig, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
    """Run Inter-Rater Reliability evaluation."""
    log.info("Running IRR evaluation...")
    try:
        from validation.IRR import IRR
        # IRR expects specific config structure, create a temporary config
        irr_cfg = _eval_cfg_with_annotated(cfg)
        irr_results = IRR(irr_cfg)
        return {'success': True, 'results': irr_results}
    except Exception as e:
        log.error(f"IRR evaluation failed: {e}")
        return {'success': False, 'error': str(e)}


def run_relate_outcomes_evaluation(cfg: DictConfig, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
    """Run relate outcomes evaluation."""
    log.info("Running relate outcomes evaluation...")
    try:
        from validation.relate_outcomes import relate_outcomes
        # relate_outcomes expects specific config structure
        ro_cfg = _eval_cfg_with_annotated(cfg)
        ro_results = relate_outcomes(ro_cfg)
        return {'success': True, 'results': ro_results}
    except Exception as e:
        log.error(f"Relate outcomes evaluation failed: {e}")
        return {'success': False, 'error': str(e)}


def _primary_fine_tuning_block(cfg: DictConfig) -> DictConfig:
    """Resolve a single fine_tuning config for paths (first enabled entry, else first list item)."""
    ft = cfg.fine_tuning
    if hasattr(ft, "configs") and ft.configs:
        for c in ft.configs:
            if c is None:
                continue
            if getattr(c, "enabled", False):
                return c
        c0 = ft.configs[0]
        if c0 is not None:
            return c0
    return ft


def load_fine_tuned_model(cfg, experiment_id: str) -> Optional[Dict]:
    """Load fine-tuned model metadata for a specific experiment."""
    block = _primary_fine_tuning_block(cfg)
    provider = getattr(block, "provider", None)
    output_dir = Path(getattr(block, "output_dir", "data/fine_tuning"))

    if provider == "local":
        for metadata_path in (
            output_dir / experiment_id / "fine_tuning_local_metadata.json",
            output_dir / f"{experiment_id}_fine_tuning_local_metadata.json",
            output_dir / "fine_tuning_local_metadata.json",
        ):
            if metadata_path.exists():
                with metadata_path.open("r", encoding="utf-8") as f:
                    return json.load(f)
    elif provider == "openai":
        if output_dir.exists():
            job_files = list(output_dir.glob(f"{experiment_id}_fine_tune_job_*.json"))
            if job_files:
                latest_job = max(job_files, key=lambda x: x.stat().st_mtime)
                with latest_job.open("r", encoding="utf-8") as f:
                    return json.load(f)
    return None


def get_model_for_inference(cfg, experiment_id: str, fine_tuned_metadata: Optional[Dict] = None) -> str:
    """Get the model identifier to use for inference."""
    block = _primary_fine_tuning_block(cfg)
    provider = getattr(block, "provider", "local")
    base_model = getattr(block, "base_model", "gpt2")
    if fine_tuned_metadata and provider == "local":
        return fine_tuned_metadata.get("model_dir", base_model)
    if fine_tuned_metadata and provider == "openai":
        return fine_tuned_metadata.get("fine_tuned_model", base_model)
    return base_model


def _local_hf_generate_label(
    model, tokenizer, device: str, speaker: str, utterance: str, max_new_tokens: int = 48
) -> str:
    """Greedy generation for prompt-completion style MI labels (local HF / PEFT)."""
    import torch

    prompt = f"Speaker: {speaker}\n\nUtterance: {utterance}\n\nLabel:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    gen_ids = out[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text.split()[0] if text else "UNKNOWN"


def _run_local_hf_predictions(
    utterances: List[str], speakers: List[str], ft_meta: Dict[str, Any]
) -> List[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = Path(ft_meta["model_dir"])
    base_model = ft_meta["base_model"]
    use_peft = bool(ft_meta.get("use_peft", False))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_peft:
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(base, str(model_dir))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
    model = model.to(device)
    model.eval()

    predictions: List[str] = []
    for utterance, speaker in tqdm(
        zip(utterances, speakers), desc="Local HF inference", total=len(utterances)
    ):
        try:
            predictions.append(
                _local_hf_generate_label(model, tokenizer, device, speaker, utterance)
            )
        except Exception as e:
            log.warning(f"Local HF inference failed: {e}")
            predictions.append("UNKNOWN")
    return predictions


def run_inference_on_utterances(
    utterances: List[str],
    speakers: List[str],
    model: str,
    provider: str,
    cfg,
    local_hf_meta: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Run inference on a list of utterances."""
    if provider == "local" and local_hf_meta:
        return _run_local_hf_predictions(utterances, speakers, local_hf_meta)

    predictions = []

    for utterance, speaker in tqdm(zip(utterances, speakers), desc="Running inference"):
        prompt_parts = [f"Speaker: {speaker}", f"Utterance: {utterance}"]
        prompt = "\n\n".join(prompt_parts) + "\n\nLabel:"

        messages = [{"role": "user", "content": prompt}]

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


def load_test_data_for_experiment(
    cfg, experiment_id: str, exp_metadata: Optional[Dict[str, Any]] = None
) -> Tuple[List[str], List[str], List[str]]:
    """Load test utterances, speakers, and ground truth labels for a specific experiment."""
    annotated_path = None
    possible_paths: List[Path] = [
        Path('data/annotated') / f"{experiment_id}_annotated.csv",
        Path('outputs') / experiment_id / "annotated.csv",
    ]
    if exp_metadata:
        parser_tail = exp_metadata["parser_model"].rsplit("/", 1)[-1]
        model_tail = exp_metadata["annotator_model"].rsplit("/", 1)[-1]
        ctx = exp_metadata.get("annotator_context_mode", "interval")
        nturns = exp_metadata.get("annotator_num_context_turns", 5) if ctx == "interval" else ""
        cls = exp_metadata.get("annotator_class_structure", "tiered")
        possible_paths.append(
            Path("data/annotated")
            / (
                f"{cfg.input_dataset.name}_{cfg.input_dataset.subset}_{cls}_"
                f"{parser_tail}_{model_tail}_{ctx}_{nturns}_annotated.csv"
            )
        )
    # Fallback using top-level cfg (single-model runs)
    if hasattr(cfg.annotator, 'model'):
        possible_paths.append(
            Path('data/annotated')
            / (
                f"{cfg.input_dataset.name}_{cfg.input_dataset.subset}_{cfg.annotator.class_structure}_"
                f"{cfg.annotator.model.rsplit('/', 1)[-1]}_{cfg.annotator.context_mode}_"
                f"{cfg.annotator.num_context_turns if cfg.annotator.context_mode == 'interval' else ''}"
                f"_annotated.csv"
            )
        )
    
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
    class_structure = (
        exp_metadata.get('annotator_class_structure', 'tiered')
        if exp_metadata
        else getattr(cfg.annotator, 'class_structure', 'tiered')
    )
    label_field = 't2_label_auto' if class_structure == 'tiered' else 'label_auto'
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
    
    # Group experiments by parser+annotator (baseline and fine-tuned share the same key)
    model_groups = {}
    for exp_id, exp_data in experiment_results.items():
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
                utterances, speakers, ground_truth = load_test_data_for_experiment(
                    cfg, ft_id, exp_metadata=ft_data
                )
                
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
                    ft_block = _primary_fine_tuning_block(cfg)
                    if getattr(ft_block, "provider", "local") == "local":
                        ft_predictions = run_inference_on_utterances(
                            utterances,
                            speakers,
                            ft_model,
                            "local",
                            cfg,
                            local_hf_meta=ft_metadata,
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
        for metadata_file in outputs_dir.rglob("experiment_metadata.json"):
            with metadata_file.open("r") as f:
                exp_data = json.load(f)
            exp_id = exp_data.get("experiment_id", metadata_file.parent.name)
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
    from main import normalize_config

    cfg = normalize_config(cfg)
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
            print(f"Average accuracy improvement (fine-tuned vs baseline): {avg_improvement:.4f}")
    
    print(f"\nDetailed results saved to: {results_dir}")
    
    log.info("Unified evaluation completed!")


if __name__ == "__main__":
    unified_eval()