"""Compare a Zero-Shot baseline vs a LoRA fine-tuned adapter on the held-out
LODO test set. Reports overall metrics, per-class metrics, and a dedicated
Change Talk / Sustain Talk long-tail breakdown.

Both runs use IDENTICAL prompts so the only difference is the model weights.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import pandas as pd
from hydra.utils import log
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

try:
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    SKLEARN = True
except Exception:  # pragma: no cover
    SKLEARN = False

from sft.data import (
    CHANGE_TALK_T2,
    CLIENT_T1,
    CLIENT_T2,
    COUNSELLOR_T1,
    COUNSELLOR_T2,
    SUSTAIN_TALK_T2,
    build_lodo,
    valid_for_speaker,
    vocab,
)


# -----------------------------------------------------------------------------
# Inference utilities (HF generate; deterministic).
# -----------------------------------------------------------------------------
def _select_device(force_cpu: bool = False):
    import torch

    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    # MPS imposes a 4 GiB per-NDArray cap that causal-LM generation can hit even
    # for small models. CPU is slower but reliable for ~hundreds of generations.
    return "cpu"


def _load_hf(
    base_model: str,
    adapter_dir: Optional[str],
    force_cpu: bool = False,
    trust_remote_code: bool = False,
):
    from components.hf_load import load_model_and_tokenizer

    return load_model_and_tokenizer(
        base_model,
        adapter_dir=adapter_dir,
        for_training=False,
        force_cpu=force_cpu,
        trust_remote_code=trust_remote_code,
    )


# Production flat.j2 templates use longer abbreviations for a few codes
# (ADWP/RCWP/CON/DIR) than the canonical vocab in `sft.data`. Map them so the
# parser accepts either form regardless of which prompt the model saw.
LABEL_ALIASES = {
    "ADWP": "ADW",
    "RCWP": "RCW",
    "CON": "CO",
    "DIR": "DI",
}


def _parse_label(generated: str, allowed: List[str]) -> str:
    """Extract the first valid MISC code from the model's generation."""
    if not generated:
        return "UNKNOWN"
    text = generated.strip()

    def _normalise(tok: str) -> str:
        return tok.strip("`*_:()[]\"' \t").strip().rstrip(".,;:")

    # Direct match on the first whitespace-token
    head = _normalise(text.split()[0]) if text else ""
    if head in allowed:
        return head
    if head in LABEL_ALIASES and LABEL_ALIASES[head] in allowed:
        return LABEL_ALIASES[head]
    # Search anywhere
    tokens = re.split(r"[\s,;.\n]+", text)
    for t in tokens:
        t = _normalise(t)
        if not t:
            continue
        if t in allowed:
            return t
        if t in LABEL_ALIASES and LABEL_ALIASES[t] in allowed:
            return LABEL_ALIASES[t]
    return "UNKNOWN"


def _predict_batch(
    model, tokenizer, device, prompts: List[str], allowed: List[str],
    max_new_tokens: int, max_input_len: int = 768,
) -> List[str]:
    import torch

    preds: List[str] = []
    for prompt in tqdm(prompts, desc="generating"):
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_input_len
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        preds.append(_parse_label(gen, allowed))
    return preds


# -----------------------------------------------------------------------------
# Metric helpers.
# -----------------------------------------------------------------------------
def _subset_metrics(
    y_true: List[str], y_pred: List[str], subset_labels: List[str]
) -> Dict:
    """Per-class precision/recall/F1 over a subset of labels (e.g. Change Talk)."""
    if not SKLEARN:
        return {"error": "sklearn not available"}
    if not subset_labels:
        return {}
    rep = classification_report(
        y_true, y_pred, labels=subset_labels, output_dict=True, zero_division=0
    )
    # macro across the subset, weighted by support
    rows = [rep[c] for c in subset_labels if c in rep]
    support = sum(r["support"] for r in rows)
    if support == 0:
        macro = {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0}
    else:
        macro = {
            "precision": sum(r["precision"] * r["support"] for r in rows) / support,
            "recall": sum(r["recall"] * r["support"] for r in rows) / support,
            "f1-score": sum(r["f1-score"] * r["support"] for r in rows) / support,
            "support": support,
        }
    return {"per_class": {c: rep.get(c, {}) for c in subset_labels}, "weighted_over_subset": macro}


def _overall_metrics(y_true: List[str], y_pred: List[str], class_structure: str) -> Dict:
    if not SKLEARN:
        return {"error": "sklearn not available"}
    labels = vocab(class_structure)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "n_examples": len(y_true),
        "per_class": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
    }
    if class_structure == "t2":
        metrics["change_talk"] = _subset_metrics(y_true, y_pred, CHANGE_TALK_T2)
        metrics["sustain_talk"] = _subset_metrics(y_true, y_pred, SUSTAIN_TALK_T2)
    return metrics


def _print_summary(name: str, m: Dict) -> None:
    print(f"\n--- {name} ---")
    print(f"  accuracy:    {m.get('accuracy'):.4f}" if "accuracy" in m else f"  {m}")
    if "f1_macro" in m:
        print(f"  f1_macro:    {m['f1_macro']:.4f}")
        print(f"  f1_weighted: {m['f1_weighted']:.4f}")
        print(f"  n_examples:  {m['n_examples']}")
    if "change_talk" in m and "weighted_over_subset" in m["change_talk"]:
        ct = m["change_talk"]["weighted_over_subset"]
        st = m["sustain_talk"]["weighted_over_subset"]
        print(f"  Change Talk  : P={ct['precision']:.3f} R={ct['recall']:.3f} F1={ct['f1-score']:.3f} (support={ct['support']})")
        print(f"  Sustain Talk : P={st['precision']:.3f} R={st['recall']:.3f} F1={st['f1-score']:.3f} (support={st['support']})")


# -----------------------------------------------------------------------------
# Driver.
# -----------------------------------------------------------------------------
def evaluate(cfg: DictConfig, adapter_dir: Optional[Path] = None) -> Dict:
    """Run zero-shot and (optionally) fine-tuned inference on the LODO test split."""
    class_structure = cfg.class_structure
    prompt_style = str(cfg.get("prompt_style", "bare"))
    _, test_examples, summary = build_lodo(
        train_datasets=list(cfg.train_datasets),
        test_dataset=cfg.test_dataset,
        class_structure=class_structure,
        prompt_style=prompt_style,
    )
    log.info(
        "Test split: %d examples (%s) | prompt_style=%s",
        len(test_examples), summary, prompt_style,
    )

    if cfg.evaluation.sample_size and cfg.evaluation.sample_size < len(test_examples):
        # Stratified-ish sample: deterministic via shuffle+seed
        import random
        rng = random.Random(int(cfg.training.seed))
        rng.shuffle(test_examples)
        test_examples = test_examples[: int(cfg.evaluation.sample_size)]
        log.info("Sampled %d examples for evaluation", len(test_examples))

    prompts = [ex["prompt"] for ex in test_examples]
    y_true = [ex["label"] for ex in test_examples]
    allowed = vocab(class_structure)

    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    eval_dir = run_dir / "sft_evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    force_cpu = bool(cfg.evaluation.get("force_cpu", False))
    max_input_len = int(cfg.evaluation.get("max_input_len", 768))
    trust_remote_code = bool(cfg.model.get("trust_remote_code", False))

    # --- Zero-shot baseline (same prompt, no adapter) ---
    log.info("Loading base model for ZERO-SHOT baseline: %s", cfg.model.base_model)
    base_model, base_tok, base_device = _load_hf(
        cfg.model.base_model,
        adapter_dir=None,
        force_cpu=force_cpu,
        trust_remote_code=trust_remote_code,
    )
    log.info("Zero-shot generation device: %s", base_device)
    zs_preds = _predict_batch(
        base_model, base_tok, base_device, prompts, allowed,
        max_new_tokens=int(cfg.evaluation.max_new_tokens),
        max_input_len=max_input_len,
    )
    zs_metrics = _overall_metrics(y_true, zs_preds, class_structure)
    _print_summary("Zero-shot baseline", zs_metrics)

    del base_model
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass

    # --- Fine-tuned model (base + LoRA adapter) ---
    ft_metrics = None
    ft_preds = None
    if adapter_dir is not None and Path(adapter_dir).exists():
        log.info("Loading fine-tuned LoRA adapter from %s", adapter_dir)
        ft_model, ft_tok, ft_device = _load_hf(
            cfg.model.base_model,
            adapter_dir=str(adapter_dir),
            force_cpu=force_cpu,
            trust_remote_code=trust_remote_code,
        )
        log.info("Fine-tuned generation device: %s", ft_device)
        ft_preds = _predict_batch(
            ft_model, ft_tok, ft_device, prompts, allowed,
            max_new_tokens=int(cfg.evaluation.max_new_tokens),
            max_input_len=max_input_len,
        )
        ft_metrics = _overall_metrics(y_true, ft_preds, class_structure)
        _print_summary("Fine-tuned (LoRA)", ft_metrics)
    else:
        log.warning("No adapter_dir provided/found; only zero-shot results will be computed.")

    # Save predictions and metrics
    df_rows = []
    for i, ex in enumerate(test_examples):
        row = {
            "dataset": ex["dataset"],
            "conv_id": ex["conv_id"],
            "speaker": ex["speaker"],
            "utt_text": ex["utt_text"],
            "human_label": ex["label"],
            "zero_shot_pred": zs_preds[i],
        }
        if ft_preds is not None:
            row["fine_tuned_pred"] = ft_preds[i]
        df_rows.append(row)
    pd.DataFrame(df_rows).to_csv(eval_dir / "predictions.csv", index=False)

    results = {
        "class_structure": class_structure,
        "prompt_style": prompt_style,
        "train_datasets": list(cfg.train_datasets),
        "test_dataset": cfg.test_dataset,
        "n_test": len(test_examples),
        "zero_shot": zs_metrics,
        "fine_tuned": ft_metrics,
    }
    (eval_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
    log.info("Saved evaluation artifacts to %s", eval_dir)
    return results


@hydra.main(config_path="../../conf", config_name="sft_config", version_base=None)
def main(cfg: DictConfig) -> None:
    adapter = cfg.evaluation.adapter_dir
    evaluate(cfg, adapter_dir=Path(adapter) if adapter else None)


if __name__ == "__main__":
    main()
