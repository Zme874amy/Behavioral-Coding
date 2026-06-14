"""Hydra entrypoint for the AutoMISC + LoRA fine-tuning fair comparison.

Pipeline:
  1. Load the human-consensus manual CSV and assign conversation-level CV folds.
  2. Zero-shot: run the original AutoMISC T1->T2 annotator once over all rows.
  3. Fine-tuned: for each fold, train T1+T2 LoRA adapters on the other folds and
     predict the held-out fold (pooled out-of-fold predictions).
  4. Score per-speaker Tier-1 accuracy and Tier-2 macro-F1/accuracy for both
     conditions and report ZS vs FT.

Usage (CUDA / HPC):
  PYTHONPATH=src python src/automisc_ft/main.py \
      hydra.run.dir=outputs/automisc_ft/qwen7b_miv63a

Smoke test (tiny model, few conversations, CPU):
  PYTHONPATH=src python src/automisc_ft/main.py \
      model.base_model=Qwen/Qwen2.5-0.5B-Instruct \
      limit_convs=3 cv.n_folds=3 inference.force_cpu=true \
      training.num_train_epochs=1
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Dict, List

import hydra
import pandas as pd
from hydra.utils import log
from omegaconf import DictConfig, OmegaConf

from automisc_ft.data import assign_folds, load_manual
from automisc_ft.eval import compute_condition_metrics, format_comparison
from automisc_ft.infer import TieredAnnotator
from automisc_ft.train import train_fold_adapters


def _free_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _maybe_limit_convs(df: pd.DataFrame, limit) -> pd.DataFrame:
    if not limit:
        return df
    keep = sorted(df["conv_id"].unique().tolist())[: int(limit)]
    out = df[df["conv_id"].isin(keep)].reset_index(drop=True)
    log.info("limit_convs=%s -> keeping %d conversations, %d rows", limit, len(keep), len(out))
    return out


@hydra.main(config_path="../../conf", config_name="automisc_ft_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    eval_dir = run_dir / "automisc_ft_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    context_mode = cfg.annotator.context_mode
    num_context_turns = int(cfg.annotator.num_context_turns)
    restrict_t2 = bool(cfg.annotator.get("restrict_t2_to_group", False))

    # 1) Data + folds
    df = load_manual(cfg.dataset.manual_csv)
    df = _maybe_limit_convs(df, cfg.get("limit_convs", None))
    fold_of = assign_folds(df, int(cfg.cv.n_folds), int(cfg.cv.seed))
    df["fold"] = df["conv_id"].map(fold_of)
    n_folds = int(cfg.cv.n_folds)
    all_positions = list(range(len(df)))
    log.info(
        "Loaded %d utterances, %d conversations, %d folds",
        len(df), df["conv_id"].nunique(), n_folds,
    )

    # 2) Zero-shot over ALL rows (no training needed; identical across folds)
    log.info("STEP 1/2: Zero-shot baseline (original AutoMISC pipeline)")
    zs_annotator = TieredAnnotator(
        base_model=cfg.model.base_model,
        t1_adapter_dir=None,
        t2_adapter_dir=None,
        force_cpu=bool(cfg.inference.force_cpu),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        max_new_tokens=int(cfg.inference.max_new_tokens),
        max_input_len=int(cfg.inference.max_input_len),
    )
    zs_results = zs_annotator.predict_rows(
        df, all_positions, context_mode, num_context_turns, restrict_t2, desc="zero-shot"
    )
    zs_annotator.close()
    _free_memory()
    zs_by_pos = {r["row_pos"]: r for r in zs_results}

    # 3) Fine-tuned, per fold
    log.info("STEP 2/2: Fine-tuned (per-fold T1+T2 LoRA adapters)")
    ft_by_pos: Dict[int, Dict] = {}
    fold_adapter_dirs: Dict[int, Dict[str, str]] = {}
    for fold in range(n_folds):
        train_positions = [i for i in all_positions if df.iloc[i]["fold"] != fold]
        test_positions = [i for i in all_positions if df.iloc[i]["fold"] == fold]
        if not test_positions:
            log.warning("Fold %d has no test rows; skipping", fold)
            continue
        log.info(
            "Fold %d/%d: train=%d test=%d", fold + 1, n_folds, len(train_positions), len(test_positions)
        )

        fold_dir = run_dir / "adapters" / f"fold{fold}"
        t1_dir, t2_dir = train_fold_adapters(cfg, df, train_positions, fold_dir)
        fold_adapter_dirs[fold] = {"t1": str(t1_dir), "t2": str(t2_dir)}
        _free_memory()

        ft_annotator = TieredAnnotator(
            base_model=cfg.model.base_model,
            t1_adapter_dir=str(t1_dir),
            t2_adapter_dir=str(t2_dir),
            force_cpu=bool(cfg.inference.force_cpu),
            trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
            max_new_tokens=int(cfg.inference.max_new_tokens),
            max_input_len=int(cfg.inference.max_input_len),
        )
        fold_results = ft_annotator.predict_rows(
            df, test_positions, context_mode, num_context_turns, restrict_t2,
            desc=f"fine-tuned fold{fold}",
        )
        ft_annotator.close()
        _free_memory()
        for r in fold_results:
            ft_by_pos[r["row_pos"]] = r

    # 4) Merge predictions into one frame
    rows: List[Dict] = []
    for pos in all_positions:
        base = df.iloc[pos]
        zs = zs_by_pos.get(pos, {})
        ft = ft_by_pos.get(pos, {})
        rows.append(
            {
                "conv_id": str(base["conv_id"]),
                "fold": int(base["fold"]),
                "speaker": base["speaker"],
                "utt_text": base["utt_text"],
                "t1_label_GT": base.get("t1_label_GT"),
                "t2_label_GT": base.get("t2_label_GT"),
                "zs_t1_pred": zs.get("t1_pred"),
                "zs_t2_pred": zs.get("t2_pred"),
                "ft_t1_pred": ft.get("t1_pred"),
                "ft_t2_pred": ft.get("t2_pred"),
            }
        )
    pred_df = pd.DataFrame(rows)
    pred_path = eval_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    log.info("Saved predictions to %s", pred_path)

    # 5) Metrics (only rows that received a fine-tuned prediction are scored for
    # FT; for a complete CV run that is every row).
    ft_scored = pred_df[pred_df["ft_t2_pred"].notna()].copy()
    zs_metrics = compute_condition_metrics(ft_scored, "zs_t1_pred", "zs_t2_pred")
    ft_metrics = compute_condition_metrics(ft_scored, "ft_t1_pred", "ft_t2_pred")

    results = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "n_utterances": int(len(pred_df)),
        "n_scored": int(len(ft_scored)),
        "n_conversations": int(pred_df["conv_id"].nunique()),
        "n_folds": n_folds,
        "fold_adapter_dirs": fold_adapter_dirs,
        "zero_shot": zs_metrics,
        "fine_tuned": ft_metrics,
    }
    (eval_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=str))

    comparison = format_comparison(zs_metrics, ft_metrics)
    (eval_dir / "comparison.txt").write_text(comparison)
    print("\n" + comparison)
    log.info("Saved metrics + comparison to %s", eval_dir)


if __name__ == "__main__":
    main()
