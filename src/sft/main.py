"""Single Hydra entrypoint: train LoRA on human labels, then run
Zero-Shot vs Fine-Tuned evaluation on the LODO held-out test set."""
from __future__ import annotations

from pathlib import Path

import hydra
from hydra.utils import log
from omegaconf import DictConfig

from sft.train import train_lora
from sft.eval import evaluate


@hydra.main(config_path="../../conf", config_name="sft_config", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info("STEP 1/2: LoRA fine-tuning on HUMAN-labeled MI utterances")
    adapter_dir = train_lora(cfg)
    log.info("LoRA adapter saved at: %s", adapter_dir)

    log.info("STEP 2/2: Zero-shot baseline vs Fine-tuned evaluation")
    results = evaluate(cfg, adapter_dir=adapter_dir)

    # Print a compact comparison
    zs = results.get("zero_shot") or {}
    ft = results.get("fine_tuned") or {}
    if zs and ft and "accuracy" in zs and "accuracy" in ft:
        print("\n" + "=" * 60)
        print(f"LODO comparison ({cfg.class_structure} | train={list(cfg.train_datasets)} -> test={cfg.test_dataset})")
        print(f"  accuracy   : zero-shot {zs['accuracy']:.4f}  |  fine-tuned {ft['accuracy']:.4f}  (Δ {ft['accuracy']-zs['accuracy']:+.4f})")
        print(f"  f1_macro   : zero-shot {zs['f1_macro']:.4f}  |  fine-tuned {ft['f1_macro']:.4f}  (Δ {ft['f1_macro']-zs['f1_macro']:+.4f})")
        print(f"  f1_weighted: zero-shot {zs['f1_weighted']:.4f}  |  fine-tuned {ft['f1_weighted']:.4f}  (Δ {ft['f1_weighted']-zs['f1_weighted']:+.4f})")
        if "change_talk" in zs and "weighted_over_subset" in zs["change_talk"]:
            ct_zs = zs["change_talk"]["weighted_over_subset"]
            ct_ft = ft["change_talk"]["weighted_over_subset"]
            print(f"  Change Talk F1 : zero-shot {ct_zs['f1-score']:.4f}  |  fine-tuned {ct_ft['f1-score']:.4f}  (Δ {ct_ft['f1-score']-ct_zs['f1-score']:+.4f})  support={ct_zs['support']}")


if __name__ == "__main__":
    main()
