"""Iterative self-training loop for Direction 1a.

Round r labels the pool with round (r-1)'s adapters (round 1 uses the baseline
ft_bare adapters), selects confident rows, retrains a fresh adapter pair on the
augmented data, and predicts on the eval set. Scoring is left to
`baseline.eval`; inspect gold T2 macro-F1 between rounds and stop when it stops
improving (confirmation-bias guard) -- pass --rounds to bound it.

Each step shells out to the existing module CLIs, so the loop adds no new
training/inference logic; it only threads adapter directories between rounds via
`paths.adapter_dir` overrides.

Usage:
    PYTHONPATH=src python -m selftrain.loop \
        --pool data/parsed/AnnoMI_parsed.csv --ctx 5 --rounds 2 \
        --work-dir data/fine_tuning/selftrain_annomi
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from baseline.local_arm import REPO_ROOT, load_config


def run(cmd: List[str]) -> None:
    print("＄ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--ctx", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--work-dir", required=True, help="base dir for per-round adapters + augmented CSVs")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--min-confidence", type=float, default=0.8)
    ap.add_argument("--per-class-cap", type=int, default=300)
    ap.add_argument("--overrides", nargs="*", default=[])
    args = ap.parse_args(argv)

    py = [sys.executable]
    cfg = load_config(args.overrides)
    # Round 0 labeler = the baseline ft_bare adapters already in cfg.paths.
    prev_adapter_dir = str((REPO_ROOT / cfg.paths.adapter_dir))
    work = Path(args.work_dir)
    tag = Path(args.pool).stem.replace("_parsed", "")

    for r in range(1, args.rounds + 1):
        pseudo = work / f"pseudo_r{r}.csv"
        aug = work / f"aug_r{r}.csv"
        adapter_dir = work / f"ctx{args.ctx}_r{r}"
        print(f"\n===== self-training round {r} (labeler adapters: {prev_adapter_dir}) =====")

        run(py + ["-m", "selftrain.label_pool", "--pool", args.pool, "--arm", "ft_bare",
                  "--ctx", str(args.ctx), "--k", str(args.k), "--temperature", str(args.temperature),
                  "--out", str(pseudo),
                  "--overrides", f"paths.adapter_dir={prev_adapter_dir}", *args.overrides])

        run(py + ["-m", "selftrain.select", "--pseudo", str(pseudo),
                  "--train-csv", "data/manual/HLQC_balanced_manual.csv",
                  "--out", str(aug), "--min-confidence", str(args.min_confidence),
                  "--per-class-cap", str(args.per_class_cap)])

        # local_arm takes OmegaConf overrides as POSITIONAL dotlist args.
        run(py + ["-m", "baseline.local_arm", "train", "--target", "bare", "--regime", "pair",
                  "--ctx", str(args.ctx),
                  f"dataset.train_csv={aug}", f"paths.adapter_dir={adapter_dir}",
                  *args.overrides])

        run(py + ["-m", "baseline.local_arm", "predict", "--arm", "ft_bare", "--inf", "bare",
                  "--ctx", str(args.ctx),
                  f"paths.adapter_dir={adapter_dir}", *args.overrides])

        prev_adapter_dir = str(adapter_dir)
        print(f"round {r} done. Score with:  PYTHONPATH=src python -m baseline.eval")

    print("\nAll rounds complete. Compare gold T2 macro-F1 across rounds before adopting.")


if __name__ == "__main__":
    main()
