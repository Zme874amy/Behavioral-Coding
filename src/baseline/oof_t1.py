"""Frozen OUT-OF-FOLD predicted Tier-1 labels for the HLQC training set.

The two-call arms train their T2 adapter conditioned on the GOLD T1 label but
run it conditioned on the PREDICTED one, so roughly one T2 call in five meets a
group spec it never saw paired with that utterance during training. Measuring
what that costs means training T2 on predicted T1 instead -- which needs a
predicted T1 for every HLQC training row.

Those predictions must be OUT OF FOLD. The production T1 adapter is trained on
all 1,925 HLQC rows for 3 epochs, so its predictions on those same rows are
partly memorised: far more accurate than the ~79% it achieves on the held-out
evaluation corpus. Training T2 against them would understate the exposure by
roughly 3x and flatten the very contrast the experiment exists to measure.

So the corpus is split into K conversation-level folds (`assign_folds`, the same
helper the cross-validation runner uses); for each fold a T1-only adapter is
trained on the OTHER folds and used to predict the held-out one. No utterance is
ever predicted by an adapter that saw its conversation.

K=5 because HLQC holds only TEN conversations, and folds are conversation-level.
At K=3 a fold adapter trains on 57-77% of the corpus against the production
adapter's 100%, so its predictions are noisier than the ones T2 meets at
inference -- the mirror image of the in-sample problem, and just as much a
mismatch. K=5 lifts that to 74-86%. K=10 would reach 83-98% but costs ten
trainings per context length, which buys little for triple the compute.

The result is frozen to JSON, per context length, for the same reason rationales
are: a prediction made with 5 volleys in view cannot be reused at 3, and a
training run has to be reproducible.

Usage:
    PYTHONPATH=src python -m baseline.oof_t1 train-fold   --ctx 5 --fold 0
    PYTHONPATH=src python -m baseline.oof_t1 predict-fold --ctx 5 --fold 0
    PYTHONPATH=src python -m baseline.oof_t1 report       --ctx 5

Smoke test on CPU with a tiny model:
    PYTHONPATH=src python -m baseline.oof_t1 predict-fold --ctx 3 --fold 0 \
        --limit 4 model.base_model=Qwen/Qwen2.5-0.5B-Instruct inference.force_cpu=true
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
OOF_DIR = REPO_ROOT / "data" / "fine_tuning" / "oof_t1"

DEFAULT_FOLDS = 5
CHECKPOINT_EVERY = 25

# Fixed here rather than read from config: the fold split must not move if a
# training seed is edited, or the store would silently mix folds from different
# partitions. Recorded in every entry so a store can be audited after the fact.
FOLD_SEED = 42

# The T1 adapter this stands in for is the `ft_bare` one, trained on label-only
# prompts, so the OOF adapters use the same prompt variant.
STRUCTURE_SUFFIX = "_bare"


def oof_t1_path(num_context_turns: int) -> Path:
    """The merged store. Written by `report`, never by a prediction job."""
    return OOF_DIR / f"hlqc_oof_t1_ctx{num_context_turns}.json"


def oof_t1_fold_path(num_context_turns: int, fold: int) -> Path:
    """One file per fold -- the only thing a prediction job writes.

    Folds are predicted CONCURRENTLY (four at a time under the lion QOS). A
    single shared JSON cannot survive that: each job loads the store at startup
    and rewrites the whole file at the end, so whichever finishes last silently
    erases every fold that landed while it was running. One writer per file
    removes the race outright rather than trying to synchronise around it.
    """
    return OOF_DIR / f"hlqc_oof_t1_ctx{num_context_turns}_fold{fold}.json"


def load_oof_t1(num_context_turns: int) -> dict:
    """Merge every per-fold file into one store keyed by `corp_utt_idx`.

    A pre-existing merged file is read first so a store written before the
    per-fold split still loads; per-fold files win on any key collision.
    """
    store: dict = {}
    legacy = oof_t1_path(num_context_turns)
    if legacy.exists():
        with open(legacy) as f:
            store.update(json.load(f))
    for path in sorted(OOF_DIR.glob(f"hlqc_oof_t1_ctx{num_context_turns}_fold*.json")):
        with open(path) as f:
            store.update(json.load(f))
    return store


def fold_adapter_root(num_context_turns: int, fold: int) -> Path:
    return OOF_DIR / f"ctx{num_context_turns}" / f"fold{fold}" / "t1"


def fold_adapter_dir(num_context_turns: int, fold: int) -> Path:
    """Where `run_local_fine_tuning` saves the fold's T1 adapter."""
    return fold_adapter_root(num_context_turns, fold) / "local_finetuned_model"


def _split(df, n_folds: int, fold: int) -> Tuple[List[int], List[int]]:
    """Return (positions to train on, positions to predict) for one fold.

    Conversation-level, so no utterance is predicted by an adapter that saw any
    other utterance from the same conversation.
    """
    from automisc_ft.data import assign_folds

    fold_of = assign_folds(df, n_folds, FOLD_SEED)
    train_pos, pred_pos = [], []
    for i in range(len(df)):
        target = pred_pos if fold_of[str(df.iloc[i]["conv_id"])] == fold else train_pos
        target.append(i)
    return train_pos, pred_pos


def _load_df(cfg):
    from automisc_ft.data import load_manual

    return load_manual(REPO_ROOT / cfg.dataset.train_csv)


# -----------------------------------------------------------------------------
# train-fold
# -----------------------------------------------------------------------------
def cmd_train_fold(args) -> None:
    from automisc_ft.train import _train_tier
    from baseline.local_arm import load_config

    cfg = load_config(args.overrides)
    ctx, fold, n_folds = args.ctx, args.fold, args.folds
    if not 0 <= fold < n_folds:
        raise SystemExit(f"--fold must be in [0, {n_folds}), got {fold}")

    df = _load_df(cfg)
    train_pos, pred_pos = _split(df, n_folds, fold)
    if args.limit:
        train_pos = train_pos[: int(args.limit)]

    out_dir = fold_adapter_root(ctx, fold)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"OOF T1 fold {fold}/{n_folds} ctx={ctx}: training on {len(train_pos)} rows, "
        f"holding out {len(pred_pos)} -> {out_dir}"
    )

    adapter = _train_tier(
        cfg, df, train_pos, out_dir, "t1", structure_suffix=STRUCTURE_SUFFIX
    )

    meta = {
        "kind": "oof_t1",
        "fold": fold,
        "n_folds": n_folds,
        "fold_seed": FOLD_SEED,
        "num_context_turns": ctx,
        "structure_suffix": STRUCTURE_SUFFIX,
        "base_model": cfg.model.base_model,
        "train_csv": cfg.dataset.train_csv,
        "n_train_rows": len(train_pos),
        "n_heldout_rows": len(pred_pos),
        "n_epochs": cfg.training.num_train_epochs,
        "learning_rate": cfg.training.learning_rate,
        "adapter": str(adapter),
    }
    (out_dir / "train_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"Fold adapter saved -> {adapter}")


# -----------------------------------------------------------------------------
# predict-fold
# -----------------------------------------------------------------------------
def cmd_predict_fold(args) -> None:
    from automisc_ft.data import build_messages_t1, t1_codes_for_speaker
    from automisc_ft.infer import TieredAnnotator, parse_label
    from baseline.local_arm import load_config

    cfg = load_config(args.overrides)
    ctx, fold, n_folds = args.ctx, args.fold, args.folds

    adapter = fold_adapter_dir(ctx, fold)
    if not adapter.exists():
        raise SystemExit(
            f"Missing fold adapter {adapter}. Train it first:\n"
            f"  PYTHONPATH=src python -m baseline.oof_t1 train-fold "
            f"--ctx {ctx} --fold {fold}"
        )

    df = _load_df(cfg)
    _, pred_pos = _split(df, n_folds, fold)
    if args.limit:
        pred_pos = pred_pos[: int(args.limit)]

    # This job owns exactly one file, so concurrent folds cannot clobber it.
    out_path = oof_t1_fold_path(ctx, fold)
    store = {}
    if out_path.exists():
        with open(out_path) as f:
            store = json.load(f)
        print(f"Resuming from {out_path}: {len(store)} utterances already predicted")

    todo = [p for p in pred_pos if str(df.iloc[p]["corp_utt_idx"]) not in store]
    if not todo:
        print(f"Fold {fold} already complete in {out_path}.")
        return

    # One adapter serving one call: passing it as the shared adapter makes it the
    # single active LoRA, which is exactly the T1-only setup. `predict_row` is
    # deliberately not used -- it always runs T2 as well.
    annotator = TieredAnnotator(
        base_model=cfg.model.base_model,
        shared_adapter_dir=str(adapter),
        force_cpu=bool(cfg.inference.force_cpu),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        max_new_tokens=int(cfg.inference.max_new_tokens),
        max_input_len=int(cfg.inference.max_input_len["ft_bare"]),
        structure_suffix=STRUCTURE_SUFFIX,
    )

    context_mode = cfg.annotator.context_mode
    done_since_save = 0

    def save() -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(store, f, indent=2)

    print(
        f"Predicting fold {fold}/{n_folds} ctx={ctx} with {adapter.parent.name}: "
        f"{len(todo)} rows -> {out_path}"
    )
    try:
        for pos in tqdm(todo, desc=f"fold{fold}", disable=not cfg.training.show_tqdm):
            row = df.iloc[pos]
            speaker = row["speaker"]
            if speaker not in {"counsellor", "client"}:
                continue
            codes = t1_codes_for_speaker(speaker)
            messages = build_messages_t1(
                df, pos, context_mode, int(ctx), STRUCTURE_SUFFIX
            )
            raw, _, _ = annotator._generate(messages)
            pred = parse_label(raw, codes)
            gold = row["t1_label_GT"]
            store[str(row["corp_utt_idx"])] = {
                "corp_utt_idx": int(row["corp_utt_idx"]),
                "conv_id": str(row["conv_id"]),
                "fold": int(fold),
                "n_folds": int(n_folds),
                "fold_seed": FOLD_SEED,
                "speaker": speaker,
                "t1_gold": gold if isinstance(gold, str) else None,
                "t1_pred": pred,
                "t1_raw": raw,
                "agree": bool(isinstance(gold, str) and pred == gold),
            }
            done_since_save += 1
            if done_since_save >= args.checkpoint_every:
                save()
                done_since_save = 0
    finally:
        save()
        annotator.close()

    print(f"Wrote {len(store)} out-of-fold T1 predictions for fold {fold} to {out_path}")
    _report(load_oof_t1(ctx), ctx)


# -----------------------------------------------------------------------------
# report -- the validity check
# -----------------------------------------------------------------------------
# `ft_bare` T1 accuracy on the held-out evaluation corpus, from
# outputs/baseline_eval/comparison.csv. Pooled out-of-fold accuracy on HLQC
# should land near this; if it lands far above, the exposure being trained on is
# gentler than the one met at inference and every downstream result is a lower
# bound rather than an estimate.
REFERENCE_T1_ACC = {3: 0.7856, 5: 0.7893}


def _report(store: dict, ctx: int) -> None:
    entries = [e for e in store.values() if e.get("t1_gold")]
    if not entries:
        print("No scoreable entries yet.")
        return

    n = len(entries)
    acc = sum(e["agree"] for e in entries) / n
    unparseable = sum(e["t1_pred"] == "UNKNOWN" for e in entries)

    print(f"\nOut-of-fold T1 on HLQC, ctx={ctx}: n={n}  accuracy={acc:.4f}")
    by_fold: Dict[int, List[dict]] = {}
    for e in entries:
        by_fold.setdefault(int(e.get("fold", -1)), []).append(e)
    for fold in sorted(by_fold):
        rows = by_fold[fold]
        print(
            f"  fold {fold}: n={len(rows):4d}  "
            f"accuracy={sum(r['agree'] for r in rows) / len(rows):.4f}"
        )
    if unparseable:
        print(f"  unparseable: {unparseable} ({unparseable / n:.1%})")

    ref = REFERENCE_T1_ACC.get(int(ctx))
    if ref is not None:
        gap = acc - ref
        print(
            f"\n  reference: ft_bare T1 on the evaluation corpus = {ref:.4f}\n"
            f"  gap: {gap:+.4f}"
        )
        if gap > 0.05:
            print(
                "  WARNING: out-of-fold accuracy sits well above the evaluation "
                "figure, so T2 will be trained on gentler T1 noise than it meets "
                "at inference. Read the teacher-forcing results as a LOWER BOUND."
            )
        else:
            print("  Exposure is calibrated; the contrast is interpretable.")


def cmd_report(args) -> None:
    store = load_oof_t1(args.ctx)
    if store:
        merged = oof_t1_path(args.ctx)
        merged.parent.mkdir(parents=True, exist_ok=True)
        with open(merged, "w") as f:
            json.dump(store, f, indent=2)
        print(f"Merged {len(store)} predictions -> {merged}")
    if not store:
        raise SystemExit(
            f"No out-of-fold store for ctx={args.ctx} at {oof_t1_path(args.ctx)}."
        )
    _report(store, args.ctx)

    # Coverage against the corpus the T2 arms will train on.
    from baseline.local_arm import load_config

    df = _load_df(load_config(args.overrides))
    covered = sum(1 for i in range(len(df)) if str(df.iloc[i]["corp_utt_idx"]) in store)
    print(f"\nCoverage: {covered}/{len(df)} training rows")
    if covered < len(df):
        missing = {f for f in range(args.folds)} - {
            int(e.get("fold", -1)) for e in store.values()
        }
        print(f"  incomplete; folds with no predictions: {sorted(missing) or 'none'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train-fold", help="train one fold's T1-only adapter")
    p_pred = sub.add_parser("predict-fold", help="predict one fold's held-out rows")
    p_rep = sub.add_parser("report", help="pooled accuracy and coverage")

    for p in (p_train, p_pred):
        p.add_argument("--fold", type=int, required=True)
    p_pred.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)

    for p in (p_train, p_pred, p_rep):
        p.add_argument("--ctx", type=int, default=None,
                       help="prior context volleys (default: config)")
        p.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
        p.add_argument("--limit", type=int, default=None,
                       help="cap rows processed (smoke tests)")
        p.add_argument("overrides", nargs="*",
                       help="OmegaConf dotlist overrides, e.g. inference.force_cpu=true")

    args = parser.parse_args()

    from baseline.local_arm import load_config

    if args.ctx is None:
        args.ctx = int(load_config(args.overrides).annotator.num_context_turns)
    else:
        args.overrides = list(args.overrides) + [
            f"annotator.num_context_turns={args.ctx}"
        ]

    {"train-fold": cmd_train_fold, "predict-fold": cmd_predict_fold,
     "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
