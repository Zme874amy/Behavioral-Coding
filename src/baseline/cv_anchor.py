"""In-distribution anchor: conversation-level CV on HLQC for the 1-adapter arm.

Every other number in the grid is HLQC-trained and scored on MIV6.3A. This
answers the reference question those numbers need: what does the SAME model
score on the corpus it was trained on, when each utterance is predicted by an
adapter that never saw its conversation? It is the top rung of the
generalization ladder (HLQC CV -> MIV6.3A -> AnnoMI -> Welivita).

The recipe is `ft1mix_bare` exactly as `baseline.local_arm` trains and predicts
it -- one shared LoRA over both tiers (`train_single_adapter`, regime `mixed`),
label-only prompts, the grid's config and context length -- restricted per fold.
The earlier anchor (`automisc_ft/main.py`) ran automisc_ft's own recipe (2
adapters, ctx3, original prompts), so it was never directly comparable to the
grid; this one is.

Folds come from `baseline.oof_t1._split` (conversation-level, fixed FOLD_SEED),
so they are the same partition the out-of-fold T1 store uses. HLQC holds 10
conversations, so K=5 puts 2 in each fold.

One job per fold, and one output file per fold (folds run concurrently, so a
shared file would race -- see `oof_t1.oof_t1_fold_path`):

    PYTHONPATH=src python -m baseline.cv_anchor train-fold   --ctx 5 --fold 0
    PYTHONPATH=src python -m baseline.cv_anchor predict-fold --ctx 5 --fold 0
    PYTHONPATH=src python -m baseline.cv_anchor report       --ctx 5

Smoke test on CPU with a tiny model (MPS off: the Mac disables grad clipping):
    PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=0 python -m baseline.cv_anchor \
        train-fold --ctx 5 --fold 0 --limit 16 \
        model.base_model=Qwen/Qwen2.5-0.5B-Instruct inference.force_cpu=true \
        training.num_train_epochs=1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from baseline.oof_t1 import FOLD_SEED, DEFAULT_FOLDS, _load_df, _split

REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_DIR = REPO_ROOT / "data" / "fine_tuning" / "cv_hlqc"
OUT_DIR = REPO_ROOT / "data" / "annotated" / "cv_hlqc"
REPORT_DIR = REPO_ROOT / "outputs" / "cv_hlqc"

ARM = "ft1mix_bare"
REGIME = "mixed"
STYLE = "bare"


def fold_adapter_root(ctx: int, fold: int) -> Path:
    return ADAPTER_DIR / f"ctx{ctx}" / f"fold{fold}"


def fold_result_path(cfg, ctx: int, fold: int) -> Path:
    return OUT_DIR / f"{cfg.tier}_{ARM}_inf_{STYLE}_ctx{ctx}_fold{fold}.csv"


# -----------------------------------------------------------------------------
# train-fold
# -----------------------------------------------------------------------------
def cmd_train_fold(args) -> None:
    from automisc_ft.train import train_single_adapter
    from baseline.local_arm import load_config, structure_suffix_for

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
        f"CV anchor fold {fold}/{n_folds} ctx={ctx} arm={ARM}: training on "
        f"{len(train_pos)} rows, holding out {len(pred_pos)} -> {out_dir}"
    )
    # Same call local_arm.cmd_train makes for a mixed-regime bare target.
    shared = train_single_adapter(
        cfg, df, train_pos, out_dir, structure_suffix_for(STYLE), None, REGIME
    )
    meta = {
        "kind": "cv_anchor",
        "arm": ARM,
        "regime": REGIME,
        "fold": fold,
        "n_folds": n_folds,
        "fold_seed": FOLD_SEED,
        "num_context_turns": ctx,
        "base_model": cfg.model.base_model,
        "train_csv": cfg.dataset.train_csv,
        "n_train_rows": len(train_pos),
        "n_heldout_rows": len(pred_pos),
        "n_epochs": cfg.training.num_train_epochs,
        "learning_rate": cfg.training.learning_rate,
        "training_seed": cfg.training.seed,
        "shared_adapter": str(shared),
    }
    (out_dir / "train_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"Fold adapter saved -> {shared}")


# -----------------------------------------------------------------------------
# predict-fold
# -----------------------------------------------------------------------------
def cmd_predict_fold(args) -> None:
    from automisc_ft.infer import TieredAnnotator
    from baseline.local_arm import load_config, structure_suffix_for

    cfg = load_config(args.overrides)
    ctx, fold, n_folds = args.ctx, args.fold, args.folds

    meta_path = fold_adapter_root(ctx, fold) / "train_metadata.json"
    if not meta_path.exists():
        raise SystemExit(
            f"No trained adapter for fold {fold} ({meta_path} missing). Train it first:\n"
            f"  PYTHONPATH=src python -m baseline.cv_anchor train-fold --ctx {ctx} --fold {fold}"
        )
    adapter = Path(json.loads(meta_path.read_text())["shared_adapter"])

    df = _load_df(cfg)
    _, pred_pos = _split(df, n_folds, fold)
    if args.limit:
        pred_pos = pred_pos[: int(args.limit)]

    out_path = fold_result_path(cfg, ctx, fold)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(out_path) if out_path.exists() else None
    done = set() if existing is None else set(existing["corp_utt_idx"].astype(int))
    if done:
        print(f"Resuming from {out_path}: {len(done)} utterances already predicted")
    # Held-out positions are whole conversations, not a contiguous index range,
    # so resume by membership rather than by a max-index checkpoint.
    todo = [p for p in pred_pos if int(df.iloc[p]["corp_utt_idx"]) not in done]
    if not todo:
        print(f"Fold {fold} already complete in {out_path}.")
        return

    suffix = structure_suffix_for(STYLE)
    max_input_len = int(cfg.inference.max_input_len[ARM])
    annotator = TieredAnnotator(
        base_model=cfg.model.base_model,
        shared_adapter_dir=str(adapter),
        force_cpu=bool(cfg.inference.force_cpu),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        max_new_tokens=int(cfg.inference.max_new_tokens),
        max_input_len=max_input_len,
        structure_suffix=suffix,
    )
    context_mode = cfg.annotator.context_mode
    restrict_t2 = bool(cfg.annotator.get("restrict_t2_to_group", False))
    rows: List[Dict] = []

    def save() -> None:
        nonlocal existing, rows
        if not rows:
            return
        out = pd.DataFrame(rows)
        if existing is not None:
            out = pd.concat([existing, out], ignore_index=True)
        out.to_csv(out_path, index=False)
        existing, rows = out, []

    print(f"Predicting fold {fold}/{n_folds} ctx={ctx}: {len(todo)} rows -> {out_path}")
    try:
        for pos in tqdm(todo, desc=f"cv fold{fold}", unit="utt"):
            row = df.iloc[pos]
            pred = annotator.predict_row(df, pos, context_mode, ctx, restrict_t2)
            rows.append({
                **row.to_dict(),
                "fold": int(fold),
                "t1_label_auto": pred["t1_pred"],
                "t2_label_auto": pred["t2_pred"],
                "t1_raw": pred["t1_raw"],
                "t2_raw": pred["t2_raw"],
                "t1_n_prompt_tokens": pred["t1_n_prompt_tokens"],
                "t2_n_prompt_tokens": pred["t2_n_prompt_tokens"],
            })
            if len(rows) >= int(cfg.checkpoint_every):
                save()
    finally:
        save()
        annotator.close()
    print(f"Fold {fold} done -> {out_path}")


# -----------------------------------------------------------------------------
# score-fold -- calibration data for self-training
# -----------------------------------------------------------------------------
def fold_scored_path(cfg, ctx: int, fold: int) -> Path:
    return OUT_DIR / f"scored_{cfg.tier}_{ARM}_inf_{STYLE}_ctx{ctx}_fold{fold}.csv"


def cmd_score_fold(args) -> None:
    """Score each held-out HLQC utterance with a full distribution over codes.

    Same fold adapter and prompts as `predict-fold`, but through
    `TieredAnnotator.predict_row_scored`, so every row carries p(code) for all
    candidates plus the gold labels. Because each row is scored by an adapter
    that never saw its conversation, this is out-of-fold evidence of how often
    the student is right at a given confidence -- what `selftrain.calibrate`
    turns into per-code pseudo-label thresholds.
    """
    from automisc_ft.infer import TieredAnnotator
    from baseline.local_arm import load_config, structure_suffix_for

    cfg = load_config(args.overrides)
    ctx, fold, n_folds = args.ctx, args.fold, args.folds
    meta_path = fold_adapter_root(ctx, fold) / "train_metadata.json"
    if not meta_path.exists():
        raise SystemExit(f"No trained adapter for fold {fold} ({meta_path} missing).")
    adapter = Path(json.loads(meta_path.read_text())["shared_adapter"])

    df = _load_df(cfg)
    _, pred_pos = _split(df, n_folds, fold)
    if args.limit:
        pred_pos = pred_pos[: int(args.limit)]
    out_path = fold_scored_path(cfg, ctx, fold)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(out_path) if out_path.exists() else None
    done = set() if existing is None else set(existing["corp_utt_idx"].astype(int))
    todo = [p for p in pred_pos if int(df.iloc[p]["corp_utt_idx"]) not in done]
    if not todo:
        print(f"Fold {fold} already scored in {out_path}.")
        return

    annotator = TieredAnnotator(
        base_model=cfg.model.base_model,
        shared_adapter_dir=str(adapter),
        force_cpu=bool(cfg.inference.force_cpu),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        max_new_tokens=int(cfg.inference.max_new_tokens),
        max_input_len=int(cfg.inference.max_input_len[ARM]),
        structure_suffix=structure_suffix_for(STYLE),
    )
    context_mode = cfg.annotator.context_mode
    rows: List[Dict] = []

    def save() -> None:
        nonlocal existing, rows
        if not rows:
            return
        out = pd.DataFrame(rows)
        if existing is not None:
            out = pd.concat([existing, out], ignore_index=True)
        out.to_csv(out_path, index=False)
        existing, rows = out, []

    print(f"Scoring fold {fold}/{n_folds} ctx={ctx}: {len(todo)} rows -> {out_path}")
    try:
        for pos in tqdm(todo, desc=f"score fold{fold}", unit="utt"):
            row = df.iloc[pos]
            r = annotator.predict_row_scored(df, pos, context_mode, ctx)
            rows.append({
                "conv_id": row["conv_id"], "corp_utt_idx": int(row["corp_utt_idx"]),
                "speaker": row["speaker"], "fold": int(fold),
                "t1_label_GT": row.get("t1_label_GT"), "t2_label_GT": row.get("t2_label_GT"),
                "t1_label_auto": r["t1_pred"], "t2_label_auto": r["t2_pred"],
                "t1_conf": r["t1_conf"], "t2_conf": r["t2_conf"], "confidence": r["confidence"],
                "t1_probs": json.dumps(r["t1_probs"]), "t2_probs": json.dumps(r["t2_probs"]),
            })
            if len(rows) >= int(cfg.checkpoint_every):
                save()
    finally:
        save()
        annotator.close()
    print(f"Fold {fold} scored -> {out_path}")


# -----------------------------------------------------------------------------
# report
# -----------------------------------------------------------------------------
def cmd_report(args) -> None:
    from automisc_ft.eval import compute_condition_metrics
    from baseline.eval import score
    from baseline.local_arm import load_config

    cfg = load_config(args.overrides)
    ctx, n_folds = args.ctx, args.folds
    df = _load_df(cfg)

    parts = []
    for fold in range(n_folds):
        p = fold_result_path(cfg, ctx, fold)
        if not p.exists():
            raise SystemExit(f"Fold {fold} has no predictions yet ({p}).")
        parts.append(pd.read_csv(p))
    pooled = pd.concat(parts, ignore_index=True)

    # Validity: every HLQC utterance predicted exactly once, by its own fold.
    dup = int(pooled["corp_utt_idx"].duplicated().sum())
    missing = len(df) - pooled["corp_utt_idx"].nunique()
    if dup or missing:
        raise SystemExit(
            f"Pooled folds are not a partition of HLQC: {dup} duplicated and "
            f"{missing} missing utterances. Re-run the incomplete fold(s)."
        )
    print(f"Pooled {len(pooled)} out-of-fold predictions over "
          f"{pooled['conv_id'].nunique()} conversations ({n_folds} folds).")

    out: Dict = {"arm": ARM, "ctx": ctx, "n_folds": n_folds, "fold_seed": FOLD_SEED,
                 "n": int(len(pooled)), "scores": []}
    for level in ("t1", "t2"):
        gt, pred = pooled[f"{level}_label_GT"], pooled[f"{level}_label_auto"]
        valid = gt.notna() & pred.notna()
        for scope in ("all", "counsellor", "client"):
            mask = valid if scope == "all" else valid & (pooled["speaker"] == scope)
            if not mask.any():
                continue
            s = score(gt[mask], pred[mask], level, conv=pooled["conv_id"][mask])
            out["scores"].append({"level": level.upper(), "scope": scope, **s})

    # The same per-speaker breakdown the 2-adapter anchor printed, so the two
    # anchors can be read side by side.
    paper_view = compute_condition_metrics(pooled, "t1_label_auto", "t2_label_auto")
    out["paper_view"] = paper_view

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"{ARM}_ctx{ctx}_report.json"
    report_path.write_text(json.dumps(out, indent=2, default=float))
    pooled.to_csv(REPORT_DIR / f"{ARM}_ctx{ctx}_pooled.csv", index=False)

    print(f"\n{ARM} on HLQC, {n_folds}-fold conversation-level CV "
          "(95% conversation-clustered CI):")
    for r in out["scores"]:
        acc = f"{r['accuracy']:.3f} [{r.get('accuracy_lo', float('nan')):.3f}–{r.get('accuracy_hi', float('nan')):.3f}]"
        f1 = f"{r['f1_macro_gold']:.3f} [{r.get('f1_macro_gold_lo', float('nan')):.3f}–{r.get('f1_macro_gold_hi', float('nan')):.3f}]"
        print(f"  {r['level']} {r['scope']:10s} n={r['n']:4d}  acc {acc}  macro-F1(gold) {f1}")
    print(f"\nWrote {report_path}")


def main() -> None:
    from baseline.local_arm import load_config

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, help_ in (
        ("train-fold", "train the shared adapter on the other folds"),
        ("predict-fold", "predict the held-out fold with that adapter"),
        ("score-fold", "score the held-out fold with full code distributions"),
        ("report", "pool the folds and score them"),
    ):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--ctx", type=int, default=None,
                       help="prior context volleys (default: config)")
        p.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
        if name != "report":
            p.add_argument("--fold", type=int, required=True)
            p.add_argument("--limit", type=int, default=None,
                           help="cap rows processed (smoke tests)")
        p.add_argument("overrides", nargs="*",
                       help="OmegaConf dotlist overrides, e.g. inference.force_cpu=true")
    args = parser.parse_args()

    # Same context handling as local_arm.main, so prompts match the grid.
    if args.ctx is None:
        args.ctx = int(load_config(args.overrides).annotator.num_context_turns)
    else:
        args.overrides = list(args.overrides) + [f"annotator.num_context_turns={args.ctx}"]

    {"train-fold": cmd_train_fold, "predict-fold": cmd_predict_fold,
     "score-fold": cmd_score_fold, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
