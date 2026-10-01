"""The `ag` arm: smart-retrieval few-shot on the local Qwen model.

Runs the identical two-call inference as baseline.local_arm's in-context arms,
except the few-shot exemplars are RETRIEVED per utterance (agentic.retriever)
instead of loaded from a frozen file. Generation, parsing, the CSV schema and the
diagnostics are all reused from baseline.local_arm, so the output
(qwen_ag_inf_<style>_ctx<N>.csv) is scored by baseline.eval as one more row.

Backbone defaults to the plain base model (0 adapters), making `ag` a clean
"fs but retrieved" comparison and letting the CPU smoke test run without any
trained adapter. Pass `--backbone ft_bare` (2 adapters, one per tier) or
`--backbone ft1mix_bare` (1 shared adapter, the main setting) to layer retrieval
on fine-tuned adapters instead (they must already be trained). The retriever
embeds the pool with adapters disabled, so the retrieval itself is identical
across backbones and only the model answering the prompt changes.

Usage:
    PYTHONPATH=src python -m agentic.arm predict --inf cot --ctx 5
    PYTHONPATH=src python -m agentic.arm predict --inf cot --ctx 5 --backbone ft_bare
    PYTHONPATH=src python -m agentic.arm predict --inf bare --ctx 5 --backbone ft1mix_bare

Smoke test on CPU with a tiny model:
    PYTHONPATH=src python -m agentic.arm predict --inf cot --ctx 5 --limit 4 \
        model.base_model=Qwen/Qwen2.5-0.5B-Instruct inference.force_cpu=true
"""
from __future__ import annotations

import argparse
import re
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from baseline.local_arm import (
    REPO_ROOT,
    STYLES,
    _report,
    _utc_now,
    _write_run_meta,
    load_config,
    resolve_adapters,
    structure_suffix_for,
)

ARM = "ag"
BACKBONES = ("none", "ft_bare", "ft1mix_bare")
RETRIEVERS = ("tfidf", "qwen")


def _run_tag(retriever: str, backbone: str) -> str:
    """Filename tag that keeps each config's result CSV distinct."""
    return f"{retriever}_{backbone}"


def _model_tag(base_model: str) -> str:
    """Cache key for pool embeddings (base-model representation, backbone-agnostic)."""
    return re.sub(r"[^A-Za-z0-9]+", "_", base_model).strip("_")


def cmd_predict(args) -> None:
    from agentic.retriever import RetrievalFewshotProvider
    from automisc_ft.data import load_manual
    from automisc_ft.infer import TieredAnnotator

    started = _utc_now()
    cfg = load_config(args.overrides)
    ctx = args.ctx
    style = args.inf
    suffix = structure_suffix_for(style)

    df = load_manual(REPO_ROOT / cfg.dataset.eval_csv)
    # The evaluation CSV ships predictions from an earlier run; produce our own.
    df = df.drop(columns=[c for c in df.columns if c.endswith("_auto")])

    limit = args.limit if args.limit is not None else cfg.limit
    n_total = len(df) if limit in (None, "null") else min(int(limit), len(df))

    # Each (retriever, backbone) config writes its own CSV so runs never collide.
    run_tag = _run_tag(args.retriever, args.backbone)
    save_path = (REPO_ROOT / cfg.paths.output_dir
                 / f"{cfg.tier}_{ARM}_{run_tag}_inf_{style}_ctx{ctx}.csv")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from an existing checkpoint, exactly like local_arm.cmd_predict.
    utt_checkpoint = -1
    existing_df = None
    if save_path.exists():
        existing_df = pd.read_csv(save_path)
        if not existing_df.empty:
            utt_checkpoint = existing_df["corp_utt_idx"].max()
            print(f"Resuming: {len(existing_df)} rows done "
                  f"(up to corp_utt_idx {utt_checkpoint})")

    positions = [
        i for i in range(n_total) if df.iloc[i]["corp_utt_idx"] > utt_checkpoint
    ]
    if not positions:
        print(f"Nothing to do; {save_path} is already complete.")
        return

    # Backbone: base model by default, the ft_bare adapter pair, or the single
    # shared ft1mix_bare adapter. resolve_adapters returns whichever applies.
    t1_adapter = t2_adapter = shared_adapter = None
    if args.backbone != "none":
        t1_adapter, t2_adapter, shared_adapter, retrain_cmd = resolve_adapters(
            cfg, args.backbone, ctx
        )
        for path in (t1_adapter, t2_adapter, shared_adapter):
            if path is not None and not path.exists():
                raise SystemExit(f"Missing adapter {path}. Train it first:\n  {retrain_cmd}")

    # ag prompts are long (multiple retrieved exemplars), so borrow the fs window.
    mil = cfg.inference.max_input_len
    max_input_len = int(mil.get(ARM, mil["fs"]))

    ag_cfg = cfg.get("agentic", {}) or {}
    rare_reserve = args.rare_reserve
    if rare_reserve is None and "rare_reserve" in ag_cfg:
        rare_reserve = int(ag_cfg["rare_reserve"])
    retriever = RetrievalFewshotProvider(
        ctx=ctx,
        method=args.retriever,
        k_t1=int(ag_cfg.get("k_t1", 8)),
        k_t2=int(ag_cfg.get("k_t2", 6)),
        rare_reserve=rare_reserve,
        model_tag=_model_tag(cfg.model.base_model),
    )

    cond = f"{cfg.tier}_{ARM}_{run_tag}_inf_{style}"
    print(
        f"Condition={cond} ctx={ctx} retriever={args.retriever} backbone={args.backbone} "
        f"rare_reserve={rare_reserve} model={cfg.model.base_model} "
        f"max_input_len={max_input_len} max_new_tokens={cfg.inference.max_new_tokens} "
        f"n={len(positions)} -> {save_path}"
    )

    annotator = TieredAnnotator(
        base_model=cfg.model.base_model,
        t1_adapter_dir=str(t1_adapter) if t1_adapter else None,
        t2_adapter_dir=str(t2_adapter) if t2_adapter else None,
        shared_adapter_dir=str(shared_adapter) if shared_adapter else None,
        force_cpu=bool(cfg.inference.force_cpu),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        max_new_tokens=int(cfg.inference.max_new_tokens),
        max_input_len=max_input_len,
        structure_suffix=suffix,
        fewshot_provider=retriever,
    )
    # qwen backend embeds the HLQC pool with the just-loaded model (adapters
    # disabled), so it must attach after the annotator is built.
    if args.retriever == "qwen":
        retriever.attach_model(annotator.model, annotator.tokenizer)

    context_mode = cfg.annotator.context_mode
    restrict_t2 = bool(cfg.annotator.get("restrict_t2_to_group", False))
    output_rows: List[Dict] = []

    def save() -> None:
        nonlocal existing_df, output_rows
        if not output_rows:
            return
        out = pd.DataFrame(output_rows)
        if existing_df is not None:
            out = pd.concat([existing_df, out], ignore_index=True)
        out.to_csv(save_path, index=False)
        existing_df = out
        output_rows = []

    try:
        for pos in tqdm(positions, desc=cond, unit="utt"):
            row = df.iloc[pos]
            # This is the one line that makes retrieval per-utterance.
            retriever.set_query(row["utt_text"], row["speaker"])
            pred = annotator.predict_row(df, pos, context_mode, ctx, restrict_t2)
            output_rows.append({
                **row.to_dict(),
                "t1_label_auto": pred["t1_pred"],
                "t2_label_auto": pred["t2_pred"],
                "t1_raw": pred["t1_raw"],
                "t2_raw": pred["t2_raw"],
                "t1_emitted_rationale": pred["t1_emitted_rationale"],
                "t2_emitted_rationale": pred["t2_emitted_rationale"],
                "t1_n_prompt_tokens": pred["t1_n_prompt_tokens"],
                "t2_n_prompt_tokens": pred["t2_n_prompt_tokens"],
                "t1_n_gen_tokens": pred["t1_n_gen_tokens"],
                "t2_n_gen_tokens": pred["t2_n_gen_tokens"],
            })
            if len(output_rows) >= int(cfg.checkpoint_every):
                save()
    finally:
        save()
        annotator.close()

    _report(existing_df, cond, save_path, int(cfg.inference.max_new_tokens), max_input_len)
    _write_run_meta(save_path, f"{ARM}_{run_tag}", style, ctx, existing_df, started)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="annotate the evaluation set with retrieval few-shot")
    p.add_argument("--inf", choices=STYLES, required=True, help="inference prompt style")
    p.add_argument("--retriever", choices=RETRIEVERS, default="qwen",
                   help="tfidf = TF-IDF cosine (+linguistic re-rank); "
                        "qwen = mean-pooled Qwen hidden-state embeddings (semantic, default)")
    p.add_argument("--backbone", choices=BACKBONES, default="none",
                   help="none = plain base model (fs-comparison, default); "
                        "ft_bare = layer retrieval on the fine-tuned adapters")
    p.add_argument("--rare-reserve", dest="rare_reserve", type=int, default=None,
                   help="0 = pure similarity (no forced rare-code coverage); "
                        "default (unset) = guarantee every candidate label")
    p.add_argument("--ctx", type=int, default=None,
                   help="prior context volleys (default: config)")
    p.add_argument("--limit", type=int, default=None, help="cap rows processed (smoke tests)")
    p.add_argument("overrides", nargs="*",
                   help="OmegaConf dotlist overrides, e.g. inference.force_cpu=true")

    args = parser.parse_args()
    if args.ctx is None:
        args.ctx = int(load_config(args.overrides).annotator.num_context_turns)
    else:
        args.overrides = list(args.overrides) + [f"annotator.num_context_turns={args.ctx}"]

    {"predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()
