"""The `agent` arm: a bounded structured agent for MISC coding (agentic.agent).

Unlike the `ag` arm (retrieval-augmented single pass), here the model directs a
bounded loop -- calling tools, self-critiquing against the codebook, and revising
(agentic.agent.MISCAgent). Reuses the arm scaffolding for config, adapters, CSV
schema, checkpoint/resume, and diagnostics, so the result is scored by the same
harness. Writes qwen_agent_<retriever>_<backbone>_inf_cot_ctx<N>.csv plus a
trajectory sidecar (.jsonl) for the first rows so decisions are auditable.

Backbone defaults to `none` (the base instruct model): the `ft_bare` adapters are
tuned to emit bare labels and fight multi-step JSON tool-use, so base is the right
reasoner for an agent. `--tail-only` restricts to rows whose gold T2 is a rare
code (SU/EC/AF/GI) for a cheap pilot.

Smoke test on CPU with a tiny model (validates loop mechanics, not quality):
    PYTHONPATH=src python -m agentic.agent_arm predict --backbone none --ctx 5 --limit 4 \
        model.base_model=Qwen/Qwen2.5-0.5B-Instruct inference.force_cpu=true
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from agentic.arm import BACKBONES, RETRIEVERS, _model_tag, _run_tag
from agentic.retriever import RARE_T2, RetrievalFewshotProvider
from baseline.local_arm import (
    REPO_ROOT,
    _report,
    _utc_now,
    _write_run_meta,
    load_config,
    resolve_adapters,
)

ARM = "agent"
AGENT_MIN_NEW_TOKENS = 384  # reasoning + JSON needs more room than a bare label


def cmd_predict(args) -> None:
    from agentic.agent import MISCAgent
    from automisc_ft.data import load_manual
    from automisc_ft.infer import TieredAnnotator

    started = _utc_now()
    cfg = load_config(args.overrides)
    ctx = args.ctx
    style = "cot"

    df = load_manual(REPO_ROOT / cfg.dataset.eval_csv)
    df = df.drop(columns=[c for c in df.columns if c.endswith("_auto")])
    if args.tail_only:
        df = df[df["t2_label_GT"].isin(RARE_T2)].reset_index(drop=True)
        print(f"tail-only pilot: {len(df)} rows with gold T2 in {RARE_T2}")

    limit = args.limit if args.limit is not None else cfg.limit
    n_total = len(df) if limit in (None, "null") else min(int(limit), len(df))

    run_tag = _run_tag(args.retriever, args.backbone) + ("_tail" if args.tail_only else "")
    save_path = (REPO_ROOT / cfg.paths.output_dir
                 / f"{cfg.tier}_{ARM}_{run_tag}_inf_{style}_ctx{ctx}.csv")
    traj_path = save_path.with_suffix(".traj.jsonl")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    utt_checkpoint = -1
    existing_df = None
    if save_path.exists():
        existing_df = pd.read_csv(save_path)
        if not existing_df.empty:
            utt_checkpoint = existing_df["corp_utt_idx"].max()
            print(f"Resuming: {len(existing_df)} rows done (up to {utt_checkpoint})")

    positions = [i for i in range(n_total) if df.iloc[i]["corp_utt_idx"] > utt_checkpoint]
    if not positions:
        print(f"Nothing to do; {save_path} is already complete.")
        return

    t1_adapter = t2_adapter = None
    if args.backbone == "ft_bare":
        t1_adapter, t2_adapter, _s, retrain = resolve_adapters(cfg, "ft_bare", ctx)
        for p in (t1_adapter, t2_adapter):
            if p is not None and not p.exists():
                raise SystemExit(f"Missing adapter {p}. Train it first:\n  {retrain}")

    mil = cfg.inference.max_input_len
    max_input_len = int(mil.get("ag", mil["fs"]))
    max_new_tokens = max(int(cfg.inference.max_new_tokens), AGENT_MIN_NEW_TOKENS)
    ag_cfg = cfg.get("agentic", {}) or {}

    annotator = TieredAnnotator(
        base_model=cfg.model.base_model,
        t1_adapter_dir=str(t1_adapter) if t1_adapter else None,
        t2_adapter_dir=str(t2_adapter) if t2_adapter else None,
        force_cpu=bool(cfg.inference.force_cpu),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        max_new_tokens=max_new_tokens,
        max_input_len=max_input_len,
        structure_suffix="",
        fewshot_provider=None,
    )
    retriever = RetrievalFewshotProvider(
        ctx=ctx, method=args.retriever, model_tag=_model_tag(cfg.model.base_model))
    if args.retriever == "qwen":
        retriever.attach_model(annotator.model, annotator.tokenizer)

    agent = MISCAgent(
        annotator, retriever,
        max_tool_calls=int(ag_cfg.get("max_tool_calls", 3)),
        max_revise=int(ag_cfg.get("max_revise", 2)),
        max_steps=int(ag_cfg.get("max_steps", 6)),
    )

    cond = f"{cfg.tier}_{ARM}_{run_tag}_inf_{style}"
    print(f"Condition={cond} ctx={ctx} retriever={args.retriever} backbone={args.backbone} "
          f"model={cfg.model.base_model} max_new_tokens={max_new_tokens} "
          f"n={len(positions)} -> {save_path}")

    context_mode = cfg.annotator.context_mode
    output_rows: List[Dict] = []
    traj_records: List[Dict] = []
    n_traj = 0            # cap trajectory logging to the first rows (interpretability)
    TRAJ_LIMIT = 30

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
        if traj_records:
            with open(traj_path, "a") as f:
                for rec in traj_records:
                    f.write(json.dumps(rec) + "\n")
            traj_records.clear()

    try:
        for pos in tqdm(positions, desc=cond, unit="utt"):
            res = agent.code_row(df, pos, context_mode, ctx)
            row = df.iloc[pos].to_dict()
            output_rows.append({
                **row,
                "t1_label_auto": res.t1,
                "t2_label_auto": res.t2,
                "t1_raw": res.rationale,
                "t2_raw": res.rationale,
                "t1_emitted_rationale": False,
                "t2_emitted_rationale": bool(res.rationale),
                "t1_n_prompt_tokens": 0, "t2_n_prompt_tokens": 0,
                "t1_n_gen_tokens": 0, "t2_n_gen_tokens": 0,
                "agent_steps": res.n_steps, "agent_tool_calls": res.n_tool_calls,
                "agent_revises": res.n_revises,
                "agent_prompt_tokens": res.n_prompt_tokens,
                "agent_gen_tokens": res.n_gen_tokens,
            })
            # Keep trajectories for the first rows only (interpretability).
            if n_traj < TRAJ_LIMIT:
                traj_records.append({"corp_utt_idx": int(row["corp_utt_idx"]),
                                     "t1": res.t1, "t2": res.t2,
                                     "trajectory": res.trajectory})
                n_traj += 1
            if len(output_rows) >= int(cfg.checkpoint_every):
                save()
    finally:
        save()
        annotator.close()

    _report(existing_df, cond, save_path, max_new_tokens, max_input_len)
    _write_run_meta(save_path, f"{ARM}_{run_tag}", style, ctx, existing_df, started)
    print(f"Trajectories -> {traj_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict", help="annotate the evaluation set with the agent")
    p.add_argument("--retriever", choices=RETRIEVERS, default="tfidf",
                   help="backend for the agent's retrieve tool (default tfidf, cheap)")
    p.add_argument("--backbone", choices=BACKBONES, default="none",
                   help="none = base instruct model (recommended for reasoning); "
                        "ft_bare = the label-only adapters (fights JSON tool-use)")
    p.add_argument("--tail-only", dest="tail_only", action="store_true",
                   help="pilot on rows whose gold T2 is a rare code (SU/EC/AF/GI)")
    p.add_argument("--ctx", type=int, default=None, help="prior context volleys")
    p.add_argument("--limit", type=int, default=None, help="cap rows (smoke tests)")
    p.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")

    args = parser.parse_args()
    if args.ctx is None:
        args.ctx = int(load_config(args.overrides).annotator.num_context_turns)
    else:
        args.overrides = list(args.overrides) + [f"annotator.num_context_turns={args.ctx}"]
    cmd_predict(args)


if __name__ == "__main__":
    main()
