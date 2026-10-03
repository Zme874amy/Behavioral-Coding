"""Screen candidate teacher models on labelled HLQC -- fast, model-agnostic.

Each candidate (served by vLLM, see scripts/mlerp_teacher_screen.slurm) labels
the HLQC utterances with the same two-call T1 -> T2 flow and the same prompts as
the student (label-only `_bare` templates, optionally the frozen HLQC few-shot
exemplars), decoded greedily. Requests are sent concurrently, so vLLM batches
them: a screen takes minutes, not the hours of per-row likelihood scoring, and
it works for reasoning models (gpt-oss, thinking modes) whose answers come after
a reasoning trace and cannot be likelihood-scored directly.

What it reports is what decides a teacher, measured on labelled data only:
  * T1/T2 accuracy and macro-F1 overall and per speaker, with
    conversation-clustered CIs -- compared against the student's out-of-fold
    T2 macro-F1 (0.399);
  * per-code precision/recall on counsellor T2, with the TAIL codes (the ones
    the student cannot label reliably) first: a teacher only adds value for
    self-training if it is reliable where the student is not;
  * unparseable rate and seconds per row.
The winner then gets the full likelihood calibration (`label_pool --confidence
likelihood` + `selftrain.calibrate`) for its per-code thresholds.

Usage (inside the slurm job, with the vLLM server up):
    PYTHONPATH=src python -m selftrain.screen_teachers --model gemma-4-31B-it \\
        --pool data/manual/HLQC_balanced_manual.csv --fewshot --exclude-exemplars \\
        --out data/selftrain/teacher/screen_gemma4_31b.csv
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
# Codes the student labels below 80% precision on HLQC (calibration_ft1mix_ctx5):
# where a teacher would have to be better to be worth distilling from.
TAIL = ["AF", "GI", "SU", "EC", "CR", "RCW", "CO", "DI", "ADW", "FA", "RF"]


def _chat(client, model: str, messages, max_tokens: int, reasoning: Optional[str]) -> str:
    extra: Dict = {"chat_template_kwargs": {"enable_thinking": False}}
    if reasoning:
        extra = {"reasoning_effort": reasoning}
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=model, messages=messages, temperature=0.0,
                max_tokens=max_tokens, extra_body=extra)
            return r.choices[0].message.content or ""
        except Exception as e:  # transient server hiccups: back off and retry
            if attempt == 3:
                return f"__ERROR__ {type(e).__name__}"
            time.sleep(2 * (attempt + 1))
    return ""


def _chat_ollama(base: str, model: str, messages, max_tokens: int, think, num_ctx: int,
                 prompt_tokens: List[int]) -> str:
    """Ollama's native /api/chat (MLeRP's shared GGUF store, /apps/ollama/models).

    Used instead of its OpenAI shim because only the native API takes `think`
    (False, or an effort level such as "low" for gpt-oss) and `num_ctx` per
    request; the shim would silently truncate long few-shot prompts at the
    server's default context. The prompt token count is recorded so truncation
    shows up in the report rather than as a quietly worse score.
    """
    import requests
    body = {"model": model, "messages": messages, "stream": False,
            "options": {"temperature": 0.0, "num_predict": max_tokens, "num_ctx": num_ctx}}
    if think is not None:
        body["think"] = think
    for attempt in range(4):
        try:
            r = requests.post(f"{base}/api/chat", json=body, timeout=1800)
            if r.status_code == 400 and "think" in body and "think" in r.text.lower():
                body.pop("think")  # model has no thinking switch: ask without it
                continue
            r.raise_for_status()
            d = r.json()
            prompt_tokens.append(int(d.get("prompt_eval_count") or 0))
            return d["message"].get("content") or ""
        except Exception as e:
            if attempt == 3:
                return f"__ERROR__ {type(e).__name__}"
            time.sleep(2 * (attempt + 1))
    return ""


def main(argv=None) -> None:
    from automisc_ft.data import (
        build_messages_t1, build_messages_t2, load_manual, t1_codes_for_speaker,
        t2_codes_for_speaker,
    )
    from automisc_ft.infer import parse_label
    from baseline.eval import score
    from components.utils import get_vllm_client
    from selftrain.label_pool import _exemplar_texts, _norm_text

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--pool", default="data/manual/HLQC_balanced_manual.csv")
    ap.add_argument("--ctx", type=int, default=5)
    ap.add_argument("--fewshot", action="store_true")
    ap.add_argument("--exclude-exemplars", action="store_true")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--reasoning", default=None, help="reasoning effort for reasoning models, e.g. low")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--backend", choices=["openai", "ollama"], default="openai",
                    help="openai = vLLM's OpenAI server; ollama = Ollama's native API")
    ap.add_argument("--think", default=None,
                    help="ollama only: 'false' to disable thinking, or an effort level (low/medium/high)")
    ap.add_argument("--num-ctx", type=int, default=12288, help="ollama only: context window per request")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    think = None if args.think is None else (False if args.think.lower() == "false" else args.think)

    df = load_manual(REPO_ROOT / args.pool)
    fewshot = None
    if args.fewshot:
        from baseline.fewshot import exemplars_path, load_exemplars
        from baseline.local_arm import _make_fewshot_provider
        fewshot = _make_fewshot_provider(load_exemplars(exemplars_path(args.ctx)), rationales=False)
    skip = set()
    if args.exclude_exemplars:
        ex = _exemplar_texts(args.ctx)
        skip = set(df.index[df["utt_text"].map(_norm_text).isin(ex)])
    pos = [i for i in range(len(df)) if i not in skip and pd.notna(df.iloc[i]["t2_label_GT"])]
    if args.limit:
        pos = pos[: args.limit]
    prompt_tokens: List[int] = []
    if args.backend == "ollama":
        import os
        base = (args.base_url or os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")).rstrip("/")
        ask = lambda msgs: _chat_ollama(base, args.model, msgs, args.max_tokens, think,  # noqa: E731
                                        args.num_ctx, prompt_tokens)
    else:
        client = get_vllm_client(args.base_url)
        ask = lambda msgs: _chat(client, args.model, msgs, args.max_tokens, args.reasoning)  # noqa: E731
    fs =(lambda spk, tier, t1: fewshot(spk, tier, t1)) if fewshot else (lambda spk, tier, t1: None)

    def run(tier: str, t1_of: Dict[int, str]) -> Dict[int, str]:
        def one(p):
            spk = df.iloc[p]["speaker"]
            if tier == "t1":
                msgs = build_messages_t1(df, p, "interval", args.ctx, "_bare", fs(spk, "t1", None))
            else:
                msgs = build_messages_t2(df, p, t1_of[p], "interval", args.ctx, "_bare", fs(spk, "t2", t1_of[p]))
            return p, ask(msgs)
        with ThreadPoolExecutor(args.concurrency) as pool:
            return dict(pool.map(one, pos))

    t0 = time.time()
    raw1 = run("t1", {})
    t1 = {p: parse_label(raw1[p], t1_codes_for_speaker(df.iloc[p]["speaker"])) for p in pos}
    # As in the pipeline: an unparseable T1 falls back to the speaker's first group.
    t1_for = {p: (t1[p] if t1[p] != "UNKNOWN" else t1_codes_for_speaker(df.iloc[p]["speaker"])[0]) for p in pos}
    raw2 = run("t2", t1_for)
    t2 = {p: parse_label(raw2[p], t2_codes_for_speaker(df.iloc[p]["speaker"])) for p in pos}
    secs = (time.time() - t0) / max(1, len(pos))

    out = df.iloc[pos][["conv_id", "corp_utt_idx", "speaker", "t1_label_GT", "t2_label_GT"]].copy()
    out["t1_label_auto"] = [t1[p] for p in pos]
    out["t2_label_auto"] = [t2[p] for p in pos]
    out["t1_raw"] = [raw1[p] for p in pos]
    out["t2_raw"] = [raw2[p] for p in pos]
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    rep: Dict = {"model": args.model, "fewshot": bool(fewshot), "n": len(out),
                 "secs_per_row": round(secs, 3),
                 "unparseable_t1": round(float((out.t1_label_auto == "UNKNOWN").mean()), 4),
                 "unparseable_t2": round(float((out.t2_label_auto == "UNKNOWN").mean()), 4),
                 "errors": int(out.t1_raw.str.startswith("__ERROR__").sum() + out.t2_raw.str.startswith("__ERROR__").sum()),
                 "scores": []}
    if prompt_tokens:  # ollama: a max at num_ctx means prompts were cut
        rep["max_prompt_tokens"] = int(max(prompt_tokens))
        rep["truncated_prompts"] = int(sum(t >= args.num_ctx - 8 for t in prompt_tokens))
    for lvl in ("t1", "t2"):
        for scope in ("all", "counsellor", "client"):
            x = out if scope == "all" else out[out.speaker == scope]
            s = score(x[f"{lvl}_label_GT"], x[f"{lvl}_label_auto"], lvl, conv=x["conv_id"])
            rep["scores"].append({"level": lvl.upper(), "scope": scope,
                                  **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in s.items()}})
    c = out[out.speaker == "counsellor"]
    per = {}
    for code in sorted(set(c.t2_label_GT.dropna()) | set(c.t2_label_auto)):
        tp = int(((c.t2_label_auto == code) & (c.t2_label_GT == code)).sum())
        npred, ngold = int((c.t2_label_auto == code).sum()), int((c.t2_label_GT == code).sum())
        per[code] = {"precision": round(tp / npred, 3) if npred else None,
                     "recall": round(tp / ngold, 3) if ngold else None,
                     "n_pred": npred, "n_gold": ngold}
    rep["counsellor_t2_per_code"] = per
    rep_path = out_path.with_suffix(".report.json")
    rep_path.write_text(json.dumps(rep, indent=2, default=float))

    print(f"SCREEN {args.model} fewshot={bool(fewshot)} n={len(out)} {secs:.2f}s/row "
          f"unparseable T1 {rep['unparseable_t1']:.1%} T2 {rep['unparseable_t2']:.1%} errors {rep['errors']}"
          + (f" max prompt {rep['max_prompt_tokens']} tok, truncated {rep['truncated_prompts']}"
             if prompt_tokens else ""))
    for s in rep["scores"]:
        print(f"  {s['level']} {s['scope']:10s} acc {s['accuracy']:.3f}  macro-F1 {s['f1_macro_gold']:.3f} "
              f"[{s.get('f1_macro_gold_lo', float('nan')):.3f}-{s.get('f1_macro_gold_hi', float('nan')):.3f}]")
    print("  tail codes (counsellor T2) precision/recall:",
          {k: (per[k]["precision"], per[k]["recall"]) for k in TAIL if k in per})
    print(f"wrote {out_path} and {rep_path}")


if __name__ == "__main__":
    main()
