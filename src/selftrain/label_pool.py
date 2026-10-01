"""Pseudo-label a real unlabeled pool with the Qwen student (Direction 1a).

Runs the two-call T1->T2 flow with self-consistency (`TieredAnnotator.
predict_row_selfconsistent`) over every utterance in a pool corpus and writes a
pseudo-labeled CSV carrying the predicted codes and a confidence signal. The
whole conversation is labeled (not a filtered subset) so that `select.py` can
retain full conversations for context while supervising only accepted rows.

Usage (self / student labeler):
    PYTHONPATH=src python -m selftrain.label_pool \
        --pool data/parsed/AnnoMI_parsed.csv --arm ft_bare --ctx 5 \
        --k 5 --temperature 0.7 \
        --out data/selftrain/pseudo/annomi_ft_bare_ctx5.csv

Smoke test with the plain base model (no adapters needed):
    PYTHONPATH=src python -m selftrain.label_pool \
        --pool data/parsed/AnnoMI_parsed.csv --zero-shot --ctx 5 --k 3 \
        --limit 20 --out /tmp/pseudo_smoke.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd
from tqdm import tqdm

from collections import Counter

from automisc_ft.data import load_manual
from baseline.local_arm import REPO_ROOT, load_config, resolve_adapters


def _remote_selfconsistent(df, pos, context_mode, num_context_turns, restrict_t2,
                           model, provider, base_url, k, temperature):
    """Self-consistency pseudo-label via a remote chat model (Direction 1b).

    Reuses the SAME tiered prompts as the student labeler (`build_messages_t1/2`)
    but sends them to a stronger open model through `call_chat_model`. With
    ``k=1`` this is a plain distillation pass (confidence defaults to 1.0);
    ``k>1`` with ``temperature>0`` recovers an agreement-based confidence.
    """
    from automisc_ft.data import (
        build_messages_t1, build_messages_t2,
        t1_codes_for_speaker, t2_codes_for_speaker,
    )
    from components.utils import call_chat_model
    from components.prompts.response_formats import (
        CounsellorUtterance_t1, ClientUtterance_t1,
        CounsellorUtterance_t2, ClientUtterance_t2,
    )

    speaker = df.iloc[pos]["speaker"]
    kwargs = {"base_url": base_url} if base_url else {}

    def _label(messages, fmt):
        r = call_chat_model(messages=messages, model=model, provider=provider,
                            temperature=temperature, response_format=fmt, **kwargs)
        return (r.model_dump() if hasattr(r, "model_dump") else r)["label"]

    t1_fmt = CounsellorUtterance_t1 if speaker == "counsellor" else ClientUtterance_t1
    t1_msgs = build_messages_t1(df, pos, context_mode, num_context_turns, "")
    t1_votes = [_label(t1_msgs, t1_fmt) for _ in range(k)]
    t1_pred, t1_count = Counter(t1_votes).most_common(1)[0]
    t1_agreement = t1_count / float(k)

    t2_fmt = CounsellorUtterance_t2 if speaker == "counsellor" else ClientUtterance_t2
    t2_msgs = build_messages_t2(df, pos, t1_pred, context_mode, num_context_turns, "")
    t2_votes = [_label(t2_msgs, t2_fmt) for _ in range(k)]
    t2_pred, t2_count = Counter(t2_votes).most_common(1)[0]
    t2_agreement = t2_count / float(k)

    return {
        "t1_pred": t1_pred, "t2_pred": t2_pred,
        "t1_agreement": t1_agreement, "t2_agreement": t2_agreement,
        "confidence": t1_agreement * t2_agreement,
    }


class _RemoteScorer:
    """Teacher-side twin of `TieredAnnotator.predict_row_scored`, over vLLM.

    Scores every candidate code by the log-likelihood of ``" {code}\\n"`` after
    the teacher's own rendering of the SAME tiered prompt, then softmaxes over the
    candidate set -- the identical confidence definition the student uses, so the
    self-vs-teacher comparison differs only in who labels. The completions
    endpoint is called with ``echo=True, logprobs=1, max_tokens=1``: the logprobs
    of the echoed candidate tail give the score, and the one generated token is
    discarded. All candidates of one row share the long prompt prefix, which
    vLLM's prefix cache computes once.
    """

    def __init__(self, model_name: str, tokenizer_path: str, base_url: Optional[str],
                 structure_suffix: str = "_bare", fewshot_provider=None):
        from transformers import AutoTokenizer

        from components.utils import get_vllm_client

        self.name = model_name
        self.tok = AutoTokenizer.from_pretrained(tokenizer_path)
        self.client = get_vllm_client(base_url)
        self.suffix = structure_suffix
        self.fewshot = fewshot_provider

    def score_codes(self, messages, codes):
        import math

        from sft.data import build_completion

        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        n_p = len(self.tok(prompt, add_special_tokens=False)["input_ids"])
        resp = self.client.completions.create(
            model=self.name, prompt=[prompt + build_completion(c) for c in codes],
            max_tokens=1, temperature=0.0, echo=True, logprobs=1,
        )
        loglik = {}
        for ch in sorted(resp.choices, key=lambda c: c.index):
            lps = ch.logprobs.token_logprobs
            tail = lps[n_p:len(lps) - 1]        # candidate tokens; drop the generated one
            loglik[codes[ch.index]] = float(sum(x for x in tail if x is not None))
        m = max(loglik.values())
        z = sum(math.exp(v - m) for v in loglik.values())
        return {c: math.exp(v - m) / z for c, v in loglik.items()}, n_p

    def greedy_t1(self, df, pos, context_mode, num_context_turns) -> str:
        """Greedy decode of the same T1 prompt, parsed -- for the argmax check."""
        from automisc_ft.data import build_messages_t1, t1_codes_for_speaker
        from automisc_ft.infer import parse_label

        spk = df.iloc[pos]["speaker"]
        fs = self.fewshot(spk, "t1", None) if self.fewshot else None
        msgs = build_messages_t1(df, pos, context_mode, num_context_turns, self.suffix, fs)
        r = self.client.chat.completions.create(model=self.name, messages=msgs,
                                                temperature=0.0, max_tokens=8)
        return parse_label(r.choices[0].message.content or "", t1_codes_for_speaker(spk))

    def predict_row_scored(self, df, pos, context_mode, num_context_turns):
        from automisc_ft.data import (
            build_messages_t1, build_messages_t2, t1_codes_for_speaker, t2_codes_for_speaker,
        )
        spk = df.iloc[pos]["speaker"]
        fs = (lambda tier, t1: self.fewshot(spk, tier, t1)) if self.fewshot else (lambda tier, t1: None)
        t1_probs, t1_n = self.score_codes(
            build_messages_t1(df, pos, context_mode, num_context_turns, self.suffix, fs("t1", None)),
            t1_codes_for_speaker(spk))
        t1 = max(t1_probs, key=t1_probs.get)
        t2_probs, t2_n = self.score_codes(
            build_messages_t2(df, pos, t1, context_mode, num_context_turns, self.suffix, fs("t2", t1)),
            t2_codes_for_speaker(spk))
        t2 = max(t2_probs, key=t2_probs.get)
        return {"t1_pred": t1, "t2_pred": t2, "t1_probs": t1_probs, "t2_probs": t2_probs,
                "t1_conf": t1_probs[t1], "t2_conf": t2_probs[t2],
                "confidence": t1_probs[t1] * t2_probs[t2],
                "t1_n_prompt_tokens": t1_n, "t2_n_prompt_tokens": t2_n}


def _norm_text(s) -> str:
    import re
    return re.sub(r"\W+", " ", str(s).lower()).strip()


def _drop_eval_like(df: pd.DataFrame, eval_csv: Path, threshold: float = 0.92) -> pd.DataFrame:
    """Remove pool utterances that duplicate or nearly duplicate test utterances.

    Some pools share a source with the test set (22 of 652 longer MIV6.3A test
    utterances appear verbatim in MIV6.3B, templated chatbot lines), so without
    this a test utterance could enter training carrying a pseudo-label. Exact
    matches after normalisation (>= 4 words) and TF-IDF cosine >= `threshold`
    (>= 6 words) are dropped -- the same criterion `synth.verify.drop_duplicates`
    uses against the eval set. Short generic lines ("okay", "yeah") are left
    alone: they are context, not evidence about the test set.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    ev = pd.read_csv(eval_csv, usecols=["utt_text"])["utt_text"].map(_norm_text)
    pool_norm = df["utt_text"].map(_norm_text)
    n_words = pool_norm.str.split().str.len()
    exact = pool_norm.isin(set(t for t in ev if len(t.split()) >= 4)) & (n_words >= 4)
    long_idx = df.index[(n_words >= 6) & ~exact]
    ev_long = [t for t in ev if len(t.split()) >= 6]
    near = pd.Series(False, index=df.index)
    if len(long_idx) and ev_long:
        vec = TfidfVectorizer(ngram_range=(1, 2)).fit(list(pool_norm[long_idx]) + ev_long)
        sim = cosine_similarity(vec.transform(pool_norm[long_idx]), vec.transform(ev_long)).max(axis=1)
        near.loc[long_idx] = sim >= threshold
    drop = exact | near
    if drop.any():
        print(f"Removed {int(drop.sum())} pool utterances matching the test set "
              f"({int(exact.sum())} exact, {int(near.sum())} near-duplicate)")
    return df[~drop].reset_index(drop=True)


def _exemplar_texts(ctx: int) -> Set[str]:
    """Normalised utterance texts used as few-shot exemplars (to exclude from scoring)."""
    import json

    from baseline.fewshot import exemplars_path

    texts: Set[str] = set()

    def walk(o):
        if isinstance(o, dict):
            if "utterance" in o:
                texts.add(_norm_text(o["utterance"]))
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(json.loads(Path(exemplars_path(ctx)).read_text()))
    return texts


def _eval_conv_ids(cfg) -> Set[str]:
    """Conversation ids of the fixed expert eval set, to keep them out of the
    pool (contamination guard). Cheap: only the conv_id column is needed."""
    try:
        ev = pd.read_csv(REPO_ROOT / cfg.dataset.eval_csv, usecols=["conv_id"])
        return {str(c) for c in ev["conv_id"].unique()}
    except Exception:
        return set()


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True, help="parsed pool CSV (conv_id, speaker, utt_text, corp_utt_idx, ...)")
    ap.add_argument("--out", required=True, help="destination pseudo-labeled CSV")
    ap.add_argument("--arm", default="ft_bare", help="which trained arm's adapters label the pool")
    ap.add_argument("--zero-shot", action="store_true", help="label with the plain base model (no adapters); for smoke tests")
    ap.add_argument("--ctx", type=int, default=None, help="context turns; defaults to cfg.annotator.num_context_turns")
    ap.add_argument("--k", type=int, default=5, help="self-consistency samples per tier")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=None, help="label at most this many utterances (testing)")
    ap.add_argument("--checkpoint-every", type=int, default=200)
    ap.add_argument("--overrides", nargs="*", default=None, help="Hydra-style cfg overrides")
    ap.add_argument("--labeler", choices=["self", "remote"], default="self",
                    help="'self' = the Qwen student (1a); 'remote' = a stronger model via call_chat_model (1b)")
    ap.add_argument("--labeler-model", default="Qwen/Qwen3-32B", help="remote labeler model id")
    ap.add_argument("--labeler-provider", default="vllm_server", help="remote labeler provider")
    ap.add_argument("--base-url", default=None, help="remote labeler endpoint (overrides $VLLM_BASE_URL)")
    # -- self-training v2 (defaults keep the v1 behaviour) ---------------------
    ap.add_argument("--confidence", choices=["vote", "likelihood"], default="vote",
                    help="vote = k-sample agreement (v1); likelihood = full distribution over codes")
    ap.add_argument("--speakers", nargs="+", default=["counsellor", "client"],
                    help="label only these speakers; other rows are kept as unlabelled context")
    ap.add_argument("--sample-convs", type=int, default=None, help="seeded sample of this many conversations")
    ap.add_argument("--seed", type=int, default=0, help="seed for --sample-convs")
    ap.add_argument("--keep-gold", action="store_true",
                    help="carry t1/t2_label_GT through (labelled pools, for calibration)")
    ap.add_argument("--allow-eval", action="store_true",
                    help="label the eval conversations too (report-only teacher evaluation; "
                         "also skips the test near-duplicate filter)")
    ap.add_argument("--fewshot", action="store_true",
                    help="remote labeler: add the frozen HLQC few-shot exemplars (as the fs arm)")
    ap.add_argument("--exclude-exemplars", action="store_true",
                    help="drop rows that are themselves few-shot exemplars (HLQC calibration)")
    ap.add_argument("--teacher-tokenizer", default=None,
                    help="remote likelihood: tokenizer/chat template of the served model (default $MODEL)")
    ap.add_argument("--greedy-check", type=int, default=0,
                    help="remote likelihood: on the first N rows also decode T1 greedily and report "
                         "argmax==greedy agreement and seconds per row (teacher smoke test)")
    args = ap.parse_args(argv)

    cfg = load_config(args.overrides)
    ctx = args.ctx if args.ctx is not None else int(cfg.annotator.num_context_turns)

    df = load_manual(REPO_ROOT / args.pool if not Path(args.pool).is_absolute() else args.pool)
    df = df.drop(columns=[c for c in df.columns if c.endswith("_auto")], errors="ignore")

    # Contamination guard: never label the fixed eval conversations -- unless this
    # is the explicit report-only evaluation of a teacher on the eval set.
    eval_ids = _eval_conv_ids(cfg)
    if eval_ids and not args.allow_eval:
        keep = ~df["conv_id"].isin(eval_ids)
        dropped = int((~keep).sum())
        if dropped:
            print(f"Excluding {dropped} rows from {len(eval_ids)} eval conversations (contamination guard)")
        df = df[keep].reset_index(drop=True)
    if args.sample_convs:
        convs = pd.Series(df["conv_id"].astype(str).unique())
        keep_c = set(convs.sample(min(int(args.sample_convs), len(convs)), random_state=args.seed))
        df = df[df["conv_id"].astype(str).isin(keep_c)].reset_index(drop=True)
        print(f"Sampled {len(keep_c)} conversations ({len(df)} rows), seed={args.seed}")
    if not args.allow_eval:
        df = _drop_eval_like(df, REPO_ROOT / cfg.dataset.eval_csv)
    # Rows never scored (kept as context so neighbouring prompts are unchanged).
    skip_idx: Set[int] = set()
    if args.exclude_exemplars:
        ex = _exemplar_texts(int(args.ctx or cfg.annotator.num_context_turns))
        hit = df["utt_text"].map(_norm_text).isin(ex)
        skip_idx = set(df.loc[hit, "corp_utt_idx"].astype(int))
        print(f"Not scoring {len(skip_idx)} rows that are few-shot exemplars (kept as context)")

    # Adapters (only for the self/student labeler, unless zero-shot smoke test).
    t1_adapter = t2_adapter = shared_adapter = None
    if args.labeler == "self" and not args.zero_shot:
        t1_adapter, t2_adapter, shared_adapter, retrain_cmd = resolve_adapters(cfg, args.arm, ctx)
        for path in (t1_adapter, t2_adapter, shared_adapter):
            if path is not None and not Path(path).exists():
                raise SystemExit(f"Missing adapter {path}. Train it first:\n  {retrain_cmd}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip rows already labeled (by corp_utt_idx high-water mark).
    existing_df = None
    utt_checkpoint = -1
    if out_path.exists():
        existing_df = pd.read_csv(out_path)
        if not existing_df.empty and "corp_utt_idx" in existing_df.columns:
            utt_checkpoint = existing_df["corp_utt_idx"].max()
            print(f"Resuming: {len(existing_df)} rows done (up to corp_utt_idx {utt_checkpoint})")

    n_total = len(df) if args.limit is None else min(int(args.limit), len(df))
    positions = [i for i in range(n_total) if int(df.iloc[i]["corp_utt_idx"]) > utt_checkpoint]
    if not positions:
        print(f"Nothing to do; {out_path} is already complete.")
        return

    context_mode = cfg.annotator.context_mode
    restrict_t2 = bool(cfg.annotator.get("restrict_t2_to_group", False))

    annotator = None
    if args.labeler == "self":
        from automisc_ft.infer import TieredAnnotator

        max_input_len = int(cfg.inference.max_input_len[args.arm]) if not args.zero_shot else int(
            cfg.inference.max_input_len.get("ft_bare", 1024)
        )
        # Self-consistency needs a couple more tokens of headroom than a single
        # bare label so a sampled decode is not clipped before the code emits.
        max_new_tokens = max(int(cfg.inference.max_new_tokens), 16)
        print(
            f"Labeling pool={args.pool} labeler=self arm={'zero-shot' if args.zero_shot else args.arm} "
            f"ctx={ctx} k={args.k} temp={args.temperature} model={cfg.model.base_model} "
            f"n={len(positions)} -> {out_path}"
        )
        annotator = TieredAnnotator(
            base_model=cfg.model.base_model,
            t1_adapter_dir=str(t1_adapter) if t1_adapter else None,
            t2_adapter_dir=str(t2_adapter) if t2_adapter else None,
            shared_adapter_dir=str(shared_adapter) if shared_adapter else None,
            force_cpu=bool(cfg.inference.force_cpu),
            trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
            max_new_tokens=max_new_tokens,
            max_input_len=max_input_len,
            structure_suffix="_bare",
        )

        if args.confidence == "likelihood":
            def label_fn(pos):
                return annotator.predict_row_scored(df, pos, context_mode, ctx)
        else:
            def label_fn(pos):
                return annotator.predict_row_selfconsistent(
                    df, pos, context_mode, ctx, restrict_t2,
                    k=int(args.k), temperature=float(args.temperature),
                )
    elif args.confidence == "likelihood":
        import os

        fewshot = None
        if args.fewshot:
            from baseline.fewshot import exemplars_path, load_exemplars
            from baseline.local_arm import _make_fewshot_provider
            fewshot = _make_fewshot_provider(load_exemplars(exemplars_path(ctx)), rationales=False)
        tok_path = args.teacher_tokenizer or os.environ.get("MODEL") or args.labeler_model
        scorer = _RemoteScorer(args.labeler_model, tok_path, args.base_url,
                               structure_suffix="_bare", fewshot_provider=fewshot)
        print(
            f"Labeling pool={args.pool} labeler=remote(likelihood) model={args.labeler_model} "
            f"fewshot={bool(fewshot)} ctx={ctx} n={len(positions)} -> {out_path}"
        )

        def label_fn(pos):
            return scorer.predict_row_scored(df, pos, context_mode, ctx)
    else:
        print(
            f"Labeling pool={args.pool} labeler=remote model={args.labeler_model} "
            f"provider={args.labeler_provider} ctx={ctx} k={args.k} temp={args.temperature} "
            f"n={len(positions)} -> {out_path}"
        )

        def label_fn(pos):
            return _remote_selfconsistent(
                df, pos, context_mode, ctx, restrict_t2,
                args.labeler_model, args.labeler_provider, args.base_url,
                int(args.k), float(args.temperature),
            )
    keep_cols = [c for c in ("corp_conv_idx", "conv_id", "speaker", "corp_vol_idx",
                             "conv_vol_idx", "vol_text", "corp_utt_idx", "conv_utt_idx",
                             "utt_text") if c in df.columns]
    output_rows: List[dict] = []

    def save() -> None:
        nonlocal existing_df, output_rows
        if not output_rows:
            return
        out = pd.DataFrame(output_rows)
        if existing_df is not None:
            out = pd.concat([existing_df, out], ignore_index=True)
        out.to_csv(out_path, index=False)
        existing_df = out
        output_rows = []

    import json
    import time

    if args.keep_gold:
        keep_cols += [c for c in ("t1_label_GT", "t2_label_GT") if c in df.columns]
    speakers = set(args.speakers)
    check = {"n": 0, "agree": 0, "secs": 0.0, "scored": 0}
    do_check = args.greedy_check and args.labeler == "remote" and args.confidence == "likelihood"
    try:
        for pos in tqdm(positions, desc="labeling", unit="utt"):
            row = df.iloc[pos]
            base = {c: row[c] for c in keep_cols}
            # Non-target speakers and excluded rows are written unlabelled: they
            # stay in the pool as context for their neighbours but never become
            # training targets (select.py treats a null label as context).
            if row["speaker"] not in speakers or int(row["corp_utt_idx"]) in skip_idx:
                output_rows.append({**base, "t1_label_auto": None, "t2_label_auto": None,
                                    "confidence": None})
            else:
                t0 = time.time()
                pred = label_fn(pos)
                check["secs"] += time.time() - t0
                check["scored"] += 1
                if do_check and check["n"] < int(args.greedy_check):
                    check["n"] += 1
                    check["agree"] += int(scorer.greedy_t1(df, pos, context_mode, ctx) == pred["t1_pred"])
                rec = {**base, "t1_label_auto": pred["t1_pred"], "t2_label_auto": pred["t2_pred"],
                       "confidence": pred["confidence"]}
                if args.confidence == "likelihood":
                    rec.update(t1_conf=pred["t1_conf"], t2_conf=pred["t2_conf"],
                               t1_probs=json.dumps(pred["t1_probs"]),
                               t2_probs=json.dumps(pred["t2_probs"]))
                else:
                    rec.update(t1_agreement=pred["t1_agreement"], t2_agreement=pred["t2_agreement"])
                output_rows.append(rec)
            if len(output_rows) >= int(args.checkpoint_every):
                save()
    finally:
        save()
        if annotator is not None:
            annotator.close()

    print(f"Wrote {len(existing_df)} pseudo-labeled rows to {out_path}")
    if check["scored"]:
        print(f"SPEED: {check['secs'] / check['scored']:.2f} s per scored row ({check['scored']} rows)")
    if check["n"]:
        print(f"GREEDY CHECK: T1 argmax == greedy on {check['agree']}/{check['n']} "
              f"({check['agree'] / check['n']:.0%}) rows")


if __name__ == "__main__":
    main()
