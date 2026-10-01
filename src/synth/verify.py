"""Independently re-code generated dialogue windows, filter, and build train CSV.

Changes from the previous verifier, which had three problems: it assumed one
target utterance per conversation, it never deduplicated (its docstring claimed
it did, 10.9% exact duplicates survived), and it was CIRCULAR -- the same model
family generated and verified, so it kept prototypes it already coded easily and
discarded ambiguous ones, selecting for exactly the examples the classifier
already gets right.

Now:
  * every turn of a window is a candidate; each is re-coded with the production
    AutoMISC classification prompt using its real in-window context;
  * agreement is RECORDED (`verifier_agreed`) rather than silently used as a
    filter, so `--keep-hard` can retain borderline-but-intended examples for the
    `boundary` variant and we can ablate them;
  * exact + near-duplicate removal is actually implemented, plus a contamination
    check against the evaluation set;
  * per-code pass rates are reported, so a code the verifier systematically
    rejects is visible instead of silently vanishing.

Use a different model (or at minimum a different prompt) for the verifier than
the generator; `--model` defaults to the generator but should be overridden.

Usage:
    PYTHONPATH=src python -m synth.verify --raw data/synth/raw/new.csv \
        --model <verifier> --provider vllm_server \
        --out data/synth/aug/new.csv [--keep-hard]
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from automisc_ft.data import COUNSELLOR_GROUPS, CLIENT_GROUPS
from components.prompts.loader import render_prompt, render_user_prompt
from synth.codes import group_of, normalise_code

REPO_ROOT = Path(__file__).resolve().parents[2]
MANUAL_COLS = ["conv_id", "speaker", "corp_utt_idx", "conv_vol_idx", "conv_utt_idx",
               "vol_text", "utt_text", "t1_label_GT", "t2_label_GT"]

CODE_TO_GROUP: Dict[tuple, str] = {}
for _spk, _groups in (("counsellor", COUNSELLOR_GROUPS), ("client", CLIENT_GROUPS)):
    for _g, _codes in _groups.items():
        for _c in _codes:
            CODE_TO_GROUP[(_spk, _c)] = _g


def recode(transcript: str, speaker: str, utterance: str, model: str,
           provider: str, base_url: Optional[str]) -> tuple:
    from components.utils import call_chat_model
    from components.prompts.response_formats import (
        CounsellorUtterance_t1, ClientUtterance_t1,
        CounsellorUtterance_t2, ClientUtterance_t2)

    kw = {"base_url": base_url} if base_url else {}
    user = render_user_prompt(transcript=transcript, speaker=speaker, utterance=utterance)

    f1 = CounsellorUtterance_t1 if speaker == "counsellor" else ClientUtterance_t1
    r1 = call_chat_model(messages=[{"role": "system", "content": render_prompt(speaker=speaker, structure="t1")},
                                   {"role": "user", "content": user}],
                         model=model, provider=provider, response_format=f1,
                         temperature=0.0, **kw)
    t1 = (r1.model_dump() if hasattr(r1, "model_dump") else r1)["label"]

    f2 = CounsellorUtterance_t2 if speaker == "counsellor" else ClientUtterance_t2
    r2 = call_chat_model(messages=[{"role": "system", "content": render_prompt(speaker=speaker, structure="t2", label=t1)},
                                   {"role": "user", "content": user}],
                         model=model, provider=provider, response_format=f2,
                         temperature=0.0, **kw)
    t2 = (r2.model_dump() if hasattr(r2, "model_dump") else r2)["label"]
    return t1, t2


def _context_for(window: pd.DataFrame, upto: int, n_turns: int = 5) -> str:
    prev = window.iloc[max(0, upto - n_turns):upto]
    return "\n".join(f"{r['speaker']}: {r['utt_text']}" for _, r in prev.iterrows())


def drop_duplicates(df: pd.DataFrame, eval_csv: Optional[str],
                    threshold: float = 0.92) -> pd.DataFrame:
    """Remove exact + near-duplicate labelled rows, and anything near the eval set."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    lab = df[df["t2_label_GT"].notna()].copy()
    if lab.empty:
        return df
    before = len(lab)
    lab = lab.drop_duplicates(subset=["utt_text"])
    texts = lab["utt_text"].astype(str).tolist()

    keep_idx = list(range(len(texts)))
    if len(texts) > 1:
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
        A = vec.fit_transform(texts)
        sim = cosine_similarity(A)
        np.fill_diagonal(sim, 0.0)
        dropped = set()
        for i in range(len(texts)):
            if i in dropped:
                continue
            for j in np.where(sim[i] >= threshold)[0]:
                if j > i:
                    dropped.add(int(j))
        keep_idx = [i for i in range(len(texts)) if i not in dropped]
    lab = lab.iloc[keep_idx]

    # contamination guard: never keep anything that mirrors the evaluation set
    if eval_csv:
        try:
            ev = pd.read_csv(REPO_ROOT / eval_csv if not Path(eval_csv).is_absolute() else eval_csv)
            ev_texts = [str(t) for t in ev["utt_text"].dropna()]
            vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
            vec.fit(lab["utt_text"].astype(str).tolist() + ev_texts)
            s = cosine_similarity(vec.transform(lab["utt_text"].astype(str)),
                                  vec.transform(ev_texts)).max(axis=1)
            lab = lab[s < threshold]
        except Exception as e:
            print(f"  (eval contamination check skipped: {type(e).__name__})")

    print(f"  dedupe: {before} labelled -> {len(lab)} kept "
          f"({before - len(lab)} exact/near-duplicate or eval-like removed)")
    keep_keys = set(lab["corp_utt_idx"])
    out = df.copy()
    mask = out["t2_label_GT"].notna() & ~out["corp_utt_idx"].isin(keep_keys)
    out.loc[mask, ["t1_label_GT", "t2_label_GT"]] = None
    return out


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-csv", default="data/manual/HLQC_balanced_manual.csv")
    ap.add_argument("--eval-csv", default="data/manual/MIV6.3A_manual.csv")
    ap.add_argument("--model", default="Qwen2.5-32B-Instruct-AWQ")
    ap.add_argument("--provider", default="vllm_server")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--keep-hard", action="store_true",
                    help="keep verifier-disagreed turns (flagged) instead of dropping them")
    ap.add_argument("--accept-all", action="store_true", help="skip re-coding (plumbing test)")
    ap.add_argument("--ctx", type=int, default=5)
    ap.add_argument("--human-repeat", type=int, default=1)
    args = ap.parse_args(argv)

    raw = pd.read_csv(REPO_ROOT / args.raw if not Path(args.raw).is_absolute() else args.raw)
    for c in ("t1_label_GT", "t2_label_GT", "t1_verified", "t2_verified"):
        raw[c] = pd.Series([None] * len(raw), dtype="object")
    raw["verifier_agreed"] = False

    agree = collections.Counter(); total = collections.Counter()
    rejected: collections.Counter = collections.Counter()
    for conv_id, win in tqdm(raw.groupby("conv_id", sort=False), desc="verifying", unit="win"):
        win = win.sort_values("conv_utt_idx")
        for pos, (idx, row) in enumerate(win.iterrows()):
            spk = row["speaker"]
            raw1, raw2 = row.get("t1_proposed"), row.get("t2_proposed")
            prop1 = normalise_code(raw1, spk, "t1")
            prop2 = normalise_code(raw2, spk, "t2")
            if prop1 is None or prop2 is None or group_of(prop2, spk) is None:
                rejected[str(raw2)] += 1
                continue
            total[prop2] += 1
            if args.accept_all:
                v1, v2 = row.get("t1_proposed"), prop2
            else:
                try:
                    v1, v2 = recode(_context_for(win, pos, args.ctx), spk,
                                    str(row["utt_text"]), args.model, args.provider,
                                    args.base_url)
                except Exception as e:
                    print(f"  recode failed [{conv_id}#{pos}]: {type(e).__name__}")
                    continue
            ok = (v1 == prop1) and (v2 == prop2)
            raw.at[idx, "t1_verified"], raw.at[idx, "t2_verified"] = v1, v2
            raw.at[idx, "verifier_agreed"] = bool(ok)
            if ok:
                agree[prop2] += 1
            if ok or args.keep_hard:
                raw.at[idx, "t1_label_GT"] = prop1
                raw.at[idx, "t2_label_GT"] = prop2

    if rejected:
        print(f"  rejected {sum(rejected.values())} candidates with unresolvable codes: "
              f"{dict(rejected.most_common(8))}")
    n_lab = int(raw["t2_label_GT"].notna().sum())
    n_ok = int(raw["verifier_agreed"].sum())
    print(f"\nverifier agreed on {n_ok}/{sum(total.values())} candidates "
          f"({100*n_ok/max(1,sum(total.values())):.0f}%); labelled rows before dedupe: {n_lab}")
    print("per-code agreement (code: agreed/total):")
    for code, tot in total.most_common():
        print(f"   {code:5s} {agree[code]:3d}/{tot:<3d} ({100*agree[code]/tot:3.0f}%)")

    raw = drop_duplicates(raw, args.eval_csv)

    keep_convs = set(raw.loc[raw["t2_label_GT"].notna(), "conv_id"])
    synth = raw[raw["conv_id"].isin(keep_convs)]

    human = pd.read_csv(REPO_ROOT / args.train_csv if not Path(args.train_csv).is_absolute() else args.train_csv)
    human = pd.concat([human] * max(1, int(args.human_repeat)), ignore_index=True)
    cols = list(dict.fromkeys(MANUAL_COLS + [c for f in (human, synth) for c in f.columns]))
    aug = pd.concat([human.reindex(columns=cols), synth.reindex(columns=cols)], ignore_index=True)

    out = Path(args.out) if Path(args.out).is_absolute() else REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    aug.to_csv(out, index=False)
    print(f"\nwrote {len(aug)} rows ({len(human)} human x{args.human_repeat} + {len(synth)} synth rows, "
          f"{int(synth['t2_label_GT'].notna().sum())} labelled synth examples) -> {out}")


if __name__ == "__main__":
    main()
