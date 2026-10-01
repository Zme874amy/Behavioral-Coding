"""Segment generated dialogue turns into MISC thought units, then code each unit.

The v2 pilot revealed the real defect: 50% of generated "utterances" held two or
more codeable behaviours, e.g.

    "Stress can definitely impact our habits. Have you noticed any triggers lately?"

which is GI + OQ but carried a single label. That inflated utterance length
(median 15 words vs 8 in the real corpus) and depressed verifier agreement, since
the verifier legitimately picked the *other* code in a compound row.

Cause: the real corpus was built `volley -> Parser -> utterances -> codes`, and
generation skipped the Parser stage, relying on a sentence in the prompt instead.
This module restores that stage, reusing the production segmentation prompt
verbatim (`components.parser.PARSER_SYSTEM_PROMPT` + its few-shots + `MISCParser`)
so synthetic units are segmented exactly like the real ones.

A turn that yields one unit keeps the generator's proposed label. A turn that
splits into several gets each unit coded individually, because the generator's
single label no longer describes the whole row.

Usage:
    PYTHONPATH=src python -m synth.segment --raw data/synth/raw/<gen>.csv \
        --model <served-name> --provider vllm_server --out data/synth/raw/<gen>_seg.csv
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from components.parser import PARSER_SYSTEM_PROMPT, few_shots, MISCParser
from synth.codes import T1Code, T2Code, codebook, group_of, normalise_code

REPO_ROOT = Path(__file__).resolve().parents[2]


class UnitCode(BaseModel):
    t1: T1Code
    t2: T2Code


def split_turn(text: str, model: str, provider: str, base_url: Optional[str]) -> List[str]:
    """Segment one turn into thought units with the production parser prompt."""
    from components.utils import call_chat_model

    kw = {"base_url": base_url} if base_url else {}
    msgs = ([{"role": "system", "content": PARSER_SYSTEM_PROMPT}] + list(few_shots)
            + [{"role": "user", "content": str(text)}])
    res = call_chat_model(messages=msgs, model=model, provider=provider,
                          temperature=0.0, response_format=MISCParser, **kw)
    units = (res.model_dump() if hasattr(res, "model_dump") else res).get("utterances", [])
    units = [u.strip() for u in units if isinstance(u, str) and u.strip()]
    return units or [str(text).strip()]


def code_unit(unit: str, speaker: str, transcript: str, model: str,
              provider: str, base_url: Optional[str]) -> Optional[UnitCode]:
    """Assign a code to one thought unit split out of a compound turn."""
    from components.utils import call_chat_model

    kw = {"base_url": base_url} if base_url else {}
    system = ("You are an expert MISC 2.5 coder. Assign exactly one Tier-1 and one "
              "Tier-2 code to the target utterance, valid for that speaker.\n\n" + codebook())
    user = (f"CONTEXT\n{transcript}\n\nTARGET ({speaker}): {unit}\n\n"
            'Return JSON: {"t1": <Tier-1 code>, "t2": <Tier-2 code>}')
    res = call_chat_model(messages=[{"role": "system", "content": system},
                                    {"role": "user", "content": user}],
                          model=model, provider=provider, temperature=0.0,
                          response_format=UnitCode, **kw)
    return res if isinstance(res, UnitCode) else UnitCode(**res)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen2.5-32B-Instruct-AWQ")
    ap.add_argument("--provider", default="vllm_server")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--ctx", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    raw = pd.read_csv(REPO_ROOT / args.raw if not Path(args.raw).is_absolute() else args.raw)
    meta_cols = [c for c in raw.columns
                 if c not in ("utt_text", "vol_text", "corp_utt_idx", "conv_utt_idx",
                              "conv_vol_idx", "t1_proposed", "t2_proposed")]

    rows: List[dict] = []
    rejected: collections.Counter = collections.Counter()
    uid = 0
    n_split = n_turns = n_units = 0
    for conv_id, win in tqdm(raw.groupby("conv_id", sort=False), desc="segmenting", unit="win"):
        win = win.sort_values("conv_utt_idx")
        history: List[str] = []
        u_idx = 0
        for _, row in win.iterrows():
            n_turns += 1
            text = str(row["utt_text"])
            units = [text] if args.dry_run else split_turn(text, args.model, args.provider,
                                                           args.base_url)
            if len(units) > 1:
                n_split += 1
            transcript = "\n".join(history[-args.ctx:])
            for u in units:
                t1, t2 = row.get("t1_proposed"), row.get("t2_proposed")
                if len(units) > 1 and not args.dry_run:
                    # the generator's single label no longer describes this unit
                    try:
                        c = code_unit(u, row["speaker"], transcript, args.model,
                                      args.provider, args.base_url)
                        t1, t2 = c.t1, c.t2
                    except Exception as e:
                        print(f"  code_unit failed: {type(e).__name__}")
                        continue
                t1 = normalise_code(t1, row["speaker"], "t1")
                t2 = normalise_code(t2, row["speaker"], "t2")
                if t1 is None or t2 is None or group_of(t2, row["speaker"]) is None:
                    rejected[str(row.get("t2_proposed"))] += 1
                    continue
                rows.append({**{c: row[c] for c in meta_cols},
                             "utt_text": u, "vol_text": text,
                             "corp_utt_idx": uid, "conv_utt_idx": u_idx, "conv_vol_idx": u_idx,
                             "t1_proposed": t1, "t2_proposed": t2})
                uid += 1
                u_idx += 1
                n_units += 1
            history.append(f"{row['speaker']}: {text}")

    out = Path(args.out) if Path(args.out).is_absolute() else REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    med = df["utt_text"].astype(str).str.split().str.len().median() if len(df) else float("nan")
    if rejected:
        print(f"  rejected {sum(rejected.values())} units with unresolvable codes: "
              f"{dict(rejected.most_common(8))}")
    print(f"\nsegmented {n_turns} turns -> {n_units} thought units "
          f"({n_split} turns split, {100*n_split/max(1,n_turns):.0f}%); "
          f"median length {med:.0f} words (real corpus: 8) -> {out}")


if __name__ == "__main__":
    main()
