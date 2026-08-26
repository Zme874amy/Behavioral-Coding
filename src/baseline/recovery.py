"""T2-recovery probe: does a two-call arm re-read the utterance when handed a
wrong T1, or blindly follow the group label?

Accuracy on the real pipeline cannot separate two very different T2 policies: one
that reasons about the utterance and one that just emits whatever fine code the
injected group makes salient. They score the same as long as the T1 it is fed is
right. The difference only shows when the T1 is WRONG -- which happens ~20% of
the time at inference and is exactly the regime a two-call arm has to survive.

The probe holds the trained T2 fixed and runs it three ways per utterance,
varying only the T1 group injected into the T2 prompt:

    gold T1   -> ceiling: how good T2 is when handed the right group
    pred T1   -> the pipeline number (T1 is the model's own prediction)
    wrong T1  -> a uniformly random INCORRECT group

and reports, mirroring `baseline.faithfulness`'s control/effect framing:

    recovery rate = P(T2 correct | wrong T1)          -- re-read the utterance
    follow rate   = P(T2 is a child of the wrong T1)  -- obeyed the wrong label

A high recovery rate with a low follow rate is the resilient policy the RL reward
is meant to produce. This probe is UNIQUE to the two-call cells: `sc_grpo` emits
T1 and T2 in one breath, so there is no committed T1 to corrupt -- its absence
here is part of what the B-vs-F comparison exposes.

Applies to the four two-call GRPO arms and their SFT warm-starts (`ft_bare`,
`ft1mix_bare`) as the untuned baseline.

Usage:
    PYTHONPATH=src python -m baseline.recovery --arm grpo_pair_dec --seed 0 --ctx 5
    PYTHONPATH=src python -m baseline.recovery --arm ft_bare --ctx 5
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "outputs" / "grpo"

# The GRPO arms carry a rationale (cot); the SFT baselines are the label-only
# arms they warm-started from, evaluated in their own bare style.
GRPO_ARMS = ("grpo_pair_dec", "grpo_pair_joint", "grpo_mix_dec", "grpo_mix_joint")
SFT_ARMS = ("ft_bare", "ft1mix_bare")
PROBE_ARMS = GRPO_ARMS + SFT_ARMS
ARM_STYLE_SUFFIX = {**{a: "" for a in GRPO_ARMS}, "ft_bare": "_bare", "ft1mix_bare": "_bare"}


def load_config(overrides: Optional[List[str]] = None) -> DictConfig:
    from baseline.grpo_tc import load_config as load_tc

    return load_tc(overrides)


def _resolve_adapters(cfg, arm, ctx, seed, variant):
    """(t1_dir, t2_dir, shared_dir) for a GRPO arm or an SFT warm-start baseline."""
    if arm in GRPO_ARMS:
        from baseline.grpo_tc import _resolve_pred_adapters

        return _resolve_pred_adapters(cfg, arm, ctx, seed, variant)
    from baseline.local_arm import load_config as load_sft, resolve_adapters

    t1, t2, shared, _ = resolve_adapters(load_sft(), arm, ctx)
    return t1, t2, shared


def _wrong_t1(speaker: str, gold: str, rng: random.Random) -> Optional[str]:
    """A uniformly random T1 group for this speaker that is NOT the gold one."""
    from automisc_ft.data import t1_codes_for_speaker

    pool = [c for c in t1_codes_for_speaker(speaker) if c != gold]
    return rng.choice(pool) if pool else None


def run_probe(cfg, arm, ctx, seed, variant, limit, wrong_seed) -> Dict[str, object]:
    from automisc_ft.data import (
        load_manual, t1_codes_for_speaker, t2_codes_for_speaker,
    )
    from automisc_ft.infer import TieredAnnotator, parse_label
    from baseline import two_call
    from baseline.two_call import is_child

    t1_dir, t2_dir, shared_dir = _resolve_adapters(cfg, arm, ctx, seed, variant)
    for p in (t1_dir, t2_dir, shared_dir):
        if p is not None and not Path(p).exists():
            raise SystemExit(f"Missing adapter {p}; train {arm} first.")

    suffix = ARM_STYLE_SUFFIX[arm]
    annot = TieredAnnotator(
        base_model=cfg.model.base_model,
        t1_adapter_dir=str(t1_dir) if t1_dir else None,
        t2_adapter_dir=str(t2_dir) if t2_dir else None,
        shared_adapter_dir=str(shared_dir) if shared_dir else None,
        force_cpu=bool(cfg.inference.force_cpu),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        max_new_tokens=int(cfg.inference.max_new_tokens),
        max_input_len=int(cfg.inference.max_input_len), structure_suffix=suffix,
    )

    df = load_manual(REPO_ROOT / cfg.dataset.eval_csv)
    ctx_mode = cfg.annotator.context_mode
    rng = random.Random(wrong_seed)
    positions = [i for i in range(len(df))
                 if df.iloc[i]["speaker"] in {"counsellor", "client"}]
    if limit:
        positions = positions[: int(limit)]

    def t2_under(pos, speaker, t1_label):
        annot._set_adapter("t2")
        msgs = two_call.t2_messages(df, pos, t1_label, ctx_mode, ctx, suffix)
        raw, _, _ = annot._generate(msgs)
        return parse_label(raw, t2_codes_for_speaker(speaker))

    records = []
    try:
        for pos in tqdm(positions, desc=f"recovery/{arm}", unit="utt"):
            row = df.iloc[pos]
            speaker = row["speaker"]
            t1_gold = row.get("t1_label_GT")
            t2_gold = row.get("t2_label_GT")
            if not isinstance(t1_gold, str) or not isinstance(t2_gold, str):
                continue
            # T1 prediction (the pipeline's own conditioning).
            annot._set_adapter("t1")
            t1_raw, _, _ = annot._generate(
                two_call.t1_messages(df, pos, ctx_mode, ctx, suffix)
            )
            t1_pred = parse_label(t1_raw, t1_codes_for_speaker(speaker))
            t1_pred_cond = t1_pred if t1_pred != two_call.UNKNOWN else t1_codes_for_speaker(speaker)[0]
            wrong = _wrong_t1(speaker, t1_gold, rng)
            if wrong is None:
                continue
            records.append({
                "corp_utt_idx": int(row["corp_utt_idx"]), "speaker": speaker,
                "t1_gold": t1_gold, "t2_gold": t2_gold, "t1_pred": t1_pred,
                "wrong_t1": wrong,
                "t2_gold_cond": t2_under(pos, speaker, t1_gold),
                "t2_pred_cond": t2_under(pos, speaker, t1_pred_cond),
                "t2_wrong_cond": t2_under(pos, speaker, wrong),
            })
    finally:
        annot.close()

    detail = pd.DataFrame(records)
    detail["follows_wrong"] = [
        is_child(r.speaker, r.wrong_t1, r.t2_wrong_cond) for r in detail.itertuples()
    ]
    return summarise(detail, arm, ctx, seed, wrong_seed)


def summarise(detail, arm, ctx, seed, wrong_seed) -> Dict[str, object]:
    n = len(detail)
    def acc(col):
        return float((detail[col] == detail["t2_gold"]).mean()) if n else 0.0
    out = {
        "arm": arm, "ctx": ctx, "seed": seed, "wrong_seed": wrong_seed, "n": n,
        "gold_cond_acc": acc("t2_gold_cond"),
        "pred_cond_acc": acc("t2_pred_cond"),
        "wrong_cond_acc": acc("t2_wrong_cond"),  # recovery rate
        "follow_rate": float(detail["follows_wrong"].mean()) if n else 0.0,
    }
    out["recovery_rate"] = out["wrong_cond_acc"]
    out["gold_minus_pred"] = out["gold_cond_acc"] - out["pred_cond_acc"]
    out["pred_minus_wrong"] = out["pred_cond_acc"] - out["wrong_cond_acc"]
    out["detail"] = detail
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arm", choices=PROBE_ARMS, required=True)
    parser.add_argument("--seed", type=int, default=None, help="GRPO seed (GRPO arms only)")
    parser.add_argument("--variant", default="weighted",
                        choices=("weighted", "unweighted", "coldstart"))
    parser.add_argument("--ctx", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--wrong-seed", type=int, default=0,
                        help="seed for the random wrong-T1 draw; fixed for reproducibility")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    ctx = args.ctx
    overrides = list(args.overrides)
    if ctx is not None:
        overrides = overrides + [f"annotator.num_context_turns={ctx}"]
    cfg = load_config(overrides)
    ctx = ctx if ctx is not None else int(cfg.annotator.num_context_turns)

    result = run_probe(cfg, args.arm, ctx, args.seed, args.variant, args.limit, args.wrong_seed)
    detail = result.pop("detail")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.arm + (f"_seed{args.seed}" if args.seed is not None else "")
    detail.to_csv(OUT_DIR / f"recovery_{tag}_ctx{ctx}.csv", index=False)
    (OUT_DIR / f"recovery_{tag}_ctx{ctx}.json").write_text(json.dumps(result, indent=2))

    print(f"\n=== T2-recovery probe: {args.arm} ctx={ctx} (n={result['n']}) ===")
    print(f"  gold-T1 cond acc      {result['gold_cond_acc']:.3f}  (ceiling)")
    print(f"  pred-T1 cond acc      {result['pred_cond_acc']:.3f}  (pipeline)")
    print(f"  wrong-T1 cond acc     {result['wrong_cond_acc']:.3f}  (recovery rate)")
    print(f"  follow rate           {result['follow_rate']:.3f}  (obeyed the wrong group)")
    print(f"\nHigh recovery with low follow = T2 re-reads the utterance rather than "
          "the injected label.")
    print(f"Written to {OUT_DIR}/recovery_{tag}_ctx{ctx}.[csv|json]")


if __name__ == "__main__":
    main()
