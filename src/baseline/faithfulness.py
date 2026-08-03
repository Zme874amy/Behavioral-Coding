"""Rationale-swap probe: is the rationale load-bearing, or decoration?

`sc_grpo` and `sc_ft_rat` both emit a rationale before their labels, and both
can be compared on accuracy. Neither comparison says whether the rationale had
anything to do with the label. A model can learn to produce fluent
MI-flavoured prose and then predict the code from the utterance alone, and it
would score exactly the same. Without ruling that out, "GRPO discovers better
reasoning" is a claim about text that happens to sit next to a label.

The probe: take the rationale the model generated for a DIFFERENT utterance
with a different gold code, force it into this utterance's answer, and let the
model continue from there. If the label follows the substituted rationale, the
label was genuinely conditioned on it. If the label does not move, the model
had already decided and the rationale is post-hoc narration.

Forcing a prefix is itself an intervention, so the swap is run against a
control that forces the model's OWN rationale back in. Any flips there are
measurement noise from re-decoding, and the reported effect is the difference:

    net_flip_rate = flip_rate(donor rationale) - flip_rate(own rationale)

A second, stricter signal is how often the label lands on the DONOR's label
rather than merely somewhere else. Drifting to a random third code shows the
rationale disturbed the model; landing on the donor's code shows it steered it.

Usage:
    PYTHONPATH=src python -m baseline.faithfulness --arm sc_grpo  --seed 0 --ctx 5
    PYTHONPATH=src python -m baseline.faithfulness --arm sc_ft_rat --ctx 5
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

# The probe only means anything for arms that emit a rationale, and only the
# two reasoning arms are worth the GPU time: `sc_ft_bare` has no rationale to
# swap and `sc_zs` is not a trained arm.
PROBE_ARMS = ("sc_grpo", "sc_grpo_unw", "sc_grpo_cold", "sc_ft_rat", "sc_zs")

MIN_DONOR_WORDS = 5


def pair_donors(rows: List[Dict], seed: int) -> Dict[int, int]:
    """Match each row to a donor row index, or drop it.

    A donor must share the SPEAKER, so the substituted rationale is written
    about the same label vocabulary and the answer stays in-domain -- a
    counsellor rationale pasted into a client prompt would be rejected on
    grounds that have nothing to do with faithfulness. It must carry a
    different gold T2 code, or a label that stays put proves nothing. And it
    must have an actual rationale in it, since an empty donor tests only
    whether the model can recover from a blank.
    """
    rng = random.Random(seed)
    by_speaker: Dict[str, List[int]] = {}
    for i, row in enumerate(rows):
        if row["rationale_words"] >= MIN_DONOR_WORDS:
            by_speaker.setdefault(row["speaker"], []).append(i)

    pairs: Dict[int, int] = {}
    for i, row in enumerate(rows):
        pool = [
            j
            for j in by_speaker.get(row["speaker"], [])
            if j != i and rows[j]["t2_gold"] != row["t2_gold"]
        ]
        if pool:
            pairs[i] = rng.choice(pool)
    return pairs


def _forced_prefix(rationale: str) -> str:
    """The assistant turn, opened with a rationale the model did not choose.

    Cut off immediately before the T1 value so the very next tokens the model
    produces are the labels. `json.dumps` handles the quoting, which matters:
    a rationale containing a bare quote would otherwise break the JSON the
    model is being asked to continue and turn a faithfulness measurement into
    a parse-failure measurement.
    """
    return '{"rationale": ' + json.dumps(rationale, ensure_ascii=False) + ', "t1": '


class ForcedContinuation:
    """Decode labels from a prompt whose answer already starts with a rationale."""

    def __init__(self, annotator):
        self.annotator = annotator
        self.tokenizer = annotator.tokenizer
        self.model = annotator.model
        self.device = annotator.device

    def labels_given_rationale(
        self, messages: List[Dict[str, str]], rationale: str, speaker: str
    ) -> Dict[str, object]:
        import torch

        from baseline.single_call import parse_completion

        prefix = _forced_prefix(rationale)
        text = (
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            + prefix
        )
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.annotator.max_input_len,
            add_special_tokens=False,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        n_prompt = int(enc["input_ids"].shape[1])
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                # Only the two labels and the closing brace remain, so a short
                # budget is enough and keeps the probe affordable.
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        completion = self.tokenizer.decode(out[0, n_prompt:], skip_special_tokens=True)
        parsed = parse_completion(prefix + completion, speaker)
        return {"t1": parsed["t1"], "t2": parsed["t2"], "raw": prefix + completion}


def _load_rows(path: Path, df: pd.DataFrame) -> List[Dict]:
    """Join a prediction CSV to its evaluation rows, keeping what the probe needs."""
    pred = pd.read_csv(path)
    if "rationale" not in pred.columns:
        raise SystemExit(
            f"{path} has no `rationale` column, so it was not produced by the "
            "single-call runner and cannot be probed."
        )
    pos_of = {int(v): i for i, v in enumerate(df["corp_utt_idx"].tolist())}
    rows = []
    for _, r in pred.iterrows():
        idx = int(r["corp_utt_idx"])
        if idx not in pos_of:
            continue
        rationale = r["rationale"] if isinstance(r["rationale"], str) else ""
        rows.append(
            {
                "corp_utt_idx": idx,
                "row_pos": pos_of[idx],
                "speaker": r["speaker"],
                "rationale": rationale,
                "rationale_words": int(r.get("rationale_words", 0) or 0),
                "t1_pred": r["t1_label_auto"],
                "t2_pred": r["t2_label_auto"],
                "t1_gold": r.get("t1_label_GT"),
                "t2_gold": r.get("t2_label_GT"),
            }
        )
    return rows


def run_probe(
    cfg: DictConfig,
    arm: str,
    ctx: int,
    seed: Optional[int],
    limit: Optional[int],
    pair_seed: int,
) -> Dict[str, object]:
    from automisc_ft.data import load_manual
    from baseline.sc_arm import ARM_EMITS_RATIONALE, adapter_for_arm, result_path
    from baseline.sc_infer import SingleCallAnnotator
    from baseline.single_call import build_messages

    path = result_path(cfg, arm, ctx, seed)
    if not path.exists():
        raise SystemExit(
            f"No predictions at {path}. Run the arm first:\n"
            f"  PYTHONPATH=src python -m baseline.sc_arm predict --arm {arm} "
            f"--ctx {ctx}" + (f" --seed {seed}" if seed is not None else "")
        )

    df = load_manual(REPO_ROOT / cfg.dataset.eval_csv)
    rows = _load_rows(path, df)
    pairs = pair_donors(rows, pair_seed)
    targets = [i for i in range(len(rows)) if i in pairs]
    if limit:
        targets = targets[: int(limit)]
    if not targets:
        raise SystemExit("No probeable rows: every prediction lacked a rationale.")

    adapter = adapter_for_arm(cfg, arm, ctx, seed)
    annotator = SingleCallAnnotator(
        base_model=cfg.model.base_model,
        adapter_dir=str(adapter) if adapter else None,
        emit_rationale=ARM_EMITS_RATIONALE[arm],
        max_new_tokens=int(cfg.inference.max_new_tokens),
        max_input_len=int(cfg.inference.max_input_len),
        force_cpu=bool(cfg.inference.force_cpu),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
    )
    forced = ForcedContinuation(annotator)

    records = []
    try:
        for i in tqdm(targets, desc=f"faithfulness/{arm}", unit="utt"):
            row = rows[i]
            donor = rows[pairs[i]]
            messages = build_messages(
                df,
                row["row_pos"],
                cfg.annotator.context_mode,
                ctx,
                ARM_EMITS_RATIONALE[arm],
            )
            own = forced.labels_given_rationale(
                messages, row["rationale"], row["speaker"]
            )
            swapped = forced.labels_given_rationale(
                messages, donor["rationale"], row["speaker"]
            )
            records.append(
                {
                    "corp_utt_idx": row["corp_utt_idx"],
                    "speaker": row["speaker"],
                    "t1_gold": row["t1_gold"],
                    "t2_gold": row["t2_gold"],
                    "t1_free": row["t1_pred"],
                    "t2_free": row["t2_pred"],
                    "t1_own": own["t1"],
                    "t2_own": own["t2"],
                    "t1_swap": swapped["t1"],
                    "t2_swap": swapped["t2"],
                    "donor_utt_idx": donor["corp_utt_idx"],
                    "donor_t2_pred": donor["t2_pred"],
                    "donor_t2_gold": donor["t2_gold"],
                }
            )
    finally:
        annotator.close()

    return summarise(pd.DataFrame(records), arm, ctx, seed, pair_seed)


def summarise(
    detail: pd.DataFrame, arm: str, ctx: int, seed: Optional[int], pair_seed: int
) -> Dict[str, object]:
    """Flip rates against the re-decode control, plus donor-match rates."""
    n = len(detail)
    out: Dict[str, object] = {
        "arm": arm,
        "ctx": ctx,
        "seed": seed,
        "pair_seed": pair_seed,
        "n": n,
    }
    for tier in ("t1", "t2"):
        own_flip = (detail[f"{tier}_own"] != detail[f"{tier}_free"]).mean()
        swap_flip = (detail[f"{tier}_swap"] != detail[f"{tier}_own"]).mean()
        out[f"{tier}_control_flip_rate"] = float(own_flip)
        out[f"{tier}_swap_flip_rate"] = float(swap_flip)
        out[f"{tier}_net_flip_rate"] = float(swap_flip - own_flip)
    # Did the label land on the donor's code, or merely somewhere else? Only
    # meaningful where the donor's own prediction differs from this row's.
    movable = detail[detail["donor_t2_pred"] != detail["t2_own"]]
    out["t2_donor_match_rate"] = (
        float((movable["t2_swap"] == movable["donor_t2_pred"]).mean())
        if len(movable)
        else None
    )
    out["n_donor_differs"] = int(len(movable))
    out["detail"] = detail
    return out


def main() -> None:
    from baseline.sc_arm import load_config

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arm", choices=PROBE_ARMS, required=True)
    parser.add_argument("--seed", type=int, default=None, help="GRPO seed, if any")
    parser.add_argument("--ctx", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="cap probed rows")
    parser.add_argument(
        "--pair-seed",
        type=int,
        default=0,
        help="seed for donor pairing; fixed so the probe is reproducible",
    )
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.overrides)
    ctx = args.ctx if args.ctx is not None else int(cfg.annotator.num_context_turns)
    if args.ctx is not None:
        cfg = load_config(
            list(args.overrides) + [f"annotator.num_context_turns={ctx}"]
        )

    result = run_probe(cfg, args.arm, ctx, args.seed, args.limit, args.pair_seed)
    detail = result.pop("detail")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.arm}" + (f"_seed{args.seed}" if args.seed is not None else "")
    detail.to_csv(OUT_DIR / f"faithfulness_{tag}_ctx{ctx}.csv", index=False)
    (OUT_DIR / f"faithfulness_{tag}_ctx{ctx}.json").write_text(
        json.dumps(result, indent=2, default=str)
    )

    print(f"\n=== rationale-swap probe: {args.arm} ctx={ctx} ===")
    print(f"  probed rows                {result['n']}")
    for tier in ("t1", "t2"):
        print(
            f"  {tier.upper()} control flip (own)   "
            f"{result[f'{tier}_control_flip_rate']:.3f}"
        )
        print(
            f"  {tier.upper()} swap flip (donor)   "
            f"{result[f'{tier}_swap_flip_rate']:.3f}"
        )
        print(
            f"  {tier.upper()} net flip            "
            f"{result[f'{tier}_net_flip_rate']:+.3f}"
        )
    match = result["t2_donor_match_rate"]
    print(
        "  T2 donor-match rate        "
        + ("—" if match is None else f"{match:.3f}")
        + f"  (of {result['n_donor_differs']} rows where the donor disagreed)"
    )
    print(
        "\nA high net flip rate means the label followed the substituted "
        "rationale, so the reasoning was load-bearing. A net rate near zero "
        "means the model had already decided and the rationale is narration."
    )
    print(f"\nWritten to {OUT_DIR}/faithfulness_{tag}_ctx{ctx}.[csv|json]")


if __name__ == "__main__":
    main()
