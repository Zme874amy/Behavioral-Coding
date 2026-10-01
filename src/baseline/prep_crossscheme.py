"""Build `load_manual`-compatible eval CSVs for the cross-scheme corpora.

These corpora are coded in a DIFFERENT MI scheme from MISC, but share a core
vocabulary (see `docs`), so the trained MISC model can be run on them and scored
on the overlap — a genuine out-of-distribution generalization test.

Two outputs, written to `data/manual/`:

  AnnoMI_eval.csv    from data/AnnoMI.csv (volley-level, gold = 'AnnoMI Label').
                     Client change/sustain/neutral -> t1_label_GT; therapist
                     question/reflection/information -> t2_label_GT.
  Welivita_eval.csv  from data/external/welivita_mi_parsed.csv (MITI, already
                     mapped to MISC T2 in `weak_t2`) -> t2_label_GT, counsellor.

Design choices that matter for a fair comparison:
  * ALL volleys are kept, so the context window the model reads is intact; gold
    is set ONLY on the scorable shared-vocab rows and left blank elsewhere, so
    `baseline.eval` scores those rows and ignores the rest.
  * Multi-label AnnoMI volleys and out-of-scheme codes get no gold (context
    only), never a guessed single label.

Usage:
    PYTHONPATH=src python -m baseline.prep_crossscheme annomi
    PYTHONPATH=src python -m baseline.prep_crossscheme welivita --sample-convs 200
    PYTHONPATH=src python -m baseline.prep_crossscheme all

Then predict + score (see docs); e.g. for AnnoMI:
    ... local_arm predict --arm ft_bare --inf bare --ctx 5 \
        dataset.eval_csv=data/manual/AnnoMI_eval.csv dataset.eval_name=annomi
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd

from baseline.local_arm import REPO_ROOT

OUT_DIR = REPO_ROOT / "data" / "manual"

# Shared vocabulary per corpus/tier — the crosswalk. Codes here are IDENTICAL in
# both schemes; see docs/EXPERIMENTS.md. Anything outside gets no gold.
ANNOMI_CLIENT_T1 = {"C", "S", "N"}
ANNOMI_COUNS_T2 = {"OQ", "CQ", "SR", "CR", "GI"}
WELIVITA_COUNS_T2 = {"GI", "ADW", "CR", "SU", "AF", "CQ", "DI",
                     "SR", "ADP", "OQ", "CO", "EC", "WA"}

# The `load_manual` / context-builder schema.
SCHEMA = [
    "corp_conv_idx", "conv_id", "speaker", "corp_vol_idx", "conv_vol_idx",
    "vol_text", "corp_utt_idx", "conv_utt_idx", "utt_text",
    "t1_label_GT", "t2_label_GT",
]


def _sample(df: pd.DataFrame, n_convs: int | None, seed: int = 0) -> pd.DataFrame:
    """Keep the first `n_convs` conversations (by a seeded shuffle), or all."""
    if not n_convs:
        return df
    convs = df["conv_id"].astype(str).drop_duplicates()
    keep = set(convs.sample(min(n_convs, len(convs)), random_state=seed))
    return df[df["conv_id"].astype(str).isin(keep)].copy()


def _index(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the corpus/volley/utterance indices `load_manual` sorts on.

    Every row here is one volley (=one utterance), in file order within each
    conversation, which is the transcript order the context builder needs.
    """
    df = df.reset_index(drop=True)
    df["corp_conv_idx"] = pd.factorize(df["conv_id"])[0]
    within = df.groupby("conv_id", sort=False).cumcount()
    df["conv_vol_idx"] = within
    df["conv_utt_idx"] = within
    df["corp_vol_idx"] = range(len(df))
    df["corp_utt_idx"] = range(len(df))
    return df


def build_annomi(sample_convs: int | None) -> pd.DataFrame:
    src = REPO_ROOT / "data" / "AnnoMI.csv"
    raw = pd.read_csv(src)
    raw = raw.rename(columns={"AnnoMI Label": "label"})
    raw = _sample(raw, sample_convs)
    raw = _index(raw)

    raw["utt_text"] = raw["vol_text"].astype(str)
    t1, t2 = [], []
    for _, r in raw.iterrows():
        codes = None
        try:
            codes = ast.literal_eval(r["label"]) if pd.notna(r["label"]) else None
        except (ValueError, SyntaxError):
            codes = [r["label"]]
        code = codes[0] if isinstance(codes, list) and len(codes) == 1 else None
        sp = str(r["speaker"]).strip().lower()
        # Client -> T1 change/sustain axis; therapist -> T2 counsellor codes.
        if sp == "client" and code in ANNOMI_CLIENT_T1:
            t1.append(code); t2.append(None)
        elif sp in ("therapist", "counsellor") and code in ANNOMI_COUNS_T2:
            t1.append(None); t2.append(code)
        else:
            t1.append(None); t2.append(None)      # context only
    raw["t1_label_GT"], raw["t2_label_GT"] = t1, t2
    return raw[SCHEMA]


def build_welivita(sample_convs: int | None) -> pd.DataFrame:
    src = REPO_ROOT / "data" / "external" / "welivita_mi_parsed.csv"
    raw = pd.read_csv(src)
    raw = _sample(raw, sample_convs)
    raw = raw.reset_index(drop=True)
    # The parsed file already carries the schema indices; only the gold columns
    # need attaching. MITI is therapist-side, so gold is T2 counsellor only.
    weak = raw.get("weak_t2")
    valid = weak.notna() & weak.isin(WELIVITA_COUNS_T2) if weak is not None else False
    is_couns = raw["speaker"].astype(str).str.lower().isin(("counsellor", "counselor"))
    raw["t2_label_GT"] = raw["weak_t2"].where(valid & is_couns)
    raw["t1_label_GT"] = pd.NA
    for col in SCHEMA:
        if col not in raw.columns:
            raw[col] = pd.NA
    return raw[SCHEMA]


BUILDERS = {"annomi": build_annomi, "welivita": build_welivita}
OUT_NAMES = {"annomi": "AnnoMI_eval.csv", "welivita": "Welivita_eval.csv"}


def _report(name: str, df: pd.DataFrame) -> None:
    n_t1 = int(df["t1_label_GT"].notna().sum())
    n_t2 = int(df["t2_label_GT"].notna().sum())
    print(
        f"{name}: {len(df)} rows, {df['conv_id'].nunique()} conversations; "
        f"scorable gold -> T1={n_t1}, T2={n_t2}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("which", choices=[*BUILDERS, "all"])
    ap.add_argument("--sample-convs", type=int, default=None,
                    help="keep only this many conversations (seeded); default all")
    args = ap.parse_args()
    targets = list(BUILDERS) if args.which == "all" else [args.which]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in targets:
        df = BUILDERS[name](args.sample_convs)
        out = OUT_DIR / OUT_NAMES[name]
        df.to_csv(out, index=False)
        _report(name, df)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
