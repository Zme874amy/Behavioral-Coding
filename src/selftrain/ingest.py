"""Ingest external MI corpora into the parsed pool schema.

`label_pool.py` consumes any CSV with the columns `conv_id, speaker, utt_text,
corp_utt_idx` (plus optional volley/context indices). This converts external
datasets into that schema so they can serve as a pseudo-labeling pool.

Currently supports:
  welivita — Welivita & Pu (2022), "Curating a Large-Scale Motivational
             Interviewing Dataset using Peer Support Forums" (COLING 2022).
             ~2k dialogues, utterance-segmented, MITI-labeled listener turns.
             github.com/anuradha1992/Motivational-Interviewing-Dataset
             Licence: CC BY-NC-SA 3.0 (non-commercial research).

Usage:
    PYTHONPATH=src python -m selftrain.ingest welivita \
        --src "/path/to/MI Dataset.csv" \
        --out data/external/welivita_mi_parsed.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Welivita listener label -> MISC 2.5 counsellor T2 (schemes.mappings). Labels
# with no clean MISC counterpart (Self-Disclose, Other) are left unmapped.
from schemes.mappings import WELIVITA_TO_MISC_T2 as MITI_TO_MISC_T2  # noqa: E402


def ingest_welivita(src: Path) -> pd.DataFrame:
    df = pd.read_csv(src)
    df["speaker"] = df["author"].map(lambda a: "counsellor" if str(a).strip() == "listener" else "client")
    df["utt_text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["utt_text"].str.len() > 0].reset_index(drop=True)

    out_rows = []
    corp_idx = 0
    for corp_conv_idx, (conv_id, grp) in enumerate(df.groupby("dialog_id", sort=False)):
        for conv_i, (_, row) in enumerate(grp.iterrows()):
            weak = MITI_TO_MISC_T2.get(str(row.get("final agreed label", "")).strip())
            out_rows.append({
                "corp_conv_idx": corp_conv_idx,
                "conv_id": str(conv_id),
                "speaker": row["speaker"],
                "corp_vol_idx": corp_idx,
                "conv_vol_idx": int(row.get("turn", conv_i)) if str(row.get("turn", "")).strip().isdigit() else conv_i,
                "vol_text": row["utt_text"],
                "corp_utt_idx": corp_idx,
                "conv_utt_idx": conv_i,
                "utt_text": row["utt_text"],
                "weak_miti": row.get("final agreed label"),
                "weak_t2": weak,
            })
            corp_idx += 1
    return pd.DataFrame(out_rows)


INGESTERS = {"welivita": ingest_welivita}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", choices=list(INGESTERS))
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    out = INGESTERS[args.dataset](Path(args.src))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    n_weak = int(out["weak_t2"].notna().sum()) if "weak_t2" in out.columns else 0
    print(
        f"Ingested {args.dataset}: {len(out)} utterances, {out['conv_id'].nunique()} conversations, "
        f"{n_weak} with a weak MISC-T2 label -> {out_path}"
    )


if __name__ == "__main__":
    main()
