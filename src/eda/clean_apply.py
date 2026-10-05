"""Build the derived "cleaned" HLQC gold copy, used as an ABLATION only.

The original `data/manual/HLQC_balanced_manual.csv` stays the default training
set and is never edited. This writes a copy with two rule-based fixes, plus a
changelog listing every changed row, so that "train on cleaned HLQC" can be run
as one extra arm (docs/RERUN_PLAN.md P1.1; user decision 2026-10-05).

Rules (each cites its evidence; nothing else is changed):
  R1  A standalone counsellor acknowledgement coded FI becomes FA.
      MISC 2.5 p.22: Facilitate (FA) covers simple acknowledgements ("mm hmm",
      "okay", "I see"); FI is for pleasantries/fillers. In HLQC 64% of such
      acknowledgements are coded FI (eda.quality.fi_content; DATASETS.md section 4).
      "Standalone" = the whole normalised utterance is in `eda.quality.ACK`.
      FA and FI are both in T1 group O, so T1 does not change.
  R2  A row whose T1 contradicts its T2 code gets T1 := the group of T2
      (schemes.misc.t1_of). T2 is the finer, more deliberate decision.
      Found by eda.clean.audit ("T1 inconsistent with T2 group").

Deliberately NOT changed (they need human judgement; documented as known noise
in DATASETS.md section 4): change talk coded N, the SR/CR boundary, SU/EC
under-coding.

    PYTHONPATH=src python -m eda.clean_apply          # writes data/clean/
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from eda.quality import ACK, norm
from schemes.misc import t1_of

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data" / "manual" / "HLQC_balanced_manual.csv"
OUT_DIR = REPO / "data" / "clean"
OUT = OUT_DIR / "misc.hlqc.gold.cleaned.csv"
CHANGELOG = OUT_DIR / "misc.hlqc.gold.cleaned.changelog.csv"


def apply_rules(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    log = []
    spk = out["speaker"].astype(str).str.strip().str.lower()

    # R1: standalone acknowledgement coded FI -> FA
    r1 = (spk == "counsellor") & (out["t2_label_GT"] == "FI") & out["utt_text"].map(norm).isin(ACK)
    for i in out.index[r1]:
        log.append({"rule": "R1 FI->FA (standalone acknowledgement, MISC 2.5 p.22)", "row": i,
                    "conv_id": out.at[i, "conv_id"], "corp_utt_idx": out.at[i, "corp_utt_idx"],
                    "speaker": spk[i], "utt_text": out.at[i, "utt_text"], "field": "t2_label_GT",
                    "before": "FI", "after": "FA"})
    out.loc[r1, "t2_label_GT"] = "FA"

    # R2: T1 := group(T2) where they contradict
    for i in out.index[out["t2_label_GT"].notna() & out["t1_label_GT"].notna()]:
        g = t1_of(spk[i], out.at[i, "t2_label_GT"])
        if g is not None and g != out.at[i, "t1_label_GT"]:
            log.append({"rule": "R2 T1:=group(T2) (T1/T2 contradiction)", "row": i,
                        "conv_id": out.at[i, "conv_id"], "corp_utt_idx": out.at[i, "corp_utt_idx"],
                        "speaker": spk[i], "utt_text": out.at[i, "utt_text"], "field": "t1_label_GT",
                        "before": out.at[i, "t1_label_GT"], "after": g})
            out.at[i, "t1_label_GT"] = g
    return out, pd.DataFrame(log)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, default=SRC)
    ap.parse_args()
    df = pd.read_csv(SRC, encoding="utf-8-sig")
    out, log = apply_rules(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    log.to_csv(CHANGELOG, index=False)
    sha = hashlib.sha256(SRC.read_bytes()).hexdigest()[:12]
    print(f"source {SRC.relative_to(REPO)} (sha256 {sha}), {len(df)} rows")
    print(log.groupby("rule").size().to_string() if len(log) else "no changes")
    print(f"wrote {OUT.relative_to(REPO)} and {CHANGELOG.relative_to(REPO)}")


if __name__ == "__main__":
    main()
