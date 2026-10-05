"""Per-fold label statistics, computed from the fold's TRAINING rows only.

Replaces the test-informed constants (retriever RARE_T2, self-training
RARE_DEFAULT, synthesis RARE_CODES, classic TAIL: all picked by looking at
MIV6.3A; EXPERIMENTS.md "Disclosure"). One rule everywhere, so "rare" means the
same thing in every arm and every fold (docs/RERUN_PLAN.md P3):

    rare        a code of the speaker's vocabulary with fewer than RARE_MIN
                labelled training rows, or under RARE_FRAC of that speaker's
                labelled rows (never-seen codes included)
    learnable   occurs at least once in the training rows (eval: Macro-F1 learnable)
    never_seen  in the vocabulary, absent from training (reported N/A, synthetic-only)
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from schemes.misc import t1_codes, t2_codes

RARE_MIN = 20
RARE_FRAC = 0.01
SPEAKERS = ("counsellor", "client")


def _vocab(level: str, speaker: str) -> List[str]:
    return t2_codes(speaker) if level == "t2" else t1_codes(speaker)


def counts(train: pd.DataFrame, level: str = "t2") -> Dict[str, pd.Series]:
    col = f"{level}_label_GT"
    out = {}
    for spk in SPEAKERS:
        lab = train.loc[train["speaker"] == spk, col].dropna().astype(str)
        out[spk] = lab.value_counts().reindex(_vocab(level, spk), fill_value=0)
    return out


def rare_codes(train: pd.DataFrame, level: str = "t2") -> Dict[str, List[str]]:
    res = {}
    for spk, c in counts(train, level).items():
        total = int(c.sum())
        res[spk] = [k for k, n in c.items() if n < RARE_MIN or (total and n / total < RARE_FRAC)]
    return res


def learnable_codes(train: pd.DataFrame, level: str = "t2") -> Dict[str, List[str]]:
    return {spk: [k for k, n in c.items() if n > 0] for spk, c in counts(train, level).items()}


def never_seen(train: pd.DataFrame, level: str = "t2") -> Dict[str, List[str]]:
    return {spk: [k for k, n in c.items() if n == 0] for spk, c in counts(train, level).items()}


def summary(train: pd.DataFrame) -> dict:
    """Everything a run records in its .meta.json about its training labels."""
    return {"rule": f"rare = < {RARE_MIN} rows or < {RARE_FRAC:.0%} of the speaker's labelled training rows",
            "rare_t2": rare_codes(train, "t2"), "never_seen_t2": never_seen(train, "t2"),
            "learnable_t2": learnable_codes(train, "t2"), "learnable_t1": learnable_codes(train, "t1")}
