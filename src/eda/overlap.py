"""Which dataset overlaps which: one explicit table instead of tribal knowledge.

Three kinds of relation are reported, and they mean different things for splits:

  subset     the gold file is a labelled slice of a pool (same sessions, same
             conv_id). Known by construction; verified by conv_id.
  derived    two files hold the same data in different formats/schemes
             (e.g. data/AnnoMI.csv -> data/manual/AnnoMI_eval.csv). Listed from
             the registry lineage; not a leak, but never count both.
  duplicate  the same session appears in two corpora under different ids,
             usually because both scraped the same YouTube video. Detected by
             word-shingle containment between conversations (below), because the
             texts differ (manual vs ASR transcripts).

Containment: for conversations A and B, shared k-gram shingles divided by the
smaller conversation's shingle count, after dropping shingles that occur in more
than `max_df` conversations (templated chatbot lines, stock MI phrases). With k=5
an unrelated pair scores ~0.00-0.02; known duplicates score 0.13-0.55 (ASR noise
keeps manual-vs-ASR pairs well below 1).
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from eda.registry import META

_TOK = re.compile(r"[a-z']+")

# Relations known by construction (subset / derived). Verified in `known_relations`.
KNOWN = [
    ("misc.hlqc.gold", "pool.hlqc", "subset", "the 10 gold sessions are pool sessions (same conv_id)"),
    ("misc.miv63a.gold", "pool.miv63a", "subset", "the 10 test sessions are pool sessions (same conv_id)"),
    ("annomi.gold", "annomi.gold", "derived",
     "AnnoMI-full -> AnnoMI-simple -> data/AnnoMI.csv (AutoMISC MISC map) -> data/manual/AnnoMI_eval.csv; parsed/AnnoMI_parsed.csv"),
    ("welivita.gold", "welivita.gold", "derived",
     "release MI_Dataset.csv -> external/welivita_mi_parsed.csv -> manual/Welivita_eval.csv; selftrain pseudo-labels are model output"),
]


# Relations of our generated sets to real data (docs/SYNTHETIC_DATA.md).
SYNTH_KNOWN = [
    ("synth.*", "misc.hlqc.gold", "derived",
     "every data/synth/aug file = the 1,925 HLQC gold rows + synthetic rows; synthesis exemplars are HLQC gold"),
    ("synth.v3_mivmix", "misc.miv63a.gold", "derived",
     "topic mix measured on the MIV6.3A test set (statistics only, no text)"),
]


def shingles(text: str, k: int = 5) -> set:
    w = _TOK.findall(text.lower())
    return {" ".join(w[i:i + k]) for i in range(max(0, len(w) - k + 1))}


def conv_shingles(df: pd.DataFrame, k: int = 5) -> Dict[str, set]:
    return {c: shingles(" ".join(g.text.astype(str)), k) for c, g in df.groupby("conv_id", sort=False)}


def pairwise(frames: Dict[str, pd.DataFrame], k: int = 5, max_df: int = 5,
             min_shared: int = 15, threshold: float = 0.05) -> pd.DataFrame:
    """All conversation pairs (across and within datasets) above `threshold`."""
    docs: Dict[Tuple[str, str], set] = {}
    for ds, df in frames.items():
        for c, sh in conv_shingles(df, k).items():
            docs[(ds, c)] = sh
    index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for d, sh in docs.items():
        for s in sh:
            index[s].append(d)
    shared: Dict[Tuple, int] = defaultdict(int)
    for s, ds in index.items():
        if 1 < len(ds) <= max_df:
            for i in range(len(ds)):
                for j in range(i + 1, len(ds)):
                    shared[(ds[i], ds[j])] += 1
    rows = []
    for (a, b), n in shared.items():
        if n < min_shared:
            continue
        cont = n / max(1, min(len(docs[a]), len(docs[b])))
        if cont >= threshold:
            rows.append({"dataset_a": a[0], "conv_a": a[1], "dataset_b": b[0], "conv_b": b[1],
                         "shared_shingles": n, "containment": round(cont, 3),
                         "size_a": len(docs[a]), "size_b": len(docs[b])})
    out = pd.DataFrame(rows)
    if not len(out):
        return out
    # Canonical order inside each pair and a total sort, so the table is identical across runs.
    swap = (out.dataset_a + "\x00" + out.conv_a) > (out.dataset_b + "\x00" + out.conv_b)
    for x, y in (("dataset_a", "dataset_b"), ("conv_a", "conv_b"), ("size_a", "size_b")):
        out.loc[swap, [x, y]] = out.loc[swap, [y, x]].values
    return out.sort_values(["containment", "dataset_a", "conv_a", "dataset_b", "conv_b"],
                           ascending=[False, True, True, True, True]).reset_index(drop=True)


def known_relations(frames: Dict[str, pd.DataFrame], synth: bool = False) -> pd.DataFrame:
    """The by-construction relations, with the conv_id check where applicable."""
    rows = []
    for a, b, rel, note in (SYNTH_KNOWN if synth else KNOWN):
        check = ""
        if rel == "subset" and a in frames and b in frames:
            ga, gb = set(frames[a].conv_id), set(frames[b].conv_id)
            check = f"{len(ga & gb)}/{len(ga)} {a} sessions found in {b}"
        rows.append({"dataset_a": a, "dataset_b": b, "relation": rel, "detail": note, "verified": check})
    return pd.DataFrame(rows)


def summarise(pairs: pd.DataFrame, frames: Dict[str, pd.DataFrame], strong: float = 0.10) -> pd.DataFrame:
    """Dataset x dataset summary of duplicate sessions (containment >= `strong`),
    excluding pairs explained by a known subset relation (same conv_id)."""
    if pairs.empty:
        return pairs
    p = pairs[(pairs.containment >= strong) & ~((pairs.conv_a == pairs.conv_b))]
    p = p[~p.apply(lambda r: {r.dataset_a, r.dataset_b} in
                   [{"misc.hlqc.gold", "pool.hlqc"}, {"misc.miv63a.gold", "pool.miv63a"}]
                   and r.conv_a == r.conv_b, axis=1)] if len(p) else p
    g = (p.groupby(["dataset_a", "dataset_b"])
         .apply(lambda x: pd.Series({
             "n_pairs": len(x),
             "sessions": "; ".join(f"{a}={b} ({c:.2f})" for a, b, c in
                                   zip(x.conv_a, x.conv_b, x.containment))}), include_groups=False)
         .reset_index())
    return g


__all__ = ["pairwise", "known_relations", "summarise", "shingles", "KNOWN", "SYNTH_KNOWN"]
