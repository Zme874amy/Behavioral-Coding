"""Cleaning / processing audit: what is wrong with each dataset, and what to do.

Every check returns rows of (dataset, check, n, share, example, severity, action).
Nothing here edits a data file. `action` says whether the issue
  - needs no change ("ok", "expected"),
  - is handled downstream already ("handled: <where>"),
  - should be fixed in a derived copy (`data/clean/`, never the original), or
  - needs a decision from the user ("decide").
"""
from __future__ import annotations

import re
from typing import Dict, List

import pandas as pd

MOJIBAKE = re.compile(r"â€|Ã.|�|Â ")


def _row(ds, check, n, total, example="", severity="info", action="ok"):
    return {"dataset": ds, "check": check, "n": int(n), "share": round(n / total, 4) if total else 0.0,
            "example": str(example)[:120], "severity": severity, "action": action}


def _norm(t: str) -> str:
    return re.sub(r"[^a-z' ]", "", str(t).lower()).strip()


def misc_vocab():
    from automisc_ft.data import CLIENT_GROUPS, COUNSELLOR_GROUPS
    t1_of = {}
    for spk, groups in (("counsellor", COUNSELLOR_GROUPS), ("client", CLIENT_GROUPS)):
        for g, cs in groups.items():
            for c in cs:
                t1_of[(spk, c)] = g
    return t1_of


def audit(ds: str, df: pd.DataFrame) -> List[dict]:
    rows, n = [], len(df)
    txt = df.text.astype(str)

    empty = txt.str.strip().isin(["", "nan", "None"])
    rows.append(_row(ds, "empty / whitespace text", empty.sum(), n, severity="warn" if empty.any() else "info",
                     action="drop in derived copy" if empty.any() else "ok"))
    moj = txt.str.contains(MOJIBAKE)
    rows.append(_row(ds, "mojibake / bad encoding", moj.sum(), n, txt[moj].head(1).tolist() or "",
                     severity="warn" if moj.any() else "info", action="fix in derived copy" if moj.any() else "ok"))

    order_bad = 0
    for _, g in df.groupby("conv_id", sort=False):
        u = pd.to_numeric(g.utt_idx, errors="coerce")
        order_bad += int((u.diff().dropna() < 0).sum() + u.duplicated().sum())
    rows.append(_row(ds, "utterance index not monotonic/unique within conv", order_bad, n,
                     severity="warn" if order_bad else "info", action="ok" if not order_bad else "decide"))

    consec = (df.conv_id.eq(df.conv_id.shift()) & df.speaker.eq(df.speaker.shift())
              & txt.str.strip().eq(txt.shift().astype(str).str.strip()) & ~empty)
    rows.append(_row(ds, "consecutive identical utterances (same speaker)", consec.sum(), n,
                     txt[consec].head(1).tolist() or "", action="inspect" if consec.sum() else "ok"))

    has_upper = txt.str.contains(r"[A-Z]")
    has_punct = txt.str.contains(r"[.?!,]")
    rows.append(_row(ds, "no casing (ASR signal)", (~has_upper & ~empty).sum(), n,
                     action="expected for ASR corpora"))
    rows.append(_row(ds, "no punctuation (ASR signal)", (~has_punct & ~empty).sum(), n,
                     action="expected for ASR corpora"))

    if df.native.notna().any():
        for spk in ("counsellor", "client"):
            s = df[df.speaker == spk]
            if len(s):
                miss = s.native.isna().sum()
                rows.append(_row(ds, f"uncoded {spk} rows", miss, len(s),
                                 action="expected (context rows / scheme does not code this speaker)"))

    if df.t2.notna().any():
        t1_of = misc_vocab()
        lab = df[df.t2.notna()]
        bad = lab[[(s, c) not in t1_of for s, c in zip(lab.speaker, lab.t2)]]
        rows.append(_row(ds, "T2 code not in MISC vocab for that speaker", len(bad), len(lab),
                         f"{bad.speaker.iloc[0]}:{bad.t2.iloc[0]}" if len(bad) else "",
                         severity="error" if len(bad) else "info", action="fix in derived copy" if len(bad) else "ok"))
        if df.t1.notna().any():
            both = lab[lab.t1.notna()]
            incons = both[[t1_of.get((s, c)) not in (None, t) for s, c, t in zip(both.speaker, both.t2, both.t1)]]
            rows.append(_row(ds, "T1 inconsistent with T2 group", len(incons), len(both),
                             f"{incons.t1.iloc[0]}/{incons.t2.iloc[0]}" if len(incons) else "",
                             severity="error" if len(incons) else "info",
                             action="fix in derived copy" if len(incons) else "ok"))

        # Label consistency of identical short utterances (a cheap annotation-noise probe).
        lab = lab.assign(k=lab.text.map(_norm))
        rep = lab[lab.k.str.split().str.len().between(1, 4)].groupby(["speaker", "k"]).t2.agg(["nunique", "size"])
        rep = rep[rep["size"] >= 3]
        if len(rep):
            conflict = rep[rep["nunique"] > 1]
            ex = ", ".join(f"'{k}'" for _, k in conflict.sort_values("size", ascending=False).head(3).index)
            rows.append(_row(ds, "short utterances (>=3 occurrences) with conflicting T2", len(conflict), len(rep),
                             ex, severity="warn" if len(conflict) else "info",
                             action="annotation noise: report, do not relabel"))
    return rows


def dataset_specific(frames: Dict[str, pd.DataFrame], pairs: pd.DataFrame) -> List[dict]:
    """Checks that only make sense for one corpus or need the overlap table."""
    rows = []
    if "misc.hlqc.gold" in frames:
        h = frames["misc.hlqc.gold"]
        rows.append(_row("misc.hlqc.gold", "row count vs paper (1,924)", len(h) - 1924, 1924, f"{len(h)} rows",
                         action="ok: off-by-one, likely a header/trailing row in the paper count"))
    if not pairs.empty:
        hq = pairs[(pairs.dataset_a == "pool.hlqc") & (pairs.dataset_b == "pool.hlqc")
                   & (pairs.containment >= 0.5)]
        rows.append(_row("pool.hlqc", "near-duplicate sessions inside the pool (containment >= 0.5)", len(hq),
                         frames["pool.hlqc"].conv_id.nunique(),
                         "; ".join(f"{a}={b}" for a, b in zip(hq.conv_a.head(3), hq.conv_b.head(3))),
                         severity="warn", action="dedupe pool in derived copy (keep one per cluster)"))
        conflict = hq[hq.conv_a.str[:3] != hq.conv_b.str[:3]]
        rows.append(_row("pool.hlqc", "duplicate sessions with CONFLICTING high/low quality label", len(conflict),
                         max(1, len(hq)), "; ".join(f"{a}={b}" for a, b in zip(conflict.conv_a.head(3), conflict.conv_b.head(3))),
                         severity="warn", action="decide (session-quality label unreliable for these)"))
        gp = pairs[(pairs.dataset_a == "misc.hlqc.gold") & (pairs.dataset_b == "pool.hlqc")
                   & (pairs.conv_a != pairs.conv_b) & (pairs.containment >= 0.5)]
        rows.append(_row("pool.hlqc", "copies of HLQC GOLD (train) sessions under another pool id", len(gp), 10,
                         "; ".join(f"{a}={b}" for a, b in zip(gp.conv_a, gp.conv_b)),
                         severity="warn", action="exclude from pool when the pool is used for evaluation"))
        wd = pairs[(pairs.dataset_a == "welivita.gold") & (pairs.dataset_b == "welivita.gold")
                   & (pairs.containment >= 0.9)]
        rows.append(_row("welivita.gold", "duplicate dialogues (containment >= 0.9)", len(wd),
                         frames["welivita.gold"].conv_id.nunique(),
                         "; ".join(f"{a}={b}" for a, b in zip(wd.conv_a.head(2), wd.conv_b.head(2))),
                         severity="warn", action="split by duplicate cluster; dedupe in derived copy"))
        for s in [k for k in frames if k.startswith("synth.")]:
            sp = pairs[(pairs.dataset_a == s) & (pairs.dataset_b == s) & (pairs.containment >= 0.5)]
            rows.append(_row(s, "near-duplicate synthetic windows (containment >= 0.5)", len(sp),
                             frames[s].conv_id.nunique(), action="dedupe in derived copy" if len(sp) else "ok",
                             severity="warn" if len(sp) else "info"))
    if "miti.casaa.gold" in frames:
        c = frames["miti.casaa.gold"]
        cs = c[c.speaker == "counsellor"]
        rows.append(_row("miti.casaa.gold", "multi-code turns left without gold (align=multi)",
                         (cs["align"] == "multi").sum(), len(cs), action="handled: no gold assigned"))
        rows.append(_row("miti.casaa.gold", "SAME continuation rows", (cs.native == "SAME").sum(), len(cs),
                         action="handled: no gold assigned"))
        rows.append(_row("miti.casaa.gold", "client rows carrying a code (source quirk)",
                         ((c.speaker == "client") & c.native.notna()).sum(), (c.speaker == "client").sum(),
                         action="handled: client rows never get gold"))
    if "annomi.gold" in frames:
        import pandas as _pd
        from eda.registry import EXT
        raw = _pd.read_csv(EXT / "annomi" / "AnnoMI-simple.csv").groupby("transcript_id").topic.first()
        variants = raw[raw != raw.str.strip()]
        rows.append(_row("annomi.gold", "topic labels with stray whitespace", len(variants), len(raw),
                         repr(variants.iloc[0]) if len(variants) else "", action="handled: stripped in eda.registry"))
    if "welivita.gold" in frames:
        w = frames["welivita.gold"]
        rows.append(_row("welivita.gold", "listener rows with '-' / no final label", (w.speaker == "counsellor").sum()
                         - w[(w.speaker == "counsellor")].native.notna().sum(), (w.speaker == "counsellor").sum(),
                         action="expected: uncodable segments"))
    return rows


def run(frames: Dict[str, pd.DataFrame], pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, df in frames.items():
        rows += audit(ds, df)
    rows += dataset_specific(frames, pairs)
    return pd.DataFrame(rows)
