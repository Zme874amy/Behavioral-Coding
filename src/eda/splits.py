"""Seeded split manifests + exclusion lists, with leakage checks.

Writes data/splits/<name>.json (small, tracked in git). Data files are never
modified: de-duplication and leakage removal are expressed as `exclude` lists
that consumers apply, so every past result stays reproducible from the originals.

    PYTHONPATH=src python -m eda.splits          # build all manifests + run checks

Splits are by conversation (never by utterance), and duplicate sessions found by
`eda.overlap` are kept in the same split (Welivita) or excluded from the test side
of any pairing where their twin is training data.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set

import pandas as pd

from eda import overlap, registry

OUT = registry.REPO / "data" / "splits"
SEED = 42
DUP = 0.10          # containment at/above which two sessions count as the same session


def _components(pairs: pd.DataFrame, nodes: Iterable[str]) -> Dict[str, str]:
    """Union-find over duplicate pairs -> cluster id per node."""
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in zip(pairs.conv_a, pairs.conv_b):
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    # Name each cluster by its smallest member, so ids do not depend on pair order
    # (pair order follows set iteration, which changes with PYTHONHASHSEED).
    members = defaultdict(list)
    for n in parent:
        members[find(n)].append(n)
    canon = {r: min(m) for r, m in members.items()}
    return {n: canon[find(n)] for n in parent}


def _allocate(groups: Dict[str, List[str]], fracs: Dict[str, float], rng: random.Random,
              forced: Dict[str, str] = None) -> Dict[str, str]:
    """Assign whole groups to splits, stratum by stratum, by largest remainder."""
    forced = forced or {}
    out = dict(forced)
    for stratum in sorted(groups):
        items = sorted(g for g in groups[stratum] if g not in forced)
        rng.shuffle(items)
        n = len(items)
        want = {s: f * n for s, f in fracs.items()}
        take = {s: int(w) for s, w in want.items()}
        for s in sorted(want, key=lambda s: want[s] - take[s], reverse=True)[: n - sum(take.values())]:
            take[s] += 1
        i = 0
        for s in fracs:
            for g in items[i:i + take[s]]:
                out[g] = s
            i += take[s]
    return out


def build(frames=None, pairs=None) -> Dict[str, dict]:
    """`pairs` (real datasets only) drives every real-data manifest, so those never
    depend on our regenerable synthetic files; the synthetic manifest is computed
    separately when synth.* frames are present."""
    frames = frames or registry.load_all(include_synth=True)
    real = {k: v for k, v in frames.items() if not k.startswith("synth.")}
    pairs = overlap.pairwise(real) if pairs is None else pairs
    pairs = pairs[~pairs.dataset_a.str.startswith("synth.") & ~pairs.dataset_b.str.startswith("synth.")]
    dup = pairs[(pairs.containment >= DUP) & (pairs.conv_a != pairs.conv_b)]
    rng = random.Random(SEED)
    man: Dict[str, dict] = {}

    def twins(ds_a: str, ids: Set[str], ds_b: str) -> Set[str]:
        """Sessions of ds_b that duplicate any of `ids` in ds_a (either column order)."""
        m1 = dup[(dup.dataset_a == ds_a) & (dup.dataset_b == ds_b) & dup.conv_a.isin(ids)].conv_b
        m2 = dup[(dup.dataset_b == ds_a) & (dup.dataset_a == ds_b) & dup.conv_b.isin(ids)].conv_a
        return set(m1) | set(m2)

    # --- MISC main setting (unchanged) -------------------------------------
    from automisc_ft.data import assign_folds
    hq = frames["misc.hlqc.gold"]
    train_ids = set(hq.conv_id)
    folds = assign_folds(hq, 5, SEED)          # == baseline.oof_t1._split (FOLD_SEED 42)
    man["misc_main"] = {
        "scheme": "MISC 2.5",
        "train": {"misc.hlqc.gold": sorted(train_ids)},
        "dev": {"misc.hlqc.gold (5-fold CV, baseline.oof_t1)": {c: int(f) for c, f in sorted(folds.items())}},
        "test": {"misc.miv63a.gold": sorted(set(frames["misc.miv63a.gold"].conv_id)),
                 "miti.casaa.gold (MISC-mapped)": sorted(
                     set(frames["miti.casaa.gold"].conv_id) - twins("misc.hlqc.gold", train_ids, "miti.casaa.gold")
                     - twins("pool.hlqc", set(frames["pool.hlqc"].conv_id), "miti.casaa.gold"))},
        "exclude": {
            "miti.casaa.gold": sorted(twins("misc.hlqc.gold", train_ids, "miti.casaa.gold")
                                      | twins("pool.hlqc", set(frames["pool.hlqc"].conv_id), "miti.casaa.gold")),
            "annomi.gold (when used as transfer test of HLQC-gold-trained models)":
                sorted(twins("misc.hlqc.gold", train_ids, "annomi.gold"), key=int),
            "annomi.gold (when the model also saw pool.hlqc: self-training/retrieval arms)":
                sorted(twins("pool.hlqc", set(frames["pool.hlqc"].conv_id), "annomi.gold"), key=int),
        },
        "notes": "Unchanged from all prior experiments. CASAA sessions duplicating any HLQC session are excluded.",
    }

    # --- PROPOSED: pooled training with MIV cross-validation -----------------
    # docs/experiments/2026-10-05-split-review.md. Alongside misc_main, not replacing it:
    # every MIV6.3A session is test exactly once, so scores still cover all 821 utterances.
    miv_folds = assign_folds(frames["misc.miv63a.gold"], 5, SEED)
    man["misc_pooled_cv"] = {
        "status": "PROPOSED (pending decision); misc_main stays the reported setting until then",
        "scheme": "MISC 2.5",
        "folds": {str(k): {
            "train": {"misc.hlqc.gold": sorted(train_ids),
                      "misc.miv63a.gold": sorted(c for c, f in miv_folds.items() if f != k)},
            "test": {"misc.miv63a.gold": sorted(c for c, f in miv_folds.items() if f == k)}}
            for k in range(5)},
        "checkpoint_selection": "HLQC validation fold as in misc_main (val_folds 7, val_fold 0); never an MIV session",
        "external_test": man["misc_main"]["test"]["miti.casaa.gold (MISC-mapped)"],
        "exclude": man["misc_main"]["exclude"],
        "notes": "Answers RQ1 as worded (a service adapting on labels it already holds). misc_main remains the "
                 "cold-start transfer setting (no in-domain labels).",
    }

    # --- AnnoMI own-scheme ---------------------------------------------------
    an = frames["annomi.gold"]
    tr = an.groupby("conv_id").agg(q=("mi_quality", "first"),
                                   ann=("annotator_id", lambda s: s[s >= 0].mode().iloc[0] if (s >= 0).any() else -1),
                                   multi=("n_annotators", "max"))
    multi = set(tr[tr.multi > 1].index)
    an_pairs = dup[(dup.dataset_a == "annomi.gold") & (dup.dataset_b == "annomi.gold")]
    comp = _components(an_pairs, tr.index)
    if len(set(comp.values())) != len(comp):
        raise RuntimeError("AnnoMI has duplicate transcripts; cluster them before splitting")
    strata = defaultdict(list)
    for c, r in tr.iterrows():
        strata[f"{r.q}|ann{int(r.ann)}"].append(c)
    n_test_extra = max(0, round(0.15 * len(tr)) - len(multi))
    assign = _allocate(strata, {"train": 0.70 / 0.85, "dev": 0.15 / 0.85}, rng,
                       forced={c: "test" for c in multi})
    # top up the test split with a stratified draw from train
    pool_tr = sorted((c for c, s in assign.items() if s == "train"), key=int)
    rng.shuffle(pool_tr)
    by_q = defaultdict(list)
    for c in pool_tr:
        by_q[tr.loc[c, "q"]].append(c)
    quota = {q: round(n_test_extra * len(v) / len(pool_tr)) for q, v in by_q.items()}
    for q, k in quota.items():
        for c in by_q[q][:k]:
            assign[c] = "test"
    man["annomi_own"] = {
        "scheme": "AnnoMI (main behaviour + subtypes; client change/neutral/sustain)",
        "split": {c: assign[c] for c in sorted(assign, key=int)},
        "test_core_10rater": sorted(multi, key=int),
        "exclude": man["misc_main"]["exclude"],
        "notes": "Transcript-level; stratified by mi_quality x annotator; the 7 ten-rater transcripts (majority labels) "
                 "are always test. For own-scheme training the HLQC overlap does not matter; for MISC transfer tests apply "
                 "the misc_main exclude lists.",
    }

    # --- Welivita own-scheme ---------------------------------------------------
    w = frames["welivita.gold"]
    lis = w[(w.speaker == "counsellor") & w.stage1_agreed]
    convs = sorted(set(w.conv_id))
    w_pairs = dup[(dup.dataset_a == "welivita.gold") & (dup.dataset_b == "welivita.gold")]
    comp = _components(w_pairs, convs)
    clusters = defaultdict(list)
    for c, root in comp.items():
        clusters[root].append(c)
    strata = defaultdict(list)
    src = w.groupby("conv_id").source.first()
    for root, members in clusters.items():
        strata[src[members[0]]].append(root)
    cl_assign = _allocate(strata, {"train": 0.8, "dev": 0.1, "test": 0.1}, rng)
    w_split = {c: cl_assign[comp[c]] for c in convs}
    man["welivita_own"] = {
        "scheme": "Welivita MITI-derived (15 codes)",
        "split": w_split,
        "label_subset": "listener rows with stage1_agreed (ann1 == ann2): "
                        f"{len(lis)} rows; the remaining rows are context only",
        "duplicate_clusters": {r: m for r, m in clusters.items() if len(m) > 1},
        "notes": "Dialogue-level; duplicate dialogues (containment >= 0.10) share a split; stratified by source.",
    }

    # --- CASAA ---------------------------------------------------------------
    man["casaa_test"] = {
        "scheme": "MITI 4 (counsellor); MISC-mapped on SR CR AF EC GI CO + T1",
        "test": man["misc_main"]["test"]["miti.casaa.gold (MISC-mapped)"],
        "exclude": man["misc_main"]["exclude"]["miti.casaa.gold"],
        "notes": "Test only. Excluded sessions duplicate HLQC (Emmy = high_121 in HLQC gold train; Rounder = high_072).",
    }

    # --- Pools ---------------------------------------------------------------
    test_miv = set(frames["misc.miv63a.gold"].conv_id)
    hp = dup[(dup.dataset_a == "pool.hlqc") & (dup.dataset_b == "pool.hlqc") & (dup.containment >= 0.5)]
    hcomp = _components(hp, sorted(set(frames["pool.hlqc"].conv_id)))
    keep_rep = {}
    for c, root in sorted(hcomp.items()):
        keep_rep.setdefault(root, c)
    hlqc_redundant = sorted(c for c, root in hcomp.items() if keep_rep[root] != c)
    man["pools"] = {
        "pool.hlqc": {
            "exclude_always": [],
            "exclude_if_annomi_is_test": sorted(twins("annomi.gold", set(frames["annomi.gold"].conv_id), "pool.hlqc")),
            "exclude_if_casaa_is_test": sorted(twins("miti.casaa.gold", set(frames["miti.casaa.gold"].conv_id), "pool.hlqc")),
            "gold_sessions_and_their_copies": sorted(train_ids | twins("misc.hlqc.gold", train_ids, "pool.hlqc")),
            "redundant_duplicates": hlqc_redundant,
            "notes": "Training-side pool, so gold copies are not a test leak; drop `redundant_duplicates` to avoid "
                     "double-weighting, and the exclude_if_* lists when the matching corpus is a test set.",
        },
        "pool.miv63a": {"exclude_always": sorted(test_miv),
                        "notes": "Contains the 10 MIV6.3A test sessions. selftrain.label_pool already drops them plus "
                                 "near-duplicate templated utterances (_drop_eval_like)."},
        "pool.miv63b": {"exclude_always": [],
                        "notes": "No session-level overlap with the test set; 22/652 long test utterances recur verbatim "
                                 "(templated chatbot lines), removed by selftrain.label_pool._drop_eval_like."},
    }

    # --- Synthetic (docs/SYNTHETIC_DATA.md) ------------------------------------
    syn = [k for k in frames if k.startswith("synth.")]
    if not syn:
        return man
    pairs = overlap.pairwise({**real, **{k: frames[k] for k in syn}})
    syn_dup = {}
    for s in syn:
        sp = pairs[(pairs.dataset_a == s) & (pairs.dataset_b == s) & (pairs.containment >= 0.5)]
        comp = _components(sp, sorted(set(frames[s].conv_id)))
        rep = {}
        for c, root in sorted(comp.items()):
            rep.setdefault(root, c)
        syn_dup[s] = sorted(c for c, root in comp.items() if rep[root] != c)
    cross = pairs[(pairs.dataset_a == "synth.v3_hlqcmix") & (pairs.dataset_b == "synth.v3_mivmix")
                  & (pairs.containment >= 0.5)]
    man["synthetic"] = {"role": "train only; never gold",
                        "redundant_duplicates": syn_dup,
                        "v3_windows_shared_by_both_mixes": [f"{a} = {b}" for a, b in zip(cross.conv_a, cross.conv_b)]}
    return man


def check(man: Dict[str, dict], frames) -> List[str]:
    errs = []
    a = man["annomi_own"]["split"]
    if set(a) != set(frames["annomi.gold"].conv_id):
        errs.append("annomi split does not cover every transcript")
    for name in ("annomi_own", "welivita_own"):
        if len(set(man[name]["split"].values()) - {"train", "dev", "test"}):
            errs.append(f"{name}: unknown split name")
    w = man["welivita_own"]
    for root, members in w["duplicate_clusters"].items():
        if len({w["split"][m] for m in members}) > 1:
            errs.append(f"welivita duplicate cluster {root} spans splits")
    mm = man["misc_main"]
    if set(mm["train"]["misc.hlqc.gold"]) & set(mm["test"]["misc.miv63a.gold"]):
        errs.append("misc train/test overlap")
    if "casaa_emmys-first-encounter" in mm["test"]["miti.casaa.gold (MISC-mapped)"]:
        errs.append("CASAA Emmy (= HLQC train high_121) is in the CASAA test split")
    if "misc_pooled_cv" in man:
        pc, tests = man["misc_pooled_cv"]["folds"], []
        for k, f in pc.items():
            if set(f["train"]["misc.miv63a.gold"]) & set(f["test"]["misc.miv63a.gold"]):
                errs.append(f"misc_pooled_cv fold {k}: an MIV session is in train and test")
            tests += f["test"]["misc.miv63a.gold"]
        if sorted(tests) != sorted(mm["test"]["misc.miv63a.gold"]) or len(tests) != len(set(tests)):
            errs.append("misc_pooled_cv: test folds do not cover every MIV session exactly once")
    if not set(man["pools"]["pool.miv63a"]["exclude_always"]) >= set(mm["test"]["misc.miv63a.gold"]):
        errs.append("pool.miv63a does not exclude every test session")
    return errs


def write(man: Dict[str, dict]) -> Dict[str, str]:
    OUT.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name, m in man.items():
        body = json.dumps({"seed": SEED, "dup_threshold": DUP, **m}, indent=1, sort_keys=True, default=str)
        (OUT / f"{name}.json").write_text(body + "\n")
        digests[name] = hashlib.sha256(body.encode()).hexdigest()[:12]
    return digests


def main() -> None:
    frames = registry.load_all(include_synth=True)   # the synthetic manifest needs the generated sets
    man = build(frames)
    errs = check(man, frames)
    for e in errs:
        print("LEAK CHECK FAILED:", e)
    d1 = write(man)
    d2 = {k: hashlib.sha256(json.dumps({"seed": SEED, "dup_threshold": DUP, **v}, indent=1, sort_keys=True,
                                       default=str).encode()).hexdigest()[:12] for k, v in build(frames).items()}
    print("deterministic:", d1 == d2, d1)
    a = pd.Series(man["annomi_own"]["split"]).value_counts().to_dict()
    w = pd.Series(man["welivita_own"]["split"]).value_counts().to_dict()
    print("annomi transcripts:", a, " welivita dialogues:", w)
    if errs:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
