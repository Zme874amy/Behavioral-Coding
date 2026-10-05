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
TRAIN_SEEDS = (42, 1, 2)
N_FOLDS = 5

# "Lucky draw" controls (docs/experiments/2026-10-05-split-review.md section 9).
VARIANCE_RULES = {
    "training_seeds": "at least 3 matched seeds (42, 1, 2) for any claimed difference; one seed = exploratory only",
    "pairing": "compare arms at the same seed and the same folds; report the mean of paired differences",
    "intervals": "95% CI from resampling sessions (clusters), on top of the across-seed spread",
    "fold_draw": "for k-fold designs, re-draw the fold assignment with each training seed (seed-tied) or use "
                 "leave-one-session-out, so no result rests on one fold assignment",
    "single_split": "do not report a single random train/test split as a headline: use the grouped k-fold CV",
}
# Exemplars (few-shot, retrieval index, synthesis style anchors); section 10 of the split review.
EXEMPLAR_RULES = {
    "source": "exemplars come only from the training side of the split: HLQC gold under misc_main; under pooled CV, "
              "HLQC gold by default (keeps the exemplar factor constant across protocols), the fold's training MIV "
              "sessions only as an explicit, labelled arm; never a test session",
    "cross_corpus": "22 of 41 few-shot exemplars (and the retrieval index and synthesis anchors) include HLQC sessions "
                    "that are AnnoMI 15/21/44/53 and CASAA Emmy: apply the AnnoMI/CASAA exclusions to ANY model that saw "
                    "HLQC gold by training, prompt exemplars, retrieval or synthetic data",
    "draws": "prompted arms use >= 3 exemplar draws (different selection seeds) and report mean +- sd; the exemplar "
             "draw is a variance source like the training seed",
    "same_corpus_scoring": "when scoring on the corpus the exemplars come from (HLQC CV, teacher screens), drop the "
                           "exemplar SESSIONS from scoring (exemplars carry 5 context volleys), or draw exemplars from "
                           "other folds",
}          # containment at/above which two sessions count as the same session


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


_SERIES_STOP = (r"\b(new video|the|a|an|how not to do|how to do|not so good|not so bad|ineffective|effective|"
                r"non[- ]motivational approach|motivational interviewing demonstration|with mi|without mi|"
                r"part (one|two|three|four|\d+)|\d+|role play|demo|demonstration)\b")


def annomi_series(an: pd.DataFrame, threshold: float = 0.85) -> Dict[str, str]:
    """conv_id -> series id. Titles are normalised (numbering, good/bad and part markers
    removed) and linked when their similarity is >= threshold; the series id is the
    smallest conv_id in the group. Deliberately conservative: over-merging only moves
    whole groups together."""
    import difflib
    t = an.groupby("conv_id").video_title.first()
    core = (t.str.lower().str.replace(_SERIES_STOP, " ", regex=True)
            .str.replace(r"[^a-z ]", " ", regex=True).str.split().str.join(" "))
    ids = sorted(t.index, key=int)
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if core[a] and core[b] and difflib.SequenceMatcher(None, core[a], core[b]).ratio() >= threshold:
                parent[find(a)] = find(b)
    groups = defaultdict(list)
    for i in ids:
        groups[find(i)].append(i)
    canon = {r: min(ms, key=int) for r, ms in groups.items()}
    return {i: canon[find(i)] for i in ids}


def build(frames=None, pairs=None) -> Dict[str, dict]:
    """`pairs` (real datasets only) drives every real-data manifest, so those never
    depend on our regenerable synthetic files; the synthetic manifest is computed
    separately when synth.* frames are present."""
    frames = frames or registry.load_all(include_synth=True)
    real = {k: v for k, v in frames.items() if not k.startswith("synth.")}
    pairs = overlap.pairwise(real) if pairs is None else pairs
    pairs = pairs[~pairs.dataset_a.str.startswith("synth.") & ~pairs.dataset_b.str.startswith("synth.")]
    dup = pairs[(pairs.containment >= DUP) & (pairs.conv_a != pairs.conv_b)]
    # Each manifest seeds its own RNG (random.Random(f"{SEED}-<name>")), so changing one split never shifts another.
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
        "exemplar_rules": EXEMPLAR_RULES,
        "variance_rules": VARIANCE_RULES,
        "exclude": {
            "miti.casaa.gold": sorted(twins("misc.hlqc.gold", train_ids, "miti.casaa.gold")
                                      | twins("pool.hlqc", set(frames["pool.hlqc"].conv_id), "miti.casaa.gold")),
            "annomi.gold (transfer test of any model that saw HLQC gold: training, exemplars, retrieval or synthetic data)":
                sorted(twins("misc.hlqc.gold", train_ids, "annomi.gold"), key=int),
            "annomi.gold (additionally, when the model saw pool.hlqc: self-training/retrieval over the pool)":
                sorted(twins("pool.hlqc", set(frames["pool.hlqc"].conv_id), "annomi.gold"), key=int),
        },
        "notes": "Unchanged from all prior experiments. CASAA sessions duplicating any HLQC session are excluded.",
    }

    # --- PROPOSED: pooled training with MIV cross-validation -----------------
    # docs/experiments/2026-10-05-split-review.md. Alongside misc_main, not replacing it:
    # every MIV6.3A session is test exactly once, so scores still cover all 821 utterances.
    miv = frames["misc.miv63a.gold"]
    miv_sessions = sorted(set(miv.conv_id))

    def fold_block(fold_of):
        return {str(k): {"train": {"misc.hlqc.gold": sorted(train_ids),
                                   "misc.miv63a.gold": sorted(c for c, f in fold_of.items() if f != k)},
                         "test": {"misc.miv63a.gold": sorted(c for c, f in fold_of.items() if f == k)}}
                for k in sorted(set(fold_of.values()))}
    man["misc_pooled_cv"] = {
        "status": "PROPOSED (pending decision); misc_main stays the reported setting until then",
        "scheme": "MISC 2.5",
        "designs": {
            "loso": fold_block({c: i for i, c in enumerate(miv_sessions)}),
            "5fold_seed_tied": {str(sd): fold_block(assign_folds(miv, 5, sd)) for sd in TRAIN_SEEDS},
        },
        "recommended_design": "loso for headline numbers (10 folds, no fold-assignment randomness, 9 MIV sessions "
                              "in every training set); 5fold_seed_tied for exploratory arms (training seed s uses "
                              "fold draw s, so the reported spread includes fold-assignment variance)",
        "checkpoint_selection": "HLQC validation fold as in misc_main (val_folds 7, val_fold 0); never an MIV session",
        "external_test": man["misc_main"]["test"]["miti.casaa.gold (MISC-mapped)"],
        "exclude": man["misc_main"]["exclude"],
        "notes": "Answers RQ1 as worded (a service adapting on labels it already holds). misc_main remains the "
                 "cold-start transfer setting (no in-domain labels).",
        "leakage_rules": {
            "frozen_config": "no hyperparameter is chosen on MIV: reuse the misc_main config as is; any new tuning uses "
                             "HLQC val fold or an inner split of the fold's TRAINING sessions",
            "fold_statistics": "label priors, self-training caps/rare-code lists, synthesis topic mixes and retrieval "
                               "indexes are rebuilt from the fold's training sessions only",
            "pools": "self-training/retrieval pools exclude all 10 MIV gold sessions (label_pool already does)",
            "scoring": "pool the out-of-fold predictions of all folds and score once (821 utterances); bootstrap by session",
        },
        "variance_rules": VARIANCE_RULES,
        "exemplar_rules": EXEMPLAR_RULES,
    }

    # --- AnnoMI own-scheme ---------------------------------------------------
    # Split by video SERIES, not transcript: parts of one session ("Daryl interviews Ricky 1-3")
    # and good/bad versions of one role-play ("The Effective / Ineffective Physician") share the
    # client and story. Series come from title similarity (annomi_series); the 7 ten-rater
    # transcripts (majority labels, the cleanest) pull their whole series into test.
    an = frames["annomi.gold"]
    tr = an.groupby("conv_id").agg(q=("mi_quality", "first"),
                                   ann=("annotator_id", lambda s: s[s >= 0].mode().iloc[0] if (s >= 0).any() else -1),
                                   multi=("n_annotators", "max"))
    series = annomi_series(an)
    members = defaultdict(list)
    for c, g in series.items():
        members[g].append(c)
    multi = set(tr[tr.multi > 1].index)
    strata = defaultdict(list)
    for g, ms in members.items():
        qs = set(tr.loc[ms, "q"])
        strata["mixed" if len(qs) > 1 else qs.pop()].append(g)
    g_fold = _allocate(strata, {str(k): 1 / N_FOLDS for k in range(N_FOLDS)}, random.Random(f"{SEED}-annomi_own"))
    man["annomi_own"] = {
        "scheme": "AnnoMI (main behaviour + subtypes; client change/neutral/sustain)",
        "design": f"{N_FOLDS}-fold CV grouped by video series, stratified by series MI quality. For test fold k, dev = fold "
                  f"(k+1) mod {N_FOLDS}, train = the other folds. Every transcript is tested exactly once (no single-split "
                  "lucky draw: one 15% split moved proxy macro-F1 by SD 0.022-0.031).",
        "folds": {c: int(g_fold[series[c]]) for c in sorted(tr.index, key=int)},
        "series": {g: sorted(ms, key=int) for g, ms in sorted(members.items(), key=lambda kv: int(kv[0])) if len(ms) > 1},
        "clean_subset_10rater": sorted(multi, key=int),
        "exclude": man["misc_main"]["exclude"],
        "variance_rules": VARIANCE_RULES,
        "notes": "Also report the score on the 7 ten-rater transcripts (majority labels, the cleanest) and per-annotator "
                 "scores (annotator effect). For MISC transfer tests apply the misc_main exclude lists.",
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
    cl_fold = _allocate(strata, {str(k): 1 / N_FOLDS for k in range(N_FOLDS)}, random.Random(f"{SEED}-welivita_own"))
    man["welivita_own"] = {
        "scheme": "Welivita, a MITI variant (15 codes; maps to MITI 4.2.1 via registry.WELIVITA_TO_MITI)",
        "design": f"{N_FOLDS}-fold CV grouped by same-post/duplicate cluster, stratified by source. For test fold k, dev = "
                  f"fold (k+1) mod {N_FOLDS}. Every dialogue is tested exactly once.",
        "folds": {c: int(cl_fold[comp[c]]) for c in convs},
        "label_subset": "listener rows with stage1_agreed (ann1 == ann2): "
                        f"{len(lis)} rows; the remaining rows are context only",
        "duplicate_clusters": {r: m for r, m in clusters.items() if len(m) > 1},
        "cross_source_robustness": {
            "train counsel_chat -> test RED": "within each fold: train on the training CounselChat dialogues, test on the test-fold RED dialogues",
            "train RED -> test counsel_chat": "the reverse",
            "why": "in-source proxy macro-F1 0.47 drops to 0.36-0.38 across sources (docs/experiments/2026-10-05-split-review.md)"},
        "variance_rules": VARIANCE_RULES,
        "notes": "Duplicate dialogues and same-post threads (containment >= 0.10) share a fold; stratified by source.",
    }

    # --- MITI 4 scheme ----------------------------------------------------------
    from eda.quality import MISC_TO_MITI
    man["miti_scheme"] = {
        "scheme": "MITI 4.2.1 (counsellor)",
        "human_data": {"miti.casaa.gold": "20 transcripts (18 clean): the only human MITI-coded spoken data we hold"},
        "test": sorted(set(frames["miti.casaa.gold"].conv_id)
                       - twins("misc.hlqc.gold", train_ids, "miti.casaa.gold")
                       - twins("pool.hlqc", set(frames["pool.hlqc"].conv_id), "miti.casaa.gold")),
        "training_options": {
            "recommended": "pooled in MITI space: welivita.gold (stage-I-agreed labels, mapped with registry.WELIVITA_TO_MITI; "
                           "Self-Disclose dropped) + MISC gold mapped with MISC_TO_MITI. Proxy on CASAA (all routes trained on "
                           "MITI-mapped labels): macro-F1 0.335 vs 0.246 (MISC gold only) and 0.303 (Welivita only)",
            "welivita_folds": "use welivita_own folds for MITI-space dev/selection; CASAA stays test only",
            "if obtained": "MI-TAGS (MITI 4.2, 242 sessions) after dedup against HLQC, AnnoMI and CASAA",
        },
        "welivita_to_miti": "registry.WELIVITA_TO_MITI (exact 57% / approximate 34% / none 8% of listener labels)",
        "misc_to_miti": MISC_TO_MITI,
        "not_mappable": ["Seek: neither MISC (folds permission-seeking into EC) nor Welivita has it",
                         "Welivita Self-Disclose (context-dependent)"],
        "notes": "CASAA is too small to train on and is the reference standard, so it is never used for training or tuning.",
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
    a = man["annomi_own"]["folds"]
    if set(a) != set(frames["annomi.gold"].conv_id):
        errs.append("annomi folds do not cover every transcript")
    for g, ms in man["annomi_own"]["series"].items():
        if len({a[m] for m in ms}) > 1:
            errs.append(f"annomi series {g} spans folds")
    w = man["welivita_own"]
    if set(w["folds"]) != set(frames["welivita.gold"].conv_id):
        errs.append("welivita folds do not cover every dialogue")
    for root, members in w["duplicate_clusters"].items():
        if len({w["folds"][m] for m in members}) > 1:
            errs.append(f"welivita duplicate cluster {root} spans folds")
    mm = man["misc_main"]
    if set(mm["train"]["misc.hlqc.gold"]) & set(mm["test"]["misc.miv63a.gold"]):
        errs.append("misc train/test overlap")
    if "casaa_emmys-first-encounter" in mm["test"]["miti.casaa.gold (MISC-mapped)"]:
        errs.append("CASAA Emmy (= HLQC train high_121) is in the CASAA test split")
    if "misc_pooled_cv" in man:
        d = man["misc_pooled_cv"]["designs"]
        for name, block in [("loso", d["loso"])] + [(f"5fold seed {k}", v) for k, v in d["5fold_seed_tied"].items()]:
            tests = []
            for k, f in block.items():
                if set(f["train"]["misc.miv63a.gold"]) & set(f["test"]["misc.miv63a.gold"]):
                    errs.append(f"misc_pooled_cv {name} fold {k}: an MIV session is in train and test")
                tests += f["test"]["misc.miv63a.gold"]
            if sorted(tests) != sorted(mm["test"]["misc.miv63a.gold"]) or len(tests) != len(set(tests)):
                errs.append(f"misc_pooled_cv {name}: test folds do not cover every MIV session exactly once")
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
    a = pd.Series(man["annomi_own"]["folds"]).value_counts().sort_index().to_dict()
    w = pd.Series(man["welivita_own"]["folds"]).value_counts().sort_index().to_dict()
    print("annomi transcripts per fold:", a, " welivita dialogues per fold:", w)
    if errs:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
