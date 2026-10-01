"""Diversity / distribution scorecard for synthetic utterance sets.

Neither source paper measures diversity quantitatively -- both measure *quality*
(SpeechDialogueFactory via an LLM-judge rubric, SocialDial via human naturalness/
coherence ratings with Cohen's kappa). Our diagnosed failure is diversity, so the
metrics here come from the diversity literature and are cited inline:

  Tier 2 (reportable)
    distinct-1 / distinct-2   Li, Galley, Brockett, Gao & Dolan, "A Diversity-
                              Promoting Objective Function for Neural Conversation
                              Models", NAACL-HLT 2016 (aclanthology.org/N16-1014).
                              Reference-free: distinct n-grams / total n-grams.
                              (Li et al. divide by total generated words; both are
                              reported below as distinct_n and distinct_n_perword.)
    Self-BLEU                 Zhu et al., "Texygen", SIGIR 2018 (arXiv:1802.01886).
                              Mean BLEU of each text against the others as
                              references; LOWER = more diverse. Implemented here
                              rather than via Texygen, whose released code has a
                              known hypothesis/reference misalignment bug.
    length distance           KS statistic + Wasserstein vs the real corpus.
    novelty vs real           nearest-neighbour TF-IDF cosine against real text:
                              are we adding anything the corpus lacks?
    self near-duplicate rate  share of texts with a within-set neighbour above a
                              cosine threshold.

  Tier 3 (exploratory triage ONLY -- unvalidated proxies, never report these)
    distinct 3-word openings, mean length, duplicate rate, topic keyword share.

Usage -- compare real, the known-bad old set, and a new set in one table:

    PYTHONPATH=src python -m synth.metrics \
        --real data/manual/HLQC_balanced_manual.csv \
        --set old=data/synth/raw/Qwen2_5-32B-Instruct-AWQ.csv \
        --set new=data/synth/raw/<new>.csv
"""
from __future__ import annotations

import argparse
import collections
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

_TOKEN = re.compile(r"[a-z']+")


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(str(text).lower())


def _ngrams(toks: Sequence[str], n: int) -> List[tuple]:
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


# ---------------------------------------------------------------- Tier 2
def distinct_n(texts: Sequence[str], n: int) -> Dict[str, float]:
    """Distinct-n (Li et al., NAACL 2016).

    Returns both conventions: `distinct_n` = distinct n-grams / total n-grams
    (the form in common use) and `per_word` = distinct n-grams / total tokens
    (the form stated in the original paper).
    """
    grams, total, words = set(), 0, 0
    for t in texts:
        toks = tokenize(t)
        words += len(toks)
        g = _ngrams(toks, n)
        grams.update(g)
        total += len(g)
    return {
        "distinct": len(grams) / total if total else 0.0,
        "per_word": len(grams) / words if words else 0.0,
    }


def _bleu(hyp: Sequence[str], refs: Sequence[Sequence[str]], max_n: int = 4) -> float:
    """Sentence BLEU against multiple references, with brevity penalty.

    Zero n-gram precisions are floored to a small epsilon rather than collapsing
    the geometric mean to 0 (equivalent in spirit to standard smoothing), so that
    short utterances -- which dominate this corpus -- remain comparable.
    """
    if not hyp:
        return 0.0
    logs = []
    for n in range(1, max_n + 1):
        h = collections.Counter(_ngrams(hyp, n))
        if not h:
            logs.append(math.log(1e-9))
            continue
        max_ref: Dict[tuple, int] = {}
        for r in refs:
            for g, c in collections.Counter(_ngrams(r, n)).items():
                if c > max_ref.get(g, 0):
                    max_ref[g] = c
        clipped = sum(min(c, max_ref.get(g, 0)) for g, c in h.items())
        logs.append(math.log(max(clipped / sum(h.values()), 1e-9)))
    # brevity penalty against the closest reference length
    h_len = len(hyp)
    r_len = min((len(r) for r in refs), key=lambda rl: (abs(rl - h_len), rl), default=h_len)
    bp = 1.0 if h_len > r_len else math.exp(1 - r_len / max(h_len, 1))
    return bp * math.exp(sum(logs) / max_n)


def self_bleu(texts: Sequence[str], n_hyp: int = 300, n_ref: int = 50,
              max_n: int = 4, seed: int = 0) -> float:
    """Self-BLEU (Zhu et al., SIGIR 2018). Lower = more diverse.

    Exhaustive Self-BLEU is O(n^2) in references; we sample `n_hyp` hypotheses
    and `n_ref` references each, which is the standard practical approximation.
    Sampling is seeded so the number is reproducible.
    """
    toks = [tokenize(t) for t in texts if str(t).strip()]
    toks = [t for t in toks if t]
    if len(toks) < 3:
        return float("nan")
    rng = random.Random(seed)
    hyps = toks if len(toks) <= n_hyp else rng.sample(toks, n_hyp)
    scores = []
    for h in hyps:
        pool = [t for t in toks if t is not h]
        refs = pool if len(pool) <= n_ref else rng.sample(pool, n_ref)
        scores.append(_bleu(h, refs, max_n))
    return float(np.mean(scores)) if scores else float("nan")


def length_distance(texts: Sequence[str], real: Sequence[str]) -> Dict[str, float]:
    from scipy.stats import ks_2samp, wasserstein_distance

    a = np.array([len(tokenize(t)) for t in texts], dtype=float)
    b = np.array([len(tokenize(t)) for t in real], dtype=float)
    if len(a) == 0 or len(b) == 0:
        return {"ks": float("nan"), "wasserstein": float("nan"), "median": float("nan")}
    return {
        "ks": float(ks_2samp(a, b).statistic),
        "wasserstein": float(wasserstein_distance(a, b)),
        "median": float(np.median(a)),
    }


def _tfidf_sims(texts: Sequence[str], other: Sequence[str]):
    """Max cosine similarity of each `texts` item against `other`, TF-IDF space."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
    vec.fit(list(texts) + list(other))
    A, B = vec.transform(texts), vec.transform(other)
    out = np.zeros(A.shape[0])
    step = 512
    for i in range(0, A.shape[0], step):
        out[i:i + step] = cosine_similarity(A[i:i + step], B).max(axis=1)
    return out


def novelty_vs_real(texts: Sequence[str], real: Sequence[str],
                    dup_threshold: float = 0.9) -> Dict[str, float]:
    if not len(texts) or not len(real):
        return {"mean_max_sim": float("nan"), "pct_near_real_dup": float("nan")}
    s = _tfidf_sims(texts, real)
    return {"mean_max_sim": float(s.mean()),
            "pct_near_real_dup": float(100 * (s >= dup_threshold).mean())}


def self_near_dup(texts: Sequence[str], threshold: float = 0.9) -> float:
    """% of texts having a within-set neighbour at/above the cosine threshold."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if len(texts) < 2:
        return float("nan")
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
    A = vec.fit_transform(texts)
    hits = np.zeros(A.shape[0], dtype=bool)
    step = 512
    for i in range(0, A.shape[0], step):
        sim = cosine_similarity(A[i:i + step], A)
        for r in range(sim.shape[0]):
            sim[r, i + r] = 0.0  # ignore self-match
        hits[i:i + step] = (sim >= threshold).any(axis=1)
    return float(100 * hits.mean())


# ---------------------------------------------------------------- Tier 3
TOPIC_PATTERNS = {
    "smoking": r"smok|cigar|nicotine|vap",
    "alcohol": r"drink|alcohol|booze|sober",
    "exercise/diet": r"exercis|diet|weight|gym",
}


def exploratory(texts: Sequence[str]) -> Dict[str, float]:
    """Unvalidated triage proxies. Do NOT report these as evidence."""
    texts = [str(t).strip() for t in texts if str(t).strip()]
    n = len(texts)
    if not n:
        return {}
    op3 = {" ".join(tokenize(t)[:3]) for t in texts}
    out = {
        "pct_unique_open3": 100 * len(op3) / n,
        "mean_words": float(np.mean([len(tokenize(t)) for t in texts])),
        "pct_exact_dup": 100 * (n - len(set(texts))) / n,
    }
    for label, pat in TOPIC_PATTERNS.items():
        out[f"pct_{label}"] = 100 * sum(bool(re.search(pat, t, re.I)) for t in texts) / n
    return out


# ---------------------------------------------------------------- driver
def scorecard(texts: Sequence[str], real: Sequence[str], seed: int = 0) -> Dict[str, float]:
    d1, d2 = distinct_n(texts, 1), distinct_n(texts, 2)
    row = {
        "n": len(texts),
        "distinct_1": d1["distinct"], "distinct_2": d2["distinct"],
        "self_bleu": self_bleu(texts, seed=seed),
        "self_near_dup_%": self_near_dup(texts),
    }
    row.update({f"len_{k}": v for k, v in length_distance(texts, real).items()})
    if list(texts) != list(real):
        row.update(novelty_vs_real(texts, real))
    row.update({f"x_{k}": v for k, v in exploratory(texts).items()})
    return row


def load_texts(path: str, col: str = "utt_text") -> List[str]:
    """Load utterances; if the CSV is a raw synth file, keep target rows only."""
    df = pd.read_csv(path)
    if "synth_target" in df.columns:
        df = df[df["synth_target"].astype(bool)]
    return [str(t) for t in df[col].dropna() if str(t).strip()]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real", required=True, help="reference corpus CSV (real human data)")
    ap.add_argument("--set", action="append", default=[], metavar="LABEL=PATH",
                    help="a set to score; repeatable")
    ap.add_argument("--col", default="utt_text")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    real = load_texts(args.real, args.col)
    cols = {"real(ref)": scorecard(real, real, args.seed)}
    for spec in args.set:
        label, _, path = spec.partition("=")
        cols[label] = scorecard(load_texts(path, args.col), real, args.seed)

    keys = list(dict.fromkeys(k for c in cols.values() for k in c))
    w = max(len(k) for k in keys) + 2
    print(f"\n{'metric'.ljust(w)}" + "".join(f"{lab:>16s}" for lab in cols))
    print("-" * (w + 16 * len(cols)))
    for k in keys:
        if k == "x_pct_unique_open3":
            print(f"{'-- Tier 3 (exploratory, do not report) --'.ljust(w)}")
        vals = "".join(
            f"{cols[lab].get(k, float('nan')):>16.3f}" if isinstance(cols[lab].get(k), float)
            else f"{str(cols[lab].get(k, '')):>16s}" for lab in cols)
        print(f"{k.ljust(w)}{vals}")
    print("\nTier 2 (reportable): distinct-1/2 = Li et al. NAACL 2016; self_bleu = Zhu et al. SIGIR 2018 (lower = more diverse).")


if __name__ == "__main__":
    main()
