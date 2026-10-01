"""Scenario seeds for synthetic MI dialogue, restricted to sourced factors.

Provenance rule for this module: every field and every value cites the MISC 2.5
manual, the AutoMISC thesis, or a measurement on the real corpora. The v2 version
sampled seven dimensions -- stage of change (Transtheoretical Model), affect,
setting, session, register, style -- that came from model pretraining or my own
judgement. Those are deleted. What remains:

  topic      measured on the real corpora (see MIX below). v2 used 14 invented
             behaviour-change domains sampled uniformly, which matched neither
             corpus.
  readiness  the Readiness Ruler (thesis section 2.3.1, Table 2.2): three
             dimensions -- importance, confidence, readiness -- each scored 0-10
             by the client. Replaces the deleted TTM `stage` field. The thesis
             notes the confidence ruler "is a known and validated predictor of
             actual behaviour change", which is why it is their primary outcome.

Scene, Narrative Flow, Character Behaviors and Emotional Progression are NOT
sampled here. They are generated per window as SpeechDialogueFactory's dialogue
script (arXiv 2503.23848 section 3.1.2), which is where that variety belongs --
v2 faked it with enums I made up.

Topic mixes, from keyword counts over the gold corpora (`synth.metrics` style
matching on utt_text):
  train  HLQC:     alcohol 150, diet/weight 73, exercise 37, smoking 28, medication 5
  eval   MIV6.3A:  smoking 222, diet/weight 50, exercise 24, alcohol 6, medication 1
The two disagree, so the mix is an explicit ablation axis rather than a guess.
`mix=eval` aligns training-data topics with the test distribution and must be
disclosed as such when reported.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Dict, List

# Behaviour-change topics, phrased for the prompt. Only topics actually present
# in the gold corpora are listed -- no invented domains.
TOPICS: List[str] = [
    "cutting down alcohol",
    "diet and weight",
    "getting regular exercise",
    "quitting smoking",
    "taking medication as prescribed",
]

# Measured proportions (counts above, normalised).
MIXES: Dict[str, List[float]] = {
    "train": [0.512, 0.249, 0.126, 0.096, 0.017],   # HLQC
    "eval":  [0.020, 0.165, 0.079, 0.733, 0.003],   # MIV6.3A
}

# Readiness Ruler is scored 0-10 (thesis Table 2.2). Sampling is done over bands
# so all three levels are represented, then a value is drawn inside the band.
BANDS: Dict[str, range] = {"low": range(0, 4), "moderate": range(4, 7), "high": range(7, 11)}
RULER_DIMS = ("importance", "confidence", "readiness")


@dataclass(frozen=True)
class ScenarioSeed:
    topic: str
    importance: int
    confidence: int
    readiness: int

    def band(self, dim: str) -> str:
        v = getattr(self, dim)
        return next(b for b, r in BANDS.items() if v in r)

    def describe(self) -> str:
        """Natural-language rendering (SocialDial section 3.3 expresses factors as
        description rather than raw slot labels)."""
        return (
            f"The target behaviour under discussion is {self.topic}. On the Readiness "
            f"Ruler the client rates importance {self.importance}/10, confidence "
            f"{self.confidence}/10, readiness {self.readiness}/10 "
            f"({self.band('importance')} importance, {self.band('confidence')} confidence, "
            f"{self.band('readiness')} readiness)."
        )

    def as_dict(self) -> dict:
        return asdict(self)


def sample_seeds(n: int, mix: str = "train", seed: int = 42) -> List[ScenarioSeed]:
    """Sample `n` seeds: topic by the measured mix, ruler dims with balanced bands.

    Topics follow the corpus proportions (so the synthetic set can be compared
    against a corpus rather than an invented uniform spread). The three ruler
    dimensions use balanced band marginals -- each of low/moderate/high appears
    about a third of the time per dimension -- so the motivational space is
    covered instead of clustering wherever the sampler happens to land.
    """
    if n <= 0:
        return []
    if mix not in MIXES:
        raise ValueError(f"mix must be one of {list(MIXES)}")

    rng = random.Random(seed)
    topics = rng.choices(TOPICS, weights=MIXES[mix], k=n)

    cols: Dict[str, List[int]] = {}
    for i, dim in enumerate(RULER_DIMS):
        r = random.Random(seed + 101 * (i + 1))          # independent stream per dim
        names = list(BANDS)
        pool = (names * ((n // len(names)) + 1))[:max(n, len(names))]
        r.shuffle(pool)
        cols[dim] = [r.choice(list(BANDS[b])) for b in pool[:n]]

    return [ScenarioSeed(topic=topics[i], **{d: cols[d][i] for d in RULER_DIMS})
            for i in range(n)]


__all__ = ["ScenarioSeed", "sample_seeds", "TOPICS", "MIXES", "BANDS", "RULER_DIMS"]
