"""Per-utterance retrieval of HLQC exemplars for the `ag` arm.

The static `fs` arm (baseline/fewshot.py) shows the model the SAME frozen
exemplars for every utterance. This retriever instead selects, per query
utterance, the most SIMILAR HLQC examples, and (optionally) guarantees every
candidate label is represented by its nearest instance so the rare tail codes
(SU/EC/AF/GI) are never missing from the demonstrations.

It plugs into `TieredAnnotator.fewshot_provider`, so the inference engine,
generation, parsing, and CSV writer are all untouched.

Two similarity backends (v2), both fully offline, no download:
  - ``tfidf`` : TF-IDF cosine over ``utt_text`` (+ light linguistic re-rank).
                Cheap, but bag-of-words is noisy on the very short MISC
                utterances ("Mm-hmm", "go on") -- this was v1's weak point.
  - ``qwen``  : mean-pooled Qwen last-hidden-state embeddings (cosine). Semantic,
                so an affirmation retrieves affirmations rather than keyword
                collisions. Reuses the already-loaded annotator model (adapters
                disabled, so the representation is stable across backbones), and
                caches the pool embeddings to avoid recompute on a requeue.

Motivation for v2 is representation: Pérez-Rosas et al. (EACL 2017) and Pellemans
et al. (JMIR 2024) both beat surface n-grams with semantic/syntactic features,
which is why v1's TF-IDF likely failed. Pellemans also found that FORCING class
balance did not help, so the rare-code guarantee here is tunable (`rare_reserve`)
rather than hardwired.

Exemplars are shown LABEL-ONLY: a retrieved HLQC row has no frozen rationale
(generating one would need gpt-4o), so demonstrations are (utterance -> label).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from automisc_ft.data import load_manual, t1_codes_for_speaker, t2_codes_for_group
from baseline.fewshot import build_fewshot_messages
from components.context import build_context_excerpt

REPO_ROOT = Path(__file__).resolve().parents[2]
HLQC_PATH = REPO_ROOT / "data" / "manual" / "HLQC_balanced_manual.csv"
EMB_CACHE_DIR = REPO_ROOT / "outputs" / "agentic"

# T2 codes scarce in HLQC but common in MIV6.3A (docs/GRPO.md).
RARE_T2 = ("SU", "EC", "AF", "GI")
CONTEXT_MODE = "interval"


def _len_bucket(n: int) -> int:
    """Coarse utterance-length bucket, an EACL-style syntactic-shape signal."""
    if n <= 15:
        return 0
    if n <= 60:
        return 1
    if n <= 200:
        return 2
    return 3


class RetrievalFewshotProvider:
    """A `fewshot_provider` that retrieves exemplars for the current query.

    Usage from the predict loop::

        provider = RetrievalFewshotProvider(ctx=5, method="qwen")
        annotator = TieredAnnotator(..., fewshot_provider=provider)
        provider.attach_model(annotator.model, annotator.tokenizer)  # qwen only
        for pos in positions:
            provider.set_query(df.iloc[pos]["utt_text"], df.iloc[pos]["speaker"])
            annotator.predict_row(df, pos, ...)
    """

    def __init__(self, ctx: int, k_t1: int = 8, k_t2: int = 6,
                 method: str = "qwen", rare_reserve: Optional[int] = None,
                 linguistic: bool = True, ling_weight: float = 0.05,
                 model_tag: str = ""):
        self.ctx = int(ctx)
        self.k_t1 = int(k_t1)
        self.k_t2 = int(k_t2)
        self.method = method
        # None -> guarantee every candidate label (v1 behaviour); 0 -> pure
        # similarity, no forced coverage (the JMIR-caution ablation).
        self.rare_reserve = rare_reserve
        self.linguistic = bool(linguistic)
        self.ling_weight = float(ling_weight)
        self.model_tag = model_tag or "base"

        # load_manual sorts by conversation/volley/utterance so a row's
        # positional index matches transcript order (build_context_excerpt).
        self.df = load_manual(HLQC_PATH).reset_index(drop=True)
        self._text = self.df["utt_text"].fillna("").astype(str)

        self._rows_by_speaker: Dict[str, np.ndarray] = {
            spk: self.df.index[self.df["speaker"] == spk].to_numpy()
            for spk in self.df["speaker"].unique()
        }
        self._t1 = self.df["t1_label_GT"].to_numpy()
        self._t2 = self.df["t2_label_GT"].to_numpy()

        # Linguistic features of the pool (cheap, method-independent).
        self._is_q = self._text.str.strip().str.endswith("?").to_numpy()
        self._lenb = np.array([_len_bucket(len(t)) for t in self._text])

        # tfidf index is built now; qwen embeddings wait for attach_model.
        self.vectorizer = None
        self.matrix = None
        self._pool_emb: Optional[np.ndarray] = None
        if method == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
            self.matrix = self.vectorizer.fit_transform(self._text)
        elif method != "qwen":
            raise ValueError(f"Unknown retriever method: {method}")

        self.model = None
        self.tokenizer = None
        self._query_text: str = ""
        self._query_is_q: bool = False
        self._query_lenb: int = 0
        self._query_emb: Optional[np.ndarray] = None

    # -- qwen backend setup -----------------------------------------------------
    def attach_model(self, model, tokenizer) -> None:
        """Give the retriever the loaded model to embed with (qwen backend)."""
        self.model = model
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.method == "qwen":
            self._pool_emb = self._pool_embeddings()

    def _pool_embeddings(self) -> np.ndarray:
        cache = EMB_CACHE_DIR / f"hlqc_emb_{self.model_tag}.npy"
        if cache.exists():
            emb = np.load(cache)
            if emb.shape[0] == len(self.df):
                print(f"Loaded pool embeddings from {cache} {emb.shape}")
                return emb
        print(f"Embedding {len(self.df)} HLQC utterances with the model "
              f"(tag={self.model_tag})...")
        emb = self._embed_texts(list(self._text))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, emb)
        print(f"Cached pool embeddings -> {cache} {emb.shape}")
        return emb

    def _embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Mean-pooled, L2-normalised last-hidden-state embeddings.

        Adapters (if any) are disabled so the representation is the base model's,
        stable across the `none` and `ft_bare` backbones.
        """
        import torch
        from contextlib import nullcontext

        device = next(self.model.parameters()).device
        adapter_off = (self.model.disable_adapter()
                       if hasattr(self.model, "disable_adapter") else nullcontext())
        out = []
        with torch.no_grad(), adapter_off:
            for i in range(0, len(texts), batch_size):
                batch = [t if t else " " for t in texts[i:i + batch_size]]
                enc = self.tokenizer(batch, return_tensors="pt", padding=True,
                                     truncation=True, max_length=256)
                enc = {k: v.to(device) for k, v in enc.items()}
                hs = self.model(**enc, output_hidden_states=True).hidden_states[-1]
                mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
                emb = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
                emb = torch.nn.functional.normalize(emb, dim=-1)
                out.append(emb.cpu().float().numpy())
        return np.vstack(out)

    # -- per-row query ----------------------------------------------------------
    def set_query(self, utt_text: Optional[str], speaker: Optional[str] = None) -> None:
        self._query_text = utt_text or ""
        self._query_is_q = self._query_text.strip().endswith("?")
        self._query_lenb = _len_bucket(len(self._query_text))
        if self.method == "qwen" and self.model is not None:
            self._query_emb = self._embed_texts([self._query_text])[0]

    # -- the fewshot_provider signature TieredAnnotator expects -----------------
    def __call__(self, speaker: str, tier: str, t1_label: Optional[str]) -> List[Dict[str, str]]:
        exemplars = self.retrieve_examples(speaker, tier, t1_label)
        return self._format(exemplars, speaker, tier, t1_label)

    def retrieve_examples(self, speaker: str, tier: str,
                          t1_label: Optional[str]) -> List[Dict[str, str]]:
        """The chosen exemplar dicts for the current query (before formatting).

        Reused both by the fewshot hook (`__call__`) and by the agent's
        `retrieve` tool, which wants the raw examples to render as an observation.
        """
        rows = self._rows_by_speaker.get(speaker)
        if rows is None or len(rows) == 0:
            return []

        if tier == "t1":
            labels, k = self._t1, self.k_t1
            must_cover = list(t1_codes_for_speaker(speaker))
        elif tier == "t2":
            group = t2_codes_for_group(speaker, t1_label) if t1_label else None
            if not group:
                return []
            rows = rows[np.isin(self._t2[rows], list(group))]
            if len(rows) == 0:
                return []
            labels, k = self._t2, self.k_t2
            must_cover = list(group)
        else:
            raise ValueError(f"Unknown tier: {tier}")

        if self.rare_reserve == 0:
            must_cover = []  # pure similarity, no forced coverage

        sims = self._similarity(rows)
        order = rows[np.argsort(-sims)]
        chosen = self._select(order, labels, must_cover, k)
        return self._to_exemplars(chosen)

    def _similarity(self, rows: np.ndarray) -> np.ndarray:
        if self.method == "tfidf":
            from sklearn.metrics.pairwise import linear_kernel
            qv = self.vectorizer.transform([self._query_text])
            sims = linear_kernel(qv, self.matrix[rows]).ravel()
        else:  # qwen
            sims = self._pool_emb[rows] @ self._query_emb
        if self.linguistic:
            bonus = (self.ling_weight * (self._is_q[rows] == self._query_is_q)
                     + self.ling_weight * (self._lenb[rows] == self._query_lenb))
            sims = sims + bonus
        return sims

    # -- selection --------------------------------------------------------------
    def _select(self, order: np.ndarray, labels: np.ndarray,
                must_cover: List[str], k: int) -> List[int]:
        """Nearest instance of each `must_cover` label, then nearest fill to k."""
        rank = {int(pos): i for i, pos in enumerate(order)}
        chosen: List[int] = []
        seen: set[int] = set()

        for lab in must_cover:
            for pos in order:
                p = int(pos)
                if p in seen or labels[p] != lab:
                    continue
                chosen.append(p)
                seen.add(p)
                break

        for pos in order:
            if len(chosen) >= k:
                break
            p = int(pos)
            if p in seen:
                continue
            chosen.append(p)
            seen.add(p)

        chosen.sort(key=lambda p: rank[p])
        return chosen

    # -- formatting -------------------------------------------------------------
    def _to_exemplars(self, positions: List[int]) -> List[Dict[str, str]]:
        out = []
        for pos in positions:
            row = self.df.iloc[pos]
            out.append({
                "transcript": build_context_excerpt(
                    self.df, int(pos), CONTEXT_MODE, self.ctx
                ),
                "speaker": row["speaker"],
                "utterance": row["utt_text"],
                "t1_label": row["t1_label_GT"],
                "t2_label": row["t2_label_GT"],
            })
        return out

    def _format(self, exemplars: List[Dict[str, str]], speaker: str,
                tier: str, t1_label: Optional[str]) -> List[Dict[str, str]]:
        """Reuse baseline.fewshot's message formatter (label-only demos)."""
        if tier == "t1":
            nested = {speaker: {"t1": exemplars, "t2": {}}}
            return build_fewshot_messages(nested, speaker, "t1", rationales=False)
        nested = {speaker: {"t1": [], "t2": {t1_label: exemplars}}}
        return build_fewshot_messages(
            nested, speaker, "t2", rationales=False, t1_label=t1_label
        )
