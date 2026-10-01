"""Feature builders for the classic (non-LLM) pathway.

Two feature families, both fit on the HLQC train slice and applied to MIV6.3A:
  - ngram : word (1-2) + char_wb (3-5) TF-IDF, hstacked with a few hand
            LINGUISTIC features (the EACL "syntactic pattern" nod, dependency-free).
            char n-grams matter because MISC utterances are very short.
  - qwen  : mean-pooled Qwen last-hidden-state embeddings (the deep-ish flavor).

Text is the utterance by default; pass a context excerpt to prepend dialogue
history (an ablation -- the papers used utterance only).
"""
from __future__ import annotations

from typing import List

import numpy as np
import scipy.sparse as sp

_FIRST_PERSON = {"i", "i'm", "im", "i've", "ive", "me", "my", "mine", "myself", "we", "our"}
_SECOND_PERSON = {"you", "you're", "youre", "your", "yours", "yourself", "you've"}


def linguistic_features(texts: List[str]) -> np.ndarray:
    """A small dense feature block, scaled to comparable magnitudes."""
    rows = []
    for t in texts:
        t = t or ""
        low = t.lower()
        toks = low.split()
        n_tok = len(toks) or 1
        rows.append([
            min(len(t), 400) / 400.0,               # char length
            min(n_tok, 100) / 100.0,                # word length
            float(t.strip().endswith("?")),          # is a question
            min(t.count("?"), 5) / 5.0,
            min(t.count("!"), 5) / 5.0,
            sum(w in _FIRST_PERSON for w in toks) / n_tok,
            sum(w in _SECOND_PERSON for w in toks) / n_tok,
        ])
    return np.asarray(rows, dtype=np.float32)


class NgramFeaturizer:
    """Word + char TF-IDF hstacked with linguistic features. Fit on train only."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                    min_df=2, sublinear_tf=True)
        self._fitted = False

    def fit(self, texts: List[str]) -> "NgramFeaturizer":
        self.word.fit(texts)
        self.char.fit(texts)
        self._fitted = True
        return self

    def transform(self, texts: List[str]):
        Xw = self.word.transform(texts)
        Xc = self.char.transform(texts)
        Xl = sp.csr_matrix(linguistic_features(texts))
        return sp.hstack([Xw, Xc, Xl]).tocsr()

    def fit_transform(self, texts: List[str]):
        return self.fit(texts).transform(texts)


def embed_texts(model, tokenizer, texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Mean-pooled, L2-normalised last-hidden-state embeddings (the `qwen` flavor).

    Standalone (not the retriever's method) so this pathway stays decoupled from
    src/agentic; same mean-pool recipe.
    """
    import torch
    from contextlib import nullcontext

    device = next(model.parameters()).device
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    adapter_off = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
    out = []
    with torch.no_grad(), adapter_off:
        for i in range(0, len(texts), batch_size):
            batch = [t if t else " " for t in texts[i:i + batch_size]]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=256)
            enc = {k: v.to(device) for k, v in enc.items()}
            hs = model(**enc, output_hidden_states=True).hidden_states[-1]
            mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
            emb = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            out.append(emb.cpu().float().numpy())
    return np.vstack(out)
