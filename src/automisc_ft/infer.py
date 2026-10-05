"""HuggingFace implementation of the AutoMISC hierarchical (T1 -> T2) annotator.

Mirrors the tiered branch of `components.annotator.Annotator` but runs through
`model.generate` so the SAME pipeline works for both the zero-shot base model
and a LoRA fine-tuned model. Zero-shot and fine-tuned inference are byte-for-
byte identical except for which adapter weights are active.

Fine-tuned mode uses TWO adapters loaded onto one base model:
  - adapter "t1" is active during the Tier-1 call
  - adapter "t2" is active during the Tier-2 call
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from automisc_ft.data import (
    build_messages_t1,
    build_messages_t2,
    t1_codes_for_speaker,
    t2_codes_for_group,
    t2_codes_for_speaker,
)
from components.hf_load import chat_template_kwargs, load_model_and_tokenizer
from schemes.misc import code_from_name
from sft.eval import LABEL_ALIASES


def _normalise_tok(tok: str) -> str:
    return tok.strip("`*_:()[]{}\"' \t").strip().rstrip(".,;:")


def _resolve(cand: str, allowed: List[str]) -> Optional[str]:
    c = _normalise_tok(cand)
    if c in allowed:
        return c
    if c in LABEL_ALIASES and LABEL_ALIASES[c] in allowed:
        return LABEL_ALIASES[c]
    return None


def parse_label(generated: str, allowed: List[str]) -> str:
    """Extract a MISC code from a model generation.

    Robust to two output shapes that arise in this experiment:
      - Fine-tuned model: emits the bare label first (``"GI"``).
      - Zero-shot model following the original prompt: emits an explanation and
        then the label last (``"... so the label is GI"`` or JSON
        ``{"explanation": ..., "label": "GI"}``).

    Strategy: prefer an explicit ``label: XX`` field (last occurrence), then the
    head token, then a scan from the END (the original format puts the label
    last). MISC ``+``/``-`` suffixes are preserved.
    """
    if not generated:
        return "UNKNOWN"
    text = generated.strip()

    # 0) the whole answer, or a "label:" line, is a code's full name ("Affirm",
    #    "label: Simple Reflection"). Zero-shot models do this often; it resolves
    #    only if the name is unique among `allowed` (added 2026-10-06, P5 pilot).
    for cand in [text] + re.findall(r'label["\']?\s*[:=]\s*["\']?([^\n"\'}]+)', text, flags=re.IGNORECASE)[::-1]:
        hit = code_from_name(cand, allowed)
        if hit:
            return hit

    # 1) explicit "label": "XX" / label = XX (take the last occurrence)
    field = re.findall(
        r'label["\']?\s*[:=]\s*["\']?([A-Za-z][A-Za-z0-9+\-]*)',
        text,
        flags=re.IGNORECASE,
    )
    for cand in reversed(field):
        hit = _resolve(cand, allowed)
        if hit:
            return hit

    tokens = [t for t in re.split(r"[\s,;.\n]+", text) if t]

    # 2) head token (bare-label / fine-tuned case)
    if tokens:
        hit = _resolve(tokens[0], allowed)
        if hit:
            return hit

    # 3) scan from the end (explanation-first case: label comes last)
    for t in reversed(tokens):
        hit = _resolve(t, allowed)
        if hit:
            return hit

    return "UNKNOWN"


_EXPLANATION_FIELD = re.compile(
    r'explanation["\']?\s*[:=]\s*(.+)', flags=re.IGNORECASE | re.DOTALL
)


def emitted_rationale(generated: str, min_words: int = 4) -> bool:
    """Whether a generation contains a rationale rather than just a label.

    This is the instruction-compliance signal for the fine-tuning arms: an
    `inf_bare` prompt should yield a bare code, an `inf_cot` prompt a rationale.
    Because the local model generates freely, either instruction can be ignored,
    and that is exactly what the FT-Bare/FT-Rat cross is measuring.

    Detection is deliberately simple: an explicit non-trivial ``explanation``
    field, or failing that enough prose words that the output cannot be a bare
    code. Codes themselves are 1-3 characters and never reach `min_words`.
    """
    if not generated:
        return False
    text = generated.strip()
    m = _EXPLANATION_FIELD.search(text)
    if m:
        return len(m.group(1).strip(" \"'{}\n").split()) >= 2
    words = [w for w in re.split(r"[\s,;.\n]+", text) if any(c.isalpha() for c in w)]
    return len(words) >= min_words


class TieredAnnotator:
    """Two-stage T1 -> T2 classifier over a HuggingFace causal LM.

    Args:
        base_model: HF model id (e.g. ``Qwen/Qwen2.5-7B-Instruct``).
        t1_adapter_dir / t2_adapter_dir: optional LoRA adapter dirs. If both are
            None the annotator is zero-shot (plain base model). If provided, the
            base model is wrapped once and both adapters are attached.
        shared_adapter_dir: a single LoRA covering both tiers, for the
            1-adapter / 2-call arms. Mutually exclusive with the pair above. The
            two-call flow is otherwise untouched, so a cell using this differs
            from the two-adapter cell in adapter count alone.
        structure_suffix: prompt variant, "" for rationale-first (``inf_cot``)
            or "_bare" for label-only (``inf_bare``).
        fewshot_provider: optional callable ``(speaker, tier, t1_label) -> list``
            of alternating user/assistant exemplar messages, spliced between the
            system prompt and the target utterance.
    """

    def __init__(
        self,
        base_model: str,
        t1_adapter_dir: Optional[str] = None,
        t2_adapter_dir: Optional[str] = None,
        shared_adapter_dir: Optional[str] = None,
        force_cpu: bool = False,
        trust_remote_code: bool = False,
        max_new_tokens: int = 8,
        max_input_len: int = 1024,
        structure_suffix: str = "",
        fewshot_provider: Optional[Callable[[str, str, Optional[str]], List[Dict[str, str]]]] = None,
    ):
        if shared_adapter_dir and (t1_adapter_dir or t2_adapter_dir):
            raise ValueError(
                "shared_adapter_dir is mutually exclusive with t1/t2_adapter_dir"
            )
        self.max_new_tokens = int(max_new_tokens)
        self.max_input_len = int(max_input_len)
        self.structure_suffix = structure_suffix
        self.fewshot_provider = fewshot_provider
        self.is_shared = bool(shared_adapter_dir)
        self.is_finetuned = bool(t1_adapter_dir and t2_adapter_dir)

        model, tokenizer, device = load_model_and_tokenizer(
            base_model,
            adapter_dir=None,
            for_training=False,
            force_cpu=force_cpu,
            trust_remote_code=trust_remote_code,
        )

        if self.is_shared:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(shared_adapter_dir))
            model.eval()
        elif self.is_finetuned:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(t1_adapter_dir), adapter_name="t1")
            model.load_adapter(str(t2_adapter_dir), adapter_name="t2")
            model.eval()

        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    # -- internals ---------------------------------------------------------
    def _set_adapter(self, name: str) -> None:
        # The shared-adapter arms have one LoRA active for both calls, so there
        # is nothing to switch.
        if self.is_finetuned:
            self.model.set_adapter(name)

    def _generate(
        self,
        messages: List[Dict[str, str]],
        do_sample: bool = False,
        temperature: Optional[float] = None,
    ) -> Tuple[str, int, int]:
        """Decode a reply, greedy by default or sampled when ``do_sample``.

        Returns ``(text, n_prompt_tokens, n_generated_tokens)``. The token counts
        let the caller tell a genuinely short reply from one clipped by
        `max_new_tokens`, and spot prompts truncated by `max_input_len`.

        ``do_sample=True`` (with ``temperature``) is used by
        `predict_row_selfconsistent` to draw independent decodes whose agreement
        is a confidence signal; every other caller keeps the greedy default so
        normal inference is byte-for-byte unchanged.
        """
        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **chat_template_kwargs(self.tokenizer),
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_len,
            add_special_tokens=False,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        n_prompt = int(inputs["input_ids"].shape[1])
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=bool(do_sample),
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if do_sample and temperature is not None:
            gen_kwargs["temperature"] = float(temperature)
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        new_ids = out[0, n_prompt:]
        gen = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return gen, n_prompt, int(new_ids.shape[0])

    def score_codes(
        self, messages: List[Dict[str, str]], codes: List[str]
    ) -> Tuple[Dict[str, float], int]:
        """Probability of each code as the answer, normalised over ``codes``.

        Each code is scored by the log-likelihood of exactly the completion the
        model was trained to emit, ``" {code}\\n"`` (`sft.data.build_completion`),
        after the same prompt `_generate` builds. Scoring the terminator too
        stops a code from winning just by being the prefix of another
        (``C`` vs ``CR``). The log-likelihoods are softmaxed over the candidate
        set, turning the generative labeller into a classifier with a full
        distribution -- the calibrated confidence self-training selection needs,
        which a vote over sampled decodes cannot give (every accepted v1 label
        had vote confidence 1.0).

        The prompt is encoded once; each candidate's few tokens are then scored
        against its cached keys/values, and the cache is cropped back to the
        prompt before the next candidate. Returns ``(probs, n_prompt_tokens)``.
        """
        import math

        import torch

        from sft.data import build_completion

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **chat_template_kwargs(self.tokenizer),
        )
        enc = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self.max_input_len, add_special_tokens=False,
        )
        ids = enc["input_ids"].to(self.device)
        n_prompt = int(ids.shape[1])
        cand_ids = [
            self.tokenizer(build_completion(c), add_special_tokens=False)["input_ids"]
            for c in codes
        ]
        loglik: Dict[str, float] = {}
        with torch.no_grad():
            out = self.model(input_ids=ids, use_cache=True)
            cache = out.past_key_values
            first = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
            can_crop = hasattr(cache, "crop")
            for code, toks in zip(codes, cand_ids):
                lp = float(first[toks[0]])
                if len(toks) > 1:
                    step = torch.tensor([toks[:-1]], device=self.device)
                    if can_crop:
                        o = self.model(input_ids=step, past_key_values=cache, use_cache=True)
                        cache.crop(n_prompt)
                    else:  # older transformers: recompute from scratch
                        o = self.model(input_ids=torch.cat([ids, step], dim=1))
                        o.logits = o.logits[:, n_prompt:]
                    lps = torch.log_softmax(o.logits[0].float(), dim=-1)
                    lp += float(sum(lps[i, t] for i, t in enumerate(toks[1:])))
                loglik[code] = lp
        m = max(loglik.values())
        z = sum(math.exp(v - m) for v in loglik.values())
        return {c: math.exp(v - m) / z for c, v in loglik.items()}, n_prompt

    def predict_row_scored(
        self,
        df: pd.DataFrame,
        row_pos: int,
        context_mode: str,
        num_context_turns: int,
    ) -> Dict[str, object]:
        """Two-call T1 -> T2 labelling with a full distribution at each tier.

        Mirrors `predict_row`: T1 over the speaker's T1 codes, then T2 prompted
        with the predicted (argmax) T1 group and scored over the speaker's full
        T2 vocabulary (unconstrained decoding allows any of them). Confidence is
        the joint ``p(T1) * p(T2 | T1)``.
        """
        speaker = df.iloc[row_pos]["speaker"]
        self._set_adapter("t1")
        t1_messages = build_messages_t1(
            df, row_pos, context_mode, num_context_turns,
            self.structure_suffix, self._fewshot(speaker, "t1", None),
        )
        t1_probs, t1_n = self.score_codes(t1_messages, t1_codes_for_speaker(speaker))
        t1_pred = max(t1_probs, key=t1_probs.get)

        self._set_adapter("t2")
        t2_messages = build_messages_t2(
            df, row_pos, t1_pred, context_mode, num_context_turns,
            self.structure_suffix, self._fewshot(speaker, "t2", t1_pred),
        )
        t2_probs, t2_n = self.score_codes(t2_messages, t2_codes_for_speaker(speaker))
        t2_pred = max(t2_probs, key=t2_probs.get)
        return {
            "t1_pred": t1_pred,
            "t2_pred": t2_pred,
            "t1_probs": t1_probs,
            "t2_probs": t2_probs,
            "t1_conf": t1_probs[t1_pred],
            "t2_conf": t2_probs[t2_pred],
            "confidence": t1_probs[t1_pred] * t2_probs[t2_pred],
            "t1_n_prompt_tokens": t1_n,
            "t2_n_prompt_tokens": t2_n,
        }

    # -- public ------------------------------------------------------------
    def _fewshot(self, speaker: str, tier: str, t1_label: Optional[str]) -> Optional[List[Dict[str, str]]]:
        if self.fewshot_provider is None:
            return None
        return self.fewshot_provider(speaker, tier, t1_label)

    def predict_row(
        self,
        df: pd.DataFrame,
        row_pos: int,
        context_mode: str,
        num_context_turns: int,
        restrict_t2_to_group: bool = False,
    ) -> Dict[str, object]:
        """Annotate the utterance at ``row_pos``.

        Returns the parsed labels plus the raw generations, generated token
        counts, and rationale-emission flags, so instruction compliance can be
        scored downstream (it is not recoverable from a parsed label alone).
        """
        speaker = df.iloc[row_pos]["speaker"]

        # Tier 1
        self._set_adapter("t1")
        t1_messages = build_messages_t1(
            df, row_pos, context_mode, num_context_turns,
            self.structure_suffix, self._fewshot(speaker, "t1", None),
        )
        t1_raw, t1_n_prompt, t1_n_gen = self._generate(t1_messages)
        t1_pred = parse_label(t1_raw, t1_codes_for_speaker(speaker))

        # Tier 2 conditioned on the predicted T1 group. If T1 was unparseable,
        # fall back to the full speaker T2 vocabulary and an empty-spec prompt.
        t1_for_prompt = t1_pred if t1_pred != "UNKNOWN" else t1_codes_for_speaker(speaker)[0]
        if restrict_t2_to_group and t1_pred != "UNKNOWN":
            t2_allowed = t2_codes_for_group(speaker, t1_pred)
        else:
            t2_allowed = t2_codes_for_speaker(speaker)

        self._set_adapter("t2")
        t2_messages = build_messages_t2(
            df, row_pos, t1_for_prompt, context_mode, num_context_turns,
            self.structure_suffix, self._fewshot(speaker, "t2", t1_for_prompt),
        )
        t2_raw, t2_n_prompt, t2_n_gen = self._generate(t2_messages)
        t2_pred = parse_label(t2_raw, t2_allowed)

        return {
            "t1_pred": t1_pred,
            "t2_pred": t2_pred,
            "t1_raw": t1_raw,
            "t2_raw": t2_raw,
            "t1_n_prompt_tokens": t1_n_prompt,
            "t2_n_prompt_tokens": t2_n_prompt,
            "t1_n_gen_tokens": t1_n_gen,
            "t2_n_gen_tokens": t2_n_gen,
            "t1_emitted_rationale": emitted_rationale(t1_raw),
            "t2_emitted_rationale": emitted_rationale(t2_raw),
        }

    def predict_row_selfconsistent(
        self,
        df: pd.DataFrame,
        row_pos: int,
        context_mode: str,
        num_context_turns: int,
        restrict_t2_to_group: bool = False,
        k: int = 5,
        temperature: float = 0.7,
    ) -> Dict[str, object]:
        """Self-consistency pseudo-label with a confidence signal.

        Draws ``k`` sampled decodes for the Tier-1 call, takes the majority code,
        then conditions the Tier-2 call on that majority T1 and draws ``k`` more.
        The per-tier agreement fraction (majority count / k) is the confidence;
        ``confidence`` is the product, i.e. the model must be consistent on BOTH
        tiers for a pseudo-label to be trusted. Used by
        `selftrain.label_pool` to gate which pseudo-labels enter training.

        Returns the same keys as `predict_row` plus ``t1_agreement``,
        ``t2_agreement`` and ``confidence``.
        """
        speaker = df.iloc[row_pos]["speaker"]

        # Tier 1: k independent samples -> majority
        self._set_adapter("t1")
        t1_messages = build_messages_t1(
            df, row_pos, context_mode, num_context_turns,
            self.structure_suffix, self._fewshot(speaker, "t1", None),
        )
        t1_allowed = t1_codes_for_speaker(speaker)
        t1_votes: List[str] = []
        t1_raw_last = ""
        for _ in range(k):
            raw, _, _ = self._generate(t1_messages, do_sample=True, temperature=temperature)
            t1_raw_last = raw
            t1_votes.append(parse_label(raw, t1_allowed))
        t1_pred, t1_count = Counter(t1_votes).most_common(1)[0]
        t1_agreement = t1_count / float(k)

        # Tier 2 conditioned on the majority T1 group.
        t1_for_prompt = t1_pred if t1_pred != "UNKNOWN" else t1_allowed[0]
        if restrict_t2_to_group and t1_pred != "UNKNOWN":
            t2_allowed = t2_codes_for_group(speaker, t1_pred)
        else:
            t2_allowed = t2_codes_for_speaker(speaker)

        self._set_adapter("t2")
        t2_messages = build_messages_t2(
            df, row_pos, t1_for_prompt, context_mode, num_context_turns,
            self.structure_suffix, self._fewshot(speaker, "t2", t1_for_prompt),
        )
        t2_votes: List[str] = []
        t2_raw_last = ""
        for _ in range(k):
            raw, _, _ = self._generate(t2_messages, do_sample=True, temperature=temperature)
            t2_raw_last = raw
            t2_votes.append(parse_label(raw, t2_allowed))
        t2_pred, t2_count = Counter(t2_votes).most_common(1)[0]
        t2_agreement = t2_count / float(k)

        return {
            "t1_pred": t1_pred,
            "t2_pred": t2_pred,
            "t1_raw": t1_raw_last,
            "t2_raw": t2_raw_last,
            "t1_agreement": t1_agreement,
            "t2_agreement": t2_agreement,
            "confidence": t1_agreement * t2_agreement,
        }

    def predict_rows(
        self,
        df: pd.DataFrame,
        row_positions: List[int],
        context_mode: str,
        num_context_turns: int,
        restrict_t2_to_group: bool = False,
        desc: str = "annotating",
    ) -> List[Dict[str, str]]:
        """Predict over many rows. Returns a list of dicts with metadata and
        both tier predictions, aligned to ``row_positions``."""
        results: List[Dict[str, str]] = []
        for pos in tqdm(row_positions, desc=desc):
            row = df.iloc[pos]
            pred = self.predict_row(
                df, pos, context_mode, num_context_turns, restrict_t2_to_group
            )
            results.append(
                {
                    "conv_id": str(row["conv_id"]),
                    "row_pos": int(pos),
                    "speaker": row["speaker"],
                    "utt_text": row["utt_text"],
                    "t1_label_GT": row.get("t1_label_GT"),
                    "t2_label_GT": row.get("t2_label_GT"),
                    **pred,
                }
            )
        return results

    def close(self) -> None:
        """Free GPU/accelerator memory held by this annotator."""
        try:
            import gc

            import torch

            del self.model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass


__all__ = ["TieredAnnotator", "parse_label", "emitted_rationale", "LABEL_ALIASES"]
