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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from automisc_ft.data import (
    build_messages_t1,
    build_messages_t2,
    t1_codes_for_speaker,
    t2_codes_for_group,
    t2_codes_for_speaker,
)
from components.hf_load import load_model_and_tokenizer
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


class TieredAnnotator:
    """Two-stage T1 -> T2 classifier over a HuggingFace causal LM.

    Args:
        base_model: HF model id (e.g. ``Qwen/Qwen2.5-7B-Instruct``).
        t1_adapter_dir / t2_adapter_dir: optional LoRA adapter dirs. If both are
            None the annotator is zero-shot (plain base model). If provided, the
            base model is wrapped once and both adapters are attached.
    """

    def __init__(
        self,
        base_model: str,
        t1_adapter_dir: Optional[str] = None,
        t2_adapter_dir: Optional[str] = None,
        force_cpu: bool = False,
        trust_remote_code: bool = False,
        max_new_tokens: int = 8,
        max_input_len: int = 1024,
    ):
        self.max_new_tokens = int(max_new_tokens)
        self.max_input_len = int(max_input_len)
        self.is_finetuned = bool(t1_adapter_dir and t2_adapter_dir)

        model, tokenizer, device = load_model_and_tokenizer(
            base_model,
            adapter_dir=None,
            for_training=False,
            force_cpu=force_cpu,
            trust_remote_code=trust_remote_code,
        )

        if self.is_finetuned:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(t1_adapter_dir), adapter_name="t1")
            model.load_adapter(str(t2_adapter_dir), adapter_name="t2")
            model.eval()

        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    # -- internals ---------------------------------------------------------
    def _set_adapter(self, name: str) -> None:
        if self.is_finetuned:
            self.model.set_adapter(name)

    def _generate(self, messages: List[Dict[str, str]]) -> str:
        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_len,
            add_special_tokens=False,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        gen = self.tokenizer.decode(
            out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return gen

    # -- public ------------------------------------------------------------
    def predict_row(
        self,
        df: pd.DataFrame,
        row_pos: int,
        context_mode: str,
        num_context_turns: int,
        restrict_t2_to_group: bool = False,
    ) -> Tuple[str, str]:
        """Return ``(t1_pred, t2_pred)`` for the utterance at ``row_pos``."""
        speaker = df.iloc[row_pos]["speaker"]

        # Tier 1
        self._set_adapter("t1")
        t1_messages = build_messages_t1(df, row_pos, context_mode, num_context_turns)
        t1_raw = self._generate(t1_messages)
        t1_pred = parse_label(t1_raw, t1_codes_for_speaker(speaker))

        # Tier 2 conditioned on the predicted T1 group. If T1 was unparseable,
        # fall back to the full speaker T2 vocabulary and an empty-spec prompt.
        t1_for_prompt = t1_pred if t1_pred != "UNKNOWN" else t1_codes_for_speaker(speaker)[0]
        if restrict_t2_to_group and t1_pred != "UNKNOWN":
            t2_allowed = t2_codes_for_group(speaker, t1_pred)
        else:
            t2_allowed = t2_codes_for_speaker(speaker)

        self._set_adapter("t2")
        t2_messages = build_messages_t2(df, row_pos, t1_for_prompt, context_mode, num_context_turns)
        t2_raw = self._generate(t2_messages)
        t2_pred = parse_label(t2_raw, t2_allowed)

        return t1_pred, t2_pred

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
            t1_pred, t2_pred = self.predict_row(
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
                    "t1_pred": t1_pred,
                    "t2_pred": t2_pred,
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


__all__ = ["TieredAnnotator", "parse_label", "LABEL_ALIASES"]
