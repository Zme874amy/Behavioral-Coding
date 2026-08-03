"""Single-call annotator: one generation yields a rationale and both labels.

The two-call `automisc_ft.infer.TieredAnnotator` swaps between a T1 and a T2
adapter per utterance. Here a single LoRA adapter (or none, for the zero-shot
arm) answers both tiers at once, so there is one adapter to load and one
generation per utterance.

Columns written mirror the two-call runner's so `baseline.eval` needs no
special case, with three additions the single-call format makes available:
`json_ok`, `hierarchy_ok`, and the raw rationale.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from baseline.single_call import (
    UNKNOWN,
    build_messages,
    is_hierarchically_consistent,
    parse_completion,
    rationale_word_count,
)
from components.hf_load import load_model_and_tokenizer


class SingleCallAnnotator:
    """Joint T1+T2 classifier over a HuggingFace causal LM.

    Args:
        base_model: HF model id (e.g. ``Qwen/Qwen2.5-7B-Instruct``).
        adapter_dir: optional LoRA adapter. None gives the zero-shot arm.
        emit_rationale: which Output Format block the prompt asks for. Only the
            output contract changes; the taxonomy is identical either way.
    """

    def __init__(
        self,
        base_model: str,
        adapter_dir: Optional[str] = None,
        emit_rationale: bool = True,
        force_cpu: bool = False,
        trust_remote_code: bool = False,
        max_new_tokens: int = 256,
        max_input_len: int = 8192,
    ):
        self.emit_rationale = bool(emit_rationale)
        self.max_new_tokens = int(max_new_tokens)
        self.max_input_len = int(max_input_len)

        model, tokenizer, device = load_model_and_tokenizer(
            base_model,
            adapter_dir=None,
            for_training=False,
            force_cpu=force_cpu,
            trust_remote_code=trust_remote_code,
        )
        if adapter_dir:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_dir))
            model.eval()

        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _generate(self, messages: List[Dict[str, str]]) -> Tuple[str, int, int]:
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
        n_prompt = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new_ids = out[0, n_prompt:]
        return (
            self.tokenizer.decode(new_ids, skip_special_tokens=True),
            n_prompt,
            int(new_ids.shape[0]),
        )

    def predict_row(
        self,
        df: pd.DataFrame,
        row_pos: int,
        context_mode: str,
        num_context_turns: int,
    ) -> Dict[str, object]:
        speaker = df.iloc[row_pos]["speaker"]
        messages = build_messages(
            df, row_pos, context_mode, num_context_turns, self.emit_rationale
        )
        raw, n_prompt, n_gen = self._generate(messages)
        parsed = parse_completion(raw, speaker)
        t1, t2 = parsed["t1"], parsed["t2"]
        n_words = rationale_word_count(parsed["rationale"])
        return {
            # Column names `baseline.eval` already expects.
            "t1_label_auto": t1,
            "t2_label_auto": t2,
            "t1_raw": raw,
            "t2_raw": raw,
            "t1_emitted_rationale": n_words > 0,
            "t2_emitted_rationale": n_words > 0,
            "t1_n_prompt_tokens": n_prompt,
            "t2_n_prompt_tokens": n_prompt,
            "t1_n_gen_tokens": n_gen,
            "t2_n_gen_tokens": n_gen,
            # Single-call extras.
            "raw": raw,
            "rationale": parsed["rationale"],
            "rationale_words": n_words,
            "emitted_rationale": n_words > 0,
            "json_ok": bool(parsed["json_ok"]),
            "hierarchy_ok": is_hierarchically_consistent(speaker, t1, t2),
            "n_prompt_tokens": n_prompt,
            "n_gen_tokens": n_gen,
        }

    def close(self) -> None:
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


__all__ = ["SingleCallAnnotator", "UNKNOWN"]
