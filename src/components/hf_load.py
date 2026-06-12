"""Shared Hugging Face model/tokenizer loading with auth for gated checkpoints."""
from __future__ import annotations

import os
from typing import Any, Optional, Tuple

from hydra.utils import log


def resolve_hf_token(explicit: Optional[str] = None) -> Optional[str]:
    """Return HF token from explicit arg, HF_TOKEN env, or huggingface-cli cache."""
    if explicit:
        return explicit
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def _ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
            or tokenizer.bos_token
            or tokenizer.cls_token
            or tokenizer.sep_token
        )


def _select_inference_device(force_cpu: bool = False) -> str:
    import torch

    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_training_dtype(
    *,
    fp16: bool,
    bf16: bool,
    for_cuda: bool,
):
    import torch

    if not for_cuda:
        return torch.float32
    if bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if fp16:
        return torch.float16
    return torch.float32


def _load_causal_lm(
    model_id: str,
    *,
    token: Optional[str],
    torch_dtype,
    device_map: Any,
    trust_remote_code: bool,
    attn_implementation: Optional[str],
):
    from transformers import AutoModelForCausalLM

    load_kwargs: dict = dict(
        token=token,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation

    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except Exception as first_exc:
        mid = model_id.lower()
        if "gemma-3n" in mid or "gemma3n" in mid:
            log.warning(
                "AutoModelForCausalLM failed for %s (%s); retrying with trust_remote_code=True",
                model_id,
                first_exc,
            )
            load_kwargs["trust_remote_code"] = True
            try:
                return AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            except Exception as second_exc:
                try:
                    from transformers import AutoModelForImageTextToText

                    log.warning(
                        "Retrying %s with AutoModelForImageTextToText", model_id
                    )
                    return AutoModelForImageTextToText.from_pretrained(
                        model_id, **load_kwargs
                    )
                except Exception:
                    raise second_exc from first_exc
        raise


def load_model_and_tokenizer(
    model_id: str,
    *,
    adapter_dir: Optional[str] = None,
    for_training: bool = False,
    force_cpu: bool = False,
    fp16: bool = False,
    bf16: bool = False,
    trust_remote_code: bool = False,
    attn_implementation: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> Tuple[Any, Any, str]:
    """Load tokenizer + causal LM (optionally with a LoRA adapter).

    Returns:
        (model, tokenizer, device_str)
    """
    import torch
    from transformers import AutoTokenizer

    token = resolve_hf_token(hf_token)
    if token is None and _looks_gated(model_id):
        raise RuntimeError(
            f"No Hugging Face token found for gated model '{model_id}'. "
            "Run `huggingface-cli login` or set HF_TOKEN, and accept the model "
            "license on huggingface.co."
        )

    log.info("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=token, use_fast=True, trust_remote_code=trust_remote_code
    )
    _ensure_pad_token(tokenizer)

    on_cuda = torch.cuda.is_available() and not force_cpu
    on_mps = (
        not on_cuda
        and not force_cpu
        and getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    )

    if for_training:
        torch_dtype = _resolve_training_dtype(
            fp16=fp16, bf16=bf16, for_cuda=on_cuda
        )
        device_map = "auto" if on_cuda else None
        device_str = "cuda" if on_cuda else ("mps" if on_mps else "cpu")
    else:
        # Half precision on CUDA keeps a 7B model near ~15GB instead of ~30GB
        # (fp32), which OOMs on a 40GB GPU. CPU stays float32 for op coverage.
        if on_cuda:
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            torch_dtype = torch.float32
        device_map = None
        device_str = _select_inference_device(force_cpu=force_cpu)

    log.info(
        "Loading model: %s (dtype=%s, device_map=%s, inference_device=%s)",
        model_id,
        torch_dtype,
        device_map,
        device_str,
    )
    model = _load_causal_lm(
        model_id,
        token=token,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation if on_cuda else None,
    )

    if adapter_dir:
        from peft import PeftModel

        log.info("Loading LoRA adapter from %s", adapter_dir)
        model = PeftModel.from_pretrained(model, adapter_dir)

    if device_map is None:
        model = model.to(device_str)

    model.eval()
    if for_training:
        model.train()
    return model, tokenizer, device_str


def _looks_gated(model_id: str) -> bool:
    mid = model_id.lower()
    return "gemma" in mid or "llama" in mid or "meta-llama" in mid
