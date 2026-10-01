"""
Utilities for invoking chat models across parser and annotator modules.
"""
import os
from typing import Optional, Type, List, Literal
from pydantic import BaseModel
try:
    import lmstudio as lms
except Exception:  # lmstudio is unavailable/broken on headless HPC nodes
    lms = None
import openai
from hydra.utils import log

_azure_client = None
_vllm_clients: dict = {}

def get_vllm_client(base_url: Optional[str] = None):
    """Lazily build an OpenAI-compatible client for a self-hosted server.

    Used by the ``vllm_server`` provider to reach a vLLM OpenAI-compatible
    endpoint (e.g. a large open teacher/generator model served on an MLeRP
    node). ``base_url`` defaults to ``$VLLM_BASE_URL``; the API key defaults to
    ``$VLLM_API_KEY`` or the literal ``"EMPTY"`` that vLLM accepts. Clients are
    cached per base_url so repeated calls reuse one connection pool.
    """
    url = (base_url or os.environ.get("VLLM_BASE_URL", "")).strip()
    if not url:
        raise EnvironmentError(
            "VLLM_BASE_URL must be set (or base_url passed) to use the "
            "'vllm_server' provider, e.g. http://localhost:8000/v1"
        )
    if url not in _vllm_clients:
        api_key = os.environ.get("VLLM_API_KEY", "EMPTY").strip() or "EMPTY"
        _vllm_clients[url] = openai.OpenAI(base_url=url, api_key=api_key)
    return _vllm_clients[url]

def get_azure_client():
    """Lazily build an AzureOpenAI client from .env / environment variables."""
    global _azure_client
    if _azure_client is None:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
        api_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview").strip()
        if not endpoint or not api_key:
            raise EnvironmentError(
                "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set "
                "(e.g. in a .env file at the repo root) to use the 'azure' provider."
            )
        _azure_client = openai.AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
    return _azure_client

def get_provider(model: str) -> Literal['openai', 'lmstudio']:
    lms_models = set()
    if lms is not None:
        lms_models = {m.model_key for m in lms.list_downloaded_models("llm")}
    openai_models = set()
    if "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"].strip():
        try:
            openai.api_key = os.environ["OPENAI_API_KEY"]
            openai_models = {m.id for m in openai.models.list().data}
        except Exception as e:
            log.warning(f"Failed to fetch OpenAI models: {e}")
    else:
        log.info("OPENAI_API_KEY not set or empty; skipping OpenAI model validation")
    
    if model in openai_models:
        return 'openai'
    elif model in lms_models:
        return 'lmstudio'
    else:
        log.info(f"Available LM Studio models: {lms_models}")
        log.info(f"Available OpenAI models: {openai_models}")
        raise ValueError(f"Model '{model}' not found in OpenAI or LM Studio models.")
    
def call_chat_model(
    messages: list[dict],
    model: str,
    provider: Literal['openai', 'lmstudio', 'azure', 'vllm_server'] = 'openai',
    temperature: float = 0.0,
    response_format: Optional[Type[BaseModel]] = None,
    **kwargs,
) -> BaseModel | str:
    """
    """
    if provider == 'vllm_server':
        # Self-hosted OpenAI-compatible endpoint (e.g. a large open teacher model
        # served with vLLM). ``base_url`` may be passed explicitly or via
        # $VLLM_BASE_URL; it is consumed here and never forwarded to the API.
        client = get_vllm_client(kwargs.pop("base_url", None))
        if response_format is not None:
            response = client.chat.completions.parse(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format=response_format,
                **kwargs,
            )
            return response.choices[0].message.parsed.model_dump()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].message.content
    if provider == 'azure':
        client = get_azure_client()
        if response_format is not None:
            response = client.chat.completions.parse(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format=response_format,
                **kwargs,
            )
            return response.choices[0].message.parsed.model_dump()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].message.content
    if provider == 'openai':
        if openai is None:
            raise ImportError("openai library is required for openai models")
        response = openai.chat.completions.parse(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format=response_format,
            **kwargs,
        )
        return response.choices[0].message.parsed.model_dump()
    elif provider == 'lmstudio':
        if lms is None:
            raise ImportError("lmstudio library is required for lmstudio models")
        lms_model = lms.llm(model)
        completion = lms_model.respond(
            {"messages": messages},
             config={"temperature": temperature}, 
             response_format=response_format
            )
        if response_format is None:
            parsed = getattr(completion, "parsed", None)
            if isinstance(parsed, str) and parsed.strip():
                return parsed.strip()
            content = getattr(completion, "content", None)
            if isinstance(content, str) and content.strip():
                return content.strip()
            text = getattr(completion, "text", None)
            if isinstance(text, str) and text.strip():
                return text.strip()
            return str(completion).strip()
        return completion.parsed
    else:
        raise ValueError(
            f"Provider '{provider}' not recognized. "
            "Use 'openai', 'lmstudio', 'azure', or 'vllm_server'."
        )
