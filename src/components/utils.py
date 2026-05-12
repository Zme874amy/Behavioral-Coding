"""
Utilities for invoking chat models across parser and annotator modules.
"""
import os
from typing import Optional, Type, List, Literal
from pydantic import BaseModel
import lmstudio as lms
import openai
from hydra.utils import log

def get_provider(model: str) -> Literal['openai', 'lmstudio']:
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
    provider: Literal['openai', 'lmstudio'] = 'openai',
    temperature: float = 0.0,
    response_format: Optional[Type[BaseModel]] = None,
    **kwargs,
) -> BaseModel | str:
    """
    """
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
        lms_model = lms.llm(model)
        completion = lms_model.respond(
            {"messages": messages},
             config={"temperature": temperature}, 
             response_format=response_format
            )
                                       
        return completion.parsed
    else:
        raise ValueError(f"Provider '{provider}' not recognized. Use 'openai' or 'lmstudio'.")
