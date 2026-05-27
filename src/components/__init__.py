"""Lightweight package init: avoid importing Parser/Annotator here (circular imports with datatypes)."""
from .utils import call_chat_model

__all__ = ["call_chat_model"]
