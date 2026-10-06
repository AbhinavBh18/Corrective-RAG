"""LLM factories shared by the graph and the compatibility pipeline.

Kept separate to avoid a graph-builder <-> pipeline import cycle.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import GEMINI_API_KEY, JUDGE_MODEL_NAME, JUDGE_TEMPERATURE, LLM_MODEL_NAME, LLM_TEMPERATURE

logger = logging.getLogger(__name__)


def get_llm(model_name: str | None = None, temperature: float | None = ...) -> ChatGoogleGenerativeAI:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set. Add it to .env; no source fallback is used.")
    temp = LLM_TEMPERATURE if temperature is ... else temperature
    kwargs: dict[str, Any] = {"model": model_name or LLM_MODEL_NAME, "google_api_key": GEMINI_API_KEY}
    if temp is not None:
        kwargs["temperature"] = temp
    return ChatGoogleGenerativeAI(**kwargs)


def get_judge_llm() -> ChatGoogleGenerativeAI:
    return get_llm(model_name=JUDGE_MODEL_NAME, temperature=JUDGE_TEMPERATURE)
