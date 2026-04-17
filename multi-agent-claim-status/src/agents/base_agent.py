"""Abstract base class shared by all agents in this system."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from src.config import get_settings

settings = get_settings()


def create_llm(**overrides: Any) -> BaseChatModel:
    """Factory that builds any OpenAI-compatible LLM from settings.

    Setting LLM_BASE_URL lets you point at Azure OpenAI, Ollama, LiteLLM,
    or any other OpenAI-compatible proxy without changing any agent code.
    """
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "model": settings.LLM_MODEL,
        "temperature": settings.LLM_TEMPERATURE,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "api_key": settings.LLM_API_KEY,
        **overrides,
    }
    if settings.LLM_BASE_URL:
        kwargs["base_url"] = settings.LLM_BASE_URL

    return ChatOpenAI(**kwargs)


class BaseClaimAgent(ABC):
    """Common interface for all specialised agents."""

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self.llm: BaseChatModel = llm or create_llm()

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def tools(self) -> list[BaseTool]:
        ...

    @abstractmethod
    async def run(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the agent's primary responsibility."""
        ...
