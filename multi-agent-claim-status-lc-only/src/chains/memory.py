"""Conversation memory management for LangChain-only architecture.

CAPABILITIES demonstrated:
  • ConversationBufferWindowMemory  – bounded sliding-window buffer
  • ConversationSummaryBufferMemory – auto-summarises old turns to save tokens
  • RunnableWithMessageHistory      – wires a BaseChatMessageHistory into any chain
  • RedisChatMessageHistory         – persists history in Redis per session_id

CHALLENGES vs LangGraph:
  1. Memory is scoped to ONE AgentExecutor. There is no shared state graph, so
     different agents in the same request each have their own isolated memory.
     You must manually copy or thread messages between agents.

  2. RunnableWithMessageHistory injects `history` into the prompt automatically
     but only works when the chain input is a DICT with a known `input_messages_key`.
     Wrapping a complex pipeline breaks the automatic injection.

  3. Persistence is the developer's responsibility: if the process restarts, any
     in-process ConversationBufferMemory is lost. RedisChatMessageHistory solves
     this but requires careful key management to avoid cross-session contamination.

  4. Token budget: ConversationBufferMemory will happily grow without bound.
     You MUST use the windowed or summary variant in production.
"""
from __future__ import annotations

from typing import Optional

import redis.asyncio as aioredis
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.config import get_settings

settings = get_settings()


# ── History backends ───────────────────────────────────────────────────────────

def get_redis_message_history(session_id: str) -> BaseChatMessageHistory:
    """Factory for RunnableWithMessageHistory.

    CHALLENGE: RedisChatMessageHistory is synchronous under the hood.
    For true async use, you'd need a custom AsyncRedisChatMessageHistory backed
    by redis.asyncio – not available out of the box in langchain-community.
    """
    return RedisChatMessageHistory(
        session_id,
        url=settings.REDIS_URL,
        key_prefix="chat_history:",
        ttl=settings.SESSION_TTL_SECONDS,
    )


# ── Memory objects (per-AgentExecutor) ────────────────────────────────────────

def make_window_memory(k: int = 10) -> ConversationBufferWindowMemory:
    """Keeps the last k conversation turns in memory.

    CHALLENGE: This is in-process only. Every new HTTP request creates a fresh
    AgentExecutor with empty memory unless you explicitly reload history.
    """
    return ConversationBufferWindowMemory(
        k=k,
        memory_key="chat_history",
        return_messages=True,
        output_key="output",
    )


def make_summary_memory(llm: BaseChatModel, max_token_limit: int = 2000) -> ConversationSummaryBufferMemory:
    """Keeps recent messages verbatim; summarises older ones with the LLM.

    CHALLENGE: Summarisation itself costs tokens and latency on every turn.
    In a high-throughput system this can double your LLM bill.
    """
    return ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=max_token_limit,
        memory_key="chat_history",
        return_messages=True,
        output_key="output",
    )


# ── RunnableWithMessageHistory helper ─────────────────────────────────────────

def wrap_with_history(
    runnable: any,
    *,
    input_messages_key: str = "input",
    history_messages_key: str = "chat_history",
    output_messages_key: str = "output",
) -> RunnableWithMessageHistory:
    """Wrap any LCEL runnable with automatic Redis-backed message history.

    Usage:
        chain_with_memory = wrap_with_history(my_chain)
        result = await chain_with_memory.ainvoke(
            {"input": "What's my claim status?"},
            config={"configurable": {"session_id": "user-123-session-abc"}},
        )

    CHALLENGE: The chain's input schema MUST contain `input_messages_key`.
    If your chain takes a complex dict (e.g. PipelineContext), you cannot use
    this wrapper directly — you'd need to reshape inputs/outputs with
    RunnablePassthrough or RunnableLambda adapters.
    """
    return RunnableWithMessageHistory(
        runnable,
        get_redis_message_history,
        input_messages_key=input_messages_key,
        history_messages_key=history_messages_key,
        output_messages_key=output_messages_key,
    )
