"""LangChain async callback handler for audit + observability.

CAPABILITY: LangChain callbacks fire at every step automatically –
LLM calls, tool calls, chain start/end, agent actions, errors.
This gives end-to-end tracing without changing any business logic.

CHALLENGE: Callbacks are fire-and-forget and run OUT-OF-BAND.
You cannot modify the chain's output from inside a callback,
and there is no guaranteed execution order when multiple callbacks
are registered on the same chain.
"""
from __future__ import annotations

import time
from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

import structlog

_logger = structlog.get_logger("chain.callback")


class AuditCallbackHandler(AsyncCallbackHandler):
    """Logs every LangChain event: LLM calls, tool use, chain errors."""

    always_verbose: bool = True

    def __init__(self, session_id: str, user_id: Optional[str] = None) -> None:
        super().__init__()
        self.session_id = session_id
        self.user_id = user_id
        self._t0: dict[UUID, float] = {}

    # ── LLM events ─────────────────────────────────────────────────────────────

    async def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], *, run_id: UUID, **_: Any
    ) -> None:
        self._t0[run_id] = time.monotonic()
        _logger.info(
            "LLM_START",
            session=self.session_id,
            model=serialized.get("name", "unknown"),
            prompt_preview=prompts[0][:120] if prompts else "",
        )

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, **_: Any) -> None:
        elapsed = time.monotonic() - self._t0.pop(run_id, time.monotonic())
        usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}
        _logger.info(
            "LLM_END",
            session=self.session_id,
            elapsed_s=round(elapsed, 3),
            total_tokens=usage.get("total_tokens"),
        )

    async def on_llm_error(self, error: BaseException, *, run_id: UUID, **_: Any) -> None:
        _logger.error("LLM_ERROR", session=self.session_id, error=str(error))

    # ── Tool events ────────────────────────────────────────────────────────────

    async def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, *, run_id: UUID, **_: Any
    ) -> None:
        self._t0[run_id] = time.monotonic()
        _logger.info(
            "TOOL_START",
            session=self.session_id,
            tool=serialized.get("name"),
            input_preview=input_str[:120],
        )

    async def on_tool_end(self, output: str, *, run_id: UUID, **_: Any) -> None:
        elapsed = time.monotonic() - self._t0.pop(run_id, time.monotonic())
        _logger.info(
            "TOOL_END",
            session=self.session_id,
            elapsed_s=round(elapsed, 3),
            output_preview=str(output)[:120],
        )

    async def on_tool_error(self, error: BaseException, *, run_id: UUID, **_: Any) -> None:
        _logger.error("TOOL_ERROR", session=self.session_id, error=str(error))

    # ── Agent events ───────────────────────────────────────────────────────────

    async def on_agent_action(self, action: Any, *, run_id: UUID, **_: Any) -> None:
        _logger.info(
            "AGENT_ACTION",
            session=self.session_id,
            tool=getattr(action, "tool", None),
            tool_input=str(getattr(action, "tool_input", ""))[:120],
        )

    async def on_agent_finish(self, finish: Any, *, run_id: UUID, **_: Any) -> None:
        _logger.info(
            "AGENT_FINISH",
            session=self.session_id,
            output_preview=str(getattr(finish, "return_values", ""))[:120],
        )

    # ── Chain events ───────────────────────────────────────────────────────────

    async def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], *, run_id: UUID, **_: Any
    ) -> None:
        self._t0[run_id] = time.monotonic()
        _logger.debug("CHAIN_START", chain=serialized.get("name"), session=self.session_id)

    async def on_chain_end(self, outputs: dict[str, Any], *, run_id: UUID, **_: Any) -> None:
        elapsed = time.monotonic() - self._t0.pop(run_id, time.monotonic())
        _logger.debug("CHAIN_END", session=self.session_id, elapsed_s=round(elapsed, 3))

    async def on_chain_error(self, error: BaseException, *, run_id: UUID, **_: Any) -> None:
        _logger.error("CHAIN_ERROR", session=self.session_id, error=str(error))
