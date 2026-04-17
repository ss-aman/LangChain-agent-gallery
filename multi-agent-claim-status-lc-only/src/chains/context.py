"""Pipeline context – the shared state dict passed between every chain.

CHALLENGE vs LangGraph:
  LangGraph enforces a TypedDict schema with Annotated merge semantics per key.
  Here we pass a plain dict. There is NO compile-time validation, no automatic
  message-append merge, and no built-in checkpointing.  Discipline in every
  chain function is the only guard against KeyError / stale values.
"""
from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class SecurityCtx(TypedDict, total=False):
    ip_address: str
    user_agent: str
    rate_limit_remaining: int
    threat_detected: bool
    threat_type: Optional[str]


class AuthCtx(TypedDict, total=False):
    user_id: str
    username: str
    email: str
    jti: str


class IntentCtx(TypedDict, total=False):
    intent: str
    claim_number: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    claim_type: Optional[str]
    limit: int
    confidence: float


class PipelineContext(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────────────
    raw_token: str
    raw_query: str
    session_id: str
    ip_address: str
    user_agent: str

    # ── Security ──────────────────────────────────────────────────────────────
    security_ctx: SecurityCtx
    security_passed: bool
    sanitized_query: str

    # ── Auth ──────────────────────────────────────────────────────────────────
    auth_ctx: AuthCtx
    is_authenticated: bool

    # ── Query ─────────────────────────────────────────────────────────────────
    intent_ctx: IntentCtx

    # ── Output ────────────────────────────────────────────────────────────────
    final_response: str
    error_code: Optional[str]
    error_message: Optional[str]

    # ── Observability ─────────────────────────────────────────────────────────
    audit_trail: list[dict[str, Any]]
    current_step: str


def new_context(
    raw_query: str,
    raw_token: str,
    session_id: str,
    ip_address: str = "unknown",
    user_agent: str = "",
) -> PipelineContext:
    """Factory: creates a clean context with safe defaults."""
    return PipelineContext(
        raw_token=raw_token,
        raw_query=raw_query,
        session_id=session_id,
        ip_address=ip_address,
        user_agent=user_agent,
        security_ctx=SecurityCtx(ip_address=ip_address, user_agent=user_agent),
        security_passed=False,
        sanitized_query="",
        auth_ctx=AuthCtx(),
        is_authenticated=False,
        intent_ctx=IntentCtx(),
        final_response="",
        error_code=None,
        error_message=None,
        audit_trail=[],
        current_step="init",
    )
