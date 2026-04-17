"""LangGraph shared state definition.

TypedDict with Annotated fields drives how LangGraph merges state across nodes.
`add_messages` uses append semantics; all other fields use last-write-wins.
"""
from __future__ import annotations

from typing import Annotated, Any, Optional, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class SecurityContext(TypedDict, total=False):
    ip_address: Optional[str]
    user_agent: Optional[str]
    rate_limit_remaining: int
    threat_detected: bool
    threat_type: Optional[str]


class AuthContext(TypedDict, total=False):
    user_id: Optional[str]
    username: Optional[str]
    email: Optional[str]
    jti: Optional[str]             # JWT ID (for revocation tracking)


class IntentContext(TypedDict, total=False):
    intent: Optional[str]
    claim_number: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    claim_type: Optional[str]
    limit: int
    confidence: float


class ClaimAgentState(TypedDict):
    # ── Conversation ──────────────────────────────────────────────────────────
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # ── Request metadata ──────────────────────────────────────────────────────
    raw_token: Optional[str]           # Bearer token from request header
    raw_query: Optional[str]           # Original user query (pre-sanitisation)
    sanitized_query: Optional[str]     # After InputSanitizer runs
    session_id: Optional[str]

    # ── Security ──────────────────────────────────────────────────────────────
    security_context: SecurityContext
    security_passed: bool

    # ── Authentication ────────────────────────────────────────────────────────
    auth_context: AuthContext
    is_authenticated: bool

    # ── Intent / Query ────────────────────────────────────────────────────────
    intent_context: IntentContext

    # ── Database result ───────────────────────────────────────────────────────
    claim_data: Optional[list[dict[str, Any]]]

    # ── Final output ──────────────────────────────────────────────────────────
    final_response: Optional[str]
    error_code: Optional[str]
    error_message: Optional[str]

    # ── Graph control ─────────────────────────────────────────────────────────
    current_node: Optional[str]
    iteration_count: int
    audit_trail: list[dict[str, Any]]
