"""LangGraph node implementations.

Each node is an async function: State → partial State update dict.
Nodes are pure functions with respect to the graph state – side effects
(DB writes, Redis writes, logging) are clearly separated.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as aioredis
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.auth_agent import AuthAgent
from src.agents.claim_query_agent import ClaimQueryAgent
from src.graph.state import ClaimAgentState
from src.security.audit_logger import AuditLogger
from src.security.rate_limiter import RateLimiter
from src.security.sanitizer import InputSanitizer
from src.session.manager import SessionManager


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1 – Security Check
# ─────────────────────────────────────────────────────────────────────────────

async def security_check_node(
    state: ClaimAgentState,
    *,
    redis_client: aioredis.Redis,
) -> dict[str, Any]:
    """Rate limiting + input sanitisation."""
    rate_limiter = RateLimiter(redis_client)
    ctx = state.get("security_context", {})
    ip = ctx.get("ip_address", "unknown")
    raw_query = state.get("raw_query", "")

    # ── Rate limiting (IP tier first, user tier later) ────────────────────────
    ip_result = await rate_limiter.check(ip, tier="ip")
    if not ip_result.allowed:
        AuditLogger.log_rate_limit(identifier=ip, tier="ip", ip_address=ip)
        return {
            "security_passed": False,
            "error_code": "RATE_LIMITED",
            "error_message": f"Too many requests. Retry after {ip_result.retry_after}s.",
            "security_context": {**ctx, "rate_limit_remaining": 0},
            "current_node": "security_check",
        }

    # ── Input sanitisation ────────────────────────────────────────────────────
    san_result = InputSanitizer.sanitize_user_query(raw_query)
    if not san_result.is_safe:
        AuditLogger.log_security_event(
            event_type=san_result.threat_type or "MALICIOUS_INPUT",
            ip_address=ip,
            details=san_result.detail,
            severity="HIGH",
        )
        return {
            "security_passed": False,
            "error_code": "INVALID_INPUT",
            "error_message": "Your query contains disallowed content.",
            "security_context": {
                **ctx,
                "threat_detected": True,
                "threat_type": san_result.threat_type,
            },
            "current_node": "security_check",
        }

    return {
        "security_passed": True,
        "sanitized_query": san_result.cleaned_value,
        "security_context": {
            **ctx,
            "rate_limit_remaining": ip_result.remaining,
            "threat_detected": False,
        },
        "current_node": "security_check",
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2 – Authentication
# ─────────────────────────────────────────────────────────────────────────────

async def auth_node(
    state: ClaimAgentState,
    *,
    redis_client: aioredis.Redis,
) -> dict[str, Any]:
    """Validate JWT and populate auth context."""
    token = state.get("raw_token", "")
    if not token:
        return {
            "is_authenticated": False,
            "error_code": "MISSING_TOKEN",
            "error_message": "Authorization token is required.",
            "current_node": "auth_node",
        }

    agent = AuthAgent(redis_client=redis_client)
    result = await agent.run(token=token)

    if result.get("valid"):
        user_id = result["user_id"]

        # Per-user rate limit (second tier)
        rate_limiter = RateLimiter(redis_client)
        user_rl = await rate_limiter.check(user_id, tier="user")
        if not user_rl.allowed:
            AuditLogger.log_rate_limit(
                identifier=user_id,
                tier="user",
                ip_address=state.get("security_context", {}).get("ip_address"),
            )
            return {
                "is_authenticated": False,
                "error_code": "RATE_LIMITED",
                "error_message": f"Rate limit exceeded for your account. Retry after {user_rl.retry_after}s.",
                "current_node": "auth_node",
            }

        AuditLogger.log_token_validation(
            user_id=user_id, success=True, jti=result.get("jti")
        )
        return {
            "is_authenticated": True,
            "auth_context": {
                "user_id": user_id,
                "username": result.get("username"),
                "email": result.get("email"),
                "jti": result.get("jti"),
            },
            "current_node": "auth_node",
        }
    else:
        AuditLogger.log_token_validation(
            user_id=None,
            success=False,
            reason=result.get("error"),
        )
        return {
            "is_authenticated": False,
            "error_code": result.get("error", "AUTH_FAILED"),
            "error_message": result.get("message", "Authentication failed."),
            "current_node": "auth_node",
        }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3 – Claim Query Processing
# ─────────────────────────────────────────────────────────────────────────────

async def query_node(
    state: ClaimAgentState,
    *,
    db: AsyncSession,
    session_manager: SessionManager,
) -> dict[str, Any]:
    """Run NLP→Intent + ClaimQueryAgent to answer the user's question."""
    auth = state.get("auth_context", {})
    user_id = auth.get("user_id", "")
    session_id = state.get("session_id", "")
    query = state.get("sanitized_query", "")

    # Load conversation history from session
    history = await session_manager.get_conversation_history(session_id)

    agent = ClaimQueryAgent(db=db)
    result = await agent.run(
        query=query,
        user_id=user_id,
        session_id=session_id,
        conversation_history=history,
    )

    response_text = result.get("response", "I was unable to process your request.")

    # Persist message to session
    session = await session_manager.get(session_id)
    if session:
        session.add_message("user", query)
        session.add_message("assistant", response_text)
        await session_manager.update(session)

    AuditLogger.log_claim_query(
        user_id=user_id,
        session_id=session_id,
        query=query,
        intent=result.get("intent"),
        success=result.get("success", False),
    )

    return {
        "final_response": response_text,
        "intent_context": result.get("intent_details", {}),
        "messages": [
            HumanMessage(content=query),
            AIMessage(content=response_text),
        ],
        "current_node": "query_node",
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4 – Error Response
# ─────────────────────────────────────────────────────────────────────────────

async def error_node(state: ClaimAgentState) -> dict[str, Any]:
    """Format a consistent error response."""
    code = state.get("error_code", "UNKNOWN_ERROR")
    message = state.get("error_message", "An unexpected error occurred.")

    error_responses = {
        "RATE_LIMITED": "You have exceeded the request limit. Please wait before trying again.",
        "INVALID_INPUT": "Your query could not be processed due to security policy.",
        "MISSING_TOKEN": "Please provide an Authorization header with a valid Bearer token.",
        "TOKEN_EXPIRED": "Your session has expired. Please log in again.",
        "TOKEN_REVOKED": "Your token has been revoked. Please log in again.",
        "AUTH_FAILED": "Authentication failed. Please check your credentials.",
    }

    friendly = error_responses.get(code, message)
    return {
        "final_response": friendly,
        "messages": [AIMessage(content=friendly)],
        "current_node": "error_node",
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 5 – Audit Log
# ─────────────────────────────────────────────────────────────────────────────

async def audit_log_node(
    state: ClaimAgentState,
    *,
    db: AsyncSession,
) -> dict[str, Any]:
    """Persist audit trail to the database (fire-and-forget pattern)."""
    from src.database.repository import AuditRepository

    auth = state.get("auth_context", {})
    ctx = state.get("security_context", {})
    error_code = state.get("error_code")
    action = "CLAIM_QUERY" if not error_code else f"BLOCKED_{error_code}"

    audit_repo = AuditRepository(db)
    await audit_repo.create_log(
        action=action,
        resource="claims",
        user_id=auth.get("user_id"),
        details={
            "session_id": state.get("session_id"),
            "intent": state.get("intent_context", {}).get("intent"),
            "error": error_code,
            "node": state.get("current_node"),
        },
        ip_address=ctx.get("ip_address"),
        user_agent=ctx.get("user_agent"),
        status_code=200 if not error_code else (429 if "RATE" in (error_code or "") else 401),
    )

    trail_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "node": state.get("current_node"),
    }
    existing = list(state.get("audit_trail", []))
    existing.append(trail_entry)

    return {
        "audit_trail": existing,
        "current_node": "audit_log_node",
    }
