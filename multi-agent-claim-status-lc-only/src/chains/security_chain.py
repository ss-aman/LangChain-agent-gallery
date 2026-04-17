"""Security check as a LangChain RunnableLambda.

CAPABILITY: RunnableLambda wraps any sync or async Python function as a
first-class LCEL component that can be chained with `|`, supports ainvoke,
astream, abatch, and receives callbacks automatically.

CHALLENGE: Unlike a LangGraph node, a RunnableLambda that fails raises an
exception that propagates up and terminates the whole pipeline immediately.
There is no per-node error handler; you must either:
  a) Catch exceptions inside the lambda and encode errors in the return value, OR
  b) Attach .with_fallbacks([fallback_chain]) on the whole segment.
Both are more manual than LangGraph's built-in error routing.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import redis.asyncio as aioredis
from langchain_core.runnables import RunnableLambda

from src.chains.context import PipelineContext
from src.security.audit_logger import AuditLogger
from src.security.rate_limiter import RateLimiter
from src.security.sanitizer import InputSanitizer


def _make_security_fn(redis_client: aioredis.Redis):
    """Closure so the lambda captures the Redis client without a class."""

    async def _security_check(ctx: PipelineContext) -> PipelineContext:
        ip = ctx.get("ip_address", "unknown")
        rate_limiter = RateLimiter(redis_client)

        # ── IP-tier rate limit ─────────────────────────────────────────────────
        ip_rl = await rate_limiter.check(ip, tier="ip")
        if not ip_rl.allowed:
            AuditLogger.log_rate_limit(identifier=ip, tier="ip", ip_address=ip)
            return {
                **ctx,
                "security_passed": False,
                "error_code": "RATE_LIMITED",
                "error_message": f"Too many requests from {ip}. Retry after {ip_rl.retry_after}s.",
                "current_step": "security_check",
            }

        # ── Input sanitisation ─────────────────────────────────────────────────
        raw = ctx.get("raw_query", "")
        san = InputSanitizer.sanitize_user_query(raw)
        if not san.is_safe:
            AuditLogger.log_security_event(
                event_type=san.threat_type or "MALICIOUS_INPUT",
                ip_address=ip,
                details=san.detail,
                severity="HIGH",
            )
            return {
                **ctx,
                "security_passed": False,
                "error_code": "INVALID_INPUT",
                "error_message": "Query rejected by security policy.",
                "security_ctx": {
                    **ctx.get("security_ctx", {}),
                    "threat_detected": True,
                    "threat_type": san.threat_type,
                },
                "current_step": "security_check",
            }

        trail = list(ctx.get("audit_trail", []))
        trail.append({
            "step": "security_check",
            "passed": True,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        return {
            **ctx,
            "security_passed": True,
            "sanitized_query": san.cleaned_value,
            "security_ctx": {
                **ctx.get("security_ctx", {}),
                "rate_limit_remaining": ip_rl.remaining,
                "threat_detected": False,
            },
            "audit_trail": trail,
            "current_step": "security_check",
        }

    return _security_check


def make_security_chain(redis_client: aioredis.Redis) -> RunnableLambda:
    """
    Returns a RunnableLambda that accepts PipelineContext and returns
    an enriched PipelineContext with security_passed set.

    This is composable with | just like any LCEL Runnable:
        pipeline = make_security_chain(redis) | next_step
    """
    return RunnableLambda(_make_security_fn(redis_client))
