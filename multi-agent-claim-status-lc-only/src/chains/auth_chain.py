"""Authentication as a LangChain RunnableLambda.

CAPABILITY: Demonstrated patterns:
  • RunnableLambda for async business logic
  • .with_retry() for transient failures (e.g. Redis blip)
  • Returning structured error payload inside the context (not raising)

CHALLENGE: LangChain has NO concept of "authenticated context" – there is no
middleware layer. Every chain that needs auth must receive the auth_ctx from
upstream via the shared PipelineContext dict. If you forget to thread it
through, you silently get empty auth data — no compiler warning.
"""
from __future__ import annotations

from datetime import datetime, timezone

import redis.asyncio as aioredis
from langchain_core.runnables import RunnableLambda
from tenacity import RetryError

from src.chains.context import PipelineContext
from src.security.audit_logger import AuditLogger
from src.security.auth import JWTManager
from src.security.rate_limiter import RateLimiter

import jwt as pyjwt


def _make_auth_fn(redis_client: aioredis.Redis):
    async def _auth_check(ctx: PipelineContext) -> PipelineContext:
        token = ctx.get("raw_token", "").strip()
        if not token:
            return {
                **ctx,
                "is_authenticated": False,
                "error_code": "MISSING_TOKEN",
                "error_message": "Authorization Bearer token is required.",
                "current_step": "auth_check",
            }

        # ── Redis token blacklist ──────────────────────────────────────────────
        revoked = await redis_client.get(f"revoked_token:{token[-16:]}")
        if revoked:
            AuditLogger.log_token_validation(user_id=None, success=False, reason="TOKEN_REVOKED")
            return {
                **ctx,
                "is_authenticated": False,
                "error_code": "TOKEN_REVOKED",
                "error_message": "Token has been revoked. Please log in again.",
                "current_step": "auth_check",
            }

        # ── JWT verification ───────────────────────────────────────────────────
        try:
            payload = JWTManager.verify_token(token, expected_type="access")
        except pyjwt.ExpiredSignatureError:
            AuditLogger.log_token_validation(user_id=None, success=False, reason="TOKEN_EXPIRED")
            return {**ctx, "is_authenticated": False, "error_code": "TOKEN_EXPIRED",
                    "error_message": "Your session has expired. Please log in again."}
        except pyjwt.PyJWTError as exc:
            AuditLogger.log_token_validation(user_id=None, success=False, reason=str(exc))
            return {**ctx, "is_authenticated": False, "error_code": "INVALID_TOKEN",
                    "error_message": "Invalid token."}

        # ── Per-user rate limit (second tier) ──────────────────────────────────
        rate_limiter = RateLimiter(redis_client)
        user_rl = await rate_limiter.check(payload.sub, tier="user")
        if not user_rl.allowed:
            AuditLogger.log_rate_limit(
                identifier=payload.sub,
                tier="user",
                ip_address=ctx.get("ip_address"),
            )
            return {
                **ctx,
                "is_authenticated": False,
                "error_code": "RATE_LIMITED",
                "error_message": f"Account rate limit exceeded. Retry after {user_rl.retry_after}s.",
            }

        AuditLogger.log_token_validation(user_id=payload.sub, success=True, jti=payload.jti)

        trail = list(ctx.get("audit_trail", []))
        trail.append({
            "step": "auth_check",
            "user_id": payload.sub,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        return {
            **ctx,
            "is_authenticated": True,
            "auth_ctx": {
                "user_id": payload.sub,
                "username": payload.username,
                "email": payload.email,
                "jti": payload.jti,
            },
            "audit_trail": trail,
            "current_step": "auth_check",
        }

    return _auth_check


def make_auth_chain(redis_client: aioredis.Redis) -> RunnableLambda:
    """
    Auth chain with automatic retry on transient Redis errors.

    CAPABILITY: .with_retry() wraps ANY Runnable with configurable backoff.
    CHALLENGE:  .with_retry() retries the whole lambda including JWT decode –
                only useful for the Redis blacklist check part. You cannot
                selectively retry sub-steps without breaking the lambda apart.
    """
    base = RunnableLambda(_make_auth_fn(redis_client))
    return base.with_retry(
        retry_if_exception_type=(ConnectionError, TimeoutError),
        wait_exponential_jitter=True,
        stop_after_attempt=3,
    )
