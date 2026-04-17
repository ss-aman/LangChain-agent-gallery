"""Sliding-window rate limiter backed by Redis.

Two tiers:
  • per-user  – generous limit for authenticated callers
  • per-IP    – strict limit to deter anonymous flooding
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import redis.asyncio as aioredis

from src.config import get_settings

settings = get_settings()

# Lua script for atomic sliding-window increment (avoids race conditions)
_SLIDING_WINDOW_SCRIPT = """
local key     = KEYS[1]
local now     = tonumber(ARGV[1])
local window  = tonumber(ARGV[2])
local limit   = tonumber(ARGV[3])

-- Remove entries older than the window
redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)

local count = redis.call('ZCARD', key)
if count < limit then
    redis.call('ZADD', key, now, now .. ':' .. math.random(1, 1000000))
    redis.call('EXPIRE', key, window)
    return {1, limit - count - 1}   -- allowed, remaining
else
    return {0, 0}                   -- denied, 0 remaining
end
"""


@dataclass
class RateLimitResult:
    allowed: bool
    remaining: int
    retry_after: Optional[int] = None  # seconds until reset


class RateLimiter:
    def __init__(self, redis_client: aioredis.Redis) -> None:
        self._redis = redis_client
        self._script = self._redis.register_script(_SLIDING_WINDOW_SCRIPT)

    async def check(
        self,
        identifier: str,
        *,
        tier: str = "user",
        window_seconds: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> RateLimitResult:
        """
        identifier: user_id or IP address
        tier: "user" | "ip"
        """
        if window_seconds is None:
            window_seconds = 60
        if limit is None:
            limit = (
                settings.RATE_LIMIT_REQUESTS_PER_MINUTE
                if tier == "user"
                else settings.RATE_LIMIT_BURST
            )

        key = f"rl:{tier}:{identifier}"
        now_ms = int(time.time() * 1000)

        result = await self._script(
            keys=[key],
            args=[now_ms, window_seconds * 1000, limit],
        )

        allowed, remaining = bool(result[0]), int(result[1])
        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            retry_after=window_seconds if not allowed else None,
        )

    async def reset(self, identifier: str, tier: str = "user") -> None:
        await self._redis.delete(f"rl:{tier}:{identifier}")
