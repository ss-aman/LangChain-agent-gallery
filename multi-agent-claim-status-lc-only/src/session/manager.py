"""Redis-backed session manager.

Stores per-session conversation context so the graph can be resumed
across HTTP requests without holding in-process state (enables
horizontal scaling).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import redis.asyncio as aioredis
from pydantic import BaseModel, Field

from src.config import get_settings

settings = get_settings()

_SESSION_PREFIX = "session:"
_CONV_HISTORY_KEY = "messages"


class SessionData(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    username: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_active: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    messages: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        # Keep only the last 20 messages to bound memory usage
        if len(self.messages) > 20:
            self.messages = self.messages[-20:]
        self.last_active = datetime.now(timezone.utc).isoformat()


class SessionManager:
    def __init__(self, redis_client: aioredis.Redis) -> None:
        self._redis = redis_client
        self._ttl = settings.SESSION_TTL_SECONDS

    def _key(self, session_id: str) -> str:
        return f"{_SESSION_PREFIX}{session_id}"

    async def create(self, user_id: str, username: str) -> SessionData:
        session = SessionData(user_id=user_id, username=username)
        await self._save(session)
        return session

    async def get(self, session_id: str) -> Optional[SessionData]:
        raw = await self._redis.get(self._key(session_id))
        if not raw:
            return None
        data = json.loads(raw)
        return SessionData(**data)

    async def update(self, session: SessionData) -> None:
        session.last_active = datetime.now(timezone.utc).isoformat()
        await self._save(session)

    async def delete(self, session_id: str) -> None:
        await self._redis.delete(self._key(session_id))

    async def extend_ttl(self, session_id: str) -> None:
        await self._redis.expire(self._key(session_id), self._ttl)

    async def _save(self, session: SessionData) -> None:
        await self._redis.setex(
            self._key(session.session_id),
            self._ttl,
            session.model_dump_json(),
        )

    async def get_conversation_history(
        self, session_id: str
    ) -> list[dict[str, Any]]:
        session = await self.get(session_id)
        return session.messages if session else []
