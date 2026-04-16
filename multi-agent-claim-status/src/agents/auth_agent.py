"""Authentication Agent.

Responsibilities:
- Validate JWT access tokens via tool calls
- Extract and return user identity context
- Flag suspicious tokens / revoked tokens via Redis blacklist
"""
from __future__ import annotations

from typing import Any

import redis.asyncio as aioredis
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from src.agents.base_agent import BaseClaimAgent
from src.tools.auth_tools import get_auth_tools


_SYSTEM_PROMPT = """You are the Authentication Agent for a secure insurance claims system.

Your ONLY job is to validate the provided JWT token and return structured authentication results.

Steps:
1. Call `validate_jwt_token` with the provided token.
2. If valid, return the user information.
3. If invalid, return the specific failure reason.

NEVER ask for passwords or sensitive data. NEVER skip token validation.
Always return a JSON object with keys: valid (bool), user_id, username, email, error (if invalid).
"""


class AuthAgent(BaseClaimAgent):
    def __init__(
        self,
        llm: BaseChatModel | None = None,
        redis_client: aioredis.Redis | None = None,
    ) -> None:
        super().__init__(llm)
        self._redis = redis_client
        self._agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=_SYSTEM_PROMPT,
        )

    @property
    def name(self) -> str:
        return "auth_agent"

    @property
    def tools(self) -> list[BaseTool]:
        return get_auth_tools()

    async def run(self, *, token: str, **_: Any) -> dict[str, Any]:
        """Validate token and optionally check Redis blacklist."""
        # Check blacklist first (zero LLM calls if revoked)
        if self._redis:
            revoked = await self._redis.get(f"revoked_token:{token[-16:]}")
            if revoked:
                return {
                    "valid": False,
                    "error": "TOKEN_REVOKED",
                    "message": "This token has been revoked",
                }

        messages = [HumanMessage(content=f"Validate this token: {token}")]
        result = await self._agent.ainvoke({"messages": messages})

        # Extract structured result from the last AI message
        last_message = result["messages"][-1]
        content = last_message.content if hasattr(last_message, "content") else str(last_message)

        # Parse JSON from response
        import json, re
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: direct tool call
        from src.security.auth import JWTManager
        import jwt
        try:
            payload = JWTManager.verify_token(token)
            return {
                "valid": True,
                "user_id": payload.sub,
                "username": payload.username,
                "email": payload.email,
                "jti": payload.jti,
            }
        except jwt.PyJWTError as exc:
            return {"valid": False, "error": "INVALID_TOKEN", "message": str(exc)}

    async def revoke_token(self, token: str, jti: str) -> None:
        """Add a token to the Redis blacklist until its natural expiry."""
        from src.config import get_settings
        settings = get_settings()
        if self._redis:
            ttl = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
            await self._redis.setex(f"revoked_token:{token[-16:]}", ttl, jti)
