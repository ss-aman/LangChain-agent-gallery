"""LangChain tools used by the Authentication Agent."""
from __future__ import annotations

from typing import Any, Optional

import jwt as pyjwt
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from src.security.auth import JWTManager


# ── Input schemas ──────────────────────────────────────────────────────────────

class ValidateTokenInput(BaseModel):
    token: str = Field(description="JWT access token to validate")


class DecodeTokenInput(BaseModel):
    token: str = Field(description="JWT token to decode (may be expired)")


# ── Tools ──────────────────────────────────────────────────────────────────────

class ValidateJWTTool(BaseTool):
    name: str = "validate_jwt_token"
    description: str = (
        "Validates a JWT access token. Returns user information on success, "
        "or an error message on failure (expired, invalid signature, wrong type)."
    )
    args_schema: type[BaseModel] = ValidateTokenInput

    def _run(self, token: str) -> dict[str, Any]:
        raise NotImplementedError("Use async version")

    async def _arun(self, token: str) -> dict[str, Any]:
        try:
            payload = JWTManager.verify_token(token, expected_type="access")
            return {
                "valid": True,
                "user_id": payload.sub,
                "username": payload.username,
                "email": payload.email,
                "jti": payload.jti,
                "expires_at": payload.exp.isoformat(),
            }
        except pyjwt.ExpiredSignatureError:
            return {"valid": False, "error": "TOKEN_EXPIRED", "message": "Token has expired"}
        except pyjwt.InvalidTokenError as exc:
            return {"valid": False, "error": "INVALID_TOKEN", "message": str(exc)}


class DecodeTokenTool(BaseTool):
    name: str = "decode_token_claims"
    description: str = (
        "Decodes JWT token claims WITHOUT verifying the signature or expiry. "
        "Useful for inspecting who the token was issued to during refresh flows."
    )
    args_schema: type[BaseModel] = DecodeTokenInput

    def _run(self, token: str) -> dict[str, Any]:
        raise NotImplementedError("Use async version")

    async def _arun(self, token: str) -> dict[str, Any]:
        try:
            claims = JWTManager.decode_without_verification(token)
            return {
                "sub": claims.get("sub"),
                "username": claims.get("username"),
                "email": claims.get("email"),
                "token_type": claims.get("token_type"),
            }
        except Exception as exc:
            return {"error": str(exc)}


def get_auth_tools() -> list[BaseTool]:
    return [ValidateJWTTool(), DecodeTokenTool()]
