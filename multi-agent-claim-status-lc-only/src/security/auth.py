"""JWT authentication + password hashing."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from src.config import get_settings

settings = get_settings()
_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


class TokenPayload(BaseModel):
    sub: str          # user_id
    username: str
    email: str
    jti: str          # unique token id (for revocation)
    exp: datetime
    iat: datetime
    token_type: str   # access | refresh


class JWTManager:
    """Stateless JWT issuer and verifier."""

    @staticmethod
    def create_access_token(
        user_id: str,
        username: str,
        email: str,
    ) -> str:
        now = datetime.now(timezone.utc)
        payload = {
            "sub": user_id,
            "username": username,
            "email": email,
            "jti": str(uuid.uuid4()),
            "iat": now,
            "exp": now + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
            "token_type": "access",
        }
        return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

    @staticmethod
    def create_refresh_token(user_id: str, username: str, email: str) -> str:
        now = datetime.now(timezone.utc)
        payload = {
            "sub": user_id,
            "username": username,
            "email": email,
            "jti": str(uuid.uuid4()),
            "iat": now,
            "exp": now + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS),
            "token_type": "refresh",
        }
        return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

    @staticmethod
    def verify_token(token: str, expected_type: str = "access") -> TokenPayload:
        """Raises jwt.PyJWTError on failure."""
        data = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            options={"require": ["exp", "iat", "sub", "jti"]},
        )
        if data.get("token_type") != expected_type:
            raise jwt.InvalidTokenError(
                f"Expected token_type={expected_type}, got {data.get('token_type')}"
            )
        return TokenPayload(**data)

    @staticmethod
    def decode_without_verification(token: str) -> dict:
        """For reading claims from expired tokens (e.g. refresh flow)."""
        return jwt.decode(
            token,
            options={"verify_signature": False, "verify_exp": False},
            algorithms=[settings.JWT_ALGORITHM],
        )


class PasswordManager:
    @staticmethod
    def hash_password(plain: str) -> str:
        return _pwd_ctx.hash(plain)

    @staticmethod
    def verify_password(plain: str, hashed: str) -> bool:
        return _pwd_ctx.verify(plain, hashed)
