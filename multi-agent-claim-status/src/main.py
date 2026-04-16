"""FastAPI application – entry point for the multi-agent claim status system.

Architecture highlights:
- Lifespan manages DB/Redis/session singletons (one per process)
- Each request gets a fresh AsyncSession (safe for concurrent queries)
- The LangGraph workflow is built per request with the live session
- Horizontal scaling: share nothing beyond Redis + DB
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

import redis.asyncio as aioredis
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.database.connection import AsyncSessionFactory, get_db, init_db
from src.database.repository import UserRepository
from src.graph.workflow import build_claim_workflow, invoke_workflow
from src.security.auth import JWTManager, PasswordManager
from src.session.manager import SessionManager

settings = get_settings()

# ── Application-level singletons ──────────────────────────────────────────────
_redis_client: Optional[aioredis.Redis] = None
_session_manager: Optional[SessionManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _redis_client, _session_manager

    # Initialise DB schema (idempotent)
    await init_db()

    # Redis connection pool
    _redis_client = aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
        max_connections=20,
    )
    _session_manager = SessionManager(_redis_client)

    yield

    # Teardown
    if _redis_client:
        await _redis_client.aclose()


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Agent Claim Status API",
    description="Insurance claim status queries powered by LangGraph multi-agent system.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Session-ID"],
)


# ── Dependencies ──────────────────────────────────────────────────────────────

def get_redis() -> aioredis.Redis:
    if _redis_client is None:
        raise RuntimeError("Redis client not initialised")
    return _redis_client


def get_session_manager() -> SessionManager:
    if _session_manager is None:
        raise RuntimeError("Session manager not initialised")
    return _session_manager


def extract_bearer_token(authorization: str = Header(default="")) -> str:
    """Extract raw JWT from 'Bearer <token>' header."""
    if not authorization.startswith("Bearer "):
        return ""
    return authorization[7:].strip()


# ── Request / Response schemas ────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class ClaimQueryRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=2000,
        description="Natural language question about your claims",
        examples=["What is the status of claim CLM000001?"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Existing session ID for conversation continuity",
    )


class ClaimQueryResponse(BaseModel):
    response: str
    session_id: str
    intent: Optional[str] = None
    is_authenticated: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check() -> dict[str, str]:
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/auth/token", response_model=TokenResponse, tags=["Authentication"])
async def login(
    body: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Issue JWT access + refresh tokens for valid credentials."""
    user_repo = UserRepository(db)
    user = await user_repo.get_by_username(body.username)

    if not user or not PasswordManager.verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access = JWTManager.create_access_token(user.id, user.username, user.email)
    refresh = JWTManager.create_refresh_token(user.id, user.username, user.email)

    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@app.post("/auth/refresh", response_model=TokenResponse, tags=["Authentication"])
async def refresh_token(
    authorization: str = Header(default=""),
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Exchange a valid refresh token for a new access token."""
    token = authorization.replace("Bearer ", "").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Refresh token required")

    try:
        payload = JWTManager.verify_token(token, expected_type="refresh")
    except Exception as exc:
        raise HTTPException(status_code=401, detail=str(exc))

    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(payload.sub)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    access = JWTManager.create_access_token(user.id, user.username, user.email)
    new_refresh = JWTManager.create_refresh_token(user.id, user.username, user.email)
    return TokenResponse(
        access_token=access,
        refresh_token=new_refresh,
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@app.post(
    "/claims/query",
    response_model=ClaimQueryResponse,
    tags=["Claims"],
    summary="Query your claim status using natural language",
)
async def query_claims(
    request: Request,
    body: ClaimQueryRequest,
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
    sm: SessionManager = Depends(get_session_manager),
    token: str = Depends(extract_bearer_token),
) -> ClaimQueryResponse:
    """
    Multi-agent pipeline:
    1. Security check (rate limit + sanitise input)
    2. Authentication (validate JWT)
    3. NLP → Intent classification
    4. Database query (parameterised, IDOR-safe)
    5. Response formatting + session persistence
    6. Audit logging
    """
    # Resolve or create session
    session_id = body.session_id
    if session_id:
        existing = await sm.get(session_id)
        if not existing:
            session_id = None   # stale – will create new after auth

    if not session_id:
        session_id = str(uuid.uuid4())   # will be populated post-auth

    # Build the graph for this request
    graph = build_claim_workflow(db=db, redis_client=redis, session_manager=sm)

    # Execute
    result = await invoke_workflow(
        graph,
        raw_query=body.query,
        raw_token=token,
        session_id=session_id,
        ip_address=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", ""),
    )

    # Map error codes to HTTP status
    error_code = result.get("error_code")
    if error_code == "RATE_LIMITED":
        raise HTTPException(status_code=429, detail=result["response"])
    if error_code == "INVALID_INPUT":
        raise HTTPException(status_code=400, detail=result["response"])
    if error_code in ("MISSING_TOKEN", "TOKEN_EXPIRED", "TOKEN_REVOKED", "AUTH_FAILED", "INVALID_TOKEN"):
        raise HTTPException(status_code=401, detail=result["response"])

    return ClaimQueryResponse(
        response=result["response"],
        session_id=result["session_id"],
        intent=result.get("intent"),
        is_authenticated=result.get("is_authenticated", False),
    )


# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    from src.security.audit_logger import AuditLogger
    AuditLogger.log_security_event(
        event_type="UNHANDLED_EXCEPTION",
        ip_address=request.client.host if request.client else None,
        details=str(exc)[:200],
        severity="HIGH",
    )
    return JSONResponse(
        status_code=500,
        content={"error": "INTERNAL_ERROR", "message": "An unexpected error occurred."},
    )
