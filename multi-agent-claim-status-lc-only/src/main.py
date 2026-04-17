"""FastAPI application for the LangChain-only claim status system.

The key difference from the LangGraph version:
  • No graph compilation step at startup
  • ClaimPipeline is built PER REQUEST (no graph-level checkpointer)
  • Streaming endpoint uses astream_events + SSE instead of graph streaming
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

import redis.asyncio as aioredis
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.chains.pipeline import ClaimPipeline
from src.config import get_settings
from src.database.connection import AsyncSessionFactory, get_db, init_db
from src.database.repository import UserRepository
from src.security.auth import JWTManager, PasswordManager
from src.session.manager import SessionManager

settings = get_settings()

_redis_client: Optional[aioredis.Redis] = None
_session_manager: Optional[SessionManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _redis_client, _session_manager
    await init_db()
    _redis_client = aioredis.from_url(
        settings.REDIS_URL, encoding="utf-8", decode_responses=True, max_connections=20
    )
    _session_manager = SessionManager(_redis_client)
    yield
    if _redis_client:
        await _redis_client.aclose()


app = FastAPI(
    title="Multi-Agent Claim Status API (LangChain-Only)",
    description="Insurance claims powered by LangChain LCEL pipelines — no LangGraph.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


# ── Dependencies ──────────────────────────────────────────────────────────────

def get_redis() -> aioredis.Redis:
    if _redis_client is None:
        raise RuntimeError("Redis not initialised")
    return _redis_client


def get_session_manager() -> SessionManager:
    if _session_manager is None:
        raise RuntimeError("Session manager not initialised")
    return _session_manager


def extract_token(authorization: str = Header(default="")) -> str:
    return authorization.removeprefix("Bearer ").strip()


# ── Schemas ────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class ClaimQueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    session_id: Optional[str] = None


class ClaimQueryResponse(BaseModel):
    response: str
    session_id: str
    intent: Optional[str] = None
    is_authenticated: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "mode": "langchain-only"}


@app.post("/auth/token", response_model=TokenResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)) -> TokenResponse:
    repo = UserRepository(db)
    user = await repo.get_by_username(body.username)
    if not user or not PasswordManager.verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return TokenResponse(
        access_token=JWTManager.create_access_token(user.id, user.username, user.email),
        refresh_token=JWTManager.create_refresh_token(user.id, user.username, user.email),
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@app.post("/claims/query", response_model=ClaimQueryResponse)
async def query_claims(
    request: Request,
    body: ClaimQueryRequest,
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
    sm: SessionManager = Depends(get_session_manager),
    token: str = Depends(extract_token),
) -> ClaimQueryResponse:
    """
    Full LangChain pipeline:
    SecurityChain | RunnableBranch(AuthChain | RunnableBranch(IntentChain | AgentExecutor))
    """
    session_id = body.session_id or str(uuid.uuid4())

    # Build pipeline — per request, no shared state
    # CHALLENGE: Building the pipeline per request is safe but slightly wasteful.
    # A pooled approach would reuse the pipeline object but requires careful
    # async session management since AsyncSession is not thread/task-safe.
    pipeline = ClaimPipeline(db=db, redis_client=redis, session_manager=sm)

    result = await pipeline.run(
        raw_query=body.query,
        raw_token=token,
        session_id=session_id,
        ip_address=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", ""),
    )

    error = result.get("error_code")
    if error == "RATE_LIMITED":
        raise HTTPException(status_code=429, detail=result["response"])
    if error == "INVALID_INPUT":
        raise HTTPException(status_code=400, detail=result["response"])
    if error in ("MISSING_TOKEN", "TOKEN_EXPIRED", "TOKEN_REVOKED", "INVALID_TOKEN", "AUTH_FAILED"):
        raise HTTPException(status_code=401, detail=result["response"])

    return ClaimQueryResponse(
        response=result["response"],
        session_id=result["session_id"],
        intent=result.get("intent"),
        is_authenticated=result.get("is_authenticated", False),
    )


@app.post("/claims/query/stream")
async def stream_query(
    request: Request,
    body: ClaimQueryRequest,
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
    sm: SessionManager = Depends(get_session_manager),
    token: str = Depends(extract_token),
) -> StreamingResponse:
    """
    Server-Sent Events stream of LLM tokens.

    CAPABILITY: LangChain's astream_events gives token-level streaming from
    any chain, including nested AgentExecutor, without changing agent code.

    CHALLENGE: The pipeline must pass the security + auth checks BEFORE any
    streaming starts. If auth fails mid-stream, the SSE connection is already
    open. We handle this with a pre-flight check.
    """
    session_id = body.session_id or str(uuid.uuid4())
    pipeline = ClaimPipeline(db=db, redis_client=redis, session_manager=sm)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async for token_chunk in pipeline.stream(
                raw_query=body.query,
                raw_token=token,
                session_id=session_id,
                ip_address=request.client.host if request.client else "unknown",
            ):
                yield f"data: {token_chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            yield f"data: [ERROR] {str(exc)[:100]}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.exception_handler(Exception)
async def unhandled_exc(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": "INTERNAL_ERROR", "message": "An unexpected error occurred."},
    )
