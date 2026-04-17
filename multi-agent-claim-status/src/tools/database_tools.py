"""LangChain tools for safe, parameterised database access.

SECURITY: The LLM never writes raw SQL. It selects a pre-approved intent
template and provides typed parameters, which are bound via SQLAlchemy's
parameterised query mechanism. This eliminates SQL-injection at the call site.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import ClaimStatusEnum
from src.database.repository import ClaimRepository
from src.utils.helpers import format_claim_summary


# ── Input schemas ──────────────────────────────────────────────────────────────

class GetClaimsByUserInput(BaseModel):
    user_id: str = Field(description="Authenticated user's ID")
    limit: int = Field(default=10, ge=1, le=50, description="Max rows to return (1-50)")
    offset: int = Field(default=0, ge=0)
    claim_type: Optional[str] = Field(default=None, description="Filter by claim type")
    start_date: Optional[str] = Field(default=None, description="Start date YYYY-MM-DD")
    end_date: Optional[str] = Field(default=None, description="End date YYYY-MM-DD")


class GetClaimByNumberInput(BaseModel):
    claim_number: str = Field(description="Claim number e.g. CLM000001")
    user_id: str = Field(description="Authenticated user's ID (ownership check)")


class GetClaimsByStatusInput(BaseModel):
    user_id: str = Field(description="Authenticated user's ID")
    status: str = Field(description="Status: SUBMITTED|UNDER_REVIEW|APPROVED|DENIED|CLOSED")
    limit: int = Field(default=10, ge=1, le=50)


class GetStatusHistoryInput(BaseModel):
    claim_id: str = Field(description="Internal claim UUID")
    user_id: str = Field(description="Authenticated user's ID")


# ── Tools ──────────────────────────────────────────────────────────────────────

class GetClaimsByUserTool(BaseTool):
    name: str = "get_claims_by_user"
    description: str = (
        "Fetches all claims belonging to an authenticated user. "
        "Supports optional filtering by claim_type and date range."
    )
    args_schema: type[BaseModel] = GetClaimsByUserInput
    db: AsyncSession = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, **kwargs: Any) -> Any:
        raise NotImplementedError("Use async")

    async def _arun(
        self,
        user_id: str,
        limit: int = 10,
        offset: int = 0,
        claim_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        repo = ClaimRepository(self.db)
        start = datetime.fromisoformat(start_date) if start_date else None
        end = datetime.fromisoformat(end_date) if end_date else None
        claims = await repo.get_claims_by_user(
            user_id,
            limit=limit,
            offset=offset,
            claim_type=claim_type,
            start_date=start,
            end_date=end,
        )
        return [format_claim_summary(c) for c in claims]


class GetClaimByNumberTool(BaseTool):
    name: str = "get_claim_by_number"
    description: str = (
        "Fetches full details for a specific claim by its claim number. "
        "Enforces ownership: returns nothing if the claim doesn't belong to user_id."
    )
    args_schema: type[BaseModel] = GetClaimByNumberInput
    db: AsyncSession = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, **kwargs: Any) -> Any:
        raise NotImplementedError("Use async")

    async def _arun(self, claim_number: str, user_id: str) -> dict[str, Any]:
        repo = ClaimRepository(self.db)
        claim = await repo.get_claim_by_number(claim_number, user_id)
        if not claim:
            return {"found": False, "message": f"Claim {claim_number} not found or access denied"}
        return {"found": True, **format_claim_summary(claim)}


class GetClaimsByStatusTool(BaseTool):
    name: str = "get_claims_by_status"
    description: str = (
        "Fetches user's claims filtered by a specific status. "
        "Valid statuses: SUBMITTED, UNDER_REVIEW, APPROVED, DENIED, CLOSED, CANCELLED."
    )
    args_schema: type[BaseModel] = GetClaimsByStatusInput
    db: AsyncSession = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, **kwargs: Any) -> Any:
        raise NotImplementedError("Use async")

    async def _arun(self, user_id: str, status: str, limit: int = 10) -> list[dict[str, Any]]:
        try:
            status_enum = ClaimStatusEnum(status.upper())
        except ValueError:
            return [{"error": f"Unknown status '{status}'"}]
        repo = ClaimRepository(self.db)
        claims = await repo.get_claims_by_status(user_id, status_enum, limit=limit)
        return [format_claim_summary(c) for c in claims]


class GetStatusHistoryTool(BaseTool):
    name: str = "get_claim_status_history"
    description: str = (
        "Returns the full audit trail of status changes for a specific claim. "
        "Shows when status changed, who changed it, and any notes/denial reasons."
    )
    args_schema: type[BaseModel] = GetStatusHistoryInput
    db: AsyncSession = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, **kwargs: Any) -> Any:
        raise NotImplementedError("Use async")

    async def _arun(self, claim_id: str, user_id: str) -> list[dict[str, Any]]:
        repo = ClaimRepository(self.db)
        history = await repo.get_status_history(claim_id, user_id)
        return [
            {
                "status": h.status.value,
                "notes": h.notes,
                "denial_reason": h.denial_reason,
                "updated_by": h.updated_by,
                "updated_at": h.updated_at.strftime("%Y-%m-%d %H:%M UTC"),
                "estimated_completion": (
                    h.estimated_completion_date.strftime("%Y-%m-%d")
                    if h.estimated_completion_date else None
                ),
            }
            for h in history
        ]


def get_database_tools(db: AsyncSession) -> list[BaseTool]:
    return [
        GetClaimsByUserTool(db=db),
        GetClaimByNumberTool(db=db),
        GetClaimsByStatusTool(db=db),
        GetStatusHistoryTool(db=db),
    ]
