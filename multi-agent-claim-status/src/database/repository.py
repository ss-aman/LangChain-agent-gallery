"""Repository pattern – all DB access goes through here (no raw SQL elsewhere)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from sqlalchemy import select, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.database.models import AuditLog, Claim, ClaimStatus, ClaimStatusEnum, User


class UserRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get_by_id(self, user_id: str) -> Optional[User]:
        result = await self._db.execute(
            select(User).where(User.id == user_id, User.is_active == True)
        )
        return result.scalar_one_or_none()

    async def get_by_username(self, username: str) -> Optional[User]:
        result = await self._db.execute(
            select(User).where(User.username == username, User.is_active == True)
        )
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> Optional[User]:
        result = await self._db.execute(
            select(User).where(User.email == email, User.is_active == True)
        )
        return result.scalar_one_or_none()

    async def create(self, **kwargs: Any) -> User:
        user = User(**kwargs)
        self._db.add(user)
        await self._db.flush()
        await self._db.refresh(user)
        return user


class ClaimRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # ── Queries (read-only, parameterised) ─────────────────────────────────────

    async def get_claims_by_user(
        self,
        user_id: str,
        *,
        limit: int = 20,
        offset: int = 0,
        claim_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Sequence[Claim]:
        filters = [Claim.user_id == user_id]
        if claim_type:
            filters.append(Claim.claim_type == claim_type)
        if start_date:
            filters.append(Claim.claim_date >= start_date)
        if end_date:
            filters.append(Claim.claim_date <= end_date)

        result = await self._db.execute(
            select(Claim)
            .options(selectinload(Claim.status_history))
            .where(and_(*filters))
            .order_by(desc(Claim.claim_date))
            .limit(min(limit, 100))   # cap at 100 to prevent abuse
            .offset(offset)
        )
        return result.scalars().all()

    async def get_claim_by_number(
        self, claim_number: str, user_id: str
    ) -> Optional[Claim]:
        """Always enforces user_id to prevent IDOR."""
        result = await self._db.execute(
            select(Claim)
            .options(selectinload(Claim.status_history))
            .where(
                Claim.claim_number == claim_number,
                Claim.user_id == user_id,   # IDOR guard
            )
        )
        return result.scalar_one_or_none()

    async def get_claims_by_status(
        self, user_id: str, status: ClaimStatusEnum, *, limit: int = 20
    ) -> Sequence[Claim]:
        result = await self._db.execute(
            select(Claim)
            .join(ClaimStatus, Claim.id == ClaimStatus.claim_id)
            .options(selectinload(Claim.status_history))
            .where(
                Claim.user_id == user_id,
                ClaimStatus.status == status,
            )
            .order_by(desc(ClaimStatus.updated_at))
            .limit(min(limit, 100))
        )
        return result.scalars().unique().all()

    async def get_status_history(
        self, claim_id: str, user_id: str
    ) -> Sequence[ClaimStatus]:
        """Verifies ownership before returning history."""
        claim = await self._db.execute(
            select(Claim).where(Claim.id == claim_id, Claim.user_id == user_id)
        )
        if not claim.scalar_one_or_none():
            return []

        result = await self._db.execute(
            select(ClaimStatus)
            .where(ClaimStatus.claim_id == claim_id)
            .order_by(desc(ClaimStatus.updated_at))
        )
        return result.scalars().all()


class AuditRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create_log(
        self,
        *,
        action: str,
        resource: str,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        status_code: Optional[int] = None,
    ) -> AuditLog:
        log = AuditLog(
            user_id=user_id,
            action=action,
            resource=resource,
            resource_id=resource_id,
            details=json.dumps(details) if details else None,
            ip_address=ip_address,
            user_agent=user_agent,
            status_code=status_code,
        )
        self._db.add(log)
        await self._db.flush()
        return log
