"""SQLAlchemy async ORM models for the claims domain."""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ── Enums ──────────────────────────────────────────────────────────────────────

class ClaimStatusEnum(str, PyEnum):
    SUBMITTED = "SUBMITTED"
    UNDER_REVIEW = "UNDER_REVIEW"
    ADDITIONAL_INFO_REQUIRED = "ADDITIONAL_INFO_REQUIRED"
    APPROVED = "APPROVED"
    PARTIALLY_APPROVED = "PARTIALLY_APPROVED"
    DENIED = "DENIED"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


class ClaimTypeEnum(str, PyEnum):
    MEDICAL = "MEDICAL"
    AUTO = "AUTO"
    HOME = "HOME"
    LIFE = "LIFE"
    DISABILITY = "DISABILITY"
    TRAVEL = "TRAVEL"


# ── Models ─────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    policy_holder_id: Mapped[str] = mapped_column(String(64), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    claims: Mapped[list[Claim]] = relationship("Claim", back_populates="user", lazy="select")
    audit_logs: Mapped[list[AuditLog]] = relationship("AuditLog", back_populates="user")

    __table_args__ = (
        Index("idx_users_username", "username"),
        Index("idx_users_email", "email"),
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username}>"


class Claim(Base):
    __tablename__ = "claims"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    claim_number: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    policy_number: Mapped[str] = mapped_column(String(64), nullable=False)
    claim_type: Mapped[ClaimTypeEnum] = mapped_column(
        Enum(ClaimTypeEnum), nullable=False
    )
    incident_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    claim_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    claimed_amount: Mapped[float] = mapped_column(Float, nullable=False)
    approved_amount: Mapped[float] = mapped_column(Float, nullable=True)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user: Mapped[User] = relationship("User", back_populates="claims")
    status_history: Mapped[list[ClaimStatus]] = relationship(
        "ClaimStatus", back_populates="claim", order_by="ClaimStatus.updated_at.desc()"
    )

    __table_args__ = (
        Index("idx_claims_user_id", "user_id"),
        Index("idx_claims_claim_number", "claim_number"),
        Index("idx_claims_policy_number", "policy_number"),
        Index("idx_claims_claim_date", "claim_date"),
    )

    @property
    def current_status(self) -> ClaimStatus | None:
        return self.status_history[0] if self.status_history else None

    def __repr__(self) -> str:
        return f"<Claim id={self.id} number={self.claim_number}>"


class ClaimStatus(Base):
    __tablename__ = "claim_status"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    claim_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("claims.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[ClaimStatusEnum] = mapped_column(Enum(ClaimStatusEnum), nullable=False)
    notes: Mapped[str] = mapped_column(Text, nullable=True)
    denial_reason: Mapped[str] = mapped_column(Text, nullable=True)
    estimated_completion_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_by: Mapped[str] = mapped_column(String(255), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    claim: Mapped[Claim] = relationship("Claim", back_populates="status_history")

    __table_args__ = (
        Index("idx_claim_status_claim_id", "claim_id"),
        Index("idx_claim_status_status", "status"),
        Index("idx_claim_status_updated_at", "updated_at"),
    )

    def __repr__(self) -> str:
        return f"<ClaimStatus claim_id={self.claim_id} status={self.status}>"


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    resource: Mapped[str] = mapped_column(String(128), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(36), nullable=True)
    details: Mapped[str] = mapped_column(Text, nullable=True)
    ip_address: Mapped[str] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[str] = mapped_column(String(512), nullable=True)
    status_code: Mapped[int] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    user: Mapped[User] = relationship("User", back_populates="audit_logs")

    __table_args__ = (
        Index("idx_audit_logs_user_id", "user_id"),
        Index("idx_audit_logs_action", "action"),
        Index("idx_audit_logs_created_at", "created_at"),
    )
