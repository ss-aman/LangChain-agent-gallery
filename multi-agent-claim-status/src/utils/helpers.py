from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from src.database.models import Claim


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def build_error_response(code: str, message: str, detail: Optional[str] = None) -> dict[str, Any]:
    return {
        "error": code,
        "message": message,
        **({"detail": detail} if detail else {}),
    }


def format_claim_summary(claim: Claim) -> dict[str, Any]:
    current = claim.current_status
    return {
        "claim_number": claim.claim_number,
        "claim_type": claim.claim_type.value if claim.claim_type else None,
        "claim_date": claim.claim_date.strftime("%Y-%m-%d") if claim.claim_date else None,
        "incident_date": claim.incident_date.strftime("%Y-%m-%d") if claim.incident_date else None,
        "claimed_amount": claim.claimed_amount,
        "approved_amount": claim.approved_amount,
        "current_status": current.status.value if current else "UNKNOWN",
        "status_notes": current.notes if current else None,
        "denial_reason": current.denial_reason if current else None,
        "estimated_completion": (
            current.estimated_completion_date.strftime("%Y-%m-%d")
            if current and current.estimated_completion_date
            else None
        ),
        "last_updated": (
            current.updated_at.strftime("%Y-%m-%d %H:%M UTC")
            if current
            else None
        ),
    }
