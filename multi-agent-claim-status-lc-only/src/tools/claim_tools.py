"""Higher-level claim domain tools (wrappers that add business logic)."""
from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class SummarizeClaimInput(BaseModel):
    claim_data: dict = Field(description="Raw claim dict from database tools")
    include_history: bool = Field(default=False)


class ClaimSummaryTool(BaseTool):
    name: str = "summarize_claim"
    description: str = (
        "Generates a human-readable narrative summary of a claim, "
        "including current status, amounts, and next steps."
    )
    args_schema: type[BaseModel] = SummarizeClaimInput

    def _run(self, claim_data: dict, include_history: bool = False) -> str:
        raise NotImplementedError("Use async")

    async def _arun(self, claim_data: dict, include_history: bool = False) -> str:
        if not claim_data.get("found", True):
            return claim_data.get("message", "Claim not found.")

        status = claim_data.get("current_status", "UNKNOWN")
        number = claim_data.get("claim_number", "N/A")
        ctype = claim_data.get("claim_type", "")
        claimed = claim_data.get("claimed_amount", 0)
        approved = claim_data.get("approved_amount")
        denial = claim_data.get("denial_reason")
        notes = claim_data.get("status_notes")
        completion = claim_data.get("estimated_completion")
        last_updated = claim_data.get("last_updated")

        lines = [
            f"Claim {number} ({ctype}) — Status: **{status}**",
            f"Claimed Amount: ${claimed:,.2f}",
        ]
        if approved is not None:
            lines.append(f"Approved Amount: ${approved:,.2f}")
        if notes:
            lines.append(f"Notes: {notes}")
        if denial:
            lines.append(f"Denial Reason: {denial}")
        if completion:
            lines.append(f"Estimated Completion: {completion}")
        if last_updated:
            lines.append(f"Last Updated: {last_updated}")

        # next-step guidance per status
        guidance: dict[str, str] = {
            "SUBMITTED": "Your claim has been received and is awaiting review.",
            "UNDER_REVIEW": "An adjuster is reviewing your claim. No action required.",
            "ADDITIONAL_INFO_REQUIRED": "Please provide the requested documents to proceed.",
            "APPROVED": "Your claim is approved. Payment will be processed shortly.",
            "PARTIALLY_APPROVED": "A portion of your claim was approved. Review the notes.",
            "DENIED": "Your claim was denied. You may file an appeal within 30 days.",
            "CLOSED": "This claim has been closed.",
            "CANCELLED": "This claim was cancelled.",
        }
        if status in guidance:
            lines.append(f"\nNext Steps: {guidance[status]}")

        return "\n".join(lines)


def get_claim_tools() -> list[BaseTool]:
    return [ClaimSummaryTool()]
