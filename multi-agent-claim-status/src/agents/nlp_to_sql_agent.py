"""NLP → SQL Intent Agent.

SECURITY MODEL: This agent never emits raw SQL.
It classifies user intent into a predefined enum and extracts typed parameters.
The graph then selects a pre-approved, parameterised query template and
binds the parameters via SQLAlchemy – eliminating injection at the call site.
"""
from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.agents.base_agent import BaseClaimAgent


# ── Intent taxonomy ────────────────────────────────────────────────────────────

class QueryIntent(str, Enum):
    LIST_ALL_CLAIMS = "LIST_ALL_CLAIMS"
    GET_CLAIM_STATUS = "GET_CLAIM_STATUS"
    GET_CLAIM_DETAILS = "GET_CLAIM_DETAILS"
    LIST_CLAIMS_BY_DATE = "LIST_CLAIMS_BY_DATE"
    LIST_DENIED_CLAIMS = "LIST_DENIED_CLAIMS"
    LIST_APPROVED_CLAIMS = "LIST_APPROVED_CLAIMS"
    LIST_PENDING_CLAIMS = "LIST_PENDING_CLAIMS"
    GET_CLAIM_HISTORY = "GET_CLAIM_HISTORY"
    GET_CLAIM_AMOUNTS = "GET_CLAIM_AMOUNTS"
    GENERAL_INQUIRY = "GENERAL_INQUIRY"


class IntentResult(BaseModel):
    intent: QueryIntent = Field(description="Classified query intent")
    claim_number: Optional[str] = Field(default=None, description="e.g. CLM000001")
    start_date: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    end_date: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    claim_type: Optional[str] = Field(default=None, description="MEDICAL|AUTO|HOME|LIFE|DISABILITY|TRAVEL")
    limit: int = Field(default=10, ge=1, le=50)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning: str = Field(default="", description="Why this intent was chosen")


# ── Prompt ─────────────────────────────────────────────────────────────────────

_INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an intent classifier for an insurance claims query system.

Classify the user's query into ONE of these intents:
- LIST_ALL_CLAIMS: User wants to see all their claims
- GET_CLAIM_STATUS: User asks about the status of a specific claim
- GET_CLAIM_DETAILS: User wants full details of a specific claim
- LIST_CLAIMS_BY_DATE: User filters claims by a date or date range
- LIST_DENIED_CLAIMS: User asks about denied claims specifically
- LIST_APPROVED_CLAIMS: User asks about approved claims
- LIST_PENDING_CLAIMS: User asks about pending/in-progress claims
- GET_CLAIM_HISTORY: User wants the status change history of a claim
- GET_CLAIM_AMOUNTS: User asks about money/amounts for a claim
- GENERAL_INQUIRY: Cannot be served by any database query

Extract these entities when present:
- claim_number: Format CLMxxxxxx (6-12 digits after CLM)
- start_date / end_date: Convert relative dates (e.g. "last month") to YYYY-MM-DD based on today {today}
- claim_type: one of MEDICAL, AUTO, HOME, LIFE, DISABILITY, TRAVEL
- limit: number of results the user wants (default 10, max 50)

Return ONLY valid JSON matching this schema:
{{
  "intent": "<INTENT>",
  "claim_number": "<CLMxxxxxx or null>",
  "start_date": "<YYYY-MM-DD or null>",
  "end_date": "<YYYY-MM-DD or null>",
  "claim_type": "<type or null>",
  "limit": <number>,
  "confidence": <0.0-1.0>,
  "reasoning": "<brief explanation>"
}}"""),
    ("human", "{query}"),
])


class NLPToSQLAgent(BaseClaimAgent):
    """Converts natural language claim queries into structured intent + parameters."""

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        super().__init__(llm)
        self._chain: RunnableSerializable = _INTENT_PROMPT | self.llm | JsonOutputParser()

    @property
    def name(self) -> str:
        return "nlp_to_sql_agent"

    @property
    def tools(self) -> list[BaseTool]:
        return []  # This agent uses an LCEL chain, not tool calls

    async def run(self, *, query: str, **_: Any) -> dict[str, Any]:
        """Parse the user's natural-language query into a structured intent."""
        from datetime import date
        today = date.today().isoformat()

        try:
            result = await self._chain.ainvoke({"query": query, "today": today})
            # Validate and coerce
            intent_result = IntentResult(**result)
            return intent_result.model_dump()
        except Exception as exc:
            # Graceful degradation: return a safe fallback intent
            return IntentResult(
                intent=QueryIntent.GENERAL_INQUIRY,
                confidence=0.0,
                reasoning=f"Parse error: {exc}",
            ).model_dump()

    def intent_to_tool_call(self, intent_result: dict[str, Any]) -> dict[str, Any]:
        """Map an intent to the correct database tool name + arguments."""
        intent = intent_result.get("intent")
        mapping: dict[str, dict[str, Any]] = {
            QueryIntent.LIST_ALL_CLAIMS: {
                "tool": "get_claims_by_user",
                "extra_args": {"limit": intent_result.get("limit", 10)},
            },
            QueryIntent.GET_CLAIM_STATUS: {
                "tool": "get_claim_by_number",
                "extra_args": {"claim_number": intent_result.get("claim_number")},
            },
            QueryIntent.GET_CLAIM_DETAILS: {
                "tool": "get_claim_by_number",
                "extra_args": {"claim_number": intent_result.get("claim_number")},
            },
            QueryIntent.LIST_CLAIMS_BY_DATE: {
                "tool": "get_claims_by_user",
                "extra_args": {
                    "start_date": intent_result.get("start_date"),
                    "end_date": intent_result.get("end_date"),
                    "limit": intent_result.get("limit", 10),
                },
            },
            QueryIntent.LIST_DENIED_CLAIMS: {
                "tool": "get_claims_by_status",
                "extra_args": {"status": "DENIED"},
            },
            QueryIntent.LIST_APPROVED_CLAIMS: {
                "tool": "get_claims_by_status",
                "extra_args": {"status": "APPROVED"},
            },
            QueryIntent.LIST_PENDING_CLAIMS: {
                "tool": "get_claims_by_status",
                "extra_args": {"status": "UNDER_REVIEW"},
            },
            QueryIntent.GET_CLAIM_HISTORY: {
                "tool": "get_claim_status_history",
                "extra_args": {"claim_number": intent_result.get("claim_number")},
            },
            QueryIntent.GET_CLAIM_AMOUNTS: {
                "tool": "get_claim_by_number",
                "extra_args": {"claim_number": intent_result.get("claim_number")},
            },
        }
        return mapping.get(intent, {"tool": None, "extra_args": {}})
