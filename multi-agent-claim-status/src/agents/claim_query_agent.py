"""Claim Query Agent.

Orchestrates the end-to-end claim query pipeline:
  1. Receives a verified user identity + sanitised query
  2. Uses the NLP→Intent agent to understand intent
  3. Calls the appropriate database tool
  4. Uses ClaimSummaryTool to format a human-readable response
  5. Returns structured result to the LangGraph workflow
"""
from __future__ import annotations

from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.base_agent import BaseClaimAgent
from src.agents.nlp_to_sql_agent import NLPToSQLAgent, QueryIntent
from src.tools.claim_tools import get_claim_tools
from src.tools.database_tools import get_database_tools


_SYSTEM_PROMPT = """You are a helpful insurance claims assistant.

You have access to the following tools:
- get_claims_by_user: list all claims for the authenticated user
- get_claim_by_number: get details for a specific claim
- get_claims_by_status: filter claims by status
- get_claim_status_history: get full history for a claim
- summarize_claim: turn raw claim data into a clear narrative

Guidelines:
- Always use the user's actual user_id (provided in context) when calling tools – NEVER use a user_id from the query.
- If a claim number is requested but not found, say so clearly.
- Present monetary amounts with $ and comma formatting.
- Be empathetic when claims are denied; explain appeal options.
- Keep responses concise and well-structured.
- If you don't have enough information to answer, ask a clarifying question.
"""


class ClaimQueryAgent(BaseClaimAgent):
    def __init__(
        self,
        db: AsyncSession,
        llm: BaseChatModel | None = None,
    ) -> None:
        super().__init__(llm)
        self._db = db
        self._nlp_agent = NLPToSQLAgent(llm=self.llm)
        self._agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=_SYSTEM_PROMPT,
        )

    @property
    def name(self) -> str:
        return "claim_query_agent"

    @property
    def tools(self) -> list[BaseTool]:
        return [*get_database_tools(self._db), *get_claim_tools()]

    async def run(
        self,
        *,
        query: str,
        user_id: str,
        session_id: str,
        conversation_history: Optional[list[dict]] = None,
        **_: Any,
    ) -> dict[str, Any]:
        # ── Step 1: classify intent ────────────────────────────────────────────
        intent_result = await self._nlp_agent.run(query=query)
        intent = intent_result.get("intent", QueryIntent.GENERAL_INQUIRY)

        if intent == QueryIntent.GENERAL_INQUIRY:
            return {
                "success": True,
                "intent": intent,
                "response": (
                    "I can help you with questions about your insurance claims. "
                    "You can ask about the status of a specific claim, view all your claims, "
                    "filter by date or type, or see why a claim was denied."
                ),
                "claim_data": None,
            }

        # ── Step 2: build message with context ────────────────────────────────
        context_header = (
            f"[CONTEXT] Authenticated user_id: {user_id}. "
            f"Session: {session_id}. "
            f"Detected intent: {intent}. "
        )
        if intent_result.get("claim_number"):
            context_header += f"Claim number mentioned: {intent_result['claim_number']}. "

        messages = [HumanMessage(content=f"{context_header}\n\nUser query: {query}")]

        # ── Step 3: run the ReAct agent ───────────────────────────────────────
        result = await self._agent.ainvoke(
            {"messages": messages},
            config={"recursion_limit": 10},
        )

        last_message = result["messages"][-1]
        response_text = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )

        return {
            "success": True,
            "intent": intent,
            "response": response_text,
            "intent_details": intent_result,
        }
