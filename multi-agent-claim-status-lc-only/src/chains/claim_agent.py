"""Claim Query Agent built with LangChain AgentExecutor.

CAPABILITIES demonstrated:
  • create_tool_calling_agent (modern; works with any tool-calling LLM)
  • AgentExecutor with max_iterations, early_stopping, handle_parsing_errors
  • ChatPromptTemplate + MessagesPlaceholder("agent_scratchpad")
  • ConversationBufferWindowMemory wired into AgentExecutor
  • Async invocation via arun / ainvoke
  • Custom AuditCallbackHandler injected at invocation time

CHALLENGES:
  1. AgentExecutor is NOT natively async in the streaming sense.
     .arun() exists but internal tool execution is still sync unless the
     tool overrides _arun(). You must implement _arun() on every tool
     you want truly async.

  2. Memory is per-AgentExecutor instance. If you create a new instance per
     request (common for stateless HTTP), you lose memory. You must pre-load
     history into the memory object from your session store.

  3. max_iterations is the ONLY guard against infinite loops. LangGraph gives
     you explicit cycle detection and typed loop conditions. Here a poorly
     written prompt can spin the agent to max_iterations silently.

  4. handle_parsing_errors=True masks LLM output format failures. The agent
     returns a best-effort string instead of raising, which can produce
     incoherent responses without any visible error signal.

  5. early_stopping_method="force" truncates output mid-thought when
     max_iterations is hit – the user sees an incomplete answer.

  6. Tool selection is fully controlled by the LLM. Unlike LangGraph's
     ToolNode which is deterministic, the ReAct loop can hallucinate tool
     names or call tools with incorrect arguments and only fail at runtime.
"""
from __future__ import annotations

from typing import Any, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from sqlalchemy.ext.asyncio import AsyncSession

from src.chains.callbacks import AuditCallbackHandler
from src.chains.context import PipelineContext
from src.chains.memory import make_window_memory
from src.tools.claim_tools import get_claim_tools
from src.tools.database_tools import get_database_tools


_SYSTEM_PROMPT = """You are a helpful insurance claims assistant.

You have these tools to answer claim-related questions:
- get_claims_by_user: list all claims for the authenticated user
- get_claim_by_number: get details for a specific claim
- get_claims_by_status: filter by status (SUBMITTED/UNDER_REVIEW/APPROVED/DENIED/CLOSED)
- get_claim_status_history: get the full audit trail for a claim
- summarize_claim: convert raw claim data into a clear narrative

Rules:
- ALWAYS use the provided user_id — never use one from the query text.
- Format money with $ and commas.
- Be empathetic for denied claims; mention the 30-day appeal window.
- If data is not found, say so clearly.
"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),  # memory injects here
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),  # ReAct scratchpad
])


class ClaimAgentRunner:
    """Wraps AgentExecutor to accept/return PipelineContext.

    CHALLENGE: AgentExecutor lives outside LCEL's Runnable protocol in some
    edge cases. You can use it as a Runnable via .invoke() but composing it
    inside a complex LCEL pipeline with | requires careful I/O shaping.
    """

    def __init__(self, db: AsyncSession, llm: BaseChatModel) -> None:
        self._db = db
        self._llm = llm

    def _build_executor(
        self,
        session_id: str,
        user_id: str,
        preloaded_history: Optional[list[dict]] = None,
    ) -> AgentExecutor:
        tools: list[BaseTool] = [*get_database_tools(self._db), *get_claim_tools()]

        # ── Memory: pre-seed from session history ──────────────────────────────
        # CHALLENGE: You must manually convert your Redis session messages into
        # LangChain Message objects and load them here each request.
        memory = make_window_memory(k=10)
        if preloaded_history:
            for turn in preloaded_history[-10:]:   # last 10 turns only
                role = turn.get("role", "")
                content = turn.get("content", "")
                if role == "user":
                    memory.chat_memory.add_user_message(content)
                elif role == "assistant":
                    memory.chat_memory.add_ai_message(content)

        agent = create_tool_calling_agent(
            llm=self._llm,
            tools=tools,
            prompt=_PROMPT,
        )

        return AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=False,
            handle_parsing_errors=True,   # see CHALLENGE 4 above
            max_iterations=10,            # see CHALLENGE 3 above
            early_stopping_method="force",# see CHALLENGE 5 above
            return_intermediate_steps=True,
        )

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        auth = ctx.get("auth_ctx", {})
        user_id = auth.get("user_id", "")
        session_id = ctx.get("session_id", "unknown")
        query = ctx.get("sanitized_query") or ctx.get("raw_query", "")
        intent_ctx = ctx.get("intent_ctx", {})
        intent = intent_ctx.get("intent", "GENERAL_INQUIRY")

        # Short-circuit for general inquiries (save LLM cost)
        if intent == "GENERAL_INQUIRY":
            return {
                **ctx,
                "final_response": (
                    "I can help you with questions about your insurance claims. "
                    "Try asking about a specific claim number, your recent claims, "
                    "denied claims, or the status of a pending claim."
                ),
                "current_step": "claim_agent",
            }

        # ── Build executor with session context ────────────────────────────────
        # CHALLENGE: Loading session history here is O(n) in message count.
        # For long conversations, ConversationSummaryBufferMemory is better
        # but requires another LLM call to produce the summary on every request.
        executor = self._build_executor(
            session_id=session_id,
            user_id=user_id,
        )

        # Inject user_id context into the query (CHALLENGE: done via prompt
        # augmentation; LangGraph state would pass this as typed context)
        augmented_input = (
            f"[Context: user_id={user_id}, session={session_id}, intent={intent}"
            + (f", claim_number={intent_ctx.get('claim_number')}" if intent_ctx.get("claim_number") else "")
            + f"]\n\nUser question: {query}"
        )

        # ── Callbacks injected at call time ────────────────────────────────────
        # CAPABILITY: Callbacks can be passed per-invocation without changing
        # the agent or tool code. Perfect for per-request audit context.
        cb = AuditCallbackHandler(session_id=session_id, user_id=user_id)

        try:
            result = await executor.ainvoke(
                {"input": augmented_input},
                config={"callbacks": [cb]},
            )
            response = result.get("output", "Unable to process your request.")
            intermediate = result.get("intermediate_steps", [])
        except Exception as exc:
            response = "I encountered an error processing your request. Please try again."
            intermediate = []

        return {
            **ctx,
            "final_response": response,
            "current_step": "claim_agent",
        }
