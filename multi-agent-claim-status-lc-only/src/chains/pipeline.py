"""Full pipeline assembly — LangChain-only, no LangGraph.

CAPABILITIES demonstrated in this file:
  • RunnableSequence via the | operator
  • RunnableParallel for concurrent independent steps
  • RunnableBranch for conditional routing
  • RunnableLambda for custom business logic
  • RunnablePassthrough to thread context through
  • .with_config() to attach callbacks pipeline-wide
  • .with_fallbacks() for graceful degradation
  • Streaming via astream_events (LangChain 0.2+)

OVERALL CHALLENGES of LangChain-only multi-agent:
─────────────────────────────────────────────────
  1. NO TYPED STATE MACHINE
     State is a plain Python dict. There's no schema validation between steps.
     Any chain can introduce typos in key names and the pipeline will silently
     pass wrong data forward. LangGraph's TypedDict state catches this.

  2. NO VISUAL GRAPH / DEBUGGING TOOL
     LangGraph has .get_graph().draw_mermaid(). Here you must read the code
     to understand the execution path. On complex pipelines this is a real
     maintenance burden.

  3. NO CYCLES
     LCEL chains are pure DAGs. If you need the LLM to clarify a misunderstood
     query and retry, you must write an explicit Python while loop outside the
     chain. LangGraph handles cycles declaratively.

  4. MEMORY FRAGMENTATION
     Each AgentExecutor carries its own memory. If you add a second agent (e.g.
     a follow-up explainer agent), their memories are completely separate. You
     must manually stitch conversation history between them.

  5. NO BUILT-IN CHECKPOINTING
     LangGraph's MemorySaver/RedisSaver lets you resume mid-graph after a crash.
     Here, if the process dies between the auth step and the DB step, the client
     gets an error and must retry from scratch.

  6. ERROR ROUTING IS MANUAL
     Every node must catch its own errors and encode them in the context dict.
     If you forget to handle an exception in one lambda, it propagates and kills
     the whole request. The .with_fallbacks() mechanism only works at the chain
     boundary level.

  7. PARALLEL AGENT EXECUTION IS VERBOSE
     RunnableParallel is clean for two independent LLM tasks (e.g. sentiment +
     intent simultaneously). But for three or more interdependent agents it
     becomes deeply nested and hard to reason about.

  8. STREAMING ACROSS MULTIPLE AGENTS
     astream_events works per-chain. To stream tokens from a ReAct agent
     INSIDE a pipeline, you need to collect events filtered by run_name,
     which is verbose and fragile.
"""
from __future__ import annotations

from typing import Any, AsyncIterator

import redis.asyncio as aioredis
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSequence,
)
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.base_agent import create_llm
from src.chains.auth_chain import make_auth_chain
from src.chains.callbacks import AuditCallbackHandler
from src.chains.claim_agent import ClaimAgentRunner
from src.chains.context import PipelineContext, new_context
from src.chains.nlp_to_sql_chain import make_context_intent_chain
from src.chains.routing import audit_append_chain, make_top_level_branch
from src.chains.security_chain import make_security_chain
from src.session.manager import SessionManager


class ClaimPipeline:
    """
    Assembles the full multi-agent claim pipeline using LangChain LCEL.

    Architecture:
        security_chain
            │
            ├── (fail) → error_chain → audit_chain
            └── (pass) →
                    auth_chain
                        │
                        ├── (fail) → error_chain → audit_chain
                        └── (pass) →
                                intent_chain (NLP→SQL)
                                    │
                                    claim_agent (AgentExecutor / ReAct)
                                        │
                                        audit_chain

    CHALLENGE: This tree is built imperatively in Python, not declared as a
    graph. Adding a new branch (e.g. a fraud-check agent between auth and query)
    requires understanding the nesting and manually inserting it. In LangGraph
    you add a node and two edges.
    """

    def __init__(
        self,
        db: AsyncSession,
        redis_client: aioredis.Redis,
        session_manager: SessionManager,
        llm: BaseChatModel | None = None,
    ) -> None:
        self._db = db
        self._redis = redis_client
        self._sm = session_manager
        self._llm = llm or create_llm()

        # ── Build individual chain components ──────────────────────────────────
        self._security_chain = make_security_chain(redis_client)
        self._auth_chain = make_auth_chain(redis_client)
        self._intent_chain = make_context_intent_chain(self._llm)
        self._claim_runner = ClaimAgentRunner(db=db, llm=self._llm)

        # ── Query sub-pipeline: intent classification → agent execution ────────
        # CAPABILITY: | chains two Runnables into a RunnableSequence.
        # Both take/return PipelineContext so no adapter needed here.
        query_chain: Runnable = (
            self._intent_chain
            | RunnableLambda(self._claim_runner.run)
        )

        # ── Top-level routing via RunnableBranch ───────────────────────────────
        routed_chain = make_top_level_branch(
            security_chain=self._security_chain,
            auth_chain=self._auth_chain,
            query_chain=query_chain,
        )

        # ── Final pipeline: security check → routing → audit ──────────────────
        # CHALLENGE: audit_append_chain always runs, even on error paths.
        # We achieve this by placing it AFTER the RunnableBranch output.
        # But if RunnableBranch itself raises (not just returns an error ctx),
        # audit would be skipped. We add .with_fallbacks() to guarantee it.
        self._pipeline: Runnable = (
            self._security_chain
            | make_top_level_branch(
                security_chain=self._security_chain,
                auth_chain=self._auth_chain,
                query_chain=query_chain,
            )
            | audit_append_chain
        ).with_fallbacks(
            [RunnableLambda(self._emergency_fallback)],
            exceptions_to_handle=(Exception,),
        )

    async def _emergency_fallback(self, ctx: Any) -> PipelineContext:
        """Last-resort fallback if the pipeline raises an unhandled exception."""
        base = ctx if isinstance(ctx, dict) else {}
        return {
            **base,
            "final_response": "A system error occurred. Please try again later.",
            "error_code": "INTERNAL_ERROR",
            "current_step": "emergency_fallback",
        }

    # ── Public API ─────────────────────────────────────────────────────────────

    async def run(
        self,
        *,
        raw_query: str,
        raw_token: str,
        session_id: str,
        ip_address: str = "unknown",
        user_agent: str = "",
    ) -> dict[str, Any]:
        """Execute the full pipeline and return a clean response dict."""
        ctx = new_context(
            raw_query=raw_query,
            raw_token=raw_token,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        # CAPABILITY: .with_config() attaches callbacks to the ENTIRE pipeline
        # without changing any chain internals.
        cb = AuditCallbackHandler(session_id=session_id)
        config = {"callbacks": [cb], "run_name": "ClaimPipeline"}

        result: PipelineContext = await self._pipeline.ainvoke(ctx, config=config)

        return {
            "response": result.get("final_response", ""),
            "session_id": session_id,
            "is_authenticated": result.get("is_authenticated", False),
            "error_code": result.get("error_code"),
            "intent": result.get("intent_ctx", {}).get("intent"),
        }

    async def stream(
        self,
        *,
        raw_query: str,
        raw_token: str,
        session_id: str,
        ip_address: str = "unknown",
    ) -> AsyncIterator[str]:
        """
        Yield token chunks as the agent generates the response.

        CAPABILITY: astream_events fires events for every sub-chain, including
        individual LLM tokens. We filter to agent output tokens only.

        CHALLENGE: Streaming ACROSS a RunnableBranch is opaque — you don't
        know which branch will be taken until it starts executing, so you
        cannot pre-announce the structure to the client.
        """
        ctx = new_context(
            raw_query=raw_query,
            raw_token=raw_token,
            session_id=session_id,
            ip_address=ip_address,
        )
        async for event in self._pipeline.astream_events(ctx, version="v2"):
            kind = event.get("event")
            # Filter: only stream LLM generation tokens from the claim agent
            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield chunk.content
