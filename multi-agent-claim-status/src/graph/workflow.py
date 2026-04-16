"""Main LangGraph workflow assembly.

Graph topology:
  START
    └─► security_check ──(fail)──► error_node ──► audit_node ──► END
                        ──(pass)──► auth_node ──(fail)──► error_node
                                              ──(pass)──► query_node
                                                            └─► audit_node ──► END

Checkpointing via MemorySaver (swap to AsyncRedisSaver for production).
"""
from __future__ import annotations

from functools import partial
from typing import Any

import redis.asyncio as aioredis
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.graph import CompiledGraph
from sqlalchemy.ext.asyncio import AsyncSession

from src.graph.edges import (
    AUTH_NODE,
    AUDIT_NODE,
    ERROR_NODE,
    QUERY_NODE,
    SECURITY_NODE,
    route_after_auth,
    route_after_security,
)
from src.graph.nodes import (
    audit_log_node,
    auth_node,
    error_node,
    query_node,
    security_check_node,
)
from src.graph.state import ClaimAgentState
from src.session.manager import SessionManager


def build_claim_workflow(
    db: AsyncSession,
    redis_client: aioredis.Redis,
    session_manager: SessionManager,
    *,
    use_persistent_checkpointer: bool = False,
) -> CompiledGraph:
    """Build and compile the multi-agent LangGraph workflow.

    Args:
        db: Async SQLAlchemy session (injected per request)
        redis_client: Shared Redis client
        session_manager: Session manager for conversation history
        use_persistent_checkpointer: Use Redis-backed checkpointer for HA deploys

    Returns:
        Compiled LangGraph graph ready for ainvoke / astream
    """
    # ── Bind infrastructure dependencies to node functions ────────────────────
    # Using functools.partial so each node remains a pure (state) → dict function
    # while still having access to I/O resources.
    bound_security = partial(security_check_node, redis_client=redis_client)
    bound_auth = partial(auth_node, redis_client=redis_client)
    bound_query = partial(query_node, db=db, session_manager=session_manager)
    bound_audit = partial(audit_log_node, db=db)

    # ── Graph builder ─────────────────────────────────────────────────────────
    builder = StateGraph(ClaimAgentState)

    # Register nodes
    builder.add_node(SECURITY_NODE, bound_security)
    builder.add_node(AUTH_NODE, bound_auth)
    builder.add_node(QUERY_NODE, bound_query)
    builder.add_node(ERROR_NODE, error_node)
    builder.add_node(AUDIT_NODE, bound_audit)

    # Entry point
    builder.add_edge(START, SECURITY_NODE)

    # Conditional routing
    builder.add_conditional_edges(SECURITY_NODE, route_after_security)
    builder.add_conditional_edges(AUTH_NODE, route_after_auth)

    # Linear edges
    builder.add_edge(QUERY_NODE, AUDIT_NODE)
    builder.add_edge(ERROR_NODE, AUDIT_NODE)
    builder.add_edge(AUDIT_NODE, END)

    # ── Checkpointer ──────────────────────────────────────────────────────────
    if use_persistent_checkpointer:
        # For production horizontal scaling: graph state is stored in Redis
        # so any replica can resume a conversation.
        # Requires: pip install langgraph-checkpoint-redis
        try:
            from langgraph.checkpoint.redis.aio import AsyncRedisSaver
            checkpointer = AsyncRedisSaver.from_conn_string(
                str(redis_client.connection_pool.connection_kwargs)
            )
        except ImportError:
            checkpointer = MemorySaver()
    else:
        checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)


async def invoke_workflow(
    graph: CompiledGraph,
    *,
    raw_query: str,
    raw_token: str,
    session_id: str,
    ip_address: str = "unknown",
    user_agent: str = "",
) -> dict[str, Any]:
    """Convenience wrapper: build initial state and invoke the graph."""
    initial_state: ClaimAgentState = {
        "messages": [],
        "raw_token": raw_token,
        "raw_query": raw_query,
        "sanitized_query": None,
        "session_id": session_id,
        "security_context": {
            "ip_address": ip_address,
            "user_agent": user_agent,
            "rate_limit_remaining": 0,
            "threat_detected": False,
            "threat_type": None,
        },
        "security_passed": False,
        "auth_context": {},
        "is_authenticated": False,
        "intent_context": {},
        "claim_data": None,
        "final_response": None,
        "error_code": None,
        "error_message": None,
        "current_node": None,
        "iteration_count": 0,
        "audit_trail": [],
    }

    config: RunnableConfig = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 20,
    }

    result = await graph.ainvoke(initial_state, config=config)
    return {
        "response": result.get("final_response", ""),
        "session_id": session_id,
        "is_authenticated": result.get("is_authenticated", False),
        "error_code": result.get("error_code"),
        "intent": result.get("intent_context", {}).get("intent"),
    }
