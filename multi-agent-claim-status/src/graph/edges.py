"""Edge condition functions for the LangGraph workflow.

Each function receives the current state and returns the name of the next node.
Pure functions – no side effects.
"""
from __future__ import annotations

from src.graph.state import ClaimAgentState

# ── Node name constants (avoids magic strings) ────────────────────────────────
SECURITY_NODE = "security_check"
AUTH_NODE = "auth_node"
QUERY_NODE = "query_node"
RESPONSE_NODE = "response_node"
AUDIT_NODE = "audit_node"
ERROR_NODE = "error_node"
END = "__end__"


def route_after_security(state: ClaimAgentState) -> str:
    """After security check: pass through or terminate with error."""
    if not state.get("security_passed", False):
        return ERROR_NODE
    return AUTH_NODE


def route_after_auth(state: ClaimAgentState) -> str:
    """After auth check: proceed if authenticated, else error."""
    if not state.get("is_authenticated", False):
        return ERROR_NODE
    return QUERY_NODE


def route_after_query(state: ClaimAgentState) -> str:
    """After query processing: always go to audit then end."""
    return AUDIT_NODE


def route_after_error(state: ClaimAgentState) -> str:
    """After error node: always audit then end."""
    return AUDIT_NODE


def route_after_audit(state: ClaimAgentState) -> str:
    """After audit: always terminate the graph."""
    return END
