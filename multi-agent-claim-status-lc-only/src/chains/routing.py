"""Conditional routing via RunnableBranch.

CAPABILITIES demonstrated:
  • RunnableBranch for multi-way if/else routing
  • Nested RunnableBranch for sequential guard checks
  • RunnableLambda as inline error/response formatters
  • .with_config() to attach callbacks to specific branches

CHALLENGES:
  1. RunnableBranch evaluates branches TOP-DOWN and takes the FIRST match.
     This is powerful but easy to mis-order, causing silent wrong routing.
     LangGraph edge functions are pure Python with explicit names — easier
     to read, test, and visualise.

  2. RunnableBranch returns the OUTPUT of the matched branch, not the whole
     context. If your branches return different shapes, downstream chains
     will see inconsistent input types. Every branch MUST return PipelineContext.

  3. There is no "else" shorthand — the last positional argument is the default.
     If you forget the default, a non-matching condition raises ValueError.

  4. Debugging RunnableBranch is hard: the branch condition functions are
     anonymous lambdas in most examples. Name them or extract them for
     readable traces.

  5. You cannot introspect "which branch was taken" from outside the branch
     without encoding it in the return value. LangGraph tracks current_node
     automatically in the state.
"""
from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.runnables import RunnableBranch, RunnableLambda

from src.chains.context import PipelineContext


# ── Named condition functions (CHALLENGE: keep these named, not inline lambdas) ─

def _security_failed(ctx: PipelineContext) -> bool:
    return not ctx.get("security_passed", False)


def _auth_failed(ctx: PipelineContext) -> bool:
    return not ctx.get("is_authenticated", False)


# ── Error formatter ────────────────────────────────────────────────────────────

_ERROR_MESSAGES: dict[str, str] = {
    "RATE_LIMITED": "You have exceeded the request limit. Please wait before trying again.",
    "INVALID_INPUT": "Your query was rejected by the security policy.",
    "MISSING_TOKEN": "Please provide an Authorization Bearer token.",
    "TOKEN_EXPIRED": "Your session has expired. Please log in again.",
    "TOKEN_REVOKED": "Your session token has been revoked. Please log in again.",
    "INVALID_TOKEN": "Authentication failed. Invalid token.",
    "AUTH_FAILED": "Authentication failed.",
}


async def _format_error(ctx: PipelineContext) -> PipelineContext:
    code = ctx.get("error_code", "UNKNOWN")
    msg = _ERROR_MESSAGES.get(code, ctx.get("error_message", "An error occurred."))
    return {
        **ctx,
        "final_response": msg,
        "current_step": "error_handler",
    }


error_chain = RunnableLambda(_format_error)


# ── Audit append (always runs last) ────────────────────────────────────────────

async def _append_audit(ctx: PipelineContext) -> PipelineContext:
    trail = list(ctx.get("audit_trail", []))
    trail.append({
        "step": "pipeline_complete",
        "error_code": ctx.get("error_code"),
        "intent": ctx.get("intent_ctx", {}).get("intent"),
        "ts": datetime.now(timezone.utc).isoformat(),
    })
    return {**ctx, "audit_trail": trail, "current_step": "done"}


audit_append_chain = RunnableLambda(_append_audit)


# ── Routing factories ──────────────────────────────────────────────────────────

def make_auth_or_error_branch(auth_chain, query_chain) -> RunnableBranch:
    """
    After security passes: check auth, then run query.

    RunnableBranch(
        (condition_1, runnable_1),
        (condition_2, runnable_2),
        default_runnable           ← required; no match = ValueError without this
    )

    CHALLENGE: The condition functions receive the EXACT same input that the
    branch receives. If a prior chain mutated the context, all conditions see
    the updated dict. Be explicit about what you're testing.
    """
    return RunnableBranch(
        # Condition: auth failed → error
        (_auth_failed, error_chain),
        # Default: auth passed → run claim query
        query_chain,
    )


def make_top_level_branch(
    security_chain,
    auth_chain,
    query_chain,
) -> RunnableBranch:
    """
    Top-level routing: security → auth → query | error at any stage.

    CHALLENGE: Nesting RunnableBranch creates a tree structure that is hard
    to visualise or debug without tooling. LangGraph's graph.draw_mermaid()
    gives you a diagram for free.
    """
    auth_or_error = make_auth_or_error_branch(auth_chain, query_chain)

    return RunnableBranch(
        # Security failed → immediate error
        (_security_failed, error_chain),
        # Security passed → run auth chain, then route again
        auth_chain | auth_or_error,
    )
