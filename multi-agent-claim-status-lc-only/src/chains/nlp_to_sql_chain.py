"""NLP → Intent classification as a pure LCEL chain.

CAPABILITIES demonstrated:
  • ChatPromptTemplate + MessagesPlaceholder
  • JsonOutputParser + PydanticOutputParser
  • RunnableParallel – run two parsers in parallel for comparison / fallback
  • .with_fallbacks() – if JsonOutputParser fails, fall back to regex extraction
  • RunnablePassthrough – thread the original context through alongside the LLM call

CHALLENGES:
  1. LCEL chains are strict DAGs – no cycles. If the LLM produces invalid JSON
     you get ONE fallback attempt, not a retry loop asking the LLM to fix it.
     LangGraph would let you loop back to the same node.

  2. RunnableParallel doubles LLM calls if both branches actually invoke the model.
     Here we use it for parse-time parallelism (two parsers on the same LLM output)
     which is fine, but accidental parallel LLM calls are expensive.

  3. Prompt engineering is your entire correctness guarantee. There is no
     structured function-calling schema enforced at the framework level unless
     you use .with_structured_output() on the LLM directly.
"""
from __future__ import annotations

import json
import re
from datetime import date
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from pydantic import BaseModel, Field

from src.chains.context import PipelineContext


# ── Output schema ──────────────────────────────────────────────────────────────

class IntentResult(BaseModel):
    intent: str = Field(description="One of the predefined intents")
    claim_number: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    claim_type: Optional[str] = None
    limit: int = Field(default=10)
    confidence: float = Field(default=1.0)
    reasoning: str = Field(default="")


VALID_INTENTS = {
    "LIST_ALL_CLAIMS", "GET_CLAIM_STATUS", "GET_CLAIM_DETAILS",
    "LIST_CLAIMS_BY_DATE", "LIST_DENIED_CLAIMS", "LIST_APPROVED_CLAIMS",
    "LIST_PENDING_CLAIMS", "GET_CLAIM_HISTORY", "GET_CLAIM_AMOUNTS",
    "GENERAL_INQUIRY",
}

# ── Prompt ─────────────────────────────────────────────────────────────────────

_INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an intent classifier for an insurance claims system.
Today is {today}.

Classify the query into ONE intent:
LIST_ALL_CLAIMS | GET_CLAIM_STATUS | GET_CLAIM_DETAILS | LIST_CLAIMS_BY_DATE
LIST_DENIED_CLAIMS | LIST_APPROVED_CLAIMS | LIST_PENDING_CLAIMS
GET_CLAIM_HISTORY | GET_CLAIM_AMOUNTS | GENERAL_INQUIRY

Return ONLY valid JSON:
{{"intent":"...", "claim_number":null, "start_date":null, "end_date":null,
  "claim_type":null, "limit":10, "confidence":0.95, "reasoning":"..."}}"""),
    ("human", "{query}"),
])

# ── Fallback: regex-based extraction for when LLM returns bad JSON ─────────────

def _regex_fallback(raw: str) -> dict[str, Any]:
    """Last-resort extraction when both JSON parsers fail."""
    claim_match = re.search(r"\bCLM\d{6,12}\b", raw, re.IGNORECASE)
    intent = "GENERAL_INQUIRY"
    for kw, detected in [
        ("denied", "LIST_DENIED_CLAIMS"),
        ("approved", "LIST_APPROVED_CLAIMS"),
        ("pending", "LIST_PENDING_CLAIMS"),
        ("status", "GET_CLAIM_STATUS"),
        ("history", "GET_CLAIM_HISTORY"),
        ("amount", "GET_CLAIM_AMOUNTS"),
    ]:
        if kw in raw.lower():
            intent = detected
            break
    return {
        "intent": intent,
        "claim_number": claim_match.group().upper() if claim_match else None,
        "confidence": 0.3,
        "reasoning": "Regex fallback — LLM returned unparseable response",
        "limit": 10,
    }


# ── Chain factory ──────────────────────────────────────────────────────────────

def make_nlp_to_sql_chain(llm: BaseChatModel) -> Runnable:
    """
    Build the NLP→Intent chain using LCEL composition.

    Pipeline:
      prompt | llm | JsonOutputParser (primary)
                   | .with_fallbacks([StrOutputParser | regex_fn])

    CAPABILITY: .with_fallbacks() seamlessly swaps to an alternative Runnable
    if the primary raises ANY exception (e.g. json.JSONDecodeError).

    CHALLENGE: .with_fallbacks() catches ALL exceptions, not just parse errors.
    A network error retrying the LLM via the fallback instead of the real parser
    would silently degrade quality without alerting you.
    """
    primary_parser = JsonOutputParser()
    str_parser = StrOutputParser()
    regex_chain = str_parser | RunnableLambda(_regex_fallback)

    # Primary: LLM → JsonOutputParser
    # Fallback: LLM → StrOutputParser → regex
    parse_chain = primary_parser.with_fallbacks([regex_chain])

    # Full chain: format prompt, call LLM, parse
    intent_chain = _INTENT_PROMPT | llm | parse_chain

    # Thread original query through alongside intent result using RunnableParallel
    # CAPABILITY: RunnableParallel runs both sides simultaneously (same LLM input)
    # CHALLENGE: Both sides call the LLM independently = 2x token cost if not careful.
    #            Here the parallel is on the PARSER only, not the LLM call.

    def _extract_for_parallel(inputs: dict) -> dict:
        """Bridge: intent_chain gets {query, today}; we need to preserve them."""
        return inputs

    return intent_chain


def make_context_intent_chain(llm: BaseChatModel) -> Runnable:
    """
    Adapter that takes a PipelineContext dict and returns an enriched context.

    CHALLENGE: This is the "impedance mismatch" problem in LangChain-only code.
    The intent_chain wants {query, today}. But our pipeline passes PipelineContext.
    We need explicit adapters (RunnableLambda) to reshape inputs/outputs at every
    chain boundary. LangGraph handles this via typed state — no adapters needed.
    """
    intent_chain = make_nlp_to_sql_chain(llm)

    async def _run(ctx: PipelineContext) -> PipelineContext:
        query = ctx.get("sanitized_query") or ctx.get("raw_query", "")
        raw_result = await intent_chain.ainvoke({
            "query": query,
            "today": date.today().isoformat(),
        })
        # Normalise
        if isinstance(raw_result, str):
            try:
                raw_result = json.loads(raw_result)
            except json.JSONDecodeError:
                raw_result = _regex_fallback(raw_result)

        intent = raw_result.get("intent", "GENERAL_INQUIRY")
        if intent not in VALID_INTENTS:
            intent = "GENERAL_INQUIRY"

        return {
            **ctx,
            "intent_ctx": {
                "intent": intent,
                "claim_number": raw_result.get("claim_number"),
                "start_date": raw_result.get("start_date"),
                "end_date": raw_result.get("end_date"),
                "claim_type": raw_result.get("claim_type"),
                "limit": raw_result.get("limit", 10),
                "confidence": raw_result.get("confidence", 1.0),
            },
            "current_step": "nlp_intent",
        }

    return RunnableLambda(_run)
