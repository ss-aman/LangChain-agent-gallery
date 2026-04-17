# Project Directory Structure

```
multi-agent-claim-status/
│
├── .env.example                    # Environment variable template
├── docker-compose.yml              # PostgreSQL + Redis + App services
├── requirements.txt                # Python package dependencies
│
├── diagrams/
│   ├── flow_diagram.md             # ← YOU ARE HERE – Mermaid architecture diagrams
│   └── directory_structure.md     # This file
│
└── src/
    ├── __init__.py
    ├── main.py                     # FastAPI application + all HTTP endpoints
    │
    ├── config/
    │   ├── __init__.py
    │   └── settings.py             # Pydantic BaseSettings (env-driven config)
    │
    ├── database/                   # ── DATA LAYER ──────────────────────────────
    │   ├── __init__.py
    │   ├── models.py               # SQLAlchemy ORM: User, Claim, ClaimStatus, AuditLog
    │   ├── connection.py           # Async engine + session factory + get_db()
    │   └── repository.py          # Repository pattern: ClaimRepository, UserRepository
    │
    ├── security/                   # ── SECURITY LAYER ──────────────────────────
    │   ├── __init__.py
    │   ├── auth.py                 # JWT issue/verify, password hashing (bcrypt)
    │   ├── sanitizer.py            # XSS / SQL-injection detection + input cleaning
    │   ├── rate_limiter.py         # Redis sliding-window rate limiter (Lua script)
    │   └── audit_logger.py        # Structured audit events with PII masking
    │
    ├── session/                    # ── SESSION LAYER ───────────────────────────
    │   ├── __init__.py
    │   └── manager.py              # Redis-backed session + conversation history
    │
    ├── tools/                      # ── LANGCHAIN TOOLS ─────────────────────────
    │   ├── __init__.py
    │   ├── auth_tools.py           # ValidateJWTTool, DecodeTokenTool
    │   ├── database_tools.py       # GetClaimsByUserTool, GetClaimByNumberTool, …
    │   └── claim_tools.py          # ClaimSummaryTool (domain-level formatting)
    │
    ├── agents/                     # ── AGENTS ──────────────────────────────────
    │   ├── __init__.py
    │   ├── base_agent.py           # Abstract base + LLM factory (any provider)
    │   ├── auth_agent.py           # AuthAgent (JWT + blacklist; create_react_agent)
    │   ├── nlp_to_sql_agent.py     # NLPToSQLAgent (LCEL chain; intent + entities)
    │   └── claim_query_agent.py    # ClaimQueryAgent (ReAct; orchestrates tools)
    │
    ├── graph/                      # ── LANGGRAPH WORKFLOW ──────────────────────
    │   ├── __init__.py
    │   ├── state.py                # ClaimAgentState TypedDict (all shared state)
    │   ├── nodes.py                # Node functions: security/auth/query/error/audit
    │   ├── edges.py                # Conditional routing functions
    │   └── workflow.py             # StateGraph assembly + compile + invoke wrapper
    │
    └── utils/
        ├── __init__.py
        ├── pii_masker.py           # Email/phone/SSN/card masking helpers
        └── helpers.py              # utc_now(), format_claim_summary(), …
```

---

## Key Design Decisions

| Concern | Decision | Rationale |
|---------|----------|-----------|
| **LLM Provider** | Generic `ChatOpenAI` via `LLM_BASE_URL` | Works with OpenAI, Azure, Ollama, LiteLLM — zero code change |
| **SQL Safety** | Intent templates + SQLAlchemy params | LLM never emits raw SQL — eliminates injection |
| **Auth** | JWT HS256 + Redis blacklist | Stateless validation; instant revocation |
| **Rate Limiting** | Sliding window in Redis (Lua) | Atomic, distributed, no race conditions |
| **Session State** | Redis `SessionManager` | Enables horizontal scaling; no in-process state |
| **Graph State** | LangGraph `MemorySaver` → `AsyncRedisSaver` | Dev = memory; Prod = Redis for multi-replica |
| **Async** | `asyncio` throughout | Single-process handles many concurrent requests |
| **IDOR** | `user_id` enforced in every DB query | Users can only see their own claims |
| **Audit** | `structlog` JSON + DB `audit_logs` | Machine-parseable; PII masked before write |
| **PII** | `PIIMasker` + `_sanitize_event` | No email/SSN/card in logs or responses |

---

## Data Flow Summary

```
HTTP Request
    │
    ▼
FastAPI endpoint (/claims/query)
    │  extracts Bearer token, session_id
    ▼
LangGraph.ainvoke(initial_state)
    │
    ├─► security_check_node
    │       Redis: sliding window rate limit (IP)
    │       InputSanitizer: XSS + SQLi patterns
    │
    ├─► auth_node
    │       Redis: token blacklist lookup
    │       JWTManager: verify_token()
    │       Redis: sliding window rate limit (user)
    │
    ├─► query_node
    │       Redis: load conversation history
    │       NLPToSQLAgent: LLM → IntentResult
    │       ClaimQueryAgent (ReAct):
    │           LLM: decide which tool to call
    │           DB Tool: parameterised SELECT
    │           LLM: format answer
    │       Redis: save updated session
    │
    ├─► audit_log_node
    │       PostgreSQL: INSERT audit_logs
    │
    └─► return final_response to FastAPI
            │
            ▼
        HTTP Response (JSON)
```
