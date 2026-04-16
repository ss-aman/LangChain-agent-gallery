# Multi-Agent Claim Status — Flow Diagrams

## 1. High-Level System Architecture

```mermaid
graph TB
    subgraph Client
        U[User / Browser / App]
    end

    subgraph FastAPI["FastAPI Application (Uvicorn)"]
        EP["/claims/query endpoint"]
        AUTH_EP["/auth/token endpoint"]
    end

    subgraph LangGraph["LangGraph Multi-Agent Workflow"]
        direction TB
        SC[🛡️ Security Check Node]
        AN[🔑 Authentication Node]
        QN[🤖 Query Node]
        EN[❌ Error Node]
        AUD[📋 Audit Log Node]
    end

    subgraph Agents["Specialised Agents"]
        AA[AuthAgent\n- JWT validation\n- Token blacklist]
        NLP[NLPToSQLAgent\n- Intent classification\n- Entity extraction]
        CQA[ClaimQueryAgent\n- ReAct loop\n- Tool orchestration]
    end

    subgraph Tools["LangChain Tools"]
        T1[validate_jwt_token]
        T2[get_claims_by_user]
        T3[get_claim_by_number]
        T4[get_claims_by_status]
        T5[get_claim_status_history]
        T6[summarize_claim]
    end

    subgraph Infrastructure
        DB[(PostgreSQL\nClaims DB)]
        RD[(Redis\nSessions / Rate Limits\nToken Blacklist)]
        LLM[Generic LLM\nOpenAI-compatible API]
    end

    U -->|POST /claims/query\nBearer token| EP
    U -->|POST /auth/token| AUTH_EP
    AUTH_EP --> DB

    EP --> LangGraph
    SC --> AA
    AN --> AA
    QN --> NLP
    QN --> CQA
    NLP --> LLM
    CQA --> LLM
    CQA --> Tools
    Tools --> DB
    SC --> RD
    AN --> RD
    AUD --> DB
```

---

## 2. LangGraph Node Flow

```mermaid
flowchart TD
    START([▶ START]) --> SC

    SC["🛡️ security_check_node\n─────────────────\n• Sliding-window rate limit\n  per IP address\n• Input sanitisation\n  - SQL injection detection\n  - XSS detection\n• PII threat detection"]

    SC -->|security_passed = True| AN
    SC -->|security_passed = False| EN

    AN["🔑 auth_node\n─────────────────\n• Check token blacklist\n  in Redis\n• JWT signature verify\n• Expiry check\n• Per-user rate limit\n• Populate auth_context"]

    AN -->|is_authenticated = True| QN
    AN -->|is_authenticated = False| EN

    QN["🤖 query_node\n─────────────────\n• Load conversation\n  history from Redis\n• Run NLPToSQLAgent\n  → classify intent\n  → extract entities\n• Run ClaimQueryAgent\n  (ReAct loop)\n• Persist to session\n• Audit log"]

    EN["❌ error_node\n─────────────────\n• Map error code to\n  user-friendly message\n• RATE_LIMITED / AUTH\n  / INVALID_INPUT"]

    AUD["📋 audit_log_node\n─────────────────\n• Persist to audit_logs\n• Append to audit_trail\n• Record IP, user agent\n• Record status code"]

    QN --> AUD
    EN --> AUD
    AUD --> STOP([⏹ END])

    style SC fill:#f0ad4e,color:#000
    style AN fill:#5bc0de,color:#000
    style QN fill:#5cb85c,color:#fff
    style EN fill:#d9534f,color:#fff
    style AUD fill:#9B59B6,color:#fff
```

---

## 3. Agent Internal Flows

### 3a. Authentication Agent

```mermaid
sequenceDiagram
    participant N as auth_node
    participant AA as AuthAgent
    participant BL as Redis Blacklist
    participant JWT as JWTManager
    participant RL as RateLimiter

    N->>AA: run(token=...)
    AA->>BL: GET revoked_token:{token[-16:]}
    BL-->>AA: nil | "revoked"

    alt token is blacklisted
        AA-->>N: {valid: false, error: TOKEN_REVOKED}
    else not blacklisted
        AA->>JWT: verify_token(token, type="access")
        JWT-->>AA: TokenPayload | PyJWTError

        alt JWT invalid / expired
            AA-->>N: {valid: false, error: ...}
        else JWT valid
            AA->>RL: check(user_id, tier="user")
            RL-->>AA: RateLimitResult

            alt rate limit exceeded
                AA-->>N: {valid: false, error: RATE_LIMITED}
            else within limit
                AA-->>N: {valid: true, user_id, username, email}
            end
        end
    end
```

### 3b. NLP → Intent Agent

```mermaid
flowchart LR
    UQ["User Query\n'show me denied claims\nfrom last month'"]

    IC["Intent Classifier\n─────────────────\nLLM + LCEL chain\nChatPromptTemplate\n→ JsonOutputParser"]

    subgraph Output["Structured IntentResult"]
        I["intent:\nLIST_DENIED_CLAIMS"]
        SD["start_date:\n2025-03-01"]
        ED["end_date:\n2025-03-31"]
        LIM["limit: 10"]
        CONF["confidence: 0.97"]
    end

    TM["Tool Mapping\n─────────────────\nintent_to_tool_call()\n→ tool: get_claims_by_status\n→ args: {status: DENIED}"]

    UQ --> IC
    IC --> Output
    Output --> TM
```

### 3c. Claim Query Agent (ReAct Loop)

```mermaid
sequenceDiagram
    participant QN as query_node
    participant CQA as ClaimQueryAgent
    participant NLP as NLPToSQLAgent
    participant LLM as Generic LLM
    participant T as DB Tools

    QN->>CQA: run(query, user_id, session_id)
    CQA->>NLP: run(query=...)
    NLP->>LLM: classify intent + extract entities
    LLM-->>NLP: IntentResult JSON
    NLP-->>CQA: {intent, claim_number, dates, ...}

    alt GENERAL_INQUIRY
        CQA-->>QN: {response: "I can help with..."}
    else needs DB
        CQA->>LLM: ReAct Think step
        LLM-->>CQA: "I need to call get_claim_by_number"
        CQA->>T: get_claim_by_number(claim_number, user_id)
        T-->>CQA: {found: true, claim_number, status, ...}
        CQA->>T: summarize_claim(claim_data)
        T-->>CQA: Formatted narrative
        CQA->>LLM: ReAct Answer step
        LLM-->>CQA: Final response string
        CQA-->>QN: {success: true, response: "..."}
    end
```

---

## 4. Security Architecture

```mermaid
flowchart TD
    subgraph Threats["Threats Mitigated"]
        SQ[SQL Injection]
        XSS[XSS / Script Injection]
        BFRC[Brute Force]
        IDOR[IDOR / Unauthorized Access]
        TOKEN[Token Theft / Replay]
        DDoS[DDoS / Flood]
    end

    subgraph Mitigations["Security Controls"]
        SQ --> SAN["InputSanitizer\nPattern matching\non all free-text"]
        XSS --> SAN
        BFRC --> RATELIM["RateLimiter\nSliding window\nper-IP + per-user"]
        DDoS --> RATELIM
        IDOR --> REPO["ClaimRepository\nuser_id enforced\non ALL queries"]
        TOKEN --> JWT_BL["JWT blacklist\nin Redis\non logout / revoke"]
        TOKEN --> JWT_EXP["Short expiry\n60 min access\n7 day refresh"]
    end

    subgraph Audit["Audit Trail"]
        A1[Every request logged]
        A2[PII masked in logs]
        A3[Threat events flagged]
        A4[DB audit_logs table]
    end
```

---

## 5. Horizontal Scaling Strategy

```mermaid
graph LR
    LB[Load Balancer] --> A1[App Instance 1]
    LB --> A2[App Instance 2]
    LB --> A3[App Instance N]

    A1 & A2 & A3 --> PG[(PostgreSQL\nRead Replicas)]
    A1 & A2 & A3 --> RD[(Redis Cluster\n- Sessions\n- Rate Limits\n- LangGraph Checkpoints\n- Token Blacklist)]
    A1 & A2 & A3 --> LLM[LLM API\nExternal]

    style RD fill:#dc322f,color:#fff
    style PG fill:#268bd2,color:#fff
```

Stateless app instances share all mutable state through Redis and PostgreSQL,
enabling zero-downtime rolling deployments and true horizontal scaling.
