```mermaid
flowchart TD
    User([User / GitHub Webhook]) -->|PR URL| API

    subgraph API["FastAPI Application"]
        direction TB
        REV[POST /review]
        SSE[POST /review/stream]
        WH[POST /webhook<br/><i>HMAC signature validation</i>]
        HEALTH[GET /health]
        METRICS[GET /metrics<br/><i>Prometheus</i>]
    end

    API --> ORCH[Reviewer Orchestrator]

    subgraph ORCH_DETAIL["Review Pipeline"]
        direction TB
        FETCH[Fetch PR Context<br/><i>metadata + file list</i>]
        PLAN[Deterministic Planner<br/><i>file priority · risk areas · checks</i>]
        AGENT[ReAct Agent Loop]
        INTEL[Intelligence Pipeline]
        RESULT[Structured ReviewResult]

        FETCH --> PLAN --> AGENT --> INTEL --> RESULT
    end

    subgraph REACT["ReAct Agent Loop"]
        direction LR
        THINK[Think<br/><i>LLM reasoning</i>]
        ACT[Act<br/><i>select + call tool</i>]
        OBS[Observe<br/><i>parse tool output</i>]
        THINK --> ACT --> OBS --> THINK
    end

    AGENT --> REACT

    subgraph LLM["LLM Providers"]
        direction LR
        ANTH[Anthropic<br/><i>Claude Sonnet 4</i>]
        OAI[OpenAI<br/><i>GPT-4o</i>]
    end

    REACT -->|API call| LLM

    subgraph BUDGET["Token Budget"]
        direction TB
        TRACK[Usage Tracking<br/><i>input + output tokens</i>]
        LIMIT[Budget Enforcement<br/><i>max 120K tokens/review</i>]
        COST[Cost Estimation<br/><i>USD per review</i>]
    end

    REACT --> BUDGET

    subgraph MCP["MCP Servers (stdio transport)"]
        direction TB
        GH[GitHub Server<br/><i>get_pr_metadata · list_pr_files<br/>get_file_contents · get_pr_diff<br/>post_review</i>]
        CA[Code Analysis Server<br/><i>run_ruff · run_mypy<br/>check_complexity</i>]
        TR[Test Runner Server<br/><i>run_tests · check_coverage</i>]
        KB[Knowledge Base Server<br/><i>ask_knowledge_base<br/>search_knowledge_base</i>]
    end

    REACT -->|MCP stdio| GH
    REACT -->|MCP stdio| CA
    REACT -->|MCP stdio| TR
    REACT -->|MCP stdio| KB

    KB -->|HTTP| RAG[RAG API<br/><i>production-hybrid-rag</i>]

    subgraph INTELLIGENCE["Intelligence Pipeline"]
        direction TB
        XF[Cross-File Analysis<br/><i>missing tests · stale imports<br/>API contracts · requirements</i>]
        SEV[Severity Re-classification<br/><i>cosmetic vs bug vs security</i>]
        SUG[Auto-Fix Suggestions<br/><i>pattern-matched fixes</i>]
        RENRICH[RAG Enrichment<br/><i>best-practice context</i>]
        CONF[Confidence Scoring<br/><i>tool score · FP adjustment<br/>corroboration · file freshness</i>]

        XF --> SEV --> SUG --> RENRICH --> CONF
    end

    INTEL --> INTELLIGENCE

    subgraph OBSERVE["Observability"]
        direction LR
        PROM[Prometheus Metrics<br/><i>counters · histograms · gauges</i>]
        TRACE[Trace Context<br/><i>per-review spans</i>]
        SLOG[structlog<br/><i>JSON logging</i>]
    end

    ORCH_DETAIL --> OBSERVE

    RESULT -->|POST| GHPR([GitHub PR Review])

    style API fill:#e3f2fd
    style REACT fill:#fff3e0
    style MCP fill:#e8f5e9
    style INTELLIGENCE fill:#f3e5f5
    style OBSERVE fill:#fce4ec
    style BUDGET fill:#fff9c4
    style LLM fill:#e0f7fa
```

## Component Details

### ReAct Agent Loop

The agent follows the classic **ReAct** (Reasoning + Acting) pattern:

1. **Think**: The LLM reasons about what information it needs next
2. **Act**: It selects and calls an MCP tool (GitHub, linter, test runner, etc.)
3. **Observe**: The tool result is parsed and fed back to the LLM
4. **Repeat** until the LLM decides it has enough information to write the review

The loop is bounded by `MAX_AGENT_STEPS` (default 20) and a token budget.

### MCP Transport

All four servers communicate via **stdio transport** — the agent spawns each server as a subprocess and exchanges JSON-RPC messages over stdin/stdout. This is the standard MCP approach and means:

- Servers are independently testable (`python servers/github_server.py`)
- No network configuration needed for local development
- Any MCP-compatible client can use the servers

### Intelligence Pipeline

Post-processing pipeline that runs after the agent loop:

1. **Cross-file analysis**: Detects issues spanning multiple files (missing tests, broken imports, stale API callers)
2. **Severity re-classification**: Promotes/demotes findings based on rule databases and code context
3. **Auto-fix suggestions**: Generates before/after code snippets for known linter rules
4. **RAG enrichment**: Queries the knowledge base for best-practice context on high-severity findings
5. **Confidence scoring**: Rates each finding's reliability (0.0–1.0) using multiple factors

### Token Budget

Every review has a token budget that tracks:

- **Input tokens**: Content sent to the LLM (PR diffs, tool results, context)
- **Output tokens**: LLM responses (reasoning, tool calls, review text)
- **Total budget**: Hard cap prevents runaway costs (default 120K tokens)
- **Cost estimation**: Per-review USD cost based on model pricing

### Evaluation Framework

Offline evaluation against labeled benchmarks:

- **Gold findings**: Human-labeled expected issues per PR
- **Fuzzy matching**: File path + line proximity + keyword matching
- **Metrics**: F1 score, recall, precision, severity accuracy, suggestion rate
- **Dataset**: 5 benchmark PR scenarios covering security, refactoring, infra, trivial, and complex changes
