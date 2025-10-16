# RAG + MCP Agent (v0.1)

**Purpose:** A local-first Retrieval-Augmented Generation (RAG) agent that:
- Retrieves content from **AWS Bedrock Knowledge Bases** via a local **MCP server**.
- (Optional) Generates **PostgreSQL SELECT** queries via a dedicated **SQL MCP server**, with **read-only** enforcement, LIMIT/timeout/result-size caps, and EXPLAIN-first validation.
- Runs locally for dev, and can be connected to an **AgentCore Gateway** when deployed.

## Contents
```
agents/rag_mcp/
├─ src/
│ ├─ mcp_server.py # KB MCP server (search across General Docs & CS KB)
│ ├─ kb_agent.py # AgentCore-compatible HTTP app using the MCP tool
│ └─ SQL_agent.py # (optional) PostgreSQL MCP server with SQL generation
├─ scripts/
│ ├─ ab_harness.py # simple model comparison
│ └─ ab_harness_plus.py # multilingual eval + metrics (F1/ROUGE-L/Jaccard)
├─ docs/
│ └─ qa_dataset.csv # optional evaluation set
├─ .env.example
├─ requirements.txt
└─ README.md
```

## Prerequisites

- Python 3.10+
- AWS credentials (`AWS_PROFILE`/`AWS_REGION`) with access to our Bedrock **Knowledge Bases**.
- (Optional) **Ollama** for local LLMs: `qwen2.5:14b-instruct`, `sqlcoder:7b`. We will need to connect to the Lambda VM later.
- (Optional) PostgreSQL read-only DSNs for SQL agent usage.

## Quick start

1. Create `.env` from `.env.example` and fill the values:
   - `KB_GENERAL_DOCS_ID`, `KB_CS_SUPPORT_ID`
   - Postgres DSNs in `DB_SOURCES` (for SQL agent)
   - (Optional) API keys for MCP servers.

2. Install:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

Launch KB MCP (port 7000):

```
python -m uvicorn mcp_server:app --app-dir src --host 127.0.0.1 --port 7000
```

(Update soon with this new MCP server)
Launch SQL MCP (port 7100):
```
python -m uvicorn SQL_agent:app --app-dir src --host 127.0.0.1 --port 7100
```

Launch the Agent (port 8080):
```
python src/kb_agent.py
```

Invoke locally:
```
curl -X POST http://127.0.0.1:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"prompt":"I have an issue with the screen of my BOX ..."}'
```

## How it works
- The KB MCP exposes kb.search that:

    - Auto-detects language (en, fr, de, zh-Hant) and routes to General Docs vs CS KB.

    - Returns chunks (title/url/text/score) for the agent to synthesize answers (with citations).

- The Agent (kb_agent.py) receives user text and:

    - Calls kb.search (via MCP), then synthesizes using Strands/Bedrock model.

    - Falls back to Bedrock Runtime if the callable path fails.

- The SQL MCP exposes:

    - db.schema, sql.generate (Ollama-backed; default sqlcoder:7b), db.explain, db.query

    - Read-only execution, hard LIMIT + timeout + size caps, and handle-based overflow.

## Safety & limits (SQL MCP)
- Only single SELECT allowed, read-only transaction (SET LOCAL default_transaction_read_only = on).

- Hard LIMIT enforced server-side.

- Statement timeout configurable (QUERY_TIMEOUT_S).

- Result-size cap with CSV handles for large outputs.

- Optional block SELECT * (BLOCK_SELECT_STAR=1).

## Evaluation (optional)
- scripts/ab_harness_plus.py runs multilingual tests (en/fr/de/zh-Hant), auto-translates golds if missing, and reports:

    - Token F1, ROUGE-L, Jaccard similarity,

    - Latencies (plan/retrieve/synthesis),

    - Citation rate, chunk counts, and more.

## Deploying behind an AgentCore Gateway (prod)
- Keep MCP servers as independent services, point MCP_URL to the MCP-Gateway route.

- Gateway can enforce API keys, IP allow-lists, and route to multiple MCP servers (KB, SQL).

- The agent runs in AgentCore runtime and calls MCP through the Gateway.

## License
Internal use. 