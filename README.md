# AgentCore + Strands + Bedrock KB (RAG) V0

This repo is my end-to-end starter for a Strands-first agent that grounds answers on an Amazon Bedrock Knowledge Base (KB) indexed from Amazon S3. I deploy the agent to AgentCore Runtime, and I test it with a tiny Streamlit UI. No MCP tools yet; the agent talks to the KB directly.

## What This Does
- RAG using Bedrock Knowledge Bases (S3 -> KB -> retrieve chunks).
- Strands provides the agent/LLM layer (model selected via environment variables for now as it's only for testing purposes).
- AgentCore Runtime hosts the agent and exposes an `/invocations` HTTP endpoint.
- Streamlit UI offers quick manual testing.
- Separate ingestion/sync script indexes S3 docs into the KB (mirrors the AWS tutorial flow).
- Optional webhook callback lets a UI/backend receive results asynchronously.

## Repo Layout
```
.
|-- src/
|   |-- strands_kb_agent.py      # AgentCore entry point; Strands LLM + Bedrock KB retrieval
|   |-- kb_ingest_sync.py        # One-shot S3 -> KB sync/ingestion helper
|   `-- app_streamlit.py         # Minimal chat UI to call the agent
|-- requirements.txt             # Runtime dependencies (agent)
|-- .env.example                 # Bash-style environment example
```

## How the Pieces Fit
```
[Docs in S3 bucket/prefix] --(sync)--> [Bedrock Knowledge Base index]
                                          ^
                                          |
                                   retrieve()/RAG
                                          |
                               [Agent (Strands) on AgentCore]
                                          ^
                                          |
                         HTTP POST { prompt } -> /invocations
                                          ^
                                          |
                             [Streamlit UI or futur backend]
```

- Upload files to S3, then run a sync so Bedrock indexes them into the KB.
- The agent retrieves KB chunks and uses Strands to synthesize an answer with inline references.
- The Streamlit UI calls the agent's `/invocations` endpoint and renders `{answer, citations}`.

## Prerequisites
- Python 3.10+
- AWS account with access to Amazon Bedrock, S3, and (optionally) SSM Parameter Store
- Knowledge Base created in the same Region as the S3 data source
- AWS CLI configured (for uploads and optional SigV4 auth from the UI)
- AgentCore CLI installed and authenticated

## Install

**Virtual Environnement**
```bash
python3 -m venv venv
```

**On Linux**
```bash
source venv/bin/activate
pip install --upgrade pip
```

**On Windows**
```bash
venv/bin/activate
pip install --upgrade pip
```

**Runtime (agent) dependencies**
```bash
pip install -r requirements.txt
```

## Environment Variables

Examples live in `.env.example` (Bash).

**Agent / runtime**
- `AWS_REGION` - Region of KB/S3 (`eu-francfort-1` for now as it is the only available region in Europe for AgentCore)
- `KB_ID` - Knowledge Base ID (Given In AWS)
- `MODEL_ID` - Bedrock text model for Strands (`anthropic.claude-sonnet-4-20250514-v1:0` for now)
- `TOP_K` (default 8), `MIN_SCORE` (default 0.0)
- `WEBHOOK_SECRET` (optional) - HMAC-sign webhook callbacks (Not clear how it works yet in my mind, keep it here for futur reminder.)

**Ingestion helper**
- `USE_SSM_PARAMS=1` to read KB/DataSource IDs from SSM, or `0` to pass them as environment variables
- `SSM_PARAM_PREFIX` (defaults to `/{account}-{region}/kb`)
- If `USE_SSM_PARAMS=0`: set `KB_ID`, `DATA_SOURCE_ID`, and optional `S3_BUCKET`

**Streamlit UI**
- `AGENT_URL` - the runtime invoke URL (or set it in the sidebar)
- For IAM-protected endpoints from the UI: the machine needs AWS credentials; choose Auth = AWS SigV4 in the sidebar

Bash: `source .env`

## General Informations

The documents ingestion and Agent Deployment should be done only for the first launch. Once they are in AWS, there is no need to redo the staps as it will be there until deletion. The document sync should be done when updating the bucket to update the KB. Same thing for futur agent updates. 
For agent updates, once an agent is deployed, it is immutable. The agent default to the last pushed version, but you need to delete by hand the previous version to avoid cost duplicate. 

## Ingesting Documents (S3 -> KB)

Upload docs:

For now, I need to Ask Guillaume to do it until we have a better understanding on how it works.

For future manual ingestions
```bash
aws s3 cp ./docs/ s3://<bucket>/kb/ --recursive
```
Or drop files directly inside the bucket in the UI.

Run a sync (indexing):

**Using SSM params (recommended)**
```bash
export USE_SSM_PARAMS=1
python kb_ingest_sync.py --sync
```

**Direct IDs**
```bash
export USE_SSM_PARAMS=0
export KB_ID=kb-xxxxxxxxxxxxxxxx
export DATA_SOURCE_ID=ds-xxxxxxxxxxxxxxxx
python kb_ingest_sync.py --sync
```

Wait for `COMPLETE`. Sync is incremental, only changes are reindexed.  
For scanned/image-heavy PDFs that need table/chart extraction, We need to enable the Data Automation parser on the KB data source. (either BDA parser or Model assisted).

## Deploying the Agent to AgentCore Runtime

Configure (one time or when entry point/requirements change):
```bash
agentcore configure -e strands_kb_agent.py -rf requirements.txt -r eu-francfort-1
```

Launch a new version with environment variables:
```bash
agentcore launch \
  --env AWS_REGION=eu-francfort-1 \
  --env KB_ID=kb-xxxxxxxxxxxxxxxx \
  --env MODEL_ID=anthropic.claude-sonnet-4-20250514-v1:0 \
  --env TOP_K=8 \
  --env MIN_SCORE=0.0
```

Fetch the invoke URL:
```bash
agentcore status -v
```

Fast test:
```bash
curl -X POST "<INVOKE_URL>" -H "Content-Type: application/json" \
  -d '{"prompt":"hello"}'
```

## Streamlit UI

Run:
```bash
streamlit run app_streamlit.py
```

Sidebar:
- Paste the Agent URL (invoke URL from AgentCore)
- Choose Auth = None or AWS SigV4 (if protected)
- Tweak `top_k` and `min_score` as needed

The UI POSTs:
```json
{ "prompt": "...", "top_k": 8, "min_score": 0.0, "polish": false }
```

and renders `{ "answer", "citations" }`.

## Optional Webhook Callbacks

- Include a `callback_url` in the agent request. The agent POSTs the result to that URL and returns `{ accepted: true, job_id }`.
- If `WEBHOOK_SECRET` is set, the agent adds `X-Signature: sha256=...` (HMAC).

Example payload:
```json
{
  "job_id": "uuid",
  "prompt": "your question",
  "answer": "grounded answer ...",
  "citations": [{ "ref": 1, "title": "Doc", "url": "...", "score": 0.87 }],
  "mode": "retrieve",
  "top_k": 8,
  "min_score": 0.0,
  "ts": 1710000000
}
```

## Switching the LLM Provider (Optional)

By default the agent uses Bedrock via Strands. To switch providers, change the Strands model class (e.g., install `strands-agents[openai]`) and control it with environment variables; KB retrieval stays on Bedrock.
We can also attach a model from Ollama but there is a need to setup the connection part.

## Invoking the Agent
Below is an example in python of a code snippet that can invoke the agent.
```
import boto3, json
client = boto3.client("bedrock-agentcore", region_name="eu-central-1")
payload = json.dumps({"input": {"prompt": "Hello"}})
resp = client.invoke_agent_runtime(
    agentRuntimeArn="AGENT_ARN",
    runtimeSessionId="x"*34,
    payload=payload,
    qualifier="DEFAULT",
)
print(json.loads(resp["response"].read()))
```

If we need to have a more open way to send the prompt, we could change the payload but I would not recommand it as keeping the current form normalize the way we pass the query.

## Security & IAM (Quick Notes)
- Keep KB, S3, and the agent in the same Region for latency/cost.
- Restrict S3 read to the KB role; restrict invocations to your callers (API Gateway/ALB/IAM).
- For multi-tenant access control, tag metadata at ingestion and enforce filters on every retrieval call.
- For webhooks, verify `X-Signature` with the shared secret on your receiver.

## Cost Basics (Non-LLM)
- S3: storage + PUT/GET requests (tiny per file)
- Ingestion: embeddings; optional Data Automation per-page parsing if enabled
- Vector backend: managed vector stores may have a monthly baseline
- Retrieval: small marginal cost; reranking (if enabled) is per query
- Start simple (no rerank, scheduled sync), measure, then tune.

## Troubleshooting
- No results -> did you run a sync after uploading? KB ID correct? Same Region?
- Auth errors -> if using IAM on the endpoint, the UI machine needs AWS credentials; choose AWS SigV4 in the sidebar.
- Latency -> co-locate S3/KB/agent in one Region.
- Fallback to `retrieve_and_generate` -> ensure the KB has a default model/inference profile, or rely on retrieve + Strands for generation.

## Roadmap
- Add memory, both session and user wise.
- Add identification through AWS, might be needed for the memory part.
- Create a MCP server and start adding tools.
- Move KB access behind MCP tools (`kb.search`, `kb.sync`) for centralized policy/cost controls and easy backend swaps.
- Automate the doc sync, either by syncing when there is an update, or on a fixed time.
- Connect the agent to a (?) database through SQL read-only querying. 
- Understand the Gateway control, the way the Security works.
- Connect the Observability to Astana (I suppose we go for Astana for Observability).
- Create tools that can interact with the frontend.
- The Agent2Agent protocol will be integrated in the future, I will need to keep an eye out for that feature since it might be needed.

## Credits
- Amazon Bedrock (Knowledge Bases & Agent Runtime) 
https://docs.aws.amazon.com/
- Strands Agents SDK
https://strandsagents.com/latest/documentation/docs/
- AgentCore Runtime
https://github.com/awslabs/amazon-bedrock-agentcore-samples/tree/main/01-tutorials/07-AgentCore-E2E

- Streamlit

## Additionnal informations

It turns out that we can use AgentCore locally if we decide to change our approach. Below are a list of accessible features on a local setup compared to a cloud setup:

### What works locally (Python or Docker)

- The entrypoint (@app.entrypoint) and all the code paths
- Bedrock KB retrieval (it still calls AWS over the network using my local AWS creds)
- LLM generation (Bedrock via Strands, or any external HTTP LLM)
- Streaming responses (over the local HTTP server)
- Webhooks (the agent can POST to an URL)
- Config via env vars (what the code reads with os.getenv)
- Local testing tools: agentcore launch --local, Docker Desktop, curl/Postman, to either Streamlit UI or a Frontend in general.

### What is cloud-only (not in local mode)

- Managed HTTPS endpoint (public URL)
- IAM/OAuth edge authorization (SigV4/OIDC enforced at the gateway) --> I need to understand this part.
- Versioned deployments & DEFAULT endpoint switching
- Autoscaling / HA (long-running or concurrent sessions at scale)
- Cloud build & image management (CodeBuild/ECR)
- Cloud observability (CloudWatch logs/metrics out-of-the-box)  ---> Need to see how we can redirect this to Astana, if it is necessary.
- Execution role separation (local uses the user creds; cloud uses the task’s IAM role)
- Header allowlist/gateway behaviors we should be able to configure during agentcore configure
- “Memory” backends managed by the platform (short-term chat context in-process is fine; long-term memory extraction/storage may rely on hosted components)
