from __future__ import annotations

import os
import json
import time
import uuid
import hmac
import hashlib
from typing import Any, Dict

import requests
import boto3
from botocore.config import Config

from strands import Agent as StrandsAgent
from strands.tools.mcp import MCPToolset

from bedrock_agentcore import BedrockAgentCoreApp

# Config / Env
REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "eu-central-1"))

# local: http://127.0.0.1:7000
# If pushed: AgentCore MCP-Gateway URL that fronts the MCP servers
MCP_URL = os.getenv("MCP_URL", "http://127.0.0.1:7000")

# If the MCP server expects a shared secret. The MCP client will pass it as X-Api-Key
MCP_API_KEY = os.getenv("MCP_API_KEY")  

# Model only used for fallback 
MODEL_ID = os.getenv("MODEL_ID")
if not MODEL_ID:
    raise RuntimeError("Missing MODEL_ID (e.g., anthropic.claude-sonnet-4-20250514-v1:0).")

# Bedrock Runtime client for fallback generation
brt = boto3.client(
    "bedrock-runtime",
    region_name=REGION,
    config=Config(retries={"max_attempts": 3, "mode": "adaptive"}),
)

PORT = int(os.getenv("PORT", "8080"))

SYSTEM_PROMPT = (
    "You are a precise enterprise assistant for a company called LivinPackets.\n"
    "- You have MCP tools available; ALWAYS try `kb.search` before answering.\n"
    "- Call `kb.search` with just the user's question; it can auto-detect language "
    "  and route to the right Knowledge Base (general-docs vs cs-support). You MAY provide kb_id/lang "
    "  if you are certain (kb_id ∈ {general-docs, cs-support}; lang ∈ {en, fr, de, zh-Hant}).\n"
    "- Answer ONLY using retrieved Knowledge Base information. Keep it concise and add inline citations like [1], [2] "
    "  in the order of sources you relied on.\n"
    "- If nothing relevant is found, say so clearly and suggest a follow-up.\n"
)

# MCP Tool wiring
# The MCPToolset supports headers, pass X-Api-Key if provided
mcp_server_cfg = {
    "name": "kb-tools",
    "transport": "http",
    "url": MCP_URL,
}

if MCP_API_KEY:
    mcp_server_cfg["headers"] = {"X-Api-Key": MCP_API_KEY}

mcp = MCPToolset(servers=[mcp_server_cfg])

agent = StrandsAgent(tools=[mcp], system_prompt=SYSTEM_PROMPT)

# Bedrock fallback 
def _bedrock_generate(prompt: str) -> str:
    """
    Safety net if Strands invocation fails. Implements Anthropic Messages on Bedrock.
    """
    if MODEL_ID.startswith("anthropic."):
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        }
        resp = brt.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(resp["body"].read().decode("utf-8"))
        parts = payload.get("content") or []
        for p in parts:
            if p.get("type") == "text" and "text" in p:
                return str(p["text"]).strip()
        return str(payload).strip()
    else:
        raise RuntimeError(
            f"Bedrock fallback implemented for Anthropic models only. MODEL_ID={MODEL_ID}"
        )

# ---------------- AgentCore entrypoint ----------------
app = BedrockAgentCoreApp()

def _sign_payload(secret: str, body: str) -> str:
    mac = hmac.new(secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"sha256={mac}"

def _post_webhook(callback_url: str, result: dict) -> None:
    body = json.dumps(result, separators=(",", ":"))
    headers = {"Content-Type": "application/json"}
    secret = os.getenv("WEBHOOK_SECRET")  # shared secret for webhook consumer
    if secret:
        headers["X-Signature"] = _sign_payload(secret, body)
    r = requests.post(callback_url, data=body, headers=headers, timeout=15)
    r.raise_for_status()

@app.entrypoint
def invoke(payload: Dict[str, Any]) -> Dict[str, Any]:
    query = (payload.get("prompt") or payload.get("input") or "").strip()
    if not query:
        return {"error": "Missing 'prompt' in payload"}

    job_id = payload.get("job_id") or str(uuid.uuid4())
    callback_url = payload.get("callback_url")

    try:
        res = agent(query)
        answer = res if isinstance(res, str) else getattr(res, "message", str(res))
        result = {
            "job_id": job_id,
            "prompt": query,
            "answer": str(answer).strip(),
            "ts": int(time.time()),
            "mode": "agent-with-mcp"
        }
    except Exception:
        answer = _bedrock_generate(query)
        result = {
            "job_id": job_id,
            "prompt": query,
            "answer": answer,
            "ts": int(time.time()),
            "mode": "fallback-bedrock"
        }

    if callback_url:
        try:
            _post_webhook(callback_url, result)
            return {"accepted": True, "job_id": job_id}
        except Exception as e:
            result["callback_error"] = str(e)[:200]

    return result

if __name__ == "__main__":
    app.run(port=PORT)
