from __future__ import annotations

import os
import json
from typing import Any, Dict, List
import requests, time, uuid

import boto3
from botocore.config import Config

# Strands agent/LLM
from strands import Agent as StrandsAgent
from strands.models import BedrockModel as StrandsBedrockModel

# AgentCore runtime wrapper
from bedrock_agentcore import BedrockAgentCoreApp



# -----------------------------
# Config in .env
# -----------------------------
REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-west-2"))
KB_ID = os.getenv("KB_ID")
MODEL_ID = os.getenv("MODEL_ID")
TOP_K = int(os.getenv("TOP_K", "8"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.0"))
PORT = int(os.getenv("PORT", "8080"))

if not KB_ID:
    raise RuntimeError("Missing KB_ID. Export your Bedrock Knowledge Base ID.")
if not MODEL_ID:
    raise RuntimeError("Missing MODEL_ID. Export your Bedrock model id for Strands.")

# Bedrock Agent Runtime client for KB retrieval
runtime = boto3.client(
    "bedrock-agent-runtime",
    region_name=REGION,
    config=Config(retries={"max_attempts": 3, "mode": "adaptive"}),
)

# Strands agent as the primary LLM
SYSTEM_PROMPT = (
    "You are a precise enterprise assistant. You receive: (a) the user's question, "
    "(b) KB excerpts, and (c) optional pre-baked KB answer text. Answer ONLY using "
    "the KB information. Add short inline citations like [1], [2]. If nothing relevant "
    "is found, say so clearly and suggest a follow-up. Keep answers concise."
)
strands_model = StrandsBedrockModel(model_id=MODEL_ID, region_name=REGION, temperature=0.2, top_p=0.9)
agent = StrandsAgent(model=strands_model, system=SYSTEM_PROMPT)


# -----------------------------
# KB Retrieval helpers
# -----------------------------
def kb_retrieve(query: str, *, top_k: int, min_score: float) -> Dict[str, Any]:
    """Retrieve KB chunks WITHOUT generation using `retrieve`.
    Falls back to `retrieve_and_generate` when `retrieve` is unavailable.
    Returns a dict: {chunks, citations, rag_text, mode}
    """
    # Primary path: KB retrieve (no generation)
    try:
        resp = runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": top_k,
                }
            },
        )
        # Parse chunks
        chunks: List[Dict[str, Any]] = []
        for item in resp.get("retrievalResults", []) or []:
            md = item.get("metadata", {})
            text = (item.get("content") or {}).get("text") or ""
            score = item.get("score")
            if score is not None and score < min_score:
                continue
            chunks.append({
                "text": text,
                "score": score,
                "title": md.get("title") or md.get("file") or md.get("source"),
                "url": md.get("url") or md.get("source"),
            })
        return {"chunks": chunks, "citations": chunks, "rag_text": None, "mode": "retrieve"}
    except Exception:
        # Fallback: RAG (retrieve_and_generate)
        rag = runtime.retrieve_and_generate(
            input={"text": query},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": KB_ID,
                    # When using retrieve_and_generate without explicit modelArn,
                    # the KB's default model/inference profile may be used.
                },
            },
        )
        out_text = ((rag.get("output") or {}).get("text") or "").strip()
        cites: List[Dict[str, Any]] = []
        for c in (rag.get("citations") or []):
            for ref in c.get("retrievedReferences", []):
                md = ref.get("metadata", {})
                cites.append({
                    "text": (ref.get("content", {}) or {}).get("text"),
                    "score": ref.get("score"),
                    "title": md.get("title") or md.get("file") or md.get("source"),
                    "url": md.get("url") or md.get("source"),
                })
        return {"chunks": cites, "citations": cites, "rag_text": out_text, "mode": "rag"}


def synthesize_with_strands(user_query: str, chunks: List[Dict[str, Any]], pre_baked: str | None) -> str:
    # Build references block for the LLM
    ref_lines = []
    for i, ch in enumerate(chunks, start=1):
        title = ch.get("title") or "Source"
        url = ch.get("url") or ""
        snippet = (ch.get("text") or "").replace("\n", " ")
        ref_lines.append(f"[{i}] {title} {('- ' + url) if url else ''}\n{snippet[:400]}")
    refs = "\n\n".join(ref_lines) if ref_lines else "(no citations)"

    baked = pre_baked or ""
    prompt = (
        f"User question:\n{user_query}\n\n"
        f"Knowledge base excerpts:\n{refs}\n\n"
        f"If there is a pre-baked KB answer, it follows between <kb_answer> tags.\n"
        f"<kb_answer>\n{baked}\n</kb_answer>\n\n"
        "Write the best possible answer using ONLY the KB information. Keep it concise, and add inline [n] citations.\n"
    )
    out = agent.tool.use_llm(prompt=prompt)
    return str(out).strip()


# ---------------- AgentCore entrypoint ----------------
app = BedrockAgentCoreApp()

def _sign_payload(secret: str, body: str) -> str:
    mac = hmac.new(secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"sha256={mac}"

def _post_webhook(callback_url: str, result: dict) -> None:
    body = json.dumps(result, separators=(",", ":"))
    headers = {"Content-Type": "application/json"}
    secret = os.getenv("WEBHOOK_SECRET")  # optional shared secret
    if secret:
        headers["X-Signature"] = _sign_payload(secret, body)
    r = requests.post(callback_url, data=body, headers=headers, timeout=15)
    r.raise_for_status()


@app.entrypoint
def invoke(payload: Dict[str, Any]) -> Dict[str, Any]:
    # --- Inputs & knobs (kept from your version)
    query = (payload.get("prompt") or payload.get("input") or "").strip()
    if not query:
        return {"error": "Missing 'prompt' in payload"}

    top_k = int(payload.get("top_k", TOP_K))
    min_score = float(payload.get("min_score", MIN_SCORE))
    callback_url = payload.get("callback_url")  # <- new (optional)
    job_id = payload.get("job_id") or str(uuid.uuid4())

    # --- Retrieve from KB (same as before)
    ret = kb_retrieve(query, top_k=top_k, min_score=min_score)
    chunks: List[Dict[str, Any]] = ret["chunks"]
    pre = ret.get("rag_text")  # may be None when using pure retrieve

    # --- No results path (same semantics)
    if not chunks and not pre:
        result = {
            "job_id": job_id,
            "prompt": query,
            "answer": "I couldn't find anything relevant in the knowledge base for that.",
            "citations": [],
            "mode": ret.get("mode"),
            "top_k": top_k,
            "min_score": min_score,
            "ts": int(time.time()),
        }
        if callback_url:
            try:
                _post_webhook(callback_url, result)
                return {"accepted": True, "job_id": job_id}
            except Exception as e:
                result["callback_error"] = str(e)[:200]
        return result

    # --- Synthesis with Strands (unchanged behavior)
    answer = synthesize_with_strands(query, chunks, pre)

    # --- Compact citations array (same shape)
    citations = []
    for i, ch in enumerate(chunks, start=1):
        citations.append({
            "ref": i,
            "title": ch.get("title") or "Source",
            "url": ch.get("url"),
            "score": ch.get("score"),
        })

    # --- Final result
    result = {
        "job_id": job_id,
        "prompt": query,
        "answer": answer,
        "citations": citations,
        "mode": ret.get("mode"),
        "top_k": top_k,
        "min_score": min_score,
        "ts": int(time.time()),
    }

    # --- Optional webhook callback
    if callback_url:
        try:
            _post_webhook(callback_url, result)
            return {"accepted": True, "job_id": job_id}
        except Exception as e:
            # Fall back to returning the result inline if webhook fails
            result["callback_error"] = str(e)[:200]
            return result

    # --- Inline response (no webhook)
    return result


if __name__ == "__main__":
    app.run(port=PORT)
