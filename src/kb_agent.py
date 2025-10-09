from __future__ import annotations

import os
import json
import time
import uuid
import hmac
import hashlib
from typing import Any, Dict, List, Optional

import requests
import boto3
from botocore.config import Config

# Strands: simple callable agent
from strands import Agent as StrandsAgent

# AgentCore runtime wrapper
from bedrock_agentcore import BedrockAgentCoreApp


# -----------------------------
# Config / Env
# -----------------------------
REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "eu-central-1"))
KB_ID = os.getenv("KB_ID")
MODEL_ID = os.getenv("MODEL_ID")  # used for Bedrock fallback (and possibly by Strands if it reads env)
TOP_K = int(os.getenv("TOP_K", "8"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.0"))
PORT = int(os.getenv("PORT", "8080"))

# Optional: allow reading KB_ID from SSM if not provided via env
if not KB_ID and os.getenv("USE_SSM_PARAMS", "0") == "1":
    ssm = boto3.client("ssm", region_name=REGION)
    sts = boto3.client("sts")
    acct = sts.get_caller_identity()["Account"]
    prefix = os.getenv("SSM_PARAM_PREFIX", f"/{acct}-{REGION}/kb")
    try:
        KB_ID = ssm.get_parameter(Name=f"{prefix}/knowledge-base-id")["Parameter"]["Value"]
    except Exception as e:
        raise RuntimeError(
            f"KB_ID is not set and could not be read from SSM at {prefix}/knowledge-base-id: {e}"
        )

if not KB_ID:
    raise RuntimeError("Missing KB_ID. Set env KB_ID or enable SSM (USE_SSM_PARAMS=1).")
if not MODEL_ID:
    # Only needed for the Bedrock fallback; Strands may pick a default model via its own config.
    # We'll keep it required so both paths are well-defined.
    raise RuntimeError("Missing MODEL_ID. Set your Bedrock model id (e.g., anthropic.claude-3-5-sonnet-20240620-v1:0).")

# Bedrock Agent Runtime client for KB retrieval
runtime = boto3.client(
    "bedrock-agent-runtime",
    region_name=REGION,
    config=Config(retries={"max_attempts": 3, "mode": "adaptive"}),
)

# Bedrock Runtime client for LLM fallback (generation)
brt = boto3.client(
    "bedrock-runtime",
    region_name=REGION,
    config=Config(retries={"max_attempts": 3, "mode": "adaptive"}),
)

# Strands agent — simple callable
# (Strands typically reads provider/model from its own config or env; we won't over-configure it here.)
agent = StrandsAgent()

SYSTEM_PROMPT = (
    "You are a precise enterprise assistant. You receive: (a) the user's question, "
    "(b) KB excerpts, and (c) optional pre-baked KB answer text. Answer ONLY using "
    "the KB information. Add short inline citations like [1], [2]. If nothing relevant "
    "is found, say so clearly and suggest a follow-up. Keep answers concise."
)


# -----------------------------
# KB Retrieval helpers
# -----------------------------
def kb_retrieve(query: str, *, top_k: int, min_score: float) -> Dict[str, Any]:
    """
    Retrieve KB chunks WITHOUT generation using `retrieve`.
    Falls back to `retrieve_and_generate` when `retrieve` is unavailable.
    Returns a dict: {chunks, citations, rag_text, mode}
    """
    # Primary path: KB retrieve (no generation)
    try:
        resp = runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
            retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": top_k}},
        )
        chunks: List[Dict[str, Any]] = []
        for item in resp.get("retrievalResults", []) or []:
            md = item.get("metadata", {})
            text = (item.get("content") or {}).get("text") or ""
            score = item.get("score")
            if score is not None and score < min_score:
                continue
            chunks.append(
                {
                    "text": text,
                    "score": score,
                    "title": md.get("title") or md.get("file") or md.get("source"),
                    "url": md.get("url") or md.get("source"),
                }
            )
        return {"chunks": chunks, "citations": chunks, "rag_text": None, "mode": "retrieve"}
    except Exception:
        # Fallback: RAG (retrieve_and_generate)
        model_arn = f"arn:aws:bedrock:{REGION}::foundation-model/{MODEL_ID}"
        rag = runtime.retrieve_and_generate(
            input={"text": query},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": KB_ID,
                    "modelArn": model_arn,   # <-- required in your account
                },
            },
        )
        out_text = ((rag.get("output") or {}).get("text") or "").strip()
        cites: List[Dict[str, Any]] = []
        for c in (rag.get("citations") or []):
            for ref in c.get("retrievedReferences", []):
                md = ref.get("metadata", {})
                cites.append(
                    {
                        "text": (ref.get("content", {}) or {}).get("text"),
                        "score": ref.get("score"),
                        "title": md.get("title") or md.get("file") or md.get("source"),
                        "url": md.get("url") or md.get("source"),
                    }
                )
        return {"chunks": cites, "citations": cites, "rag_text": out_text, "mode": "rag"}


# -----------------------------
# LLM generation
# -----------------------------
def _bedrock_generate(prompt: str) -> str:
    """
    Safety net if Strands isn't callable: call Bedrock Runtime directly (Anthropic Messages).
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


def synthesize_with_strands(user_query: str, chunks: List[Dict[str, Any]], pre_baked: Optional[str]) -> str:
    # Build references block for the LLM
    ref_lines: List[str] = []
    for i, ch in enumerate(chunks, start=1):
        title = ch.get("title") or "Source"
        url = ch.get("url") or ""
        snippet = (ch.get("text") or "").replace("\n", " ")
        ref_lines.append(f"[{i}] {title} {('- ' + url) if url else ''}\n{snippet[:400]}")
    refs = "\n\n".join(ref_lines) if ref_lines else "(no citations)"

    baked = pre_baked or ""
    prompt = (
        "SYSTEM:\n" + SYSTEM_PROMPT + "\n\n"
        f"USER QUESTION:\n{user_query}\n\n"
        f"KB EXCERPTS:\n{refs}\n\n"
        "If there is a pre-baked KB answer, it follows between <kb_answer> tags.\n"
        f"<kb_answer>\n{baked}\n</kb_answer>\n\n"
        "Write the best possible answer using ONLY the KB information. Keep it concise, and add inline [n] citations.\n"
    )

    # Preferred path: Strands callable Agent
    try:
        res = agent(prompt)
        # common return shapes: str, or object with .message
        if isinstance(res, str):
            return res.strip()
        msg = getattr(res, "message", None)
        if msg is not None:
            return str(msg).strip()
        return str(res).strip()
    except Exception:
        # Fallback to Bedrock Runtime if Strands call fails for any reason
        return _bedrock_generate(prompt)


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
    # Inputs & knobs
    query = (payload.get("prompt") or payload.get("input") or "").strip()
    if not query:
        return {"error": "Missing 'prompt' in payload"}

    top_k = int(payload.get("top_k", TOP_K))
    min_score = float(payload.get("min_score", MIN_SCORE))
    callback_url = payload.get("callback_url")  # optional
    job_id = payload.get("job_id") or str(uuid.uuid4())

    # Retrieve from KB
    ret = kb_retrieve(query, top_k=top_k, min_score=min_score)
    chunks: List[Dict[str, Any]] = ret["chunks"]
    pre = ret.get("rag_text")  # may be None when using pure retrieve

    # No results path
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

    # Synthesis with Strands (with Bedrock fallback)
    answer = synthesize_with_strands(query, chunks, pre)

    # Compact citations
    citations = []
    for i, ch in enumerate(chunks, start=1):
        citations.append(
            {
                "ref": i,
                "title": ch.get("title") or "Source",
                "url": ch.get("url"),
                "score": ch.get("score"),
            }
        )

    # Final result
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

    # Optional webhook callback
    if callback_url:
        try:
            _post_webhook(callback_url, result)
            return {"accepted": True, "job_id": job_id}
        except Exception as e:
            result["callback_error"] = str(e)[:200]
            return result

    # Inline response (no webhook)
    return result


if __name__ == "__main__":
    app.run(port=PORT)
