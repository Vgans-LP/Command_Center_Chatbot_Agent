# -------------------------------------------------------------------
# Local MCP server that wraps Bedrock KB retrieval.
# For now it only exposes a single tool `kb.search` where `kb_id` and `lang` are optional.
# The other tools will probably be added here later.
# The language is auto-detecte and routed to the right KB when it is omitted.
# -------------------------------------------------------------------

from fastapi import FastAPI, Header
from pydantic import BaseModel
from typing import Literal, Dict, Any, List, Optional
import boto3, os, re
from functools import lru_cache
from botocore.config import Config
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # deterministic language detection

# Config / Env
REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "eu-central-1"))

KB_GENERAL_DOCS_ID = os.getenv("KB_GENERAL_DOCS_ID", "REPLACE")
KB_CS_SUPPORT_ID   = os.getenv("KB_CS_SUPPORT_ID",   "REPLACE")

TOP_K_DEFAULT = int(os.getenv("TOP_K", "8"))
MIN_SCORE_DEFAULT = float(os.getenv("MIN_SCORE", "0.0"))

# Simple shared secret to allow only trusted callers (e.g. the Gateway)
# If set, must include header: X-Api-Key: <value>
API_KEY = os.getenv("MCP_SERVER_API_KEY")

KB_IDS = {
    "general-docs": KB_GENERAL_DOCS_ID,
    "cs-support":   KB_CS_SUPPORT_ID,
}

LANG_ENUM = ["en", "fr", "de", "zh-Hant"]
SUPPORTED = set(LANG_ENUM)

runtime = boto3.client(
    "bedrock-agent-runtime",
    region_name=REGION,
    config=Config(retries={"max_attempts": 3, "mode": "adaptive"}),
)

app = FastAPI(title="KB MCP Server", version="1.0.0")

# Helpers
def _normalize_lang(code: Optional[str]) -> str:
    if not code:
        return "en"
    code = code.lower()
    if code.startswith("zh"):
        return "zh-Hant"
    if code in {"en", "fr", "de"}:
        return code
    return "en"  # fallback for any other language. 

def _detect_lang(text: str) -> str:
    try:
        raw = detect(text or "")
    except Exception:
        raw = None
    return _normalize_lang(raw)

def _auto_route_kb(lang: Optional[str], text: str) -> str:
    """
    Simple routing:
      - If language != en -> general-docs
      - If English:
          If troubleshooting/support-ish text -> cs-support
          Else -> general-docs
    """
    if lang and lang != "en":
        return "general-docs"
    t = (text or "").lower()
    support_signals = [
        "issue", "error", "didn't", "doesn't", "failed", "cannot", "can't",
        "how do i fix", "troubleshoot", "support", "screen", "label", "rma",
        "ticket", "won't", "not working", "handle that", "broken"
    ]
    if any(s in t for s in support_signals):
        return "cs-support"
    return "general-docs"

def _clean_item(it: Dict[str, Any]) -> Dict[str, Any]:
    content = (it.get("content") or {})
    meta    = (it.get("metadata") or {})
    return {
        "score":  it.get("score"),
        "text":   content.get("text"),
        "source": it.get("location"),  # to include s3 info
        "metadata": meta,
        "title": meta.get("title") or meta.get("file") or meta.get("source"),
        "url":   meta.get("url") or meta.get("source"),
    }

def _post_filter_lang(items: List[Dict[str, Any]], kb_key: str, lang: Optional[str]) -> List[Dict[str, Any]]:
    if not lang or kb_key == "cs-support":
        return items
    out = []
    for it in items:
        md = it.get("metadata") or {}
        item_lang = (md.get("lang") or md.get("language") or "").strip()
        if not item_lang:
            out.append(it) # If KB lacks language metadata, the model judge relevance.
        elif item_lang == lang:
            out.append(it)
    return out

@lru_cache(maxsize=16)
def _kb_enum() -> List[str]:
    return list(KB_IDS.keys())

@app.get("/health")
def health():
    return {
        "region": REGION,
        "general_docs_set": KB_GENERAL_DOCS_ID not in (None, "", "REPLACE"),
        "cs_support_set":   KB_CS_SUPPORT_ID   not in (None, "", "REPLACE"),
        "kb_ids": {
            "general-docs": KB_GENERAL_DOCS_ID,
            "cs-support":   KB_CS_SUPPORT_ID,
        }
    }

# MCP schema endpoints
@app.get("/mcp/tools")
def list_tools():
    """
    Advertise a single tool with optional args; LLM can call with just 'query'.
    """
    return {
        "tools": [
            {
                "name": "kb.search",
                "description": (
                    "Search across Knowledge Bases. If kb_id/lang are omitted, the server auto-detects language "
                    "and routes to the right KB (general-docs vs cs-support)."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query":    {"type": "string", "description": "User question or search text."},
                        "kb_id":    {"type": "string", "enum": _kb_enum(), "description": "Optional. Force a specific KB."},
                        "lang":     {"type": "string", "enum": LANG_ENUM, "description": "Optional. Force language filter."},
                        "top_k":    {"type": "integer", "minimum": 1, "maximum": 50, "default": TOP_K_DEFAULT},
                        "min_score":{"type": "number", "minimum": 0, "maximum": 1, "default": MIN_SCORE_DEFAULT}
                    },
                    "required": ["query"]
                }
            }
        ]
    }

class ToolCall(BaseModel):
    name: Literal["kb.search"]
    arguments: Dict[str, Any]


@app.post("/mcp/call")
def call_tool(call: ToolCall, x_api_key: Optional[str] = Header(default=None)):
    # Optional shared-secret guard (when fronted by Gateway once it is in AWS and not local)
    if API_KEY and x_api_key != API_KEY:
        return {"error": "unauthorized"}

    if call.name != "kb.search":
        return {"error": "unknown tool"}

    args = call.arguments or {}
    q: str = str(args.get("query", "")).strip()
    if not q:
        return {"error": "query is required"}

    lang: Optional[str] = args.get("lang")
    if not lang:
        lang = _detect_lang(q)  # always returns one of the supported language {"en","fr","de","zh-Hant"}
    else:
        lang = _normalize_lang(lang)

    kb_id: Optional[str] = args.get("kb_id")
    if not kb_id:
        kb_key = _auto_route_kb(lang, q)
    else:
        if kb_id not in _kb_enum():
            return {"error": f"Unknown kb_id '{kb_id}'"}
        kb_key = kb_id

    top_k = int(args.get("top_k", TOP_K_DEFAULT))
    min_score = float(args.get("min_score", MIN_SCORE_DEFAULT))

    resp = runtime.retrieve(
        knowledgeBaseId=KB_IDS[kb_key],
        retrievalQuery={"text": q},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": top_k}},
    )
    items = [_clean_item(it) for it in resp.get("retrievalResults", []) or []]
    if min_score > 0:
        items = [it for it in items if (it.get("score") or 0) >= min_score]
    items = _post_filter_lang(items, kb_key, lang)

    return {
        "content": items,
        "routing": {"kb_id": kb_key, "lang": lang, "top_k": top_k, "min_score": min_score}
    }
