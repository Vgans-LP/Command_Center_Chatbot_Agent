import os
import json
import time
import requests
import streamlit as st

# Optional AWS SigV4 (comment out if you don't need it)
try:
    from requests_aws4auth import AWS4Auth  # pip install requests-aws4auth
    import boto3
except Exception:
    AWS4Auth = None
    boto3 = None


# ------------------------ UI CONFIG ------------------------
st.set_page_config(page_title="AgentCore KB Chat", page_icon="🤖", layout="centered")

st.title("🤖 AgentCore • KB Chat")

with st.sidebar:
    st.header("Settings")

    # Agent URL: works with local dev (http://localhost:8080/invocations) or your deployed HTTPS endpoint
    agent_url = st.text_input(
        "Agent URL",
        value=os.getenv("AGENT_URL", "http://localhost:8080/invocations"),
        help="POST JSON here: { 'prompt': '...' }"
    )

    # Retrieval knobs (forwarded to your agent payload)
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.number_input("Top-K", min_value=1, max_value=50, value=int(os.getenv("TOP_K", "8")))
    with col2:
        min_score = st.number_input("Min score", min_value=0.0, max_value=1.0, value=float(os.getenv("MIN_SCORE", "0.0")))

    polish = st.toggle("Polish answer (agent option)", value=False)

    st.divider()

    # Auth mode
    auth_mode = st.selectbox("Auth", ["None", "AWS SigV4"])
    aws_service = st.text_input("AWS Service (SigV4)", value=os.getenv("AWS_SERVICE", "execute-api"))
    aws_region = st.text_input("AWS Region", value=os.getenv("AWS_REGION", "eu-west-3"))

    st.caption("Tip: If your endpoint is public and needs no AWS auth, leave Auth=None.")

    st.divider()
    show_raw = st.toggle("Show raw JSON responses", value=False)

# ------------------------ STATE ------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: {role: "user"/"assistant", content: str, citations: [...]}

def post_json(url: str, payload: dict) -> requests.Response:
    """POST JSON with optional AWS SigV4 signing."""
    headers = {"Content-Type": "application/json"}

    if auth_mode == "AWS SigV4":
        if AWS4Auth is None or boto3 is None:
            raise RuntimeError("requests-aws4auth and boto3 are required for SigV4. Install with: pip install requests-aws4auth boto3")
        session = boto3.Session(region_name=aws_region or None)
        creds = session.get_credentials()
        if creds is None:
            raise RuntimeError("No AWS credentials found (profile/ENV). Configure AWS CLI or env vars.")
        awsauth = AWS4Auth(creds.access_key, creds.secret_key, aws_region, aws_service, session_token=creds.token)
        return requests.post(url, headers=headers, data=json.dumps(payload), auth=awsauth, timeout=60)
    else:
        return requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)


def render_citations(citations):
    if not citations:
        return
    with st.expander("Citations"):
        for i, c in enumerate(citations, start=1):
            title = c.get("title") or c.get("ref") or "Source"
            url = c.get("url")
            score = c.get("score")
            line = f"[{i}] {title}"
            if url:
                line += f" — {url}"
            if score is not None:
                line += f" (score: {score:.3f})"
            st.write(line)


# ------------------------ CHAT UI ------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_citations(msg.get("citations"))

user_input = st.chat_input("Ask something grounded in your KB…")
if user_input:
    # show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # compose payload expected by your AgentCore entrypoint
    payload = {
        "prompt": user_input,
        "top_k": top_k,
        "min_score": min_score,
        "polish": polish,
    }

    # call agent
    try:
        resp = post_json(agent_url, payload)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Request failed: {e}")
        st.stop()

    # Figure out the shape of the response (your agent typically returns {answer, citations})
    answer = data.get("answer") or data.get("result", {}).get("answer") or json.dumps(data, indent=2)
    citations = data.get("citations") or data.get("result", {}).get("citations") or []

    with st.chat_message("assistant"):
        st.markdown(answer)
        render_citations(citations)
        if show_raw:
            st.code(json.dumps(data, indent=2), language="json")

    st.session_state.messages.append({"role": "assistant", "content": answer, "citations": citations})
