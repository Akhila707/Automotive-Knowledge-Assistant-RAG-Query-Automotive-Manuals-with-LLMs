"""
app.py — Streamlit UI for Automotive Knowledge Assistant
Run: streamlit run app.py
"""

import os
import streamlit as st

# ── Auto-load .env file FIRST before anything else ───────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    # If python-dotenv not installed, manually read .env file
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ[key.strip()] = value.strip()

from rag_pipeline import RAGPipeline

st.set_page_config(page_title="Automotive Knowledge Assistant", page_icon="🔧", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #0d1117; color: #e6edf3; }
section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; color: #f0a500 !important; }
.answer-box { background: #161b22; border: 1px solid #f0a500; border-left: 4px solid #f0a500; border-radius: 6px; padding: 20px 24px; font-size: 15px; line-height: 1.75; color: #e6edf3; }
.chunk-card { background: #1c2128; border: 1px solid #30363d; border-radius: 6px; padding: 14px 18px; margin-bottom: 12px; font-size: 13px; font-family: 'IBM Plex Mono', monospace; color: #8b949e; line-height: 1.65; }
.chunk-header { color: #58a6ff; font-weight: 600; margin-bottom: 8px; font-size: 12px; text-transform: uppercase; }
.score-badge { background: #21262d; border: 1px solid #30363d; border-radius: 3px; padding: 1px 7px; font-size: 11px; color: #3fb950; margin-left: 8px; }
.stTextInput > div > div > input { background-color: #161b22 !important; border: 1px solid #30363d !important; color: #e6edf3 !important; border-radius: 6px !important; font-size: 15px !important; }
.stButton > button { background: #f0a500 !important; color: #0d1117 !important; border: none !important; border-radius: 6px !important; font-weight: 700 !important; font-family: 'IBM Plex Mono', monospace !important; padding: 10px 28px !important; }
hr { border-color: #30363d !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_pipeline(api_key: str):
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
    pipe = RAGPipeline()
    ok, err = pipe.load()
    return (pipe, "") if ok else (None, err)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔧 Configuration")
    st.markdown("---")

    existing_key = os.environ.get("GROQ_API_KEY", "")
    if existing_key:
        st.success("✅ API Key loaded from .env")
    else:
        st.error("❌ No API Key found")

    api_key_input = st.text_input("Groq API Key", type="password", placeholder="gsk_...", help="Get free key at console.groq.com")
    active_key = api_key_input.strip() if api_key_input.strip() else existing_key

    st.markdown("---")
    st.markdown("### 💡 Example Queries")
    for q in ["How does ABS braking system work?", "What causes engine overheating?",
               "How to diagnose fuel injector problems?", "Explain the function of the alternator.",
               "What are common transmission failure symptoms?"]:
        if st.button(q, use_container_width=True, key=q):
            st.session_state["prefill_query"] = q

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# 🚗 Automotive Knowledge Assistant")
st.markdown("<p style='color:#8b949e; font-size:15px; margin-top:-12px;'>RAG-powered technical assistant — ask anything about your vehicle manuals.</p>", unsafe_allow_html=True)
st.markdown("---")

if not active_key:
    st.warning("⚠️ Paste your Groq API key in the sidebar to continue.", icon="🔑")
    st.info("Get a **free** key at https://console.groq.com → API Keys → Create Key", icon="ℹ️")
    st.stop()

pipeline, load_error = get_pipeline(active_key)

if load_error:
    st.error(f"⚠️ {load_error}", icon="🚨")
    if "FAISS" in load_error or "index" in load_error.lower():
        st.code("python ingest.py", language="bash")
        st.info("Run the command above in your terminal, then refresh this page.", icon="ℹ️")
    st.stop()

default_query = st.session_state.pop("prefill_query", "")
col_input, col_btn = st.columns([5, 1])
with col_input:
    user_question = st.text_input("Question", value=default_query, placeholder="e.g. How does the hybrid braking system work?", label_visibility="collapsed")
with col_btn:
    ask_clicked = st.button("Ask ⚡", use_container_width=True)

if ask_clicked and user_question.strip():
    with st.spinner("Searching manuals and generating answer…"):
        result = pipeline.query(user_question.strip())
    if result["error"]:
        st.error(f"Error: {result['error']}")
    else:
        st.markdown("### 💡 Answer")
        st.markdown(f"<div class='answer-box'>{result['answer']}</div>", unsafe_allow_html=True)
        st.markdown("---")
        if result["chunks"]:
            st.markdown("### 📄 Retrieved Context")
            for i, chunk in enumerate(result["chunks"], start=1):
                display_text = chunk["text"][:600] + ("…" if len(chunk["text"]) > 600 else "")
                st.markdown(f"""<div class='chunk-card'><div class='chunk-header'>Chunk {i} &nbsp;·&nbsp; {chunk['source']} &nbsp;·&nbsp; Page {chunk['page']} <span class='score-badge'>sim {chunk['score']*100:.1f}%</span></div>{display_text}</div>""", unsafe_allow_html=True)

elif ask_clicked:
    st.warning("Please enter a question first.", icon="⚠️")
else:
    st.markdown("<div style='text-align:center; padding:60px 20px;'><div style='font-size:64px;'>🔩</div><div style='font-family: IBM Plex Mono, monospace; font-size:18px; color:#484f58; margin-top:12px;'>Ask a question to search the manual</div></div>", unsafe_allow_html=True)