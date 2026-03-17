"""
app.py — Toyota Yaris Hybrid Showroom Assistant
A friendly guide for customers exploring their new car.
"""

import os
import streamlit as st

# ── Auto-load .env ────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ[k.strip()] = v.strip()

from rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="Toyota Yaris Hybrid Guide",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #f8f9fa; color: #1a1a2e; }

section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #eb0a1e 0%, #a00012 100%);
    color: white;
}
section[data-testid="stSidebar"] * { color: white !important; }
section[data-testid="stSidebar"] .stTextInput > div > div > input {
    background: rgba(255,255,255,0.15) !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    color: white !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] .stTextInput > div > div > input::placeholder { color: rgba(255,255,255,0.6) !important; }

h1 { font-family: 'Space Grotesk', sans-serif !important; color: #1a1a2e !important; font-size: 2.2rem !important; }
h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: #eb0a1e !important; }

.hero-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 24px;
    color: white;
}
.hero-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.8rem; font-weight: 700; color: white; margin: 0; }
.hero-sub { font-size: 1rem; color: rgba(255,255,255,0.7); margin-top: 6px; }
.hero-badge {
    display: inline-block; background: #eb0a1e; color: white;
    padding: 4px 14px; border-radius: 20px; font-size: 12px;
    font-weight: 600; margin-bottom: 12px; letter-spacing: 0.05em;
}

.answer-box {
    background: white;
    border: none;
    border-left: 5px solid #eb0a1e;
    border-radius: 12px;
    padding: 24px 28px;
    font-size: 15px;
    line-height: 1.85;
    color: #1a1a2e;
    box-shadow: 0 2px 16px rgba(0,0,0,0.08);
    white-space: pre-wrap;
}

.chunk-card {
    background: white;
    border: 1px solid #e9ecef;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
    font-size: 13px;
    color: #495057;
    line-height: 1.7;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.chunk-header {
    color: #eb0a1e;
    font-weight: 600;
    margin-bottom: 8px;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-family: 'Space Grotesk', sans-serif;
}
.score-badge {
    background: #e8f5e9; color: #2e7d32;
    border-radius: 4px; padding: 1px 8px;
    font-size: 11px; font-weight: 600; margin-left: 8px;
}

.stTextInput > div > div > input {
    background: white !important;
    border: 2px solid #dee2e6 !important;
    border-radius: 10px !important;
    font-size: 15px !important;
    padding: 12px 16px !important;
    color: #1a1a2e !important;
    transition: border-color 0.2s;
}
.stTextInput > div > div > input:focus { border-color: #eb0a1e !important; }

.stButton > button {
    background: #eb0a1e !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 15px !important;
    padding: 12px 28px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

.topic-chip {
    display: inline-block;
    background: #fff3f3;
    border: 1px solid #ffcdd2;
    color: #c62828;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    margin: 3px;
    cursor: pointer;
}
hr { border-color: #dee2e6 !important; }
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
    st.markdown("## 🔧 Settings")
    st.markdown("---")

    existing_key = os.environ.get("GROQ_API_KEY", "")
    if existing_key:
        st.success("✅ API Key ready")
    else:
        st.error("❌ API Key missing")

    api_key_input = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    active_key    = api_key_input.strip() if api_key_input.strip() else existing_key

    st.markdown("---")
    st.markdown("### 💡 Try asking about...")

    # Organised topic buttons for showroom customers
    topics = {
        "🔋 Hybrid System": [
            "What is hybrid?", "How does EV mode work?", "How does the car charge itself?"
        ],
        "🛡️ Safety": [
            "What is ABS?", "What is VSC?", "How do airbags work?", "What is pre-crash safety?"
        ],
        "⚠️ Warning Lights": [
            "Warning lights on dashboard", "Check engine light", "What does HV battery warning mean?"
        ],
        "⛽ Fuel & Economy": [
            "How to save fuel?", "What is the fuel tank size?", "How far can I drive?"
        ],
        "🚗 Driving": [
            "What is Eco mode?", "How does cruise control work?", "What is traction control?"
        ],
        "🔧 Maintenance": [
            "When to service the car?", "How to check engine oil?", "Tyre pressure tips"
        ],
    }

    for section, questions in topics.items():
        st.markdown(f"**{section}**")
        for q in questions:
            if st.button(q, use_container_width=True, key=q):
                st.session_state["prefill_query"] = q

# ── Hero Section ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-box">
    <div class="hero-badge">🚗 TOYOTA YARIS HYBRID</div>
    <div class="hero-title">Your Personal Car Guide</div>
    <div class="hero-sub">Ask anything about your Yaris Hybrid — in plain, simple language. No jargon, no confusing manuals!</div>
</div>
""", unsafe_allow_html=True)

if not active_key:
    st.warning("⚠️ Please add your Groq API key in the sidebar to start chatting.", icon="🔑")
    st.info("Get a free key at **https://console.groq.com** → API Keys → Create Key", icon="ℹ️")
    st.stop()

pipeline, load_error = get_pipeline(active_key)

if load_error:
    st.error(f"⚠️ {load_error}", icon="🚨")
    if "FAISS" in load_error or "index" in load_error.lower():
        st.code("python ingest.py", language="bash")
        st.info("Run the above command in your terminal first, then refresh.", icon="ℹ️")
    st.stop()

# ── Query Input ───────────────────────────────────────────────────────────────
default_query = st.session_state.pop("prefill_query", "")
col_input, col_btn = st.columns([5, 1])
with col_input:
    user_question = st.text_input(
        "Question",
        value=default_query,
        placeholder="💬  e.g. What is hybrid? How does ABS work? What does this warning light mean?",
        label_visibility="collapsed"
    )
with col_btn:
    ask_clicked = st.button("Ask 🚗", use_container_width=True)

# ── Answer ────────────────────────────────────────────────────────────────────
if ask_clicked and user_question.strip():
    with st.spinner("🔍 Looking through your Yaris manual..."):
        result = pipeline.query(user_question.strip())

    if result["error"]:
        st.error(f"Something went wrong: {result['error']}")
    else:
        st.markdown("### 💡 Here's your answer")
        st.markdown(f"<div class='answer-box'>{result['answer']}</div>", unsafe_allow_html=True)

        if result["chunks"]:
            st.markdown("---")
            with st.expander("📄 View manual sections used to answer this", expanded=False):
                st.caption(f"Retrieved {len(result['chunks'])} relevant sections from the Toyota Yaris Hybrid manual:")
                for i, chunk in enumerate(result["chunks"], start=1):
                    display_text = chunk["text"][:500] + ("…" if len(chunk["text"]) > 500 else "")
                    st.markdown(f"""
                    <div class='chunk-card'>
                        <div class='chunk-header'>
                            Section {i} &nbsp;·&nbsp; Page {chunk['page']}
                            <span class='score-badge'>match {chunk['score']*100:.1f}%</span>
                        </div>
                        {display_text}
                    </div>
                    """, unsafe_allow_html=True)

elif ask_clicked:
    st.warning("Please type a question first!", icon="⚠️")

else:
    # Welcome state
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style='background:white; border-radius:12px; padding:20px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
            <div style='font-size:36px;'>🔋</div>
            <div style='font-weight:600; color:#1a1a2e; margin-top:8px;'>Hybrid System</div>
            <div style='font-size:13px; color:#6c757d; margin-top:4px;'>Understand how your car switches between electric and petrol</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='background:white; border-radius:12px; padding:20px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
            <div style='font-size:36px;'>🛡️</div>
            <div style='font-weight:600; color:#1a1a2e; margin-top:8px;'>Safety Features</div>
            <div style='font-size:13px; color:#6c757d; margin-top:4px;'>Learn about ABS, airbags, stability control and more</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style='background:white; border-radius:12px; padding:20px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
            <div style='font-size:36px;'>⚠️</div>
            <div style='font-weight:600; color:#1a1a2e; margin-top:8px;'>Warning Lights</div>
            <div style='font-size:13px; color:#6c757d; margin-top:4px;'>Know exactly what every dashboard symbol means</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center; padding:30px; color:#adb5bd; font-size:14px;'>
        👆 Type your question above or pick a topic from the sidebar
    </div>
    """, unsafe_allow_html=True)