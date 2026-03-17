"""
app.py — Volvo Premium Automotive Knowledge Assistant
Swedish minimalism meets AI — designed to impress.
"""

import os
import streamlit as st

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
    page_title="MyCar Guide",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* ══════════════════════════════════════════
   GLOBAL BACKGROUND
══════════════════════════════════════════ */
.stApp {
    background-color: #f5f3ef;
    background-image:
        radial-gradient(ellipse at 10% 20%, rgba(30,60,114,0.04) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 80%, rgba(30,60,114,0.03) 0%, transparent 50%);
}

/* ══════════════════════════════════════════
   SIDEBAR — Deep Volvo Navy
══════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1c3a 0%, #091528 60%, #060e1f 100%) !important;
    border-right: none !important;
}
section[data-testid="stSidebar"]::after {
    content: '';
    position: absolute; top: 0; right: 0;
    width: 1px; height: 100%;
    background: linear-gradient(180deg, transparent, rgba(196,164,100,0.3), rgba(196,164,100,0.6), rgba(196,164,100,0.3), transparent);
}
section[data-testid="stSidebar"] * { color: #c8d4e8 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #c4a464 !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
}
section[data-testid="stSidebar"] .stTextInput > div > div > input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(196,164,100,0.25) !important;
    color: #e8dfc8 !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.03em !important;
}
section[data-testid="stSidebar"] .stTextInput > div > div > input:focus {
    border-color: rgba(196,164,100,0.6) !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: rgba(200,212,232,0.7) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 6px !important;
    font-size: 12px !important;
    font-weight: 400 !important;
    text-align: left !important;
    transition: all 0.25s ease !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 8px 12px !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(196,164,100,0.1) !important;
    border-color: rgba(196,164,100,0.35) !important;
    color: #c4a464 !important;
    transform: translateX(3px) !important;
}

/* ══════════════════════════════════════════
   HERO SECTION
══════════════════════════════════════════ */
.volvo-hero {
    background: linear-gradient(135deg, #0c1c3a 0%, #1a2f5a 50%, #0c1c3a 100%);
    border-radius: 4px;
    padding: 64px 68px 56px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.volvo-hero::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        radial-gradient(ellipse at 80% 50%, rgba(196,164,100,0.08) 0%, transparent 55%),
        radial-gradient(ellipse at 20% 80%, rgba(30,80,160,0.15) 0%, transparent 50%);
    pointer-events: none;
}
.volvo-hero::after {
    content: '';
    position: absolute; top: 0; right: 0;
    width: 400px; height: 100%;
    background: url("data:image/svg+xml,%3Csvg width='400' height='300' viewBox='0 0 400 300' xmlns='http://www.w3.org/2000/svg'%3E%3Ccircle cx='320' cy='150' r='180' fill='none' stroke='rgba(196,164,100,0.06)' stroke-width='60'/%3E%3Ccircle cx='320' cy='150' r='120' fill='none' stroke='rgba(196,164,100,0.04)' stroke-width='1'/%3E%3C/svg%3E") no-repeat center right;
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 10px; font-weight: 500;
    color: #c4a464;
    letter-spacing: 0.2em; text-transform: uppercase;
    margin-bottom: 20px;
    display: flex; align-items: center; gap: 12px;
}
.hero-eyebrow::after {
    content: ''; display: block; width: 40px; height: 1px;
    background: linear-gradient(90deg, #c4a464, transparent);
}
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.8rem; font-weight: 300;
    color: #ffffff;
    line-height: 1.1; letter-spacing: -0.01em;
    margin-bottom: 6px;
}
.hero-title span {
    font-weight: 500;
    background: linear-gradient(135deg, #c4a464, #e8d09a);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-desc {
    font-size: 1rem; font-weight: 300;
    color: rgba(200,212,232,0.65);
    line-height: 1.7; max-width: 500px;
    margin-top: 20px;
}
.hero-divider {
    width: 48px; height: 2px;
    background: linear-gradient(90deg, #c4a464, transparent);
    margin: 28px 0;
}
.hero-meta {
    display: flex; gap: 36px; align-items: center;
}
.meta-item { display: flex; flex-direction: column; gap: 2px; }
.meta-val {
    font-family: 'DM Mono', monospace;
    font-size: 1.1rem; font-weight: 500; color: #c4a464;
}
.meta-key {
    font-size: 10px; font-weight: 400;
    color: rgba(200,212,232,0.4);
    text-transform: uppercase; letter-spacing: 0.12em;
}
.meta-sep { width: 1px; height: 32px; background: rgba(196,164,100,0.2); }

/* ══════════════════════════════════════════
   SEARCH AREA
══════════════════════════════════════════ */
.search-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px; color: rgba(12,28,58,0.4);
    letter-spacing: 0.15em; text-transform: uppercase;
    margin-bottom: 10px;
}
.stTextInput > div > div > input {
    background: #ffffff !important;
    border: 1.5px solid rgba(12,28,58,0.12) !important;
    border-radius: 4px !important;
    color: #0c1c3a !important;
    font-size: 15px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 400 !important;
    padding: 16px 20px !important;
    box-shadow: 0 2px 12px rgba(12,28,58,0.06) !important;
    transition: all 0.25s ease !important;
}
.stTextInput > div > div > input:focus {
    border-color: #1e3c72 !important;
    box-shadow: 0 4px 20px rgba(12,28,58,0.12) !important;
}
.stTextInput > div > div > input::placeholder { color: rgba(12,28,58,0.3) !important; }

/* ══════════════════════════════════════════
   BUTTON
══════════════════════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg, #1e3c72, #0c1c3a) !important;
    color: #c4a464 !important;
    border: 1px solid rgba(196,164,100,0.3) !important;
    border-radius: 4px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.05em !important;
    padding: 16px 28px !important;
    box-shadow: 0 4px 16px rgba(12,28,58,0.25) !important;
    transition: all 0.25s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2a4f96, #1a3060) !important;
    box-shadow: 0 6px 24px rgba(12,28,58,0.35) !important;
    transform: translateY(-1px) !important;
    border-color: rgba(196,164,100,0.5) !important;
}

/* ══════════════════════════════════════════
   ANSWER PANEL
══════════════════════════════════════════ */
.answer-panel {
    background: #ffffff;
    border-radius: 4px;
    border-top: 3px solid #1e3c72;
    padding: 36px 40px;
    box-shadow: 0 8px 40px rgba(12,28,58,0.08);
    font-size: 15px; line-height: 1.95;
    color: #1a2a40;
    white-space: pre-wrap;
}
.answer-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 9px; font-weight: 500;
    color: #c4a464; letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 1px solid #f0ebe0;
    display: flex; align-items: center; gap: 10px;
}
.answer-eyebrow::before {
    content: ''; display: block;
    width: 20px; height: 1px;
    background: #c4a464;
}

/* ══════════════════════════════════════════
   SOURCE CHUNKS
══════════════════════════════════════════ */
.source-card {
    background: #faf9f6;
    border: 1px solid #ede8dc;
    border-left: 3px solid #c4a464;
    border-radius: 4px;
    padding: 18px 22px;
    margin-bottom: 12px;
    font-size: 12.5px;
    color: rgba(12,28,58,0.55);
    line-height: 1.75;
    font-family: 'DM Mono', monospace;
}
.source-header {
    font-family: 'DM Sans', sans-serif;
    font-size: 10px; font-weight: 600;
    color: #1e3c72; text-transform: uppercase;
    letter-spacing: 0.1em; margin-bottom: 10px;
}
.relevance-tag {
    display: inline-block;
    background: rgba(30,60,114,0.08);
    color: #1e3c72;
    border-radius: 3px; padding: 1px 8px;
    font-size: 10px; font-weight: 600;
    margin-left: 8px; font-family: 'DM Mono', monospace;
}

/* ══════════════════════════════════════════
   FEATURE GRID (Welcome state)
══════════════════════════════════════════ */
.feat-row { display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin-top: 4px; }
.feat-tile {
    background: #ffffff;
    border: 1px solid rgba(12,28,58,0.08);
    border-radius: 4px;
    padding: 28px 24px;
    transition: all 0.25s ease;
    cursor: default;
}
.feat-tile:hover {
    box-shadow: 0 8px 32px rgba(12,28,58,0.1);
    transform: translateY(-2px);
    border-color: rgba(196,164,100,0.4);
}
.feat-ico { font-size: 1.8rem; margin-bottom: 14px; }
.feat-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.1rem; font-weight: 600;
    color: #0c1c3a; margin-bottom: 8px;
}
.feat-blurb { font-size: 12.5px; color: rgba(12,28,58,0.45); line-height: 1.6; font-weight: 300; }

/* ══════════════════════════════════════════
   MISC
══════════════════════════════════════════ */
.section-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 10px; color: rgba(12,28,58,0.35);
    letter-spacing: 0.18em; text-transform: uppercase;
    margin-bottom: 16px;
    display: flex; align-items: center; gap: 10px;
}
.section-eyebrow::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(12,28,58,0.1), transparent);
}
.stSpinner > div { border-top-color: #1e3c72 !important; }
.streamlit-expanderHeader {
    background: #faf9f6 !important;
    border: 1px solid #ede8dc !important;
    border-radius: 4px !important;
    color: rgba(12,28,58,0.5) !important;
    font-size: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
}
hr { border-color: rgba(12,28,58,0.08) !important; }
.stAlert { border-radius: 4px !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_pipeline(api_key: str):
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
    pipe = RAGPipeline()
    ok, err = pipe.load()
    return (pipe, "") if ok else (None, err)


# ══════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown("### Configuration")
    st.markdown("---")
    existing_key = os.environ.get("GROQ_API_KEY", "")
    if existing_key:
        st.success("✅ System ready")
    else:
        st.error("❌ API key required")
    api_key_input = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    active_key    = api_key_input.strip() if api_key_input.strip() else existing_key

    st.markdown("---")
    st.markdown("### Explore Topics")

    cats = {
        "🔋 Hybrid & Electric": [
            "What is hybrid?", "How does EV mode work?",
            "How does the car charge itself?", "What is regenerative braking?"
        ],
        "🛡️ Safety Systems": [
            "What is ABS?", "How do airbags work?",
            "What is VSC?", "What is pre-crash safety?"
        ],
        "⚠️ Warning Indicators": [
            "Warning lights on dashboard",
            "Check engine light meaning",
            "HV battery warning light"
        ],
        "⛽ Fuel & Efficiency": [
            "How to save fuel?", "What is Eco mode?",
            "How far can I drive on full tank?"
        ],
        "🚗 Driving & Comfort": [
            "How does cruise control work?",
            "What is Sport mode?",
            "How do parking sensors work?"
        ],
        "🔧 Care & Maintenance": [
            "When to service my car?",
            "How to check engine oil?",
            "What tyre pressure should I use?"
        ],
    }
    for label, qs in cats.items():
        with st.expander(label, expanded=False):
            for q in qs:
                if st.button(q, use_container_width=True, key=q):
                    st.session_state["prefill_query"] = q

    st.markdown("---")
    st.markdown("""
    <div style='font-size:10px; color:rgba(200,212,232,0.25); font-family:DM Mono,monospace; line-height:2; letter-spacing:0.06em;'>
    RAG · FAISS · SENTENCE TRANSFORMERS<br>
    GROQ LLAMA-3.1 · STREAMLIT
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════
# HERO
# ══════════════════════════════════════════
st.markdown("""
<div class="volvo-hero">
    <div class="hero-eyebrow">MyCar Guide</div>
    <div class="hero-title">Plain Answers<br><span>For Every Driver.</span></div>
    <div class="hero-desc">
        Ask any question about your vehicle — in plain language.
        Powered by AI trained on your official owner's manual,
        so every answer is accurate, grounded, and easy to understand.
    </div>
    <div class="hero-divider"></div>
    <div class="hero-meta">
        <div class="meta-item">
            <span class="meta-val">300+</span>
            <span class="meta-key">Manual Pages</span>
        </div>
        <div class="meta-sep"></div>
        <div class="meta-item">
            <span class="meta-val">AI</span>
            <span class="meta-key">Powered Answers</span>
        </div>
        <div class="meta-sep"></div>
        <div class="meta-item">
            <span class="meta-val">Plain</span>
            <span class="meta-key">English Always</span>
        </div>
        <div class="meta-sep"></div>
        <div class="meta-item">
            <span class="meta-val">∞</span>
            <span class="meta-key">Questions Welcome</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if not active_key:
    st.warning("Please add your Groq API key in the sidebar to activate the assistant.", icon="🔑")
    st.info("Get a free key at **https://console.groq.com** → API Keys → Create Key", icon="ℹ️")
    st.stop()

pipeline, load_error = get_pipeline(active_key)
if load_error:
    st.error(f"{load_error}", icon="⚠️")
    if "FAISS" in load_error or "index" in load_error.lower():
        st.code("python ingest.py", language="bash")
        st.info("Run the command above in your terminal, then refresh this page.", icon="ℹ️")
    st.stop()

# ══════════════════════════════════════════
# SEARCH
# ══════════════════════════════════════════
st.markdown('<div class="search-label">Your Question</div>', unsafe_allow_html=True)
default_query = st.session_state.pop("prefill_query", "")
col_q, col_btn = st.columns([5, 1])
with col_q:
    user_question = st.text_input(
        "q", value=default_query,
        placeholder="e.g.  What is hybrid?  ·  How does ABS work?  ·  What does this warning light mean?",
        label_visibility="collapsed"
    )
with col_btn:
    ask_clicked = st.button("Ask →", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════
# RESULT
# ══════════════════════════════════════════
if ask_clicked and user_question.strip():
    with st.spinner("Consulting the manual..."):
        result = pipeline.query(user_question.strip())

    if result["error"]:
        st.error(f"Something went wrong: {result['error']}")
    else:
        st.markdown(f"""
        <div class="answer-panel">
            <div class="answer-eyebrow">Answer</div>
            {result['answer']}
        </div>
        """, unsafe_allow_html=True)

        if result["chunks"]:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("View source references from the manual", expanded=False):
                st.markdown(f"""
                <div style='font-family:DM Mono,monospace; font-size:10px;
                color:rgba(12,28,58,0.35); letter-spacing:0.12em; text-transform:uppercase;
                margin-bottom:16px;'>
                {len(result['chunks'])} sections retrieved
                </div>
                """, unsafe_allow_html=True)
                for i, chunk in enumerate(result["chunks"], start=1):
                    txt = chunk["text"][:480] + ("…" if len(chunk["text"]) > 480 else "")
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-header">
                            Reference {i} &nbsp;·&nbsp; Page {chunk['page']}
                            <span class="relevance-tag">{chunk['score']*100:.1f}% relevant</span>
                        </div>
                        {txt}
                    </div>
                    """, unsafe_allow_html=True)

elif ask_clicked:
    st.warning("Please enter a question to continue.", icon="⚠️")

else:
    st.markdown('<div class="section-eyebrow">How can we help you today</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="feat-row">
        <div class="feat-tile">
            <div class="feat-ico">🔋</div>
            <div class="feat-name">Hybrid System</div>
            <div class="feat-blurb">Understand how electric and petrol work seamlessly together to maximise efficiency and performance.</div>
        </div>
        <div class="feat-tile">
            <div class="feat-ico">🛡️</div>
            <div class="feat-name">Safety Features</div>
            <div class="feat-blurb">Every safety system explained clearly — ABS, airbags, stability control, pre-crash technology.</div>
        </div>
        <div class="feat-tile">
            <div class="feat-ico">⚠️</div>
            <div class="feat-name">Warning Indicators</div>
            <div class="feat-blurb">Understand every dashboard symbol instantly — what it means, and exactly what to do.</div>
        </div>
        <div class="feat-tile">
            <div class="feat-ico">⛽</div>
            <div class="feat-name">Fuel Efficiency</div>
            <div class="feat-blurb">Practical guidance on driving modes, tips, and habits that save fuel on every journey.</div>
        </div>
        <div class="feat-tile">
            <div class="feat-ico">🚗</div>
            <div class="feat-name">Driving Modes</div>
            <div class="feat-blurb">Eco, Sport, EV — discover when and how to use each mode for the best driving experience.</div>
        </div>
        <div class="feat-tile">
            <div class="feat-ico">🔧</div>
            <div class="feat-name">Maintenance</div>
            <div class="feat-blurb">Simple, clear guidance on keeping your vehicle in perfect condition between service visits.</div>
        </div>
    </div>
    <div style='text-align:center; margin-top:40px;
    font-family:DM Mono,monospace; font-size:10px;
    color:rgba(12,28,58,0.2); letter-spacing:0.15em; text-transform:uppercase;'>
        Type your question above · or select a topic from the sidebar
    </div>
    """, unsafe_allow_html=True)