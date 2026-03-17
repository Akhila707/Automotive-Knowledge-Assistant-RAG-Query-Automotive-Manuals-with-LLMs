# 🚗 Automotive Knowledge Assistant (RAG)

> A Retrieval-Augmented Generation system that lets engineers and technicians query automotive manuals using natural language — powered by FAISS, SentenceTransformers, and Groq LLaMA-3.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE (ingest.py)               │
│                                                                 │
│  docs/*.pdf  →  PyPDF  →  Text Chunks  →  SentenceTransformers │
│                                              ↓                  │
│                                     FAISS Index (disk)          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    QUERY PIPELINE (rag_pipeline.py)             │
│                                                                 │
│  User Question  →  Embed Query  →  FAISS Search (top-3 chunks) │
│                                              ↓                  │
│  Structured Prompt  →  Groq API (LLaMA-3 8B)  →  Answer        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         UI (app.py)                             │
│                                                                 │
│  Streamlit  →  Text Input  →  RAGPipeline.query()              │
│                  ↓                                              │
│         Display Answer + Retrieved Chunks + Sources             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
automotive_rag_assistant/
│
├── docs/                     ← Place your PDF manuals here
│   └── sample_manual.pdf
│
├── ingest.py                 ← Step 1: Ingestion pipeline
├── rag_pipeline.py           ← Step 2: RAG core logic
├── app.py                    ← Step 3: Streamlit UI
│
├── faiss_index.bin           ← Auto-generated after ingest
├── chunks_metadata.pkl       ← Auto-generated after ingest
│
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone / download the project

```bash
cd automotive_rag_assistant
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get a Groq API key

1. Visit [console.groq.com](https://console.groq.com) and sign up (free).
2. Create an API key.
3. Export it:

```bash
export GROQ_API_KEY="gsk_your_key_here"    # Linux / macOS
set GROQ_API_KEY=gsk_your_key_here         # Windows CMD
```

Alternatively, you can paste it directly into the Streamlit sidebar.

---

## How to Run

### Step 1 — Add your PDFs

Place any automotive PDF manuals into the `docs/` folder.

```
docs/
├── toyota_corolla_manual.pdf
├── ford_f150_repair_guide.pdf
└── obd2_diagnostic_guide.pdf
```

### Step 2 — Ingest the documents

```bash
python ingest.py
```

This will:
- Extract text from all PDFs
- Split text into ~600-token chunks with overlap
- Generate embeddings with `all-MiniLM-L6-v2`
- Save `faiss_index.bin` and `chunks_metadata.pkl`

### Step 3 — Launch the UI

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## Example Queries

| Query | What it tests |
|---|---|
| `How does ABS braking system work?` | Safety systems |
| `What causes engine overheating?` | Engine diagnostics |
| `How to diagnose fuel injector problems?` | Fuel system |
| `Explain the function of the alternator.` | Electrical systems |
| `What are common transmission failure symptoms?` | Drivetrain |
| `What is the torque spec for cylinder head bolts?` | Spec lookup |

---

## Configuration

All tunable parameters are at the top of each file:

| File | Parameter | Default | Description |
|---|---|---|---|
| `ingest.py` | `CHUNK_SIZE` | 600 | Target tokens per chunk |
| `ingest.py` | `CHUNK_OVERLAP` | 100 | Overlap between chunks |
| `ingest.py` | `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence embedding model |
| `rag_pipeline.py` | `TOP_K` | 3 | Number of chunks to retrieve |
| `rag_pipeline.py` | `GROQ_MODEL` | `llama3-8b-8192` | Groq LLM model |
| `rag_pipeline.py` | `TEMPERATURE` | 0.2 | LLM temperature |

---

## How It Works

1. **Ingestion** — PyPDF reads every page of each PDF. Text is split into overlapping word windows (≈600 tokens each). SentenceTransformers converts each chunk into a 384-dimensional embedding vector. Vectors are stored in a FAISS `IndexFlatIP` (inner-product, equivalent to cosine similarity after L2 normalisation).

2. **Retrieval** — The user's question is embedded with the same model. FAISS performs an approximate nearest-neighbour search and returns the 3 most similar chunks, along with their source filename and page number.

3. **Generation** — The retrieved chunks are formatted into a structured prompt and sent to the Groq API. The LLM is instructed to answer only from the provided context and cite the source chunk.

4. **UI** — Streamlit renders the answer, the retrieved chunks, and their similarity scores in a clean dark-themed interface.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `FAISS index not found` | Run `python ingest.py` first |
| `GROQ_API_KEY not set` | Export the env variable or paste in sidebar |
| `No PDF files found` | Add PDFs to the `docs/` folder |
| Answer says "Information not found" | The manuals may not cover this topic — try different PDFs |
| Slow ingestion | Normal for large PDFs — embedding runs on CPU |