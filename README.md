# 📚 DocRAG — Local PDF Question Answering with LLaMA-3

> Retrieval-Augmented Generation (RAG) over your own PDF documents — powered by FAISS, SentenceTransformers, and Groq's blazing-fast LLaMA-3 inference.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-009688)
![LLaMA](https://img.shields.io/badge/LLM-LLaMA--3%208B-7C3AED)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ What It Does

Drop in any PDF documents, ask questions in plain English, and get grounded, cited answers — without hallucination-prone LLMs making things up. The system retrieves the most relevant passages from your documents before generating a response.

```
You: "What are the key findings in Q3?"
Bot: "According to page 4 of report.pdf: ..."
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   INGESTION  (ingest.py)                    │
│                                                             │
│  docs/*.pdf  →  PyPDF  →  Text Chunks  →  SentenceTransformers │
│                                              ↓              │
│                                     FAISS Index (disk)      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   QUERY  (rag_pipeline.py)                  │
│                                                             │
│  User Question  →  Embed Query  →  FAISS Search (top-3)    │
│                                              ↓              │
│  Structured Prompt  →  Groq API (LLaMA-3 8B)  →  Answer   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   UI  (app.py)                              │
│                                                             │
│  Streamlit  →  Text Input  →  RAGPipeline.query()          │
│                  ↓                                          │
│     Answer  +  Retrieved Chunks  +  Source Citations        │
└─────────────────────────────────────────────────────────────┘
```

| Component | Technology |
|-----------|-----------|
| PDF Parsing | PyPDF |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS (persisted to disk) |
| LLM Inference | Groq API — LLaMA-3 8B |
| UI | Streamlit |

---

## 🚀 Quickstart

### 1. Clone & install

```bash
git clone https://github.com/your-username/docrag.git
cd docrag
pip install -r requirements.txt
```

### 2. Set your Groq API key

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

Get a free key at [console.groq.com](https://console.groq.com).

### 3. Add your PDFs

```bash
mkdir docs
cp /path/to/your/files/*.pdf docs/
```

### 4. Ingest documents

```bash
python ingest.py
```

This reads all PDFs in `docs/`, chunks the text, generates embeddings, and saves a FAISS index to `faiss_index/`.

### 5. Launch the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) and start asking questions.

---

## 📁 Project Structure

```
docrag/
├── docs/                  # 📂 Place your PDF files here
├── faiss_index/           # 🗄️  Auto-generated vector index (git-ignored)
├── ingest.py              # 🔄  PDF → chunks → embeddings → FAISS
├── rag_pipeline.py        # 🧠  Query embedding + retrieval + LLM call
├── app.py                 # 🖥️  Streamlit UI
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration

Key parameters can be tuned at the top of each script:

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `CHUNK_SIZE` | `ingest.py` | `512` | Characters per text chunk |
| `CHUNK_OVERLAP` | `ingest.py` | `64` | Overlap between chunks |
| `TOP_K` | `rag_pipeline.py` | `3` | Number of chunks retrieved per query |
| `MODEL` | `rag_pipeline.py` | `llama3-8b-8192` | Groq model name |
| `EMBEDDING_MODEL` | `ingest.py` | `all-MiniLM-L6-v2` | SentenceTransformer model |

---

## 🔍 How It Works (Step by Step)

**Ingestion**

1. PyPDF reads every `.pdf` in `docs/` and extracts raw text page by page.
2. Text is split into overlapping chunks to preserve context across boundaries.
3. Each chunk is embedded using `SentenceTransformers` into a 384-dimensional vector.
4. All vectors are stored in a FAISS flat index and saved to disk.

**Querying**

1. The user's question is embedded with the same model used during ingestion.
2. FAISS performs an approximate nearest-neighbour search, returning the top-3 most semantically similar chunks.
3. Those chunks are injected into a structured prompt alongside the question.
4. Groq's API runs inference on LLaMA-3 8B and streams back the answer.

**UI**

1. Streamlit renders a text input and a submit button.
2. On submit, `RAGPipeline.query()` is called and the answer, retrieved chunks, and source filenames are displayed.

---

## 📦 Requirements

```
streamlit
pypdf
sentence-transformers
faiss-cpu
groq
```

Install all at once:

```bash
pip install -r requirements.txt
```

> **Note:** Use `faiss-gpu` instead of `faiss-cpu` if you have a CUDA-capable GPU for faster indexing and search.

---

## 🛠️ Re-indexing

Whenever you add or remove PDFs, re-run ingestion to rebuild the index:

```bash
python ingest.py
```

The old index in `faiss_index/` will be overwritten.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change, then submit a pull request.

1. Fork the repo
2. Create your branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push and open a PR

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Groq](https://groq.com) for ultra-low-latency LLaMA-3 inference
- [FAISS](https://github.com/facebookresearch/faiss) by Meta AI Research
- [Sentence Transformers](https://www.sbert.net/) by UKP Lab
- [Streamlit](https://streamlit.io) for the frictionless UI
