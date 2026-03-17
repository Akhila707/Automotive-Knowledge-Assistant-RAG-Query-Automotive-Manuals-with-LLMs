"""
rag_pipeline.py — Retrieval-Augmented Generation Core
Handles: loading the FAISS index, retrieving top-k chunks, calling Groq LLM.
"""

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# ── Configuration ────────────────────────────────────────────────────────────
INDEX_PATH     = "faiss_index.bin"
METADATA_PATH  = "chunks_metadata.pkl"
EMBED_MODEL    = "all-MiniLM-L6-v2"
GROQ_MODEL     = "llama-3.1-8b-instant"
TOP_K          = 3          # number of chunks to retrieve
MAX_TOKENS     = 1024       # max LLM response tokens
TEMPERATURE    = 0.2        # low temperature → factual, deterministic answers
# ─────────────────────────────────────────────────────────────────────────────

# Structured system prompt for the automotive assistant
SYSTEM_PROMPT = """You are an expert automotive technical assistant helping engineers and technicians.
Answer the question using ONLY the provided context from the automotive manual.

Rules:
- Be concise, precise, and technical.
- If the answer is not found in the context, respond exactly with: 'Information not found in manual.'
- Always mention which source chunk the information came from (e.g., Source: filename.pdf, Page X).
- Do not hallucinate or add information beyond what is given in the context.
"""

USER_PROMPT_TEMPLATE = """Context from automotive manual:
{context}

Technician's Question:
{question}

Answer:"""


class RAGPipeline:
    """
    Encapsulates the full RAG workflow:
      1. Load FAISS index + metadata
      2. Embed user query
      3. Retrieve top-k relevant chunks
      4. Generate grounded answer via Groq LLM
    """

    def __init__(self):
        self._index    = None
        self._metadata = None
        self._embedder = None
        self._groq     = None
        self._ready    = False

    # ── Lazy initialisation ──────────────────────────────────────────────────
    def load(self) -> tuple[bool, str]:
        """
        Load all required artefacts. Returns (success, error_message).
        Call this once before invoking `query()`.
        """
        # 1. Check index files exist
        if not os.path.exists(INDEX_PATH):
            return False, f"FAISS index not found at '{INDEX_PATH}'. Run `python ingest.py` first."
        if not os.path.exists(METADATA_PATH):
            return False, f"Metadata not found at '{METADATA_PATH}'. Run `python ingest.py` first."

        # 2. Load FAISS index
        self._index = faiss.read_index(INDEX_PATH)

        # 3. Load chunk metadata
        with open(METADATA_PATH, "rb") as f:
            self._metadata = pickle.load(f)

        # 4. Load sentence embedder
        self._embedder = SentenceTransformer(EMBED_MODEL)

        # 5. Initialise Groq client (reads GROQ_API_KEY from environment)
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            return False, "GROQ_API_KEY environment variable not set."
        self._groq  = Groq(api_key=api_key)
        self._ready = True
        return True, ""

    # ── Retrieval ────────────────────────────────────────────────────────────
    def retrieve(self, question: str) -> list[dict]:
        """
        Embed the question, search FAISS, return top-k metadata dicts
        (each dict contains 'source', 'page', 'text', 'score').
        """
        # Embed and normalise query vector
        vec = self._embedder.encode([question], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(vec)

        # Search FAISS index
        scores, indices = self._index.search(vec, TOP_K)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:          # FAISS returns -1 when fewer results exist
                continue
            chunk_meta = dict(self._metadata[idx])  # copy so we don't mutate
            chunk_meta["score"] = float(score)
            results.append(chunk_meta)

        return results

    # ── Generation ───────────────────────────────────────────────────────────
    def generate(self, question: str, retrieved_chunks: list[dict]) -> str:
        """
        Build a prompt from retrieved chunks and call the Groq LLM.
        Returns the generated answer string.
        """
        # Build the context block with source labels
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            header = f"[Chunk {i} | {chunk['source']} | Page {chunk['page']}]"
            context_parts.append(f"{header}\n{chunk['text']}")
        context = "\n\n".join(context_parts)

        user_message = USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )

        # Call Groq API
        response = self._groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message}
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

        return response.choices[0].message.content.strip()

    # ── Unified query interface ───────────────────────────────────────────────
    def query(self, question: str) -> dict:
        """
        End-to-end RAG: retrieve + generate.
        Returns a dict with keys: 'answer', 'chunks', 'error'.
        """
        if not self._ready:
            return {"answer": "", "chunks": [], "error": "Pipeline not loaded. Call load() first."}

        try:
            chunks = self.retrieve(question)
            if not chunks:
                return {"answer": "Information not found in manual.", "chunks": [], "error": ""}

            answer = self.generate(question, chunks)
            return {"answer": answer, "chunks": chunks, "error": ""}

        except Exception as exc:
            return {"answer": "", "chunks": [], "error": str(exc)}