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
TOP_K          = 5          # increased from 3 → 5 for better coverage
MAX_TOKENS     = 1024
TEMPERATURE    = 0.3        # slightly higher for more natural explanations
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a friendly and helpful automotive assistant — like a knowledgeable mechanic 
explaining things to a car owner in simple, easy-to-understand language.

Your job is to explain automotive concepts clearly using the provided context from the vehicle manual.

Rules:
- Always explain in simple, plain English that any car owner can understand — avoid heavy jargon.
- If a technical term is used, immediately explain what it means in brackets.
- Give complete, helpful explanations — don't be too brief.
- If the user asks a single keyword like "ABS", "battery", "engine", treat it as "explain what this is and how it works".
- Use the context provided to ground your answer, but explain it conversationally.
- If the context has partial information, use it and explain what you know from it.
- Only say "Information not found in manual" if the context has absolutely zero relevance to the question.
- Always mention the source at the end (Source: filename, Page X).
- Structure longer answers with clear points for easy reading.
"""

USER_PROMPT_TEMPLATE = """Here is relevant information from the vehicle manual:

{context}

The user is asking about: "{question}"

Please explain this clearly and simply, as if talking to a car owner who is not a mechanic.
If it's a single word or short phrase, give a full explanation of what it is, how it works, 
and why it matters for the car."""


# ── Query expansion ───────────────────────────────────────────────────────────
KEYWORD_EXPANSIONS = {
    "abs":          "ABS anti-lock braking system wheel lock",
    "ebs":          "EBS electronic braking system",
    "trc":          "TRC traction control system wheel spin",
    "vsc":          "VSC vehicle stability control skid",
    "hybrid":       "hybrid system electric motor battery regeneration",
    "battery":      "battery charging electric power",
    "engine":       "engine motor power combustion",
    "brake":        "brake braking system stopping",
    "fuel":         "fuel injector consumption economy",
    "transmission": "transmission gear gearbox shifting",
    "steering":     "steering wheel direction control",
    "coolant":      "coolant temperature cooling engine overheat",
    "oil":          "engine oil lubrication pressure",
    "tire":         "tire tyre pressure wear",
    "airbag":       "airbag SRS safety supplemental restraint",
    "ac":           "air conditioning climate control cooling",
    "alternator":   "alternator charging generator electricity",
}

def expand_query(question: str) -> str:
    """Expand short keywords into richer search queries for better retrieval."""
    q_lower = question.lower().strip()
    # Check if query is a short keyword (1-2 words)
    if len(q_lower.split()) <= 2:
        for keyword, expansion in KEYWORD_EXPANSIONS.items():
            if keyword in q_lower:
                return expansion
    return question


class RAGPipeline:
    def __init__(self):
        self._index    = None
        self._metadata = None
        self._embedder = None
        self._groq     = None
        self._ready    = False

    def load(self) -> tuple[bool, str]:
        if not os.path.exists(INDEX_PATH):
            return False, f"FAISS index not found at '{INDEX_PATH}'. Run `python ingest.py` first."
        if not os.path.exists(METADATA_PATH):
            return False, f"Metadata not found at '{METADATA_PATH}'. Run `python ingest.py` first."

        self._index    = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            self._metadata = pickle.load(f)
        self._embedder = SentenceTransformer(EMBED_MODEL)

        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            return False, "GROQ_API_KEY environment variable not set."
        self._groq  = Groq(api_key=api_key)
        self._ready = True
        return True, ""

    def retrieve(self, question: str) -> list[dict]:
        """Embed query (with expansion), search FAISS, return top-k chunks."""
        # Expand short keyword queries for better semantic matching
        expanded = expand_query(question)

        # Embed both original and expanded query, average them
        queries  = list(set([question, expanded]))  # deduplicate if same
        vecs     = self._embedder.encode(queries, convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(vecs)
        vec      = vecs.mean(axis=0, keepdims=True).astype("float32")
        faiss.normalize_L2(vec)

        scores, indices = self._index.search(vec, TOP_K)

        results = []
        seen    = set()
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk_meta = dict(self._metadata[idx])
            # Deduplicate by text content
            text_key = chunk_meta["text"][:100]
            if text_key in seen:
                continue
            seen.add(text_key)
            chunk_meta["score"] = float(score)
            results.append(chunk_meta)

        return results

    def generate(self, question: str, retrieved_chunks: list[dict]) -> str:
        """Build prompt from chunks and call Groq LLM."""
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            header = f"[Section {i} | {chunk['source']} | Page {chunk['page']}]"
            context_parts.append(f"{header}\n{chunk['text']}")
        context = "\n\n".join(context_parts)

        user_message = USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )

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

    def query(self, question: str) -> dict:
        """End-to-end RAG: retrieve + generate."""
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