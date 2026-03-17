"""
ingest.py — Document Ingestion Pipeline
Loads PDFs from docs/, chunks text, generates embeddings, and saves FAISS index.
"""

import os
import pickle
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ── Configuration ────────────────────────────────────────────────────────────
DOCS_FOLDER    = "docs"
INDEX_PATH     = "faiss_index.bin"
METADATA_PATH  = "chunks_metadata.pkl"
CHUNK_SIZE     = 600   # target tokens per chunk (approx. chars / 4)
CHUNK_OVERLAP  = 100   # overlap between consecutive chunks (tokens)
EMBED_MODEL    = "all-MiniLM-L6-v2"   # fast, accurate, 384-dim embeddings
# ─────────────────────────────────────────────────────────────────────────────


def load_pdfs(folder: str) -> list[dict]:
    """
    Walk through every PDF in `folder`, extract page text,
    and return a list of { 'source': filename, 'text': page_text } dicts.
    """
    documents = []
    if not os.path.exists(folder):
        print(f"[ERROR] Folder '{folder}' not found. Create it and add PDF manuals.")
        return documents

    pdf_files = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"[WARN] No PDF files found in '{folder}'.")
        return documents

    for filename in pdf_files:
        path = os.path.join(folder, filename)
        print(f"[INFO] Loading: {filename}")
        reader = PdfReader(path)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                documents.append({
                    "source": filename,
                    "page":   page_num + 1,
                    "text":   text
                })

    print(f"[INFO] Loaded {len(documents)} pages from {len(pdf_files)} PDF(s).")
    return documents


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split raw text into overlapping token-approximate chunks.
    We use words as a proxy for tokens (1 word ≈ 1.3 tokens).
    """
    words      = text.split()
    word_chunk = int(chunk_size / 1.3)     # words per chunk
    word_step  = int((chunk_size - overlap) / 1.3)

    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + word_chunk, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += word_step

    return chunks


def build_chunks(documents: list[dict]) -> tuple[list[str], list[dict]]:
    """
    Iterate over all document pages, split each into chunks,
    and build parallel lists: raw text chunks + metadata dicts.
    """
    all_chunks   = []
    all_metadata = []

    for doc in documents:
        page_chunks = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(page_chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "source":      doc["source"],
                "page":        doc["page"],
                "chunk_index": i,
                "text":        chunk
            })

    print(f"[INFO] Created {len(all_chunks)} chunks total.")
    return all_chunks, all_metadata


def embed_and_index(chunks: list[str], metadata: list[dict]) -> None:
    """
    Generate sentence embeddings for all chunks,
    build a FAISS flat L2 index, and persist both to disk.
    """
    print(f"[INFO] Loading embedding model: {EMBED_MODEL}")
    model      = SentenceTransformer(EMBED_MODEL)

    print("[INFO] Generating embeddings (this may take a moment)…")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    embeddings = np.array(embeddings, dtype="float32")

    # Normalize vectors for cosine-like retrieval with L2 index
    faiss.normalize_L2(embeddings)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner-product = cosine similarity after normalisation
    index.add(embeddings)

    # Persist index and metadata side-by-side
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"[INFO] FAISS index saved → {INDEX_PATH}")
    print(f"[INFO] Metadata saved   → {METADATA_PATH}")
    print(f"[INFO] Total vectors indexed: {index.ntotal}")


def main():
    print("=" * 55)
    print("  Automotive RAG — Document Ingestion Pipeline")
    print("=" * 55)
    documents              = load_pdfs(DOCS_FOLDER)
    if not documents:
        return
    chunks, metadata       = build_chunks(documents)
    embed_and_index(chunks, metadata)
    print("\n[DONE] Ingestion complete. Run `streamlit run app.py` to start.")


if __name__ == "__main__":
    main()