"""
rag_pipeline.py — RAG Core for Toyota Yaris Hybrid Showroom Assistant
Tailored for customers visiting a car showroom who need simple, friendly explanations.
"""

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# ── Configuration ─────────────────────────────────────────────────────────────
INDEX_PATH    = "faiss_index.bin"
METADATA_PATH = "chunks_metadata.pkl"
EMBED_MODEL   = "all-MiniLM-L6-v2"
GROQ_MODEL    = "llama-3.1-8b-instant"
TOP_K         = 6       # retrieve more chunks for richer context
MAX_TOKENS    = 1500    # longer answers for full explanations
TEMPERATURE   = 0.4     # friendly, natural tone
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a warm, friendly Toyota Yaris Hybrid showroom assistant helping customers 
who are interested in buying or have just bought a Toyota Yaris Hybrid car.

These customers are NOT mechanics or engineers. They are everyday people who want to understand 
their new car in simple, clear language — like a helpful friend explaining things over a cup of coffee.

Your personality:
- Warm, patient, and encouraging
- Use simple everyday language — NO technical jargon without explanation
- When you must use a technical term, ALWAYS explain it simply in brackets
- Use real-life analogies to explain how things work (e.g. "think of it like...")
- Be thorough but not overwhelming

How to answer:
- Start with a simple 1-2 sentence summary of what the thing IS
- Then explain HOW it works in simple terms
- Then explain WHY it matters / what benefit it gives the customer
- If relevant, mention any tips or things the customer should know
- End with the source page from the manual so they can read more

Special cases:
- Single keywords like "hybrid", "ABS", "VSC", "battery" → give a full friendly explanation
- Questions about warning lights → explain what it means and what to do calmly
- Questions about safety features → reassure and explain clearly
- Questions about fuel economy → explain in practical real-world terms
- If context has partial info → use it and explain what you know, don't just say "not found"
- Only say information is unavailable if context has absolutely nothing relevant

Remember: This person is excited about their new Toyota Yaris Hybrid. Make them feel confident and informed!
"""

USER_PROMPT_TEMPLATE = """Here is relevant information from the Toyota Yaris Hybrid Owner's Manual:

{context}

Customer's question: "{question}"

Please explain this in simple, friendly language that any car owner can easily understand.
- If it's a single word or short phrase, give a complete explanation: what it is, how it works, and why it's useful.
- Use a real-life analogy if it helps.
- Keep it clear, warm, and encouraging.
- Mention the manual page at the end."""


# ── Keyword expansions tailored for Toyota Yaris Hybrid manual ────────────────
KEYWORD_EXPANSIONS = {
    # Hybrid system
    "hybrid":        "hybrid system electric motor petrol engine battery regeneration fuel economy",
    "ev":            "EV mode electric vehicle motor drive battery",
    "ev mode":       "EV mode electric only driving motor battery",
    "regeneration":  "regenerative braking energy recovery battery charging",
    "hv":            "HV hybrid vehicle system battery electric motor",
    "hv battery":    "hybrid vehicle battery high voltage charging",

    # Safety systems
    "abs":           "ABS anti-lock braking system wheel lock prevent",
    "vsc":           "VSC vehicle stability control skid slide cornering",
    "trc":           "TRC traction control wheel spin acceleration",
    "pre-crash":     "pre-crash safety system collision warning brake",
    "pcs":           "PCS pre-crash safety collision avoidance automatic brake",
    "airbag":        "airbag SRS supplemental restraint safety inflation",
    "srs":           "SRS airbag supplemental restraint system safety",
    "lda":           "LDA lane departure alert warning road marking",
    "rsa":           "RSA road sign assist speed limit recognition",

    # Dashboard & warning lights
    "warning light": "warning indicator light dashboard meaning action",
    "warning":       "warning light indicator symbol dashboard",
    "light":         "warning indicator light meaning dashboard",
    "check engine":  "check engine malfunction indicator light MIL",
    "dashboard":     "dashboard warning indicator symbols meaning",

    # Engine & power
    "engine":        "engine petrol motor power start stop",
    "start":         "engine start stop button power on",
    "stop":          "engine stop start button power off",
    "power":         "power engine performance driving output",
    "acceleration":  "acceleration power performance driving speed",

    # Fuel & economy
    "fuel":          "fuel consumption economy efficiency petrol tank",
    "fuel economy":  "fuel consumption economy efficiency driving range",
    "mileage":       "fuel consumption economy mileage range efficiency",
    "tank":          "fuel tank capacity refuel petrol",
    "refuel":        "refuelling fuel tank petrol cap",

    # Battery
    "battery":       "battery charging 12V hybrid HV power",
    "charging":      "battery charging hybrid regenerative energy",

    # Brakes
    "brake":         "brake braking system pedal stopping ABS",
    "braking":       "braking system ABS stopping distance pedal",
    "handbrake":     "parking brake handbrake electric hold",
    "parking brake": "electric parking brake hold automatic",

    # Transmission & driving
    "gear":          "gear transmission drive reverse park neutral",
    "transmission":  "transmission automatic CVT gear shift drive",
    "cvt":           "CVT continuously variable transmission automatic gear",
    "reverse":       "reverse gear parking camera assist",
    "drive mode":    "drive mode sport eco normal power",
    "eco mode":      "eco mode fuel economy driving efficiency",
    "sport mode":    "sport mode performance power driving response",

    # Comfort & features
    "ac":            "air conditioning climate control temperature cooling",
    "air con":       "air conditioning climate control temperature cooling",
    "climate":       "climate control air conditioning heating temperature",
    "heating":       "heating climate temperature comfort",
    "cruise":        "cruise control adaptive speed maintain highway",
    "cruise control":"adaptive cruise control speed maintain distance",
    "acc":           "adaptive cruise control radar distance speed",

    # Steering & handling
    "steering":      "steering wheel power assisted control direction",
    "power steering":"electric power steering assist control",
    "turning":       "turning radius steering manoeuvre parking",

    # Tyres & wheels
    "tyre":          "tyre tire pressure wear rotation spare",
    "tire":          "tyre tire pressure wear rotation spare",
    "pressure":      "tyre tire pressure inflation TPMS warning",
    "tpms":          "tyre pressure monitoring system warning low",
    "puncture":      "puncture flat tyre spare repair kit",

    # Lights
    "headlight":     "headlight automatic LED daytime running light",
    "led":           "LED headlight automatic light sensor",
    "fog light":     "fog light front rear visibility",

    # Parking & cameras
    "parking":       "parking sensor camera assist reverse proximity",
    "camera":        "reverse parking camera display assist",
    "sensor":        "parking sensor proximity detection alert",

    # Maintenance
    "oil":           "engine oil level check change maintenance",
    "coolant":       "coolant temperature cooling engine overheat level",
    "maintenance":   "maintenance service schedule check inspection",
    "service":       "service maintenance schedule inspection interval",
    "overheat":      "overheating coolant temperature engine warning",

    # Connectivity
    "bluetooth":     "Bluetooth connectivity phone music audio",
    "usb":           "USB charging connectivity audio media",
    "display":       "display audio screen navigation system",
    "navigation":    "navigation GPS map route direction",
    "apple":         "Apple CarPlay connectivity phone",
    "android":       "Android Auto connectivity phone",

    # Seats & comfort
    "seat":          "seat adjustment position comfort lumbar",
    "seatbelt":      "seatbelt safety pretensioner load limiter",
    "belt":          "seatbelt safety wear fasten",

    # Doors & locks
    "door":          "door lock unlock child safety smart entry",
    "lock":          "door lock central locking smart entry key",
    "key":           "smart key entry start proximity keyless",
    "keyless":       "keyless smart entry start proximity key fob",
}


def expand_query(question: str) -> str:
    """
    Expand short or keyword queries into richer search strings.
    This helps FAISS find relevant chunks even for vague questions.
    """
    q_lower = question.lower().strip().rstrip("?")

    # Direct match first
    if q_lower in KEYWORD_EXPANSIONS:
        return KEYWORD_EXPANSIONS[q_lower]

    # Partial match for short queries (1-3 words)
    if len(q_lower.split()) <= 3:
        for keyword, expansion in KEYWORD_EXPANSIONS.items():
            if keyword in q_lower:
                return expansion

    return question


class RAGPipeline:
    """
    Full RAG pipeline:
    1. Load FAISS index + metadata
    2. Embed + expand user query
    3. Retrieve top-k relevant chunks
    4. Generate friendly layman explanation via Groq LLM
    """

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

        self._index = faiss.read_index(INDEX_PATH)
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
        """
        Embed the query (with expansion), search FAISS, return top-k unique chunks.
        Averages embeddings of original + expanded query for best coverage.
        """
        expanded = expand_query(question)
        queries  = list({question, expanded})   # deduplicate

        vecs = self._embedder.encode(queries, convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(vecs)

        # Average all query vectors → single representative vector
        vec = vecs.mean(axis=0, keepdims=True).astype("float32")
        faiss.normalize_L2(vec)

        scores, indices = self._index.search(vec, TOP_K)

        results = []
        seen    = set()
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk_meta = dict(self._metadata[idx])
            # Deduplicate by first 120 chars of text
            key = chunk_meta["text"][:120]
            if key in seen:
                continue
            seen.add(key)
            chunk_meta["score"] = float(score)
            results.append(chunk_meta)

        return results

    def generate(self, question: str, retrieved_chunks: list[dict]) -> str:
        """Build a friendly prompt and call Groq LLM."""
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            header = f"[Section {i} — {chunk['source']} | Page {chunk['page']}]"
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
        """End-to-end: retrieve + generate. Returns answer, chunks, error."""
        if not self._ready:
            return {"answer": "", "chunks": [], "error": "Pipeline not loaded. Call load() first."}
        try:
            chunks = self.retrieve(question)
            if not chunks:
                return {"answer": "I couldn't find information about that in the manual. Try rephrasing your question!", "chunks": [], "error": ""}
            answer = self.generate(question, chunks)
            return {"answer": answer, "chunks": chunks, "error": ""}
        except Exception as exc:
            return {"answer": "", "chunks": [], "error": str(exc)}