"""
RAG Pipeline - Edxso AI Engineer Intern Assignment
Author: Anshul Sharma
Domain: Advanced Beekeeping Techniques

Stack:
  Embeddings  : TF-IDF vectorizer (zero-dependency, no internet needed)
  Vector Store: FAISS IndexFlatIP  (cosine search after L2 normalisation)
  LLM         : Groq / LLaMA-3.1-8b-instant  (optional; falls back gracefully)
"""

import os
import re
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ─────────────────────────────────────────────────────────
# 1.  DATASET  (provided in assignment spec)
# ─────────────────────────────────────────────────────────

DOCUMENTS = [
    "Winterization of Langstroth Hives requires strict temperature management. "
    "The internal hive temperature must be maintained above 40 degrees Fahrenheit "
    "to prevent the colony from freezing.",

    "Beekeepers often use insulated wraps and moisture quilt boxes to control "
    "condensation inside the hive during winter months.",

    "Entrance reducers are placed to prevent field mice from entering during the "
    "colder months. This protects the colony from rodent damage.",
]

# ─────────────────────────────────────────────────────────
# 2.  EMBEDDINGS  (TF-IDF + L2 normalisation)
# ─────────────────────────────────────────────────────────

print("[INFO] Fitting TF-IDF vectorizer on corpus...")
vectorizer = TfidfVectorizer()
doc_matrix = vectorizer.fit_transform(DOCUMENTS).toarray().astype("float32")
doc_matrix = normalize(doc_matrix, norm="l2")   # unit vectors → IP = cosine

# ─────────────────────────────────────────────────────────
# 3.  FAISS INDEX
# ─────────────────────────────────────────────────────────

dim   = doc_matrix.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(doc_matrix)
print(f"[INFO] FAISS index ready — {index.ntotal} chunks | vocab dim={dim}")

# ─────────────────────────────────────────────────────────
# 4.  RETRIEVER
# ─────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = 1) -> list[dict]:
    """Return top-k chunks ranked by cosine similarity."""
    q_vec = vectorizer.transform([query]).toarray().astype("float32")
    q_vec = normalize(q_vec, norm="l2")
    scores, indices = index.search(q_vec, top_k)
    return [
        {"chunk": DOCUMENTS[idx], "score": round(float(s), 4)}
        for s, idx in zip(scores[0], indices[0])
    ]

# ─────────────────────────────────────────────────────────
# 5.  GENERATOR
#     With Groq: concise LLM answer grounded in context.
#     Without:   returns the first sentence of best chunk
#                (still a valid retrieval-based answer).
# ─────────────────────────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

def _best_sentence(chunk: str, query: str) -> str:
    """Return the sentence in a chunk most relevant to the query (TF-IDF cosine)."""
    sentences = re.split(r"(?<=[.!?])\s+", chunk.strip())
    if len(sentences) == 1:
        return sentences[0]
    from sklearn.metrics.pairwise import cosine_similarity as _cos
    combined = sentences + [query]
    mat = TfidfVectorizer().fit_transform(combined).toarray()
    sims = _cos([mat[-1]], mat[:-1])[0]
    return sentences[int(np.argmax(sims))]

def generate_answer(query: str, context_chunks: list[dict]) -> str:
    context_text = "\n".join(c["chunk"] for c in context_chunks)

    if not GROQ_API_KEY:
        # Fallback: return the single most query-relevant sentence
        return _best_sentence(context_chunks[0]["chunk"], query)

    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    prompt = (
        "You are a concise assistant. Answer the question in ONE short sentence "
        "using ONLY the context below. Do NOT add extra information.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\nAnswer:"
    )
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=60,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()

# ─────────────────────────────────────────────────────────
# 6.  FULL PIPELINE
# ─────────────────────────────────────────────────────────

def rag_query(question: str, top_k: int = 1, verbose: bool = True) -> str:
    chunks = retrieve(question, top_k=top_k)
    answer = generate_answer(question, chunks)

    if verbose:
        sep = "=" * 62
        print(f"\n{sep}")
        print(f"Q: {question}")
        print(f"\n  [Retrieved — score={chunks[0]['score']}]")
        print(f"  {chunks[0]['chunk'][:100]}...")
        print(f"\nA: {answer}")
        print(sep)
    return answer

# ─────────────────────────────────────────────────────────
# 7.  DEMO
# ─────────────────────────────────────────────────────────

EVAL_QUESTIONS = [
    "What is the minimum internal temperature for a Langstroth Hive in winter?",
    "Why are entrance reducers used?",
    "How do beekeepers control condensation?",
]

if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  Edxso AI Engineer Assignment — RAG Pipeline Demo")
    print("=" * 62)
    for q in EVAL_QUESTIONS:
        rag_query(q)
    print("\n[INFO] Pipeline demo complete. Run `python evaluate.py` for metrics.")
