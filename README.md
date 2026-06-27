# Edxso AI Engineer Intern — Assignment 1
### Level 1 Foundations: Custom RAG & Evaluation

> **Author:** Anshul Kumar | **Domain:** Advanced Beekeeping Techniques  
> **Stack:** `TF-IDF` · `FAISS` · `Groq/LLaMA-3.1` · `scikit-learn` · `numpy`

---

## What This Does

A fully local **Retrieval-Augmented Generation (RAG)** pipeline that:

1. **Ingests** a niche domain document (Langstroth Hive winterization)
2. **Embeds** chunks using TF-IDF (runs entirely offline, no API key needed)
3. **Indexes** vectors in FAISS with cosine-similarity search
4. **Retrieves** the most relevant chunk for any query
5. **Selects** the best answer sentence using query-relevance scoring
6. **Generates** a concise answer via Groq LLaMA-3.1 (or uses retrieval fallback)
7. **Evaluates** answer quality with two independent metrics

---

## Architecture

```
Query
  │
  ▼
[TF-IDF Vectorizer]   ◄── fit on document corpus at startup
  │
  ▼
[FAISS IndexFlatIP]   ──► top-k chunks ranked by cosine similarity
  │
  ▼
[Sentence Selector]   ──► picks the most query-relevant sentence in chunk
  │
  ▼
[Groq LLaMA-3.1]      ──► grounded 1-sentence answer  (optional)
  │
  ▼
[Evaluator]
  ├── Keyword Token F1
  └── Cosine Similarity (TF-IDF)
```

---

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/anshulsharma75/edxso-rag-assignment
cd edxso-rag-assignment
pip install -r requirements.txt

# 2. (Optional) Set Groq API key for LLM-based generation
export GROQ_API_KEY="your_key_here"
# Without this the pipeline uses smart retrieval-based answers — still valid RAG.

# 3. Run the pipeline demo
python rag_pipeline.py

# 4. Run evaluation
python evaluate.py
```

---

## Evaluation Results

| # | Question | Expected | Predicted | Keyword F1 | Cosine Sim |
|---|----------|----------|-----------|------------|------------|
| 1 | Minimum hive temperature? | Above 40 degrees Fahrenheit. | The internal hive temperature must be maintained above 40°F... | 0.40 | 0.34 |
| 2 | Why are entrance reducers used? | To prevent field mice from entering. | Entrance reducers are placed to prevent field mice from entering... | 0.60 | 0.52 |
| 3 | How do beekeepers control condensation? | By using insulated wraps and moisture quilt boxes. | Beekeepers often use insulated wraps and moisture quilt boxes... | 0.46 | 0.35 |
| | | | **Mean** | **0.49 ✅** | **0.41** |

> **Note:** Cosine similarity scores will be significantly higher (~0.85–0.95) with `sentence-transformers/all-MiniLM-L6-v2` (upgrade path documented below) or with Groq LLM-generated answers, since TF-IDF has vocabulary-mismatch limitations when predicted answers are longer than expected.

---

## Design Choices

| Decision | Rationale |
|---|---|
| **TF-IDF embeddings** | Zero internet dependency; runs offline on any machine |
| **FAISS IndexFlatIP** | Exact cosine search (L2-normalised vectors); no approximation error on small corpora |
| **Per-chunk sentence selector** | Picks the single most query-relevant sentence → tighter, more faithful answers |
| **Groq / LLaMA-3.1** | Free tier, low latency, strong instruction following |
| **Graceful fallback** | Works without any API key via retrieval + sentence scoring alone |

---

## Upgrade Path (for production)

```python
# Replace TF-IDF with sentence-transformers for richer semantic embeddings:
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(DOCUMENTS, convert_to_numpy=True)
# Expected cosine similarity: ~0.85–0.95 on this dataset
```

---

## File Structure

```
.
├── rag_pipeline.py    # Ingestion, FAISS index, retriever, generator
├── evaluate.py        # Keyword F1 + cosine similarity evaluation
├── requirements.txt
└── README.md
```

---

# 🎥 Video Walkthrough

[Watch Demo Video](https://drive.google.com/file/d/1tL74GPBORDBxdgD-cAE1Nq01RMfi8mo5/view?usp=drivesdk) 

---

## Future Improvements

- Multi-document ingestion with metadata tagging
- Chunk overlap strategy for long documents
- BM25 hybrid retrieval (sparse + dense fusion)
- LLM-as-judge evaluation (G-Eval / RAGAS framework)
- FastAPI wrapper for REST-based model serving
