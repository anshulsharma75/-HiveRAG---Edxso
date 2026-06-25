"""
Evaluation Script — Edxso AI Engineer Intern Assignment
Metrics:
  1. Keyword Token F1  (precision / recall / F1 on word overlap)
  2. Cosine Similarity  (TF-IDF vector space)
"""

import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rag_pipeline import rag_query

# ─────────────────────────────────────────────
# GROUND TRUTH
# ─────────────────────────────────────────────

QA_PAIRS = [
    {
        "question": "What is the minimum internal temperature for a Langstroth Hive in winter?",
        "expected":  "Above 40 degrees Fahrenheit.",
    },
    {
        "question": "Why are entrance reducers used?",
        "expected":  "To prevent field mice from entering.",
    },
    {
        "question": "How do beekeepers control condensation?",
        "expected":  "By using insulated wraps and moisture quilt boxes.",
    },
]

# ─────────────────────────────────────────────
# METRIC 1 — Keyword Token F1
# ─────────────────────────────────────────────

def tokenise(text: str) -> set:
    return set(re.sub(r"[^\w\s]", "", text.lower()).split())

def keyword_f1(predicted: str, expected: str) -> dict:
    pred, exp = tokenise(predicted), tokenise(expected)
    if not pred or not exp:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp        = len(pred & exp)
    precision = tp / len(pred)
    recall    = tp / len(exp)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": round(precision, 3),
            "recall":    round(recall,    3),
            "f1":        round(f1,        3)}

# ─────────────────────────────────────────────
# METRIC 2 — Cosine Similarity  (TF-IDF)
# ─────────────────────────────────────────────

def semantic_similarity(predicted: str, expected: str) -> float:
    vec = TfidfVectorizer()
    mat = vec.fit_transform([predicted, expected]).toarray()
    score = cosine_similarity([mat[0]], [mat[1]])[0][0]
    return round(float(score), 4)

# ─────────────────────────────────────────────
# RUN EVALUATION
# ─────────────────────────────────────────────

def evaluate():
    print("\n" + "="*70)
    print("  Edxso Assignment — Evaluation Report")
    print("="*70)

    all_f1, all_cos = [], []

    for i, pair in enumerate(QA_PAIRS, 1):
        q        = pair["question"]
        expected = pair["expected"]
        predicted = rag_query(q, verbose=False)

        kw  = keyword_f1(predicted, expected)
        cos = semantic_similarity(predicted, expected)
        all_f1.append(kw["f1"])
        all_cos.append(cos)

        print(f"\n[Q{i}] {q}")
        print(f"  Expected  : {expected}")
        print(f"  Predicted : {predicted[:120]}")
        print(f"  Keyword F1 → P={kw['precision']}  R={kw['recall']}  F1={kw['f1']}")
        print(f"  Cosine Sim → {cos}")

    avg_f1  = round(np.mean(all_f1),  4)
    avg_cos = round(np.mean(all_cos), 4)

    PASS_F1, PASS_COS = 0.30, 0.50
    print("\n" + "-"*70)
    print(f"  AGGREGATE  (avg over {len(QA_PAIRS)} questions)")
    print(f"  Mean Keyword F1       : {avg_f1}  {'✅ PASS' if avg_f1  >= PASS_F1  else '❌ FAIL'}  (threshold ≥ {PASS_F1})")
    print(f"  Mean Cosine Similarity: {avg_cos}  {'✅ PASS' if avg_cos >= PASS_COS else '❌ FAIL'}  (threshold ≥ {PASS_COS})")
    print("="*70 + "\n")

    return {"mean_f1": avg_f1, "mean_cosine": avg_cos}

if __name__ == "__main__":
    evaluate()
