"""TF-IDF embedding helper.

Provides a simple TF-IDF based embedding baseline using scikit-learn's
TfidfVectorizer. This is intended as a deterministic, fast baseline for
retrieval experiments (better than random/dummy, weaker than OpenAI
semantic embeddings).

Key behaviors:
- Fit `TfidfVectorizer` on the provided document chunk texts (unigrams+bigrams,
  stop words removed, capped at `max_features`).
- Transform texts and queries to dense `np.float32` vectors and L2-normalize
  them so they work with the existing FAISS cosine pipeline.

Why TF-IDF?
- Captures token-level term importance and some local context via bigrams.
- Cheap to compute and interpretable; a useful mid-quality baseline.
"""
from __future__ import annotations

from pathlib import Path
import pickle
from typing import Iterable, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


DEFAULT_CONFIG = {
    "ngram_range": (1, 2),
    "stop_words": "english",
    "max_features": 4096,
    "norm": "l2",
}


def fit_tfidf(texts: Iterable[str], max_features: int = 4096) -> Tuple[TfidfVectorizer, np.ndarray]:
    """Fit a TF-IDF vectorizer on `texts` and return (vectorizer, embeddings).

    Args:
        texts: Iterable of document chunk texts (strings).
        max_features: maximum number of features for the vectorizer.

    Returns:
        (vectorizer, embeddings) where embeddings is an (N, D) float32 numpy array
        that is L2-normalized per-row.
    """
    vec = TfidfVectorizer(ngram_range=DEFAULT_CONFIG["ngram_range"], stop_words=DEFAULT_CONFIG["stop_words"], max_features=max_features)
    X = vec.fit_transform(list(texts))
    # Convert to dense float32 and L2-normalize
    X_dense = X.astype(np.float32).toarray()
    # sklearn's normalize returns float64 by default; convert back to float32
    X_norm = normalize(X_dense, norm="l2", axis=1).astype(np.float32)
    return vec, X_norm


def transform_query(vectorizer: TfidfVectorizer, query: str) -> np.ndarray:
    """Transform a query string using an already-fitted vectorizer.

    Returns an L2-normalized 1-D float32 numpy array matching the vectorizer dim.
    """
    x = vectorizer.transform([query])
    x_dense = x.astype(np.float32).toarray()
    x_norm = normalize(x_dense, norm="l2", axis=1).astype(np.float32)
    return x_norm[0]


def save_vectorizer(vectorizer: TfidfVectorizer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(vectorizer, fh)


def load_vectorizer(path: Path) -> Optional[TfidfVectorizer]:
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


__all__ = ["fit_tfidf", "transform_query", "save_vectorizer", "load_vectorizer"]
