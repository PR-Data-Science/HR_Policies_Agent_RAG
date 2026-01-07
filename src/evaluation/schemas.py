from __future__ import annotations

"""Pydantic schemas for RAG retrieval evaluation.

This module defines a small, versioned gold dataset schema used to
evaluate retrieval-only performance (recall/precision/MRR/nDCG). This is
intentionally NOT for generation evaluation.
"""

from typing import List
from pydantic import BaseModel


class RAGTestSample(BaseModel):
    """A single RAG retrieval test sample.

    Attributes:
        query: The user's retrieval query text.
        relevant_doc_ids: List of stable document/chunk IDs (strings) that are
            considered relevant for this query. These IDs should match the
            `id` field used in the FAISS documents JSON.
    """

    query: str
    relevant_doc_ids: List[str]


class RAGTestSet(BaseModel):
    """A collection of `RAGTestSample` entries forming a gold test set.

    Use `RAGTestSet.parse_file(path)` or `RAGTestSet.parse_raw(json_text)` to
    load the dataset from a JSON file. The file must contain an array of
    objects matching `RAGTestSample`.
    """

    samples: List[RAGTestSample]

    @classmethod
    def load_from_file(cls, path: str) -> "RAGTestSet":
        """Convenience loader to read a JSON file and parse into models."""
        return cls.parse_file(path)
