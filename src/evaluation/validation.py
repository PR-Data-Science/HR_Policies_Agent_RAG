from __future__ import annotations

from typing import List, Dict

from src.evaluation.schemas import RAGTestSet


def validate_testset_ids(testset: RAGTestSet, documents: List[Dict]) -> None:
    """Validate that every relevant_doc_id in `testset` exists in `documents`.

    Raises:
        ValueError: if any ids are missing. The error message lists missing ids
            and the query (for easier debugging).
    """
    doc_ids = {d.get("id") for d in documents if isinstance(d, dict) and d.get("id")}

    missing_overall = {}

    for sample in testset.samples:
        missing = [rid for rid in sample.relevant_doc_ids if rid not in doc_ids]
        if missing:
            missing_overall[sample.query] = missing

    if missing_overall:
        msg_lines = ["ID Validation Error: Some relevant_doc_ids are missing from the vector store:"]
        for q, missing in missing_overall.items():
            msg_lines.append(f"- Query: {q}")
            for m in missing:
                msg_lines.append(f"    Missing id: {m}")

        full_msg = "\n".join(msg_lines)
        raise ValueError(full_msg)


def confirm_ids(testset: RAGTestSet, documents: List[Dict]) -> str:
    """Return a short confirmation string listing number of validated ids."""
    doc_ids = {d.get("id") for d in documents if isinstance(d, dict) and d.get("id")}
    total_checks = sum(len(s.relevant_doc_ids) for s in testset.samples)
    total_found = sum(1 for s in testset.samples for rid in s.relevant_doc_ids if rid in doc_ids)
    return f"Validated {total_found}/{total_checks} relevant_doc_id entries present in vector store."
