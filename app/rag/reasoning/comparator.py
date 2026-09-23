import re
from typing import List, Dict, Any, Optional
from app.rag.ingestion.models import DocumentChunk

class DeterministicComparator:
    """
    Deterministic Comparison Engine.
    Structures multi-entity side-by-side attribute comparisons from retrieved evidence.
    """
    @staticmethod
    def compare_entities(
        entity_a: str,
        entity_b: str,
        chunks_a: List[DocumentChunk],
        chunks_b: List[DocumentChunk]
    ) -> str:
        """
        Builds a structured comparison between entity A and entity B from their respective retrieved chunks.
        """
        text_a = chunks_a[0].text.strip() if chunks_a else "No specific details found."
        text_b = chunks_b[0].text.strip() if chunks_b else "No specific details found."

        doc_a = chunks_a[0].document_name if chunks_a else ""
        doc_b = chunks_b[0].document_name if chunks_b else ""

        src_info = f"Sources: {doc_a}" if doc_a == doc_b else f"Sources: {doc_a}, {doc_b}"

        return (
            f"Comparison between '{entity_a.title()}' and '{entity_b.title()}':\n\n"
            f"1. {entity_a.title()}:\n{text_a}\n\n"
            f"2. {entity_b.title()}:\n{text_b}\n\n"
            f"({src_info.strip(', ')})"
        )
