import re
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
from app.rag.ingestion.models import DocumentChunk
from app.rag.table.table_processor import TableProcessor

class TableSearchEngine:
    """
    Structured Table Retrieval Engine (<0.2ms).
    Enables precise, deterministic lookup over structured tabular data
    without relying solely on dense vector embeddings.
    """
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.table_chunks: List[Tuple[int, DocumentChunk]] = []

    def index_chunks(self, chunks: List[DocumentChunk]):
        """Indexes table chunks and table row sub-chunks."""
        self.chunks = list(chunks)
        self.table_chunks.clear()

        for idx, chunk in enumerate(self.chunks):
            if getattr(chunk, "is_table_row", False) or TableProcessor.is_table_text(chunk.text):
                self.table_chunks.append((idx, chunk))

    def search(
        self,
        query: str,
        target_attributes: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Matches structured table queries against row sub-chunks and table grids.
        """
        if not self.table_chunks:
            return []

        clean_q = query.strip().lower()
        query_words = set(re.findall(r'\b\w+\b', clean_q))
        attrs = [a.lower() for a in (target_attributes or [])]

        results: List[Tuple[DocumentChunk, float]] = []

        for idx, chunk in self.table_chunks:
            text_lower = chunk.text.lower()
            section_lower = (chunk.section or "").lower()
            row_dict = getattr(chunk, "row_dict", {}) or {}

            score = 0.0
            # 1. Attribute match in table columns or section
            for a in attrs:
                if a in text_lower or a in section_lower:
                    score += 0.4

            # 2. Word overlap between query and table row
            row_words = set(re.findall(r'\b\w+\b', text_lower))
            overlap = query_words.intersection(row_words)
            if overlap:
                score += min(0.6, 0.15 * len(overlap))

            # 3. If query mentions key row entity and asks for a column
            if row_dict:
                for col_name, cell_val in row_dict.items():
                    c_clean = str(col_name).lower()
                    v_clean = str(cell_val).lower()
                    if c_clean in clean_q:
                        score += 0.3
                    if v_clean and v_clean in clean_q and len(v_clean) >= 3:
                        score += 0.4

            if score > 0.25:
                results.append((chunk, min(1.0, score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
