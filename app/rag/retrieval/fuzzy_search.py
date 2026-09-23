from typing import List, Tuple, Optional
from rapidfuzz import fuzz
from app.rag.ingestion.models import DocumentChunk
from app.rag.query.normalizer import QueryNormalizer

class FuzzySearchEngine:
    """Fuzzy matching engine for spelling mistakes and typos using RapidFuzz."""

    def __init__(self, normalizer: Optional[QueryNormalizer] = None):
        self.normalizer = normalizer or QueryNormalizer()
        self.chunks: List[DocumentChunk] = []
        self._preprocessed: List[Tuple[DocumentChunk, str, str]] = []

    def index_chunks(self, chunks: List[DocumentChunk]):
        self.chunks = list(chunks)
        self._preprocessed = [
            (
                c,
                c.section.lower() if c.section else "",
                c.text[:500].lower() if c.text else ""
            )
            for c in chunks
        ]

    def search(
        self,
        query: str,
        top_k: int = 15,
        threshold: float = 65.0
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Computes fuzzy match ratio between query string and chunk text / headings.
        Returns top_k (DocumentChunk, score_0_to_1).
        """
        if not query or not self._preprocessed:
            return []

        q_clean = query.strip().lower()
        scored_pairs: List[Tuple[DocumentChunk, float]] = []

        for chunk, sec_lower, text_sample in self._preprocessed:
            # Check section heading match (high signal for typos in section names)
            section_score = 0.0
            if sec_lower:
                section_score = fuzz.token_sort_ratio(q_clean, sec_lower)

            # Check text match using token_set_ratio for partial match resilience
            text_score = fuzz.token_set_ratio(q_clean, text_sample)

            best_ratio = max(section_score, text_score)

            if best_ratio >= threshold:
                norm_score = best_ratio / 100.0
                scored_pairs.append((chunk, norm_score))

        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return scored_pairs[:top_k]
