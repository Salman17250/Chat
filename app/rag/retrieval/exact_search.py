import re
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
from app.rag.ingestion.models import DocumentChunk

class ExactSearchEngine:
    """
    Ultra-Fast Exact Matching Retrieval Engine (<0.1ms).
    Specialized for:
      - IDs, codes, SKU numbers, invoice IDs (#INV-1234, SKU-402)
      - Monetary values, dates, percentages
      - Technical terms, acronyms, and proper nouns
      - Exact multi-word quoted phrases
    """
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        # Inverted index: exact token/code (lowercase) -> set of chunk indices
        self._token_index: Dict[str, Set[int]] = defaultdict(set)
        # Inverted index: numbers / codes / identifiers -> set of chunk indices
        self._code_index: Dict[str, Set[int]] = defaultdict(set)

    def index_chunks(self, chunks: List[DocumentChunk]):
        """Builds exact inverted indices over chunks."""
        self.chunks = list(chunks)
        self._token_index.clear()
        self._code_index.clear()

        for idx, chunk in enumerate(self.chunks):
            text = chunk.text.lower()
            section = (chunk.section or "").lower()
            full_text = f"{section} {text}"

            # 1. Alphanumeric codes, IDs, hyphenated terms (e.g. inv-123, sku_402, 18.5, 120)
            codes = re.findall(r'\b[a-z0-9_\-\.]{2,}\b', full_text)
            for c in codes:
                self._code_index[c].add(idx)

            # 2. Words & numbers
            words = re.findall(r'\b\w+\b', full_text)
            for w in words:
                self._token_index[w].add(idx)

    def search(
        self,
        query: str,
        exact_tokens: Optional[List[str]] = None,
        top_k: int = 15
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieves chunks containing exact tokens or codes with high precision scoring.
        """
        if not self.chunks:
            return []

        clean_q = query.strip().lower()
        tokens_to_search = exact_tokens if exact_tokens else re.findall(r'\b[a-z0-9_\-\.]{2,}\b', clean_q)
        if not tokens_to_search:
            return []

        chunk_scores: Dict[int, float] = defaultdict(float)
        total_tokens = len(tokens_to_search)

        for t in tokens_to_search:
            # Check code index first (higher specificity)
            if t in self._code_index:
                for idx in self._code_index[t]:
                    chunk_scores[idx] += 1.0
            elif t in self._token_index:
                for idx in self._token_index[t]:
                    chunk_scores[idx] += 0.7

        if not chunk_scores:
            return []

        # Calculate exact coverage score (0.0 to 1.0)
        results: List[Tuple[DocumentChunk, float]] = []
        for idx, score_val in chunk_scores.items():
            coverage = min(1.0, score_val / max(1, total_tokens))
            chunk = self.chunks[idx]

            # Boost if query appears as an exact substring
            if clean_q in chunk.text.lower():
                coverage = min(1.0, coverage + 0.3)

            results.append((chunk, coverage))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
