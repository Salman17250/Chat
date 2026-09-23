from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi
from app.rag.ingestion.models import DocumentChunk
from app.rag.query.normalizer import QueryNormalizer

class BM25SearchEngine:
    """Lexical keyword search engine using BM25Okapi."""

    def __init__(self, normalizer: Optional[QueryNormalizer] = None):
        self.normalizer = normalizer or QueryNormalizer()
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[DocumentChunk] = []
        self._corpus_tokens: List[List[str]] = []

    def index_chunks(self, chunks: List[DocumentChunk]):
        """Builds or updates BM25 index over the provided document chunks."""
        self.chunks = list(chunks)
        self._corpus_tokens = []
        self._corpus_tokens_set = []
        for c in self.chunks:
            # Combine heading and body text for richer lexical index
            full_text = f"{c.section} {c.text}" if c.section else c.text
            norm_res = self.normalizer.normalize(full_text)
            toks = norm_res.tokens
            self._corpus_tokens.append(toks)
            self._corpus_tokens_set.append(set(toks))

        if self._corpus_tokens:
            self.bm25 = BM25Okapi(self._corpus_tokens)
        else:
            self.bm25 = None

    def search(
        self,
        query_tokens: List[str],
        top_k: int = 25,
        base_query_tokens: Optional[List[str]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Executes BM25 ranking for the given query tokens.
        Returns list of (DocumentChunk, normalized_score in 0.0-1.0).
        """
        if not self.bm25 or not self.chunks or not query_tokens:
            return []

        raw_scores = self.bm25.get_scores(query_tokens)
        max_score = max(raw_scores) if len(raw_scores) > 0 else 0.0

        q_tokens_set = set(t.lower() for t in query_tokens if len(t) >= 2)
        base_concepts = [t.lower() for t in (base_query_tokens or query_tokens) if len(t) >= 2]
        concept_stems = [(bt, bt[:-1] if bt.endswith("e") else bt) for bt in base_concepts]

        scored_pairs: List[Tuple[DocumentChunk, float]] = []
        for idx, score in enumerate(raw_scores):
            if score <= 0.0:
                continue

            doc_tokens = self._corpus_tokens_set[idx] if idx < len(self._corpus_tokens_set) else set(self._corpus_tokens[idx])

            if base_query_tokens:
                concept_covered = 0
                for bt, stem in concept_stems:
                    if bt in doc_tokens:
                        concept_covered += 1
                    elif len(stem) >= 4:
                        matched = any(
                            dt.startswith(stem) or (len(dt) >= 4 and stem.startswith(dt[:4]) and dt.startswith(stem[:4]))
                            for dt in doc_tokens
                        )
                        if matched:
                            concept_covered += 1
                overlap_score = (concept_covered / len(base_concepts)) if base_concepts else 0.0
                overlap_count = concept_covered
                total_req = len(base_concepts)
            else:
                overlap_count = len(q_tokens_set.intersection(doc_tokens)) if q_tokens_set else 0
                overlap_score = (overlap_count / len(q_tokens_set)) if q_tokens_set else 0.0
                total_req = len(q_tokens_set)

            if max_score > 0:
                norm_bm25 = max(0.0, score / max_score)
                lexical_score = (norm_bm25 * 0.50) + (overlap_score * 0.50)
            else:
                lexical_score = overlap_score

            # If the user asked a multi-concept query but only 1 isolated word matched, penalize
            if total_req >= 3 and overlap_count <= 1:
                lexical_score *= 0.35

            if lexical_score > 0.05:
                scored_pairs.append((self.chunks[idx], min(1.0, lexical_score)))

        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return scored_pairs[:top_k]
