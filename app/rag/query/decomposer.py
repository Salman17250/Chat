import re
from typing import List
from app.rag.query.normalizer import STOP_WORDS

class QueryDecomposer:
    """
    Deterministic rule-based query decomposer for multi-part questions.
    Splits compound questions into individual sub-queries without an LLM.
    """

    # Clause separators that indicate distinct sub-questions
    SPLIT_PATTERNS = [
        r'\s+(?:and|as\s+well\s+as|also|plus)\s+(?=(?:what|how|where|when|can|do|does|which|is|are|why)\b)',
        r'\s*;\s*',
        r'\?\s+(?=[A-Z0-9])',
    ]

    def decompose(self, query: str) -> List[str]:
        """
        Decomposes a query into sub-queries if it contains multiple independent questions.
        If no meaningful multi-part structure is found, returns [query].
        """
        q = query.strip()
        if not q or len(q.split()) < 6:
            return [q]

        candidates = [q]

        for pattern in self.SPLIT_PATTERNS:
            new_candidates = []
            for cand in candidates:
                parts = re.split(pattern, cand, flags=re.IGNORECASE)
                for p in parts:
                    clean_p = p.strip().rstrip("?").strip()
                    if clean_p:
                        new_candidates.append(clean_p)
            candidates = new_candidates

        # Validate that sub-parts are substantive questions/clauses (at least 3 words, not just noise)
        valid_sub_queries = []
        for cand in candidates:
            words = [w.lower() for w in cand.split() if w.lower() not in STOP_WORDS]
            if len(words) >= 2 or len(cand.split()) >= 3:
                valid_sub_queries.append(cand)

        if len(valid_sub_queries) >= 2:
            return valid_sub_queries

        return [q]
