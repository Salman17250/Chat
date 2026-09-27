import re
from typing import List, Optional
from app.rag.query.normalizer import STOP_WORDS
from app.rag.query.coreference import CoreferenceResolver

class QueryDecomposer:
    """
    Deterministic NLP Query Decomposer for Compound & Multi-Part Questions.
    Splits multi-clause questions into independent, self-contained sub-queries without an LLM.
    Propagates subjects and entities across sub-questions so pronouns like 'it' are resolved.
    
    Example:
      'What is a Pipeline, how do I configure it, and what happens after creating it?'
      ->
      Sub-question 1: 'What is a Pipeline'
      Sub-question 2: 'How do I configure Pipeline'
      Sub-question 3: 'What happens after creating Pipeline'
    """

    # Clause separators that indicate distinct sub-questions
    SPLIT_PATTERNS = [
        # Serial comma or conjunction followed by interrogative word
        re.compile(r',\s*(?:and\s+)?(?=(?:how|what|where|when|can|could|do|does|did|which|is|are|why)\b)', re.IGNORECASE),
        re.compile(r'\s+(?:and|as\s+well\s+as|also|plus)\s+(?=(?:how|what|where|when|can|could|do|does|did|which|is|are|why)\b)', re.IGNORECASE),
        re.compile(r'\s*;\s*'),
        re.compile(r'\?\s+(?=[A-Za-z0-9])'),
        # Sequential multi-question markers
        re.compile(r'\s+(?:then|after\s+that|next|subsequently)\s*,?\s*(?=(?:how|what|where|when|can|do|does)\b)', re.IGNORECASE),
    ]

    def _extract_primary_subject(self, text: str) -> Optional[str]:
        """Heuristic to extract the primary subject/entity introduced in the first clause."""
        # Check patterns like "what is a/an [Entity]", "what are [Entities]", "[Entity] kya hai"
        m_def = re.search(r'\b(?:what\s+is|what\s+are|explain|define|overview\s+of)\s+(?:a|an|the)?\s*([A-Za-z0-9_\-\s]{2,30}?)(?:\?|,|\band\b|$)', text, re.I)
        if m_def:
            cand = m_def.group(1).strip()
            words = [w for w in cand.split() if w.lower() not in STOP_WORDS]
            if words:
                return " ".join(words)

        # Check capitalized proper nouns
        prop = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        if prop:
            clean_props = [p for p in prop if p.lower() not in STOP_WORDS]
            if clean_props:
                return clean_props[0]

        # Check first prominent non-stopword
        tokens = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', text) if w.lower() not in STOP_WORDS]
        if tokens:
            return tokens[0]

        return None

    def decompose(self, query: str) -> List[str]:
        """
        Decomposes a query into self-contained sub-queries.
        Propagates subjects from earlier clauses to resolve references in later clauses.
        """
        q = query.strip()
        if not q or len(q.split()) < 4:
            return [q]

        candidates = [q]

        for pattern in self.SPLIT_PATTERNS:
            new_candidates = []
            for cand in candidates:
                parts = pattern.split(cand)
                for p in parts:
                    clean_p = p.strip().rstrip("?").strip()
                    if clean_p:
                        new_candidates.append(clean_p)
            candidates = new_candidates

        # Validate that sub-parts are substantive questions/clauses (at least 2 content words)
        valid_sub_queries: List[str] = []
        for cand in candidates:
            words = [w.lower() for w in re.findall(r'\b\w+\b', cand) if w.lower() not in STOP_WORDS]
            if len(words) >= 1 or len(cand.split()) >= 3:
                valid_sub_queries.append(cand)

        if len(valid_sub_queries) <= 1:
            return [q]

        # Cross-clause context propagation:
        # Determine the primary subject introduced in the early sub-queries
        primary_subject = None
        for sub_q in valid_sub_queries:
            subj = self._extract_primary_subject(sub_q)
            if subj:
                primary_subject = subj
                break

        # Resolve coreferences in subsequent sub-queries using primary_subject
        resolved_sub_queries: List[str] = []
        for idx, sub_q in enumerate(valid_sub_queries):
            if idx > 0 and primary_subject:
                # Resolve pronouns like "it", "this", "that"
                resolved_sub, was_res, _ = CoreferenceResolver.resolve_references(
                    query=sub_q,
                    context_entity=primary_subject
                )
                if was_res:
                    resolved_sub_queries.append(resolved_sub)
                    continue

                # If sub-query has no explicit entity, append the primary subject context
                has_entity = bool(self._extract_primary_subject(sub_q))
                if not has_entity and primary_subject.lower() not in sub_q.lower():
                    resolved_sub_queries.append(f"{sub_q} {primary_subject}")
                    continue

            resolved_sub_queries.append(sub_q)

        return resolved_sub_queries
