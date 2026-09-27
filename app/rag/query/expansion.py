import re
from typing import List, Dict, Set, Optional, Tuple, Union
from app.rag.vocabulary.dynamic_vocabulary import DynamicCorpusVocabulary
from app.rag.query.normalizer import STOP_WORDS

class QueryExpander:
    """
    Deterministic Semantic Query Expansion Engine.
    Dynamic and document-aware:
      - Uses domain vocabulary, acronyms, and terminology extracted from uploaded documents
      - Morphological root variants (e.g. configure <-> configuration <-> configuring)
      - Heading terminology and section-specific keywords
      - ZERO hardcoded business answers or hallucinated concepts.
    """

    def __init__(self, vocab: Optional[DynamicCorpusVocabulary] = None):
        self.vocab = vocab or DynamicCorpusVocabulary()

    def set_vocabulary(self, vocab: DynamicCorpusVocabulary):
        """Updates the active dynamic corpus vocabulary."""
        self.vocab = vocab

    def expand(
        self,
        query_or_tokens: Union[List[str], str],
        vocab: Optional[Union[DynamicCorpusVocabulary, int]] = None,
        max_expansions_per_token: int = 2,
        include_headings: bool = True
    ) -> List[str]:
        """
        Expands query tokens using dynamic document vocabulary:
        - Typo corrections from document vocabulary
        - Morphological variants found in the corpus (e.g. quote <-> quotation, pipeline <-> pipelines)
        - Acronym expansion (e.g. CRM -> Customer Relationship Management)
        - Statistically co-occurring terms in the corpus
        """
        # Handle polymorphic argument passing
        active_vocab = self.vocab
        limit = max_expansions_per_token
        if isinstance(vocab, DynamicCorpusVocabulary):
            active_vocab = vocab
        elif isinstance(vocab, int):
            limit = vocab

        if isinstance(query_or_tokens, str):
            raw_tokens = [w for w in re.findall(r'\b\w+\b', query_or_tokens.lower())]
        else:
            raw_tokens = list(query_or_tokens)

        expanded: Set[str] = set()
        clean_tokens = [t.lower().strip() for t in raw_tokens if t.strip() and t.lower() not in STOP_WORDS]

        for token in clean_tokens:
            expanded.add(token)

            # 1. Typo correction against corpus vocabulary
            typo_fix = active_vocab.correct_typo(token)
            if typo_fix and typo_fix not in expanded:
                expanded.add(typo_fix)
                token = typo_fix

            # 2. Acronym expansion
            acronym_expansion = active_vocab.expand_acronym(token)
            if acronym_expansion:
                for word in acronym_expansion.lower().split():
                    if word not in STOP_WORDS:
                        expanded.add(word)

            # 3. Morphological variants from document corpus
            morphs = active_vocab.get_morphological_variants(token)
            for m in morphs[:limit]:
                if m not in STOP_WORDS:
                    expanded.add(m)

            # 4. Related co-occurring terms
            related = active_vocab.get_related_terms(token, top_k=limit)
            for r in related:
                if r not in STOP_WORDS:
                    expanded.add(r)

        return list(expanded)

    def build_expanded_query_string(self, original_query: str, tokens: List[str]) -> str:
        """Appends corpus-derived expanded terms to the query while preserving original query weight."""
        expanded_terms = self.expand(tokens)
        orig_lower = original_query.lower()
        new_terms = [t for t in expanded_terms if t not in orig_lower]
        if new_terms:
            return f"{original_query} {' '.join(new_terms)}"
        return original_query

    def expand_query(self, query: str) -> Tuple[str, List[str]]:
        tokens = [t.strip() for t in query.split() if t.strip() and t.lower() not in STOP_WORDS]
        expanded_terms = self.expand(tokens)
        new_terms = [t for t in expanded_terms if t.lower() not in [tok.lower() for tok in tokens]]
        expanded_query_str = self.build_expanded_query_string(query, tokens)
        return expanded_query_str, new_terms

# Alias for consistent naming
SynonymExpander = QueryExpander
