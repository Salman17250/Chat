from typing import List, Dict, Set, Optional, Tuple
from app.rag.vocabulary.dynamic_vocabulary import DynamicCorpusVocabulary

class QueryExpander:
    """
    Dynamic query expansion driven strictly by the document corpus vocabulary.
    Zero hardcoded business dictionaries.
    Uses corpus n-grams, morphological clusters, and co-occurrences discovered from uploaded files.
    """

    def __init__(self, vocab: Optional[DynamicCorpusVocabulary] = None):
        self.vocab = vocab or DynamicCorpusVocabulary()

    def set_vocabulary(self, vocab: DynamicCorpusVocabulary):
        """Updates the active dynamic corpus vocabulary."""
        self.vocab = vocab

    def expand(self, tokens: List[str], max_expansions_per_token: int = 2) -> List[str]:
        """
        Expands query tokens using dynamic document vocabulary:
        - Typo corrections from document vocabulary
        - Morphological variants found in the corpus (e.g. quote <-> quotation)
        - Statistically co-occurring terms in the corpus
        """
        expanded: Set[str] = set(tokens)

        for token in tokens:
            # 1. Check for typos against corpus vocabulary
            typo_fix = self.vocab.correct_typo(token)
            if typo_fix and typo_fix not in expanded:
                expanded.add(typo_fix)
                token = typo_fix

            # 2. Add corpus-derived related terms (morphology & co-occurrence)
            related = self.vocab.get_related_terms(token, top_k=max_expansions_per_token)
            for r in related:
                expanded.add(r)

        return list(expanded)

    def build_expanded_query_string(self, original_query: str, tokens: List[str]) -> str:
        """Appends corpus-derived expanded terms to the query while preserving original query weight."""
        expanded_terms = self.expand(tokens)
        combined = [original_query]
        orig_lower = original_query.lower()
        for term in expanded_terms:
            if term not in orig_lower:
                combined.append(term)
        return " ".join(combined)

    def expand_query(self, query: str) -> Tuple[str, List[str]]:
        tokens = [t.strip() for t in query.split() if t.strip()]
        expanded_terms = self.expand(tokens)
        new_terms = [t for t in expanded_terms if t.lower() not in [tok.lower() for tok in tokens]]
        expanded_query_str = self.build_expanded_query_string(query, tokens)
        return expanded_query_str, new_terms

# Alias for consistent naming
SynonymExpander = QueryExpander
