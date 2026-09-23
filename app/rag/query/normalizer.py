import re
import unicodedata
from typing import List, Set, Tuple

# Pure grammatical stopwords that carry zero semantic specificity in document retrieval
STOP_WORDS: Set[str] = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "about", "into", "through", "during", "before", "after", "above", "below",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "done",
    "can", "could", "shall", "should", "will", "would", "may", "might", "must",
    "how", "where", "what", "when", "why", "which", "who", "whom", "whose",
    "and", "or", "but", "nor", "so", "yet", "if", "that",
    "this", "these", "those", "it", "its",
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself",
    "he", "him", "his", "she", "her", "hers",
    "they", "them", "their", "theirs",
    "we", "us", "our", "ours",
    "please", "tell", "explain", "describe", "show", "give", "know", "want", "let",
    "also", "just", "like", "really", "kindly", "need", "find"
}

# Standard English irregular plural mapping (strictly linguistic, zero domain bias)
IRREGULAR_PLURAL_MAP = {
    "children": "child",
    "people": "person",
    "men": "man",
    "women": "woman",
    "criteria": "criterion",
    "analyses": "analysis",
    "theses": "thesis",
    "crises": "crisis",
    "hypotheses": "hypothesis",
    "indices": "index",
    "matrices": "matrix",
    "vertices": "vertex",
    "appendices": "appendix",
    "feet": "foot",
    "teeth": "tooth",
    "geese": "goose",
    "mice": "mouse",
    "lives": "life",
    "halves": "half",
    "leaves": "leaf",
    "knives": "knife",
    "wives": "wife",
    "shelves": "shelf",
    "thieves": "thief",
}

def singularize(word: str) -> str:
    """
    Deterministic algorithmic singularization without external dependencies.
    Domain-agnostic: operates by standard morphological inflection rules.
    """
    w = word.lower()
    if w in IRREGULAR_PLURAL_MAP:
        return IRREGULAR_PLURAL_MAP[w]
    if len(w) > 4:
        # e.g. policies -> policy, categories -> category, inquiries -> inquiry
        if w.endswith("ies") and not w.endswith("eies"):
            return w[:-3] + "y"
        # e.g. processes -> process, boxes -> box, branches -> branch
        if w.endswith("es") and w[-3] in ("s", "x", "z") or w.endswith(("ches", "shes")):
            return w[:-2]
        # e.g. quotations -> quotation, users -> user, requirements -> requirement
        if w.endswith("s") and not w.endswith("ss"):
            return w[:-1]
    elif len(w) == 4 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w

from dataclasses import dataclass

@dataclass
class NormalizedQueryResult:
    normalized_text: str
    tokens: List[str]

    def __iter__(self):
        return iter((self.normalized_text, self.tokens))

    def __getitem__(self, item):
        return (self.normalized_text, self.tokens)[item]

class QueryNormalizer:
    """
    Deterministic NLP query normalizer.
    Cleans punctuation, normalizes whitespace, handles singular/plurals, and filters noise words.
    """

    def normalize(self, query: str) -> NormalizedQueryResult:
        if not query:
            return NormalizedQueryResult("", [])

        # 1. Unicode normalization & lowercasing
        text = unicodedata.normalize("NFKC", query).lower().strip()

        # 2. Punctuation removal (preserve hyphen within compound terms like multi-stage)
        text = re.sub(r'[^\w\s\-]', ' ', text)

        # 3. Tokenization & whitespace collapse
        tokens = [t.strip('-') for t in text.split() if t.strip('-')]

        # 4. Singularization & noise reduction
        normalized_tokens: List[str] = []
        for t in tokens:
            sing = singularize(t)
            if sing not in STOP_WORDS and len(sing) >= 2:
                normalized_tokens.append(sing)

        # If all tokens were filtered (e.g. query was "who is it"), keep original tokens
        if not normalized_tokens:
            normalized_tokens = [singularize(t) for t in tokens if t]

        normalized_query_str = " ".join(normalized_tokens)
        return NormalizedQueryResult(normalized_query_str, normalized_tokens)

