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
    # Protect words that end in 's' but are not plural nouns
    if w in ("does", "goes", "has", "was", "this", "thus", "lens", "boss", "pass", "mass", "news", "plus", "yes", "his"):
        return w
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

# Contractions mapping (English standard)
CONTRACTIONS_MAP = {
    "what's": "what is",
    "whats": "what is",
    "how's": "how is",
    "hows": "how is",
    "where's": "where is",
    "wheres": "where is",
    "who's": "who is",
    "whos": "who is",
    "that's": "that is",
    "thats": "that is",
    "there's": "there is",
    "theres": "there is",
    "here's": "here is",
    "it's": "it is",
    "can't": "cannot",
    "cant": "cannot",
    "won't": "will not",
    "wont": "will not",
    "don't": "do not",
    "dont": "do not",
    "doesn't": "does not",
    "doesnt": "does not",
    "didn't": "did not",
    "didnt": "did not",
    "haven't": "have not",
    "hasn't": "has not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "i'm": "i am",
    "im": "i am",
    "you're": "you are",
    "they're": "they are",
    "we're": "we are",
    "let's": "let us",
}

from dataclasses import dataclass, field

@dataclass
class NormalizedQueryResult:
    normalized_text: str
    tokens: List[str]
    original_text: str = ""
    semantic_text: str = ""
    original_tokens: List[str] = field(default_factory=list)

    def __iter__(self):
        return iter((self.normalized_text, self.tokens))

    def __getitem__(self, item):
        return (self.normalized_text, self.tokens)[item]

class QueryNormalizer:
    """
    Deterministic NLP query normalizer.
    Cleans punctuation, expands contractions, normalizes whitespace, handles singular/plurals,
    and isolates core semantic tokens while preserving original query representation.
    """

    def expand_contractions(self, text: str) -> str:
        words = text.split()
        expanded = []
        for w in words:
            clean_w = w.lower().strip(".,;:?!")
            if clean_w in CONTRACTIONS_MAP:
                expanded.append(CONTRACTIONS_MAP[clean_w])
            else:
                expanded.append(w)
        return " ".join(expanded)

    def normalize(self, query: str) -> NormalizedQueryResult:
        if not query:
            return NormalizedQueryResult("", [], original_text="")

        original = query.strip()

        # 1. Expand contractions
        expanded = self.expand_contractions(original)

        # 2. Unicode normalization & lowercasing
        text = unicodedata.normalize("NFKC", expanded).lower().strip()

        # 3. Punctuation removal (preserve hyphen within compound terms like multi-stage)
        clean_text = re.sub(r'[^\w\s\-]', ' ', text)

        # 4. Tokenization & whitespace collapse
        orig_tokens = [t.strip('-') for t in clean_text.split() if t.strip('-')]

        # 5. Singularization & noise reduction
        normalized_tokens: List[str] = []
        semantic_tokens: List[str] = []
        for t in orig_tokens:
            sing = singularize(t)
            if sing not in STOP_WORDS and len(sing) >= 2:
                semantic_tokens.append(sing)
            if len(sing) >= 1:
                normalized_tokens.append(sing)

        # If all tokens were filtered (e.g. query was "who is it"), keep normalized tokens
        if not semantic_tokens:
            semantic_tokens = list(normalized_tokens)

        normalized_query_str = " ".join(normalized_tokens)
        semantic_query_str = " ".join(semantic_tokens)

        return NormalizedQueryResult(
            normalized_text=semantic_query_str,
            tokens=semantic_tokens,
            original_text=original,
            semantic_text=semantic_query_str,
            original_tokens=orig_tokens
        )


