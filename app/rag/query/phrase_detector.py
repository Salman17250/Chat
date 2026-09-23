import re
from typing import List, Set, Optional
from app.rag.vocabulary.dynamic_vocabulary import DynamicCorpusVocabulary

class PhraseDetector:
    """
    Detects and preserves multi-word concepts (bigrams, trigrams, and quoted phrases)
    dynamically derived from uploaded document vocabulary and user query quotes.
    Zero hardcoded domain phrases.
    """

    def __init__(self, vocab: Optional[DynamicCorpusVocabulary] = None):
        self.vocab = vocab or DynamicCorpusVocabulary()

    def set_vocabulary(self, vocab: DynamicCorpusVocabulary):
        self.vocab = vocab

    def detect_phrases(self, query: str, tokens: Optional[List[str]] = None) -> List[str]:
        phrases: List[str] = []
        q_lower = query.lower()
        if tokens is None:
            tokens = [t for t in re.sub(r'[^\w\s\-]', ' ', q_lower).split() if t]

        # 1. Quoted exact phrases (e.g. "sales order", "quotation module")
        quoted = re.findall(r'"([^"]+)"', query)
        for qp in quoted:
            c = qp.strip().lower()
            if c and c not in phrases:
                phrases.append(c)

        # 2. Dynamic document phrases appearing in query
        if self.vocab and self.vocab.significant_phrases:
            for dp in self.vocab.significant_phrases:
                if dp in q_lower and dp not in phrases and len(dp) > 3:
                    phrases.append(dp)

        # 3. Adjacency bigrams from tokens matching corpus vocabulary
        if len(tokens) >= 2 and self.vocab:
            for i in range(len(tokens) - 1):
                bigram = f"{tokens[i]} {tokens[i+1]}"
                if bigram in self.vocab.significant_phrases and bigram not in phrases:
                    phrases.append(bigram)

        return phrases
