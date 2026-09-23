import re
import os
import json
import math
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple, Optional, Any
from rapidfuzz import fuzz
from app.rag.ingestion.models import DocumentChunk, DocumentBlock
from app.rag.query.normalizer import STOP_WORDS, singularize
from app.rag.vocabulary.acronym_extractor import AcronymExtractor

class DynamicCorpusVocabulary:
    """
    100% Dynamic Document Vocabulary Layer.
    Extracts terminology, n-grams, acronyms, co-occurrence associations, and section-specific keywords
    directly from uploaded document chunks without any hardcoded business dictionaries.
    Supports isolated tenant-level serialization and restoration.
    """

    def __init__(self):
        # Term frequencies across corpus
        self.term_freq: Counter = Counter()
        # Document/Chunk frequency (number of chunks containing the term)
        self.chunk_freq: Counter = Counter()
        # Total chunks indexed
        self.total_chunks: int = 0

        # Acronym mappings: ACRONYM -> Full Phrase and vice-versa
        self.acronyms: Dict[str, str] = {}
        self.phrase_to_acronym: Dict[str, str] = {}

        # Frequent multi-word phrases (bigrams, trigrams)
        self.phrase_freq: Counter = Counter()
        self.significant_phrases: Set[str] = set()

        # Section-specific vocabulary: section -> set of keywords
        self.section_keywords: Dict[str, Set[str]] = defaultdict(set)

        # Morphological root clusters: stem/prefix -> list of corpus terms
        self.stem_clusters: Dict[str, Set[str]] = defaultdict(set)

        # All unique vocabulary tokens
        self.vocab_tokens: Set[str] = set()

    def build_from_chunks(self, chunks: List[DocumentChunk]):
        """Builds lightweight in-memory statistical vocabulary, acronyms, and clusters from chunks."""
        self.term_freq.clear()
        self.chunk_freq.clear()
        self.acronyms.clear()
        self.phrase_to_acronym.clear()
        self.phrase_freq.clear()
        self.significant_phrases.clear()
        self.section_keywords.clear()
        self.stem_clusters.clear()
        self.vocab_tokens.clear()
        self.total_chunks = len(chunks)

        if not chunks:
            return

        for chunk in chunks:
            raw_text = f"{chunk.section} {chunk.text}" if chunk.section else chunk.text
            sec = (chunk.section or "General").strip()

            # 1. Extract acronyms from raw text
            chunk_acronyms = AcronymExtractor.extract_from_text(raw_text)
            for acr, phrase in chunk_acronyms.items():
                self.acronyms[acr] = phrase
                self.phrase_to_acronym[phrase.lower()] = acr
                self.significant_phrases.add(phrase.lower())

            # 2. Tokenize and normalize for term statistics
            # Split text into sentences for localized analysis
            sentences = re.split(r'[.\n!?]+', raw_text)
            chunk_seen_tokens: Set[str] = set()

            for sent in sentences:
                sent_clean = re.sub(r'[^\w\s\-]', ' ', sent.lower())
                raw_words = [w.strip('-') for w in sent_clean.split() if len(w.strip('-')) >= 2]

                filtered_words = [singularize(w) for w in raw_words if w not in STOP_WORDS and len(w) >= 3]

                # Unigram statistics & morphological clustering
                for w in filtered_words:
                    self.term_freq[w] += 1
                    chunk_seen_tokens.add(w)
                    self.vocab_tokens.add(w)
                    if sec:
                        self.section_keywords[sec].add(w)

                    prefix = w[:4] if len(w) >= 4 else w
                    self.stem_clusters[prefix].add(w)

                # Bigrams and Trigrams
                if len(filtered_words) >= 2:
                    for i in range(len(filtered_words) - 1):
                        bigram = f"{filtered_words[i]} {filtered_words[i+1]}"
                        self.phrase_freq[bigram] += 1
                if len(filtered_words) >= 3:
                    for i in range(len(filtered_words) - 2):
                        trigram = f"{filtered_words[i]} {filtered_words[i+1]} {filtered_words[i+2]}"
                        self.phrase_freq[trigram] += 1

            for tok in chunk_seen_tokens:
                self.chunk_freq[tok] += 1

        # Keep phrases that occur more than once or appear in section titles
        for phrase, count in self.phrase_freq.items():
            if count >= 2 or any(phrase in sec.lower() for sec in self.section_keywords):
                self.significant_phrases.add(phrase)

        # Include section titles directly as significant phrases
        for sec in self.section_keywords:
            clean_sec = re.sub(r'[^\w\s]', ' ', sec.lower()).strip()
            if len(clean_sec.split()) >= 2:
                self.significant_phrases.add(clean_sec)

    def expand_acronym(self, token: str) -> Optional[str]:
        """Expands an acronym token to its learned full phrase."""
        return self.acronyms.get(token.upper())

    def get_acronym_for_phrase(self, phrase: str) -> Optional[str]:
        """Looks up an acronym for a given multi-word phrase."""
        return self.phrase_to_acronym.get(phrase.lower())

    def get_morphological_variants(self, token: str) -> List[str]:
        """
        Dynamically finds morphological variants in corpus sharing the same root.
        E.g., quote <-> quotation, invoice <-> invoicing, call <-> calling, create <-> creation.
        """
        token_sing = singularize(token.lower())
        variants: List[str] = []

        # Generic English stem: strip silent trailing 'e' (e.g. quote -> quot, create -> creat)
        stem = token_sing[:-1] if token_sing.endswith('e') and len(token_sing) > 3 else token_sing
        prefix = stem[:4] if len(stem) >= 4 else stem

        cluster = self.stem_clusters.get(prefix, set())
        for variant in cluster:
            v_stem = variant[:-1] if variant.endswith('e') and len(variant) > 3 else variant
            if variant != token_sing and variant not in variants:
                if (
                    variant.startswith(stem)
                    or token_sing.startswith(v_stem)
                    or fuzz.ratio(token_sing, variant) >= 55
                ):
                    variants.append(variant)

        return variants

    def get_related_terms(self, token: str, top_k: int = 2) -> List[str]:
        """Returns morphological variants as primary term equivalence."""
        return self.get_morphological_variants(token)[:top_k]

    def correct_typo(self, token: str, threshold: float = 78.0) -> Optional[str]:
        """
        Deterministic typo correction against the document's actual vocabulary.
        E.g., "quotaton" -> "quotation", "creat" -> "create".
        """
        t = token.lower()
        if t in self.vocab_tokens or len(t) < 4:
            return None

        best_match = None
        best_score = 0.0

        for vocab_word in self.vocab_tokens:
            if abs(len(vocab_word) - len(t)) > 3:
                continue

            score = fuzz.ratio(t, vocab_word)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = vocab_word

        return best_match

    def extract_document_phrases(self, text: str) -> List[str]:
        """Finds any known corpus phrases in the input text."""
        t_lower = text.lower()
        found = []
        for phrase in self.significant_phrases:
            if phrase in t_lower:
                found.append(phrase)
        return found

    def to_dict(self) -> Dict[str, Any]:
        """Serializes dynamic vocabulary state for tenant storage."""
        return {
            "total_chunks": self.total_chunks,
            "term_freq": dict(self.term_freq),
            "chunk_freq": dict(self.chunk_freq),
            "acronyms": self.acronyms,
            "phrase_to_acronym": self.phrase_to_acronym,
            "phrase_freq": dict(self.phrase_freq),
            "significant_phrases": list(self.significant_phrases),
            "section_keywords": {k: list(v) for k, v in self.section_keywords.items()},
            "stem_clusters": {k: list(v) for k, v in self.stem_clusters.items()},
            "vocab_tokens": list(self.vocab_tokens)
        }

    def from_dict(self, data: Dict[str, Any]):
        """Deserializes dynamic vocabulary state."""
        self.total_chunks = data.get("total_chunks", 0)
        self.term_freq = Counter(data.get("term_freq", {}))
        self.chunk_freq = Counter(data.get("chunk_freq", {}))
        self.acronyms = data.get("acronyms", {})
        self.phrase_to_acronym = data.get("phrase_to_acronym", {})
        self.phrase_freq = Counter(data.get("phrase_freq", {}))
        self.significant_phrases = set(data.get("significant_phrases", []))
        self.section_keywords = defaultdict(set, {k: set(v) for k, v in data.get("section_keywords", {}).items()})
        self.stem_clusters = defaultdict(set, {k: set(v) for k, v in data.get("stem_clusters", {}).items()})
        self.vocab_tokens = set(data.get("vocab_tokens", []))

    def save_to_file(self, file_path: str):
        """Saves vocabulary to JSON file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load_from_file(self, file_path: str):
        """Loads vocabulary from JSON file."""
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                self.from_dict(json.load(f))
