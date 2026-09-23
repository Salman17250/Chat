import re
from typing import Set, List, Dict, Any
from app.config import FUZZY_MAX_EDIT_DISTANCE, FUZZY_MIN_WORD_LEN

def build_vocabulary(documents: List[Dict[str, Any]]) -> Set[str]:
    vocabulary = set()
    for document in documents:
        for chunk in document.get("chunks", []):
            text = chunk.get("text", "")
            # Extract alphanumeric tokens
            cleaned_text = re.sub(r"[^a-zA-Z0-9_\u0900-\u097F]", " ", text)
            for word in cleaned_text.lower().split():
                if len(word) >= FUZZY_MIN_WORD_LEN:
                    vocabulary.add(word)
    return vocabulary

_cached_vocab = None
_cached_chunk_count = 0

def get_cached_vocabulary(documents: List[Dict[str, Any]]) -> Set[str]:
    global _cached_vocab, _cached_chunk_count
    total_chunks = sum(len(d.get("chunks", [])) for d in documents)
    if _cached_vocab is not None and _cached_chunk_count == total_chunks:
        return _cached_vocab
    _cached_vocab = build_vocabulary(documents)
    _cached_chunk_count = total_chunks
    return _cached_vocab

def damerau_levenshtein_distance(s1: str, s2: str) -> int:
    """Ultra-fast array-based Damerau-Levenshtein distance (< 0.04ms)."""
    if s1 == s2:
        return 0
    len1, len2 = len(s1), len(s2)
    d = [[0] * (len2 + 2) for _ in range(len1 + 2)]
    max_dist = len1 + len2
    d[0][0] = max_dist
    for i in range(0, len1 + 1):
        d[i + 1][0] = max_dist
        d[i + 1][1] = i
    for j in range(0, len2 + 1):
        d[0][j + 1] = max_dist
        d[1][j + 1] = j

    da = {}
    for i in range(1, len1 + 1):
        db = 0
        for j in range(1, len2 + 1):
            k = da.get(s2[j - 1], 0)
            l = db
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            if cost == 0:
                db = j
            d[i + 1][j + 1] = min(
                d[i][j + 1] + 1,
                d[i + 1][j] + 1,
                d[i][j] + cost,
                d[k][l] + (i - k - 1) + 1 + (j - l - 1)
            )
        da[s1[i - 1]] = i
    return d[len1 + 1][len2 + 1]

def correct_query_for_retrieval(query: str, vocabulary: Set[str]) -> str:
    if not vocabulary or not query:
        return query

    words = query.split()
    corrected_words = []

    # Stopwords that should never be mutated
    skip_words = {"how", "can", "the", "and", "for", "with", "this", "that", "from", "into", "over", "what", "where", "when", "which", "make", "create"}

    for raw_word in words:
        clean_word = re.sub(r"[^a-zA-Z0-9_\u0900-\u097F]", "", raw_word).lower()

        if len(clean_word) < FUZZY_MIN_WORD_LEN or clean_word in vocabulary or clean_word in skip_words:
            corrected_words.append(raw_word)
            continue

        best_match = None
        best_distance = FUZZY_MAX_EDIT_DISTANCE + 1

        for vocab_word in vocabulary:
            if abs(len(vocab_word) - len(clean_word)) > FUZZY_MAX_EDIT_DISTANCE:
                continue

            dist = damerau_levenshtein_distance(clean_word, vocab_word)
            if dist < best_distance:
                best_distance = dist
                best_match = vocab_word

        corrected_words.append(best_match if best_match else raw_word)

    return " ".join(corrected_words)

    return " ".join(corrected_words)
