import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from app.rag.ingestion.models import DocumentChunk
from app.rag.query.intent import QueryIntent, RuleBasedIntentDetector
from app.rag.config import (
    SEMANTIC_WEIGHT,
    KEYWORD_WEIGHT,
    PHRASE_WEIGHT,
    FUZZY_WEIGHT,
    HEADING_WEIGHT,
    INTENT_WEIGHT,
    METADATA_WEIGHT,
)

@dataclass
class ScoredChunk:
    chunk: DocumentChunk
    final_score: float
    semantic_score: float
    keyword_score: float
    phrase_score: float
    fuzzy_score: float
    heading_score: float
    intent_score: float
    metadata_score: float
    independent_signals_count: int
    matched_terms: List[str] = field(default_factory=list)
    matched_phrases: List[str] = field(default_factory=list)
    explanation: Dict[str, float] = field(default_factory=dict)

class DeterministicScorer:
    """
    Normalized, mathematically consistent multi-signal reranker.
    All component signals are bounded to [0.0, 1.0].
    Applies independent signal convergence bonus and penalties to prevent single-signal domination.
    """

    def __init__(
        self,
        semantic_weight: float = SEMANTIC_WEIGHT,
        keyword_weight: float = KEYWORD_WEIGHT,
        phrase_weight: float = PHRASE_WEIGHT,
        fuzzy_weight: float = FUZZY_WEIGHT,
        heading_weight: float = HEADING_WEIGHT,
        intent_weight: float = INTENT_WEIGHT,
        metadata_weight: float = METADATA_WEIGHT,
    ):
        self.w_sem = semantic_weight
        self.w_kw = keyword_weight
        self.w_phrase = phrase_weight
        self.w_fuzz = fuzzy_weight
        self.w_head = heading_weight
        self.w_intent = intent_weight
        self.w_meta = metadata_weight
        self.intent_detector = RuleBasedIntentDetector()

    def calculate_phrase_score(
        self,
        chunk: DocumentChunk,
        detected_phrases: List[str]
    ) -> Tuple[float, List[str]]:
        """Checks if multi-word phrases appear in chunk body or section."""
        if not detected_phrases:
            return 0.50, []  # Neutral baseline

        body_lower = chunk.text.lower()
        section_lower = (chunk.section or "").lower()
        matched: List[str] = []

        for phrase in detected_phrases:
            p_lower = phrase.lower()
            if p_lower in section_lower or p_lower in body_lower:
                matched.append(phrase)

        score = min(1.0, len(matched) / len(detected_phrases)) if detected_phrases else 0.50
        return score, matched

    def calculate_heading_score(
        self,
        chunk: DocumentChunk,
        query_tokens: List[str],
        expanded_terms: List[str]
    ) -> Tuple[float, List[str]]:
        """
        Boosts chunk if query tokens or expanded synonyms match section/heading,
        and verifies if body contains supporting information.
        """
        section = (chunk.section or "").lower()
        if not section:
            return 0.0, []

        all_query_terms = set(t.lower() for t in (query_tokens + expanded_terms) if len(t) >= 2)
        if not all_query_terms:
            return 0.0, []

        section_words = set(section.split())
        matched_words = list(all_query_terms.intersection(section_words))

        # Check if body also supports the query
        body_lower = chunk.text.lower()
        body_has_support = any(w in body_lower for w in all_query_terms)

        if len(matched_words) >= 2:
            base_score = 1.0
        elif len(matched_words) == 1:
            base_score = 0.80
        elif any(term in section for term in all_query_terms if len(term) > 3):
            base_score = 0.60
        else:
            return 0.0, []

        # Synergistic boost if both heading and body contain query evidence
        if body_has_support and base_score > 0:
            base_score = min(1.0, base_score + 0.10)

        return base_score, matched_words

    def calculate_metadata_score(
        self,
        chunk: DocumentChunk,
        filter_doc_id: Optional[str] = None,
        filter_section: Optional[str] = None
    ) -> float:
        """Scores metadata alignment for optional filtering."""
        if not filter_doc_id and not filter_section:
            return 0.50  # Neutral

        score = 0.50
        if filter_doc_id:
            score = 1.0 if chunk.document_id == filter_doc_id else 0.0
        if filter_section and chunk.section:
            if filter_section.lower() in chunk.section.lower():
                score = min(1.0, score + 0.5)
        return score

    def score_candidate(
        self,
        chunk: DocumentChunk,
        semantic_score: float,
        keyword_score: float,
        fuzzy_score: float,
        detected_phrases: List[str],
        query_tokens: List[str],
        expanded_terms: List[str],
        intent: QueryIntent,
        filter_doc_id: Optional[str] = None,
        filter_section: Optional[str] = None,
    ) -> ScoredChunk:
        """
        Computes composite final score with independent signal agreement bonus.
        All signals normalized to [0.0, 1.0].
        """
        # Ensure all incoming signals are strictly bounded
        sem_norm = max(0.0, min(1.0, float(semantic_score)))
        kw_norm = max(0.0, min(1.0, float(keyword_score)))
        fuzz_norm = max(0.0, min(1.0, float(fuzzy_score)))

        phrase_score, matched_phrases = self.calculate_phrase_score(chunk, detected_phrases)
        heading_score, matched_heading_terms = self.calculate_heading_score(chunk, query_tokens, expanded_terms)
        intent_score = self.intent_detector.calculate_intent_alignment(intent, chunk.text)
        metadata_score = self.calculate_metadata_score(chunk, filter_doc_id, filter_section)

        # Base weighted sum
        weighted_base = (
            (sem_norm * self.w_sem)
            + (kw_norm * self.w_kw)
            + (phrase_score * self.w_phrase)
            + (fuzz_norm * self.w_fuzz)
            + (heading_score * self.w_head)
            + (intent_score * self.w_intent)
            + (metadata_score * self.w_meta)
        )

        # Requirement 6: Measure independent signal convergence
        signals_active = 0
        if sem_norm >= 0.50:
            signals_active += 1
        if kw_norm >= 0.25:
            signals_active += 1
        if phrase_score >= 0.65:
            signals_active += 1
        if heading_score >= 0.50:
            signals_active += 1
        if fuzz_norm >= 0.60:
            signals_active += 1

        # Signal convergence adjustments:
        # Multiple independent signals agree -> strong confidence multiplier
        # Only 1 isolated signal active and others 0 -> dampening
        if signals_active >= 3:
            multiplier = 1.06
        elif signals_active == 2:
            multiplier = 1.02
        elif signals_active <= 1 and kw_norm < 0.10 and heading_score == 0.0 and fuzz_norm == 0.0:
            multiplier = 0.88  # Isolated semantic-only signal dampening
        else:
            multiplier = 1.00

        final_score = max(0.0, min(1.0, weighted_base * multiplier))

        # Collect all matched terms for explainability
        matched_terms: Set[str] = set(matched_heading_terms)
        body_words = set(re.sub(r'[^\w\s]', ' ', chunk.text.lower()).split())
        for qt in query_tokens:
            if qt.lower() in body_words:
                matched_terms.add(qt)

        explanation = {
            "semantic": round(sem_norm, 4),
            "keyword": round(kw_norm, 4),
            "phrase": round(phrase_score, 4),
            "fuzzy": round(fuzz_norm, 4),
            "heading": round(heading_score, 4),
            "intent": round(intent_score, 4),
            "metadata": round(metadata_score, 4),
            "signals_count": signals_active,
            "final": round(final_score, 4),
        }

        return ScoredChunk(
            chunk=chunk,
            final_score=final_score,
            semantic_score=sem_norm,
            keyword_score=kw_norm,
            phrase_score=phrase_score,
            fuzzy_score=fuzz_norm,
            heading_score=heading_score,
            intent_score=intent_score,
            metadata_score=metadata_score,
            independent_signals_count=signals_active,
            matched_terms=list(matched_terms),
            matched_phrases=matched_phrases,
            explanation=explanation,
        )
