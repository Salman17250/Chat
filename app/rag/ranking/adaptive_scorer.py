import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from rapidfuzz import fuzz
from app.rag.ingestion.models import DocumentChunk
from app.rag.query.analyzer import QueryProfile, AnalyzedQuery
from app.rag.query.intent import QueryIntent, RuleBasedIntentDetector
from app.rag.query.normalizer import STOP_WORDS, singularize

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
    query_coverage: float = 0.0
    matched_terms: List[str] = field(default_factory=list)
    matched_phrases: List[str] = field(default_factory=list)
    explanation: Dict[str, float] = field(default_factory=dict)

class AdaptiveScorer:
    """
    Adaptive Multi-Signal Reranker with Query-Profile Dynamic Weighting.
    Adjusts signal weights dynamically based on whether the query is:
      - EXACT_CODE: High BM25 + Exact lexical match
      - CONCEPTUAL_PARAPHRASE: High Semantic (Nomic 768) + Concept corroboration
      - PROCEDURAL: Balanced Semantic + Action verbs + Step intents
      - TABLE_LOOKUP: Row sub-chunk prioritization + Attribute-header bindings
    Protects conceptual paraphrases from artificial lexical dampening.
    """

    # Dynamic weight profiles (summing to 1.0)
    PROFILE_WEIGHTS: Dict[QueryProfile, Dict[str, float]] = {
        QueryProfile.EXACT_CODE: {
            "sem": 0.20, "kw": 0.45, "phrase": 0.10, "fuzz": 0.15, "head": 0.05, "intent": 0.03, "meta": 0.02
        },
        QueryProfile.CONCEPTUAL_PARAPHRASE: {
            "sem": 0.65, "kw": 0.10, "phrase": 0.10, "fuzz": 0.03, "head": 0.07, "intent": 0.03, "meta": 0.02
        },
        QueryProfile.PROCEDURAL: {
            "sem": 0.35, "kw": 0.25, "phrase": 0.10, "fuzz": 0.05, "head": 0.12, "intent": 0.10, "meta": 0.03
        },
        QueryProfile.TABLE_LOOKUP: {
            "sem": 0.35, "kw": 0.25, "phrase": 0.10, "fuzz": 0.05, "head": 0.15, "intent": 0.05, "meta": 0.05
        },
        QueryProfile.MULTI_PART: {
            "sem": 0.45, "kw": 0.25, "phrase": 0.10, "fuzz": 0.05, "head": 0.10, "intent": 0.03, "meta": 0.02
        },
        QueryProfile.WORKFLOW: {
            "sem": 0.40, "kw": 0.20, "phrase": 0.15, "fuzz": 0.03, "head": 0.12, "intent": 0.08, "meta": 0.02
        },
        QueryProfile.RELATIONSHIP: {
            "sem": 0.40, "kw": 0.20, "phrase": 0.15, "fuzz": 0.03, "head": 0.12, "intent": 0.07, "meta": 0.03
        }
    }

    def __init__(self):
        self.intent_detector = RuleBasedIntentDetector()

    def calculate_phrase_score(
        self,
        chunk: DocumentChunk,
        detected_phrases: List[str]
    ) -> Tuple[float, List[str]]:
        """Checks if multi-word phrases appear in chunk body or section."""
        if not detected_phrases:
            return 0.50, []

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
        """Boosts chunk if query tokens or synonyms match section/heading."""
        section = (chunk.section or "").lower()
        if not section:
            return 0.0, []

        all_query_terms = set(
            t.lower() for t in (query_tokens + expanded_terms)
            if len(t) >= 3 and t.lower() not in STOP_WORDS
        )
        if not all_query_terms:
            return 0.0, []

        section_words = set(re.sub(r'[^\w\s]', ' ', section).split())
        matched_words = list(all_query_terms.intersection(section_words))

        body_words = set(re.sub(r'[^\w\s]', ' ', chunk.text.lower()).split())
        body_has_support = any(w in body_words for w in all_query_terms)

        if len(matched_words) >= 2:
            base_score = 1.0
        elif len(matched_words) == 1:
            base_score = 0.75
        else:
            return 0.0, []

        if body_has_support and base_score > 0:
            base_score = min(1.0, base_score + 0.10)

        return base_score, matched_words

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
        profile: QueryProfile = QueryProfile.CONCEPTUAL_PARAPHRASE,
        target_attributes: Optional[List[str]] = None,
        filter_doc_id: Optional[str] = None,
        filter_section: Optional[str] = None,
        core_query: Optional[str] = None,
        core_tokens: Optional[List[str]] = None,
    ) -> ScoredChunk:
        """
        Calculates adaptive composite score with query-profile-specific dynamic weights.
        """
        sem_norm = max(0.0, min(1.0, float(semantic_score)))
        kw_norm = max(0.0, min(1.0, float(keyword_score)))
        fuzz_norm = max(0.0, min(1.0, float(fuzzy_score)))

        phrase_score, matched_phrases = self.calculate_phrase_score(chunk, detected_phrases)
        heading_score, matched_heading_terms = self.calculate_heading_score(chunk, query_tokens, expanded_terms)
        intent_score = self.intent_detector.calculate_intent_alignment(intent, chunk.text)
        metadata_score = 1.0 if (filter_doc_id and chunk.document_id == filter_doc_id) else 0.50

        weights = self.PROFILE_WEIGHTS.get(profile, self.PROFILE_WEIGHTS[QueryProfile.CONCEPTUAL_PARAPHRASE])

        # Base weighted sum using profile-specific weights
        weighted_base = (
            (sem_norm * weights["sem"])
            + (kw_norm * weights["kw"])
            + (phrase_score * weights["phrase"])
            + (fuzz_norm * weights["fuzz"])
            + (heading_score * weights["head"])
            + (intent_score * weights["intent"])
            + (metadata_score * weights["meta"])
        )

        # Table-specific enhancement: if table row matches requested attribute
        if profile == QueryProfile.TABLE_LOOKUP and chunk.is_table_row:
            if target_attributes and chunk.table_headers:
                header_match = any(
                    any(attr in h.lower() for attr in target_attributes)
                    for h in chunk.table_headers
                )
                if header_match:
                    weighted_base = min(1.0, weighted_base + 0.12)

        # Structural Q&A Alignment:
        # If the chunk is a detected Q&A unit, compare user query directly against the question text!
        qa_match_score = 0.0
        if chunk.is_qa_pair and chunk.question_text:
            q_clean = chunk.question_text.lower().strip('?.,;:! ')
            user_clean = " ".join([t for t in query_tokens if len(t) >= 2])
            qa_fuzz = fuzz.token_set_ratio(user_clean, q_clean) / 100.0

            q_words = set(re.sub(r'[^\w\s]', ' ', q_clean).split())
            user_content = [t for t in query_tokens if len(t) >= 3 and t.lower() not in STOP_WORDS]
            if user_content:
                overlap = sum(1 for t in user_content if t in q_words or singularize(t) in q_words)
                qa_overlap = overlap / len(user_content)
            else:
                qa_overlap = 0.0

            qa_match_score = (qa_fuzz * 0.50) + (qa_overlap * 0.50)

            # If core_query or core_tokens are provided, check structural alignment against core query as well
            if core_query or core_tokens:
                core_clean = (core_query or " ".join(core_tokens)).lower().strip('?.,;:! ')
                core_fuzz = fuzz.token_set_ratio(core_clean, q_clean) / 100.0
                c_content = [t for t in (core_tokens or core_clean.split()) if len(t) >= 3 and t.lower() not in STOP_WORDS]
                if c_content:
                    c_overlap = sum(1 for t in c_content if t in q_words or singularize(t) in q_words) / len(c_content)
                else:
                    c_overlap = 0.0
                core_qa_score = (core_fuzz * 0.50) + (c_overlap * 0.50)
                qa_match_score = max(qa_match_score, core_qa_score)

            if qa_match_score >= 0.70:
                weighted_base = min(1.0, weighted_base + (0.15 * qa_match_score))

        # Direct Match Baseline Safeguard:
        # If chunk is a direct structural QA match or has very high keyword + fuzzy alignment,
        # ensure it maintains a strong baseline floor regardless of dynamic profile weighting.
        if qa_match_score >= 0.85:
            weighted_base = max(weighted_base, 0.65 * qa_match_score)
        elif kw_norm >= 0.85 and fuzz_norm >= 0.85:
            weighted_base = max(weighted_base, 0.60 * ((kw_norm + fuzz_norm) / 2.0))

        # Signal convergence count
        signals_active = 0
        if sem_norm >= 0.50:
            signals_active += 1
        if kw_norm >= 0.20:
            signals_active += 1
        if phrase_score >= 0.65:
            signals_active += 1
        if heading_score >= 0.50:
            signals_active += 1
        if fuzz_norm >= 0.60:
            signals_active += 1
        if qa_match_score >= 0.70:
            signals_active += 1

        # Paraphrase Protection:
        # If profile is CONCEPTUAL and semantic similarity is high (>= 0.65), DO NOT penalize for 0 BM25.
        if profile == QueryProfile.CONCEPTUAL_PARAPHRASE and sem_norm >= 0.65:
            multiplier = 1.05  # High-confidence conceptual paraphrase boost
        elif signals_active >= 3:
            multiplier = 1.06
        elif signals_active == 2:
            multiplier = 1.02
        elif signals_active <= 1 and kw_norm < 0.10 and heading_score == 0.0 and fuzz_norm == 0.0:
            # Only dampen if profile is NOT conceptual and semantic score is mediocre
            if profile != QueryProfile.CONCEPTUAL_PARAPHRASE and sem_norm < 0.65:
                multiplier = 0.90
            else:
                multiplier = 1.00
        else:
            multiplier = 1.00

        final_score = max(0.0, min(1.0, weighted_base * multiplier))

        matched_terms: Set[str] = set(matched_heading_terms)
        body_words = set(re.sub(r'[^\w\s]', ' ', chunk.text.lower()).split())
        body_lemmas = {singularize(w) for w in body_words}
        for qt in query_tokens:
            q_sing = singularize(qt.lower())
            if qt.lower() in body_words or q_sing in body_lemmas:
                matched_terms.add(qt)

        # Calculate non-stopword query concept coverage
        content_tokens = [t.lower() for t in query_tokens if len(t) >= 3 and t.lower() not in STOP_WORDS]
        if content_tokens:
            matched_content = [
                t for t in content_tokens
                if (
                    t in matched_terms
                    or t in body_words
                    or singularize(t) in body_lemmas
                    or (len(t) >= 5 and any(w.startswith(t[:5]) for w in body_words))
                )
            ]
            query_coverage = round(len(matched_content) / len(content_tokens), 4)
        else:
            query_coverage = 1.0

        explanation = {
            "profile": profile.value,
            "semantic": round(sem_norm, 4),
            "keyword": round(kw_norm, 4),
            "phrase": round(phrase_score, 4),
            "fuzzy": round(fuzz_norm, 4),
            "heading": round(heading_score, 4),
            "intent": round(intent_score, 4),
            "metadata": round(metadata_score, 4),
            "query_coverage": query_coverage,
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
            query_coverage=query_coverage,
            matched_terms=list(matched_terms),
            matched_phrases=matched_phrases,
            explanation=explanation,
        )
