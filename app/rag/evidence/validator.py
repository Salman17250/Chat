import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from app.rag.ingestion.models import DocumentChunk
from app.rag.ranking.adaptive_scorer import ScoredChunk
from app.rag.query.planner import QueryPlan
from app.rag.query.normalizer import STOP_WORDS

GENERIC_TOPIC_NOUNS = {
    "policy", "period", "document", "information", "detail", "details",
    "rule", "rules", "guideline", "guidelines", "process", "procedure",
    "system", "plan", "manual", "guide", "record", "records"
}

@dataclass
class ValidationResult:
    is_valid: bool
    confidence: str          # HIGH, MEDIUM, LOW, VERY_LOW
    confidence_score: float
    rejection_reason: Optional[str] = None
    validated_chunk: Optional[DocumentChunk] = None

class EvidenceValidator:
    """
    Mandatory Strict Evidence Validation Engine.
    Ensures zero-hallucination and eliminates false-positive retrieval by verifying:
      1. Qualifying subjects and key entities from the query are actually present in evidence
      2. No matching on isolated generic words (e.g. 'period', 'policy') without their modifier
      3. Tenant and document scope match
      4. Safe fallback to 'insufficient evidence' message on any doubt
    """
    @classmethod
    def validate_candidate(
        cls,
        scored_chunk: ScoredChunk,
        plan: QueryPlan,
        tenant_id: str = "default"
    ) -> ValidationResult:
        chunk = scored_chunk.chunk
        score = scored_chunk.final_score
        chunk_text_lower = chunk.text.lower()
        chunk_section_lower = (chunk.section or "").lower()
        combined_text = f"{chunk_section_lower} {chunk_text_lower}"

        # 1. Tenant Verification
        if chunk.tenant_id and chunk.tenant_id != tenant_id and tenant_id != "default":
            return ValidationResult(
                is_valid=False,
                confidence="VERY_LOW",
                confidence_score=0.0,
                rejection_reason="Tenant mismatch"
            )

        # 2. Extract Key Non-Generic Query Qualifiers
        # e.g., for "what is the leave policy", qualifiers = {"leave"}. "policy" is generic.
        query_words = set(re.findall(r'\b[a-z]{3,}\b', plan.clean_query.lower()))
        specific_qualifiers = [
            w for w in query_words
            if w not in STOP_WORDS and w not in GENERIC_TOPIC_NOUNS
        ]

        # Check attribute corroboration from dynamic chunk attributes
        chunk_attrs = getattr(chunk, "attributes", {}) or {}
        has_attribute_corroboration = False
        if plan.target_attributes:
            for ta in plan.target_attributes:
                if ta in chunk_attrs or any(ta in k for k in chunk_attrs.keys()):
                    has_attribute_corroboration = True
                    break

        # 3. Specific Qualifier Verification Rule:
        # If the user specified distinctive qualifiers (e.g. "leave", "probation", "notice", "maternity", "loan"),
        # at least one qualifier must be present, OR semantic score must demonstrate strong conceptual match (>= 0.70),
        # OR dynamic attribute corroboration must exist.
        if specific_qualifiers and not has_attribute_corroboration:
            qualifiers_found = [q for q in specific_qualifiers if q in combined_text]
            if not qualifiers_found and scored_chunk.semantic_score < 0.70:
                return ValidationResult(
                    is_valid=False,
                    confidence="VERY_LOW",
                    confidence_score=score,
                    rejection_reason=f"Key query qualifier(s) {specific_qualifiers} not present in candidate"
                )

        # 3b. Compound Bigram Qualifier Verification (e.g. 'notice period', 'leave policy')
        # Prevents matching 'Header Notice' when the query specifically requested 'notice period'
        words_ordered = [w for w in re.findall(r'\b[a-z]{3,}\b', plan.clean_query.lower()) if w not in STOP_WORDS]
        for i in range(len(words_ordered) - 1):
            w1, w2 = words_ordered[i], words_ordered[i+1]
            if w2 in GENERIC_TOPIC_NOUNS or w1 in GENERIC_TOPIC_NOUNS:
                bigram = f"{w1} {w2}"
                if bigram not in combined_text and scored_chunk.semantic_score < 0.70 and not has_attribute_corroboration:
                    tokens_chunk = re.findall(r'\b[a-z]{3,}\b', combined_text)
                    co_occur = False
                    for idx_t, tok in enumerate(tokens_chunk):
                        if tok == w1:
                            window = tokens_chunk[max(0, idx_t - 6):min(len(tokens_chunk), idx_t + 6)]
                            if w2 in window:
                                co_occur = True
                                break
                    if not co_occur:
                        return ValidationResult(
                            is_valid=False,
                            confidence="VERY_LOW",
                            confidence_score=score,
                            rejection_reason=f"Compound concept '{bigram}' not corroborated in candidate"
                        )

        # 4. Entity Verification Rule:
        # If query specifies a specific named entity (e.g. Person, Company, Product), it should be present
        if plan.entities:
            entity_match = False
            chunk_ents = [ce.lower() for ce in getattr(chunk, "entities", [])]
            for e in plan.entities:
                e_clean = e.lower()
                if e_clean in combined_text:
                    entity_match = True
                    break
                for sub in e_clean.split():
                    if len(sub) >= 3 and sub not in STOP_WORDS and sub in combined_text:
                        entity_match = True
                        break
                if any(e_clean in ce or ce in e_clean for ce in chunk_ents):
                    entity_match = True
                    break

            if not entity_match and scored_chunk.semantic_score < 0.70:
                return ValidationResult(
                    is_valid=False,
                    confidence="VERY_LOW",
                    confidence_score=score,
                    rejection_reason=f"Specified entities {plan.entities} not found in candidate"
                )

        # 5. Composite Score Threshold Verification
        if score < 0.38:
            return ValidationResult(
                is_valid=False,
                confidence="VERY_LOW",
                confidence_score=score,
                rejection_reason=f"Final score {score:.4f} below confidence threshold"
            )

        # Determine confidence category
        if score >= 0.72:
            conf = "HIGH"
        elif score >= 0.50:
            conf = "MEDIUM"
        else:
            conf = "LOW"

        return ValidationResult(
            is_valid=True,
            confidence=conf,
            confidence_score=score,
            validated_chunk=chunk
        )
