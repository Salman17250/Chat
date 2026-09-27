import re
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
from app.rag.ingestion.models import DocumentChunk
from app.rag.query.planner import QueryPlan
from app.rag.query.normalizer import STOP_WORDS, singularize

@dataclass
class EvidenceCoverageReport:
    is_fully_covered: bool
    concept_coverage: float
    entity_coverage: float
    qualifier_coverage: float
    relationship_coverage: float
    numeric_coverage: float
    missing_concepts: List[str]
    missing_entities: List[str]
    missing_qualifiers: List[str]
    corroborated_terms: List[str]

class EvidenceCoverageChecker:
    """
    Deterministic Evidence Coverage Analyzer.
    Verifies that retrieved document candidates actually cover all necessary aspects of the user query:
      - Core query concepts (non-generic words)
      - Explicit named entities
      - Modifiers and qualifiers (e.g. 'after', 'before', 'maximum', 'discount')
      - Quantitative numbers / percentages
    Eliminates partial false-positives and ungrounded hallucinations.
    """

    @classmethod
    def evaluate_coverage(
        cls,
        evidence_text: str,
        plan: QueryPlan,
        evidence_chunk: Optional[DocumentChunk] = None
    ) -> EvidenceCoverageReport:
        lower_evidence = evidence_text.lower()
        evidence_words = set(re.findall(r'\b[a-z0-9_\-]{2,}\b', lower_evidence))
        evidence_lemmas = {singularize(w) for w in evidence_words}

        # 1. Concept Coverage
        content_tokens = [
            t.lower() for t in plan.keywords
            if len(t) >= 3 and t.lower() not in STOP_WORDS
        ]
        matched_concepts = []
        missing_concepts = []
        for t in content_tokens:
            t_sing = singularize(t)
            if t in evidence_words or t_sing in evidence_lemmas or any(w.startswith(t[:4]) for w in evidence_words if len(t) >= 4):
                matched_concepts.append(t)
            else:
                missing_concepts.append(t)

        concept_cov = len(matched_concepts) / max(1, len(content_tokens))

        # 2. Entity Coverage
        matched_entities = []
        missing_entities = []
        for ent in plan.entities:
            ent_clean = ent.lower().strip()
            ent_words = [w for w in ent_clean.split() if w not in STOP_WORDS]
            if ent_clean in lower_evidence or any(w in evidence_words for w in ent_words):
                matched_entities.append(ent)
            else:
                missing_entities.append(ent)

        entity_cov = len(matched_entities) / max(1, len(plan.entities)) if plan.entities else 1.0

        # 3. Qualifier Coverage
        matched_qualifiers = []
        missing_qualifiers = []
        for q in plan.qualifiers:
            q_clean = q.lower()
            if q_clean in evidence_words or singularize(q_clean) in evidence_lemmas:
                matched_qualifiers.append(q)
            else:
                missing_qualifiers.append(q)

        qualifier_cov = len(matched_qualifiers) / max(1, len(plan.qualifiers)) if plan.qualifiers else 1.0

        # 4. Numeric Coverage
        matched_numbers = 0
        for num in plan.numbers:
            num_str = str(int(num)) if num.is_integer() else str(num)
            if num_str in lower_evidence:
                matched_numbers += 1
        num_cov = matched_numbers / max(1, len(plan.numbers)) if plan.numbers else 1.0

        # 5. Relationship Coverage
        rel_cov = 1.0
        if plan.relationships:
            if "next_step" in plan.relationships:
                # Check for step indicators, numbers, 'next', 'then', 'after', 'configure'
                has_next = any(w in lower_evidence for w in ["next", "then", "after", "step", "configure", "stages", "follow"])
                rel_cov = 1.0 if has_next else 0.5
            elif "prerequisite" in plan.relationships:
                has_req = any(w in lower_evidence for w in ["require", "prerequisite", "before", "must", "need"])
                rel_cov = 1.0 if has_req else 0.5

        # Strict full coverage determination
        is_fully_covered = (
            (entity_cov >= 1.0 or not plan.entities)
            and concept_cov >= 0.50
            and (qualifier_cov >= 0.50 or not plan.qualifiers)
            and (num_cov >= 1.0 or not plan.numbers)
        )

        return EvidenceCoverageReport(
            is_fully_covered=is_fully_covered,
            concept_coverage=round(concept_cov, 4),
            entity_coverage=round(entity_cov, 4),
            qualifier_coverage=round(qualifier_cov, 4),
            relationship_coverage=round(rel_cov, 4),
            numeric_coverage=round(num_cov, 4),
            missing_concepts=missing_concepts,
            missing_entities=missing_entities,
            missing_qualifiers=missing_qualifiers,
            corroborated_terms=matched_concepts
        )
