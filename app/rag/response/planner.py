"""
app/rag/response/planner.py
Structured Response Planning Engine.
Separates understanding, retrieval, and reasoning from presentation by creating
an intermediate AnswerPlan with evidence-grounded claims, confidence, and reasoning paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.rag.evidence.validator import ValidationResult, EvidenceStatus
from app.rag.ingestion.models import DocumentChunk
from app.rag.query.planner import QueryPlan


@dataclass
class AnswerClaim:
    """An atomic, evidence-backed statement in the answer plan."""
    claim: str
    evidence_ids: List[str] = field(default_factory=list)
    document_names: List[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class AnswerPlan:
    """Structured intermediate plan before final text composition."""
    answer_type: str  # definition, explanation, how_to, workflow, comparison, calculation, list, table, multi_part, multi_hop, entity_lookup, insufficient
    query: str
    subject: str = ""
    claims: List[AnswerClaim] = field(default_factory=list)
    confidence: float = 0.0
    confidence_level: str = "HIGH_CONFIDENCE"  # HIGH_CONFIDENCE, MEDIUM_CONFIDENCE, LOW_CONFIDENCE, INSUFFICIENT_EVIDENCE
    missing_information: List[str] = field(default_factory=list)
    reasoning_path: List[str] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    raw_structured_data: Optional[Dict[str, Any]] = None


class ResponsePlanner:
    """
    Constructs an AnswerPlan from validated evidence, query intent, and reasoning outputs.
    """

    @classmethod
    def create_answer_plan(
        cls,
        query_plan: QueryPlan,
        validation_report: ValidationResult,
        selected_chunks: List[DocumentChunk],
        reasoning_output: Optional[Dict[str, Any]] = None,
    ) -> AnswerPlan:
        # Determine confidence level
        missing = []
        if validation_report.coverage_report:
            missing = validation_report.coverage_report.missing_concepts + validation_report.coverage_report.missing_entities
        elif validation_report.rejection_reason:
            missing = [validation_report.rejection_reason]

        if validation_report.status == EvidenceStatus.INSUFFICIENT or not selected_chunks or not validation_report.is_valid:
            return AnswerPlan(
                answer_type="insufficient",
                query=query_plan.original_query,
                subject=query_plan.subject or "",
                confidence=0.0,
                confidence_level="INSUFFICIENT_EVIDENCE",
                missing_information=missing,
                reasoning_path=["Validation check failed: insufficient document evidence."],
            )

        if validation_report.status == EvidenceStatus.SUPPORTED:
            conf_level = "HIGH_CONFIDENCE"
            conf_val = min(1.0, max(0.85, validation_report.confidence_score))
        elif validation_report.status == EvidenceStatus.PARTIALLY_SUPPORTED:
            conf_level = "MEDIUM_CONFIDENCE"
            conf_val = max(0.50, min(0.84, validation_report.confidence_score))
        else:
            conf_level = "LOW_CONFIDENCE"
            conf_val = 0.35

        # Infer answer type from intent or reasoning
        answer_type = "explanation"
        if reasoning_output and "reasoning_type" in reasoning_output:
            rtype = reasoning_output["reasoning_type"]
            if rtype == "calculation":
                answer_type = "calculation"
            elif rtype == "comparison":
                answer_type = "comparison"
            elif rtype == "temporal" or rtype == "workflow":
                answer_type = "workflow"
            elif rtype == "multi_hop":
                answer_type = "multi_hop"
        elif query_plan.intent == "workflow_explanation" or "workflow" in query_plan.action:
            answer_type = "workflow"
        elif query_plan.intent == "definition":
            answer_type = "definition"
        elif query_plan.intent == "comparison":
            answer_type = "comparison"
        elif query_plan.intent == "calculation":
            answer_type = "calculation"
        elif query_plan.intent == "table_lookup":
            answer_type = "table"

        # Build atomic claims from validated sentences in chunks
        claims: List[AnswerClaim] = []
        sources: List[Dict[str, Any]] = []
        seen_chunks = set()

        for c in selected_chunks:
            if c.chunk_id not in seen_chunks:
                seen_chunks.add(c.chunk_id)
                sources.append({
                    "document_id": c.document_name,
                    "chunk_id": c.chunk_id,
                    "section": c.section or "General",
                    "score": getattr(c, "score", 0.0)
                })

            # Create claim from leading sentence or summary
            text_lines = [l.strip() for l in c.text.split("\n") if l.strip()]
            for line in text_lines[:3]:
                if len(line) > 20 and not any(line == cl.claim for cl in claims):
                    claims.append(AnswerClaim(
                        claim=line,
                        evidence_ids=[c.chunk_id],
                        document_names=[c.document_name] if c.document_name else [],
                        confidence=conf_val
                    ))

        reasoning_path = []
        if reasoning_output and "explanation" in reasoning_output:
            reasoning_path.append(reasoning_output["explanation"])
        cov_pct = (validation_report.coverage_report.concept_coverage * 100.0) if validation_report.coverage_report else 100.0
        reasoning_path.append(f"Validated with status {validation_report.status.value} (coverage: {cov_pct:.1f}%)")

        return AnswerPlan(
            answer_type=answer_type,
            query=query_plan.original_query,
            subject=query_plan.subject or "",
            claims=claims,
            confidence=conf_val,
            confidence_level=conf_level,
            missing_information=validation_report.missing_elements,
            reasoning_path=reasoning_path,
            sources=sources,
            raw_structured_data=reasoning_output,
        )
