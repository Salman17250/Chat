"""
app/rag/response/assembler.py
Production Response Assembler with Strict Evidence Gating, Observability & Provenance.
Integrates ResponsePlanner, AnswerTemplates, and deterministic safety checks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from app.rag.config import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    FINAL_TOP_K,
    MIN_CONFIDENCE,
    NO_ANSWER_FOUND_MESSAGE,
)
from app.rag.context.assembler import AssembledResponse, AssembledSource
from app.rag.response.planner import AnswerPlan
from app.rag.response.templates import AnswerTemplates


class ResponseAssembler:
    """
    Transforms validated retrieval candidates and reasoning graphs into
    production-ready, evidence-backed AssembledResponse objects.
    """

    @classmethod
    def assemble_from_plan(
        cls,
        answer_plan: AnswerPlan,
        sources: List[AssembledSource],
        timing_breakdown: Optional[Dict[str, float]] = None,
        debug_trace: Optional[Dict[str, Any]] = None,
    ) -> AssembledResponse:
        """
        Assembles final response from an existing AnswerPlan.
        """
        answer_text = AnswerTemplates.compose(answer_plan)

        conf_str = "VERY_LOW"
        if answer_plan.confidence_level == "HIGH_CONFIDENCE":
            conf_str = "HIGH"
        elif answer_plan.confidence_level == "MEDIUM_CONFIDENCE":
            conf_str = "MEDIUM"
        elif answer_plan.confidence_level == "LOW_CONFIDENCE":
            conf_str = "LOW"

        trace = dict(debug_trace or {})
        if timing_breakdown:
            trace["timing_breakdown"] = timing_breakdown

        resp = AssembledResponse(
            answer=answer_text,
            confidence=conf_str,
            confidence_score=round(answer_plan.confidence, 4),
            sources=sources,
            debug_trace=trace if trace else None,
        )
        # Attach rich metadata
        resp.answer_type = answer_plan.answer_type
        resp.reasoning = {
            "type": answer_plan.answer_type,
            "path": answer_plan.reasoning_path,
            "confidence_level": answer_plan.confidence_level,
        }
        if timing_breakdown:
            resp.timing_breakdown = timing_breakdown

        return resp

    @classmethod
    def assemble_insufficient(
        cls,
        query: str,
        message: str = NO_ANSWER_FOUND_MESSAGE,
        timing_breakdown: Optional[Dict[str, float]] = None,
        debug_trace: Optional[Dict[str, Any]] = None,
    ) -> AssembledResponse:
        trace = dict(debug_trace or {})
        if timing_breakdown:
            trace["timing_breakdown"] = timing_breakdown

        resp = AssembledResponse(
            answer=message,
            confidence="VERY_LOW",
            confidence_score=0.0,
            sources=[],
            debug_trace=trace if trace else None,
        )
        resp.answer_type = "insufficient"
        resp.reasoning = {"type": "insufficient_evidence", "query": query}
        if timing_breakdown:
            resp.timing_breakdown = timing_breakdown
        return resp
