"""
app/rag/response/templates.py
Deterministic Natural Answer Templates.
Generates polished, evidence-backed answers according to answer type
(Definition, Explanation, How-to, Workflow, Comparison, Calculation, List, Table, Multi-hop, Entity lookup)
without introducing hallucinations or generative models.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.rag.response.planner import AnswerPlan


class AnswerTemplates:
    """
    Renders an AnswerPlan into a clean, human-readable response strictly bounded by evidence.
    """

    @classmethod
    def render_insufficient(cls, plan: AnswerPlan) -> str:
        return "I couldn't find enough supporting information in the uploaded documents to answer that accurately."

    @classmethod
    def render_definition(cls, plan: AnswerPlan) -> str:
        if not plan.claims:
            return cls.render_insufficient(plan)
        
        subject = plan.subject.title() if plan.subject else "The requested item"
        primary_claim = plan.claims[0].claim
        
        # Check if claim already starts with subject
        if primary_claim.lower().startswith(subject.lower()):
            body = primary_claim
        else:
            body = f"{subject} is defined as follows: {primary_claim}"

        additional = []
        for c in plan.claims[1:3]:
            additional.append(c.claim)

        if additional:
            return f"{body}\n\nKey Details:\n" + "\n".join(f"• {a}" for a in additional)
        return body

    @classmethod
    def render_workflow(cls, plan: AnswerPlan) -> str:
        if plan.raw_structured_data and "formatted_text" in plan.raw_structured_data:
            return plan.raw_structured_data["formatted_text"]
        
        if not plan.claims:
            return cls.render_insufficient(plan)

        header = f"Workflow for {plan.subject.title()}:" if plan.subject else "Document Workflow & Steps:"
        steps = []
        for idx, cl in enumerate(plan.claims[:6], 1):
            text = cl.claim.strip().rstrip(".")
            # Clean leading numbers if already present
            text = re.sub(r"^(?:step\s*\d+|[\d]+[\.\)]|\([a-z\d]+\))\s*", "", text, flags=re.IGNORECASE)
            steps.append(f"{idx}. {text}.")

        return f"{header}\n\n" + "\n".join(steps)

    @classmethod
    def render_how_to(cls, plan: AnswerPlan) -> str:
        header = f"Instructions to {plan.subject or 'proceed'}:"
        steps = []
        for idx, cl in enumerate(plan.claims[:5], 1):
            text = cl.claim.strip().rstrip(".")
            steps.append(f"{idx}. {text}.")
        return f"{header}\n\n" + "\n".join(steps) if steps else cls.render_insufficient(plan)

    @classmethod
    def render_explanation(cls, plan: AnswerPlan) -> str:
        if not plan.claims:
            return cls.render_insufficient(plan)
        lines = [c.claim for c in plan.claims[:4]]
        return "\n\n".join(lines)

    @classmethod
    def render_list(cls, plan: AnswerPlan) -> str:
        if not plan.claims:
            return cls.render_insufficient(plan)
        lines = [f"• {c.claim.strip()}" for c in plan.claims[:8]]
        header = f"Information regarding {plan.subject.title()}:" if plan.subject else "Document Points:"
        return f"{header}\n\n" + "\n".join(lines)

    @classmethod
    def render_multi_hop(cls, plan: AnswerPlan) -> str:
        if plan.raw_structured_data and "formatted_text" in plan.raw_structured_data:
            return plan.raw_structured_data["formatted_text"]
        if not plan.claims:
            return cls.render_insufficient(plan)
        return "\n\n".join([c.claim for c in plan.claims[:5]])

    @classmethod
    def render_calculation(cls, plan: AnswerPlan) -> str:
        if plan.raw_structured_data and "formatted_text" in plan.raw_structured_data:
            return plan.raw_structured_data["formatted_text"]
        if plan.claims:
            return plan.claims[0].claim
        return cls.render_insufficient(plan)

    @classmethod
    def render_comparison(cls, plan: AnswerPlan) -> str:
        if plan.raw_structured_data and "formatted_text" in plan.raw_structured_data:
            return plan.raw_structured_data["formatted_text"]
        if plan.claims:
            return "\n\n".join([c.claim for c in plan.claims[:4]])
        return cls.render_insufficient(plan)

    @classmethod
    def render_table(cls, plan: AnswerPlan) -> str:
        if plan.raw_structured_data and "formatted_text" in plan.raw_structured_data:
            return plan.raw_structured_data["formatted_text"]
        if not plan.claims:
            return cls.render_insufficient(plan)
        return "\n".join([f"• {c.claim}" for c in plan.claims[:5]])

    @classmethod
    def render_entity_lookup(cls, plan: AnswerPlan) -> str:
        if not plan.claims:
            return cls.render_insufficient(plan)
        return plan.claims[0].claim

    @classmethod
    def compose(cls, plan: AnswerPlan) -> str:
        """
        Dispatches to appropriate renderer based on answer_type.
        """
        if plan.confidence_level == "INSUFFICIENT_EVIDENCE" or plan.answer_type == "insufficient":
            return cls.render_insufficient(plan)

        dispatch_map = {
            "definition": cls.render_definition,
            "workflow": cls.render_workflow,
            "how_to": cls.render_how_to,
            "explanation": cls.render_explanation,
            "list": cls.render_list,
            "multi_hop": cls.render_multi_hop,
            "calculation": cls.render_calculation,
            "comparison": cls.render_comparison,
            "table": cls.render_table,
            "entity_lookup": cls.render_entity_lookup,
        }

        renderer = dispatch_map.get(plan.answer_type, cls.render_explanation)
        return renderer(plan)
