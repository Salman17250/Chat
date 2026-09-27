"""
app/rag/reasoning/logical.py
Deterministic Logical Reasoning Engine.
Handles deterministic evaluation of conditional constraints (IF-THEN), Boolean operations (AND/OR/NOT),
and prerequisites/dependencies found directly within retrieved document evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.rag.ingestion.models import DocumentChunk


@dataclass
class LogicalRule:
    """A conditional rule or prerequisite extracted from evidence."""
    condition: str
    consequence: str
    rule_type: str  # 'if_then', 'prerequisite', 'exclusive_or', 'negation'
    evidence_chunk_id: Optional[str] = None
    document_name: Optional[str] = None
    raw_sentence: str = ""


@dataclass
class LogicalReasoningResult:
    """Result of deterministic logical evaluation."""
    query: str
    evaluated_condition: Optional[str] = None
    supported: bool = False
    rules_found: List[LogicalRule] = field(default_factory=list)
    verdict: str = "INSUFFICIENT_INFORMATION"  # 'SATISFIED', 'VIOLATED', 'CONDITIONAL', 'INSUFFICIENT_INFORMATION'
    explanation: str = ""
    evidence_claims: List[str] = field(default_factory=list)

    def to_formatted_answer(self) -> str:
        if not self.supported or not self.explanation:
            return ""
        lines = [self.explanation]
        if self.rules_found:
            lines.append("\nDocument Rules & Conditions:")
            for idx, r in enumerate(self.rules_found, 1):
                lines.append(f"• If {r.condition}, then {r.consequence}.")
        return "\n".join(lines)


class DeterministicLogicalReasoner:
    """
    Extracts conditional dependencies, prerequisites, and logical rules from document chunks.
    Evaluates:
      - "Can I [Action X] without [Requirement Y]?" -> Checks for prerequisite/dependency.
      - "What happens if [Condition C]?" -> Extracts consequence from IF-THEN statements.
      - "Is [Feature A] required for [Feature B]?" -> Evaluates necessity.
    """

    IF_THEN_PATTERNS = [
        re.compile(r"\bif\s+(.+?),\s*(?:then\s+)?(.+)", re.IGNORECASE),
        re.compile(r"\bin\s+case\s+(?:of\s+)?(.+?),\s*(.+)", re.IGNORECASE),
        re.compile(r"\bwhen\s+(.+?),\s*(.+)", re.IGNORECASE),
        re.compile(r"\bprovided\s+(?:that\s+)?(.+?),\s*(.+)", re.IGNORECASE),
        re.compile(r"\bunless\s+(.+?),\s*(.+)", re.IGNORECASE),
    ]

    PREREQUISITE_PATTERNS = [
        re.compile(r"\b(?:requires|prerequisite|requires\s+that|must\s+first)\s+(.+)", re.IGNORECASE),
        re.compile(r"(.+?)\s+is\s+required\s+(?:for|to|before)\s+(.+)", re.IGNORECASE),
        re.compile(r"\bcannot\s+(?:be\s+done|proceed|be\s+used)\s+without\s+(.+)", re.IGNORECASE),
    ]

    @classmethod
    def detect_logical_intent(cls, query: str) -> Optional[Tuple[str, str]]:
        """
        Detects if query involves a logical/conditional inquiry.
        Returns: (query_type, condition_clause) e.g. ('can_without', 'configure without saving')
        """
        q = query.lower().strip()

        # "Can I [X] without [Y]?"
        m = re.search(r"\bcan\s+(?:i|we|a\s+user)\s+(.+?)\s+without\s+(.+?)(?:\?|$)", q)
        if m:
            return "can_without", f"{m.group(1).strip()} WITHOUT {m.group(2).strip()}"

        # "What happens if [X]?"
        m = re.search(r"\bwhat\s+happens\s+if\s+(.+?)(?:\?|$)", q)
        if m:
            return "if_condition", m.group(1).strip()

        # "Is [X] required for [Y]?"
        m = re.search(r"\bis\s+(.+?)\s+(?:required|mandatory|necessary)\s+(?:for|to)\s+(.+?)(?:\?|$)", q)
        if m:
            return "is_required", f"{m.group(1).strip()} FOR {m.group(2).strip()}"

        # "Does [X] depend on [Y]?"
        m = re.search(r"\bdoes\s+(.+?)\s+depend\s+on\s+(.+?)(?:\?|$)", q)
        if m:
            return "dependency", f"{m.group(1).strip()} ON {m.group(2).strip()}"

        return None

    @classmethod
    def evaluate_logic(
        cls,
        query: str,
        evidence_chunks: List[DocumentChunk],
    ) -> LogicalReasoningResult:
        """
        Evaluates logical relationships and conditional statements against retrieved evidence.
        """
        intent = cls.detect_logical_intent(query)
        result = LogicalReasoningResult(query=query)

        if not intent or not evidence_chunks:
            return result

        logic_type, clause = intent
        result.evaluated_condition = clause
        rules: List[LogicalRule] = []

        for chunk in evidence_chunks:
            sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
            for sent in sentences:
                sent_clean = sent.strip()
                if not sent_clean or len(sent_clean) < 15:
                    continue

                # 1. Check IF-THEN rules
                for pat in cls.IF_THEN_PATTERNS:
                    m = pat.search(sent_clean)
                    if m:
                        cond = m.group(1).strip()
                        cons = m.group(2).strip()
                        if len(cond) > 5 and len(cons) > 5:
                            rules.append(LogicalRule(
                                condition=cond,
                                consequence=cons,
                                rule_type="if_then",
                                evidence_chunk_id=chunk.chunk_id,
                                document_name=chunk.document_name,
                                raw_sentence=sent_clean
                            ))

                # 2. Check Prerequisite rules
                for pat in cls.PREREQUISITE_PATTERNS:
                    m = pat.search(sent_clean)
                    if m:
                        rules.append(LogicalRule(
                            condition=m.group(1).strip(),
                            consequence=m.group(2).strip() if m.lastindex >= 2 else "Operation allowed",
                            rule_type="prerequisite",
                            evidence_chunk_id=chunk.chunk_id,
                            document_name=chunk.document_name,
                            raw_sentence=sent_clean
                        ))

        result.rules_found = rules

        # Specific resolution based on logic_type
        if logic_type == "if_condition":
            # Find closest matching condition
            cond_words = set(re.findall(r"\b\w+\b", clause.lower()))
            best_rule = None
            best_overlap = 0
            for r in rules:
                rule_words = set(re.findall(r"\b\w+\b", r.condition.lower()))
                overlap = len(cond_words.intersection(rule_words))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_rule = r

            if best_rule and best_overlap >= 1:
                result.supported = True
                result.verdict = "CONDITIONAL"
                result.explanation = (
                    f"Based on the document: If {best_rule.condition}, {best_rule.consequence}."
                )
                result.evidence_claims.append(best_rule.raw_sentence)

        elif logic_type == "can_without":
            parts = clause.split(" WITHOUT ")
            action, missing = parts[0], parts[1]
            # Check if missing element is marked as required or prerequisite
            missing_words = set(re.findall(r"\b\w+\b", missing.lower()))
            is_mandatory = False
            mand_sentence = ""

            for chunk in evidence_chunks:
                text_lower = chunk.text.lower()
                for mw in missing_words:
                    if len(mw) < 3:
                        continue
                    if f"{mw} is required" in text_lower or f"mandatory" in text_lower or f"cannot be without" in text_lower:
                        is_mandatory = True
                        mand_sentence = chunk.text[:200]
                        break

            if is_mandatory:
                result.supported = True
                result.verdict = "VIOLATED"
                result.explanation = (
                    f"No. According to the document, '{missing}' is required and cannot be omitted."
                )
                if mand_sentence:
                    result.evidence_claims.append(mand_sentence)
            elif rules:
                result.supported = True
                result.verdict = "CONDITIONAL"
                result.explanation = f"Reviewing requirements for '{action}', conditional dependencies apply."

        elif logic_type in ("is_required", "dependency"):
            # Check if mentioned as prerequisite
            clause_words = set(re.findall(r"\b\w+\b", clause.lower()))
            for chunk in evidence_chunks:
                lower = chunk.text.lower()
                if "require" in lower or "mandatory" in lower or "depend" in lower:
                    # Check token overlap
                    tokens = set(re.findall(r"\b\w+\b", lower))
                    if len(clause_words.intersection(tokens)) >= 2:
                        result.supported = True
                        result.verdict = "SATISFIED"
                        result.explanation = (
                            f"Yes. The documentation indicates that {clause.replace(' FOR ', ' is required for ').replace(' ON ', ' depends on ')}."
                        )
                        result.evidence_claims.append(chunk.text[:200])
                        break

        return result
