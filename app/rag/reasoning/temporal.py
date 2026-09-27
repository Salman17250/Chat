"""
app/rag/reasoning/temporal.py
Deterministic Temporal & Sequential Reasoning Engine.
Analyzes temporal expressions (before, after, next, previous, first, last, sequence, prior, following)
and orders extracted steps or events strictly based on document evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.rag.ingestion.models import DocumentChunk


@dataclass
class TemporalEvent:
    """A distinct step, event, or condition identified with temporal metadata."""
    description: str
    order_index: int
    marker: str  # e.g., 'after', 'before', 'step 1', 'first', 'next', 'then'
    evidence_chunk_id: Optional[str] = None
    document_name: Optional[str] = None
    raw_sentence: str = ""


@dataclass
class TemporalReasoningResult:
    """Structured result of temporal reasoning."""
    target_event: str
    relation: str  # 'after', 'before', 'sequence', 'next', 'prerequisite'
    ordered_events: List[TemporalEvent] = field(default_factory=list)
    direct_answers: List[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence_found: bool = False
    explanation: str = ""

    def to_formatted_answer(self) -> str:
        """Produce clean, evidence-backed natural response."""
        if not self.evidence_found or not self.direct_answers:
            return ""
        
        lines = []
        if self.relation in ("after", "next", "following"):
            lines.append(f"Following or after {self.target_event}:")
            for idx, ans in enumerate(self.direct_answers, 1):
                clean_ans = ans.strip().rstrip(".")
                lines.append(f"{idx}. {clean_ans}.")
        elif self.relation in ("before", "prior", "prerequisite"):
            lines.append(f"Prior to or before {self.target_event}:")
            for idx, ans in enumerate(self.direct_answers, 1):
                clean_ans = ans.strip().rstrip(".")
                lines.append(f"{idx}. {clean_ans}.")
        else:
            lines.append(f"Timeline / Sequence for {self.target_event}:")
            for idx, ans in enumerate(self.direct_answers, 1):
                clean_ans = ans.strip().rstrip(".")
                lines.append(f"{idx}. {clean_ans}.")
        
        return "\n".join(lines)


class DeterministicTemporalReasoner:
    """
    Extracts procedural or chronological sequences from evidence text deterministically.
    Recognizes:
      - Explicit step numbering: "1.", "Step 1:", "(i)"
      - Temporal prepositions: "After creating...", "Once the pipeline is created, configure..."
      - Sequence adverbs: "First", "Next", "Then", "Subsequently", "Finally"
      - Prerequisite clauses: "Before X can be used, Y must be completed"
    """

    TEMPORAL_MARKERS = {
        "after": [r"\bafter\b", r"\bonce\b", r"\bfollowing\b", r"\bsubsequent(?:ly)?\b", r"\bnext\b", r"\bthen\b"],
        "before": [r"\bbefore\b", r"\bprior to\b", r"\bprerequisite\b", r"\binitially\b", r"\bfirst\b"],
        "during": [r"\bduring\b", r"\bwhile\b", r"\bas\b"],
    }

    STEP_PATTERNS = [
        re.compile(r"^(?:step\s*\d+|[\d]+[\.\)]|\([a-z\d]+\))\s*(.+)$", re.IGNORECASE),
        re.compile(r"\b(?:step\s*\d+|[\d]+[\.\)]|\([a-z\d]+\))\s*[:\-]?\s*([^.;\n]+)", re.IGNORECASE),
    ]

    @classmethod
    def detect_temporal_intent(cls, query: str) -> Optional[Tuple[str, str]]:
        """
        Detects if query is asking for temporal/sequential information.
        Returns: (relation, anchor_subject) e.g. ('after', 'creating a pipeline')
        """
        q = query.lower().strip()

        # "What happens after creating a pipeline?" / "What to do after creating a pipeline?"
        m = re.search(r"\b(?:what\s+happens|what\s+to\s+do|what\s+should\s+be\s+done|what\s+is\s+the\s+next\s+step)\s+after\s+(.+?)(?:\?|$)", q)
        if m:
            return "after", m.group(1).strip()

        # "after [action/event], what happens?"
        m = re.search(r"\bafter\s+(.+?),\s*(?:what\s+happens|what\s+next|what\s+to\s+do)(?:\?|$)", q)
        if m:
            return "after", m.group(1).strip()

        # "before [action/event], what ..." / "what is needed before ..."
        m = re.search(r"\b(?:what\s+(?:is|are)\s+(?:needed|required)|what\s+to\s+do)\s+before\s+(.+?)(?:\?|$)", q)
        if m:
            return "before", m.group(1).strip()

        # General "after X"
        m = re.search(r"\bafter\s+([a-z0-9\s]{3,35}?)(?:\?|\s+how|\s+what|$)", q)
        if m:
            return "after", m.group(1).strip()

        # General "before X"
        m = re.search(r"\bbefore\s+([a-z0-9\s]{3,35}?)(?:\?|\s+how|\s+what|$)", q)
        if m:
            return "before", m.group(1).strip()

        # "next step for X"
        m = re.search(r"\bnext\s+step(?:s)?\s+(?:for|in|of|after)\s+(.+?)(?:\?|$)", q)
        if m:
            return "next", m.group(1).strip()

        return None

    @classmethod
    def reason_sequence(
        cls,
        target_subject: str,
        relation: str,
        evidence_chunks: List[DocumentChunk],
    ) -> TemporalReasoningResult:
        """
        Extracts temporal order and steps related to target_subject and relation.
        """
        result = TemporalReasoningResult(
            target_event=target_subject,
            relation=relation,
            ordered_events=[],
            direct_answers=[],
            confidence=0.0,
            evidence_found=False,
        )

        if not evidence_chunks:
            return result

        clean_subject_words = [w for w in re.findall(r"\b\w+\b", target_subject.lower()) if len(w) > 2]
        extracted_steps: List[TemporalEvent] = []
        direct_step_texts: List[str] = []

        for chunk in evidence_chunks:
            text = chunk.text
            # Split into sentences or lines
            lines = [l.strip() for l in re.split(r"[\n\r]+", text) if l.strip()]

            # 1. Look for numbered step lists in the chunk
            chunk_steps = []
            for line in lines:
                for pat in cls.STEP_PATTERNS:
                    sm = pat.match(line)
                    if sm:
                        step_desc = sm.group(1).strip()
                        if len(step_desc) > 8:
                            chunk_steps.append((step_desc, line))
                            break

            if chunk_steps and len(chunk_steps) >= 2:
                # We found a multi-step sequence
                for idx, (sdesc, raw_line) in enumerate(chunk_steps, 1):
                    extracted_steps.append(TemporalEvent(
                        description=sdesc,
                        order_index=idx,
                        marker=f"Step {idx}",
                        evidence_chunk_id=chunk.chunk_id,
                        document_name=chunk.document_name,
                        raw_sentence=raw_line
                    ))
                    direct_step_texts.append(sdesc)

            # 2. Look for sentence-level temporal transitions
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for s_idx, sent in enumerate(sentences):
                sent_clean = sent.strip()
                if not sent_clean or len(sent_clean) < 15:
                    continue
                sent_lower = sent_clean.lower()

                if relation in ("after", "next", "following"):
                    # Match patterns like:
                    # "After creating the pipeline, configure the stages..."
                    # "Once created, the user can..."
                    # "The next step is to..."
                    after_patterns = [
                        r"\bafter\s+(?:creating|creation of|configuring|setting up|the)?\s*([a-z0-9\s]+?),\s*(.+)",
                        r"\bonce\s+(?:created|configured|saved|setup|active),\s*(.+)",
                        r"\bthe\s+next\s+step\s+is\s+to\s+(.+)",
                        r"\bthen,?\s*(.+)",
                        r"\bsubsequently,?\s*(.+)",
                    ]
                    for ap in after_patterns:
                        am = re.search(ap, sent_clean, re.IGNORECASE)
                        if am:
                            # Group could be following action
                            consequence = am.group(am.lastindex).strip()
                            if len(consequence) > 10 and consequence not in direct_step_texts:
                                direct_step_texts.append(consequence)
                                extracted_steps.append(TemporalEvent(
                                    description=consequence,
                                    order_index=len(extracted_steps) + 1,
                                    marker="after",
                                    evidence_chunk_id=chunk.chunk_id,
                                    document_name=chunk.document_name,
                                    raw_sentence=sent_clean
                                ))
                                break

                elif relation in ("before", "prior", "prerequisite"):
                    # Match patterns like:
                    # "Before creating a pipeline, ensure..."
                    # "Prerequisites include..."
                    before_patterns = [
                        r"\bbefore\s+(?:creating|configuring|using|starting)?\s*([a-z0-9\s]+?),\s*(.+)",
                        r"\bprior\s+to\s+(.+?),\s*(.+)",
                        r"\bprerequisite[s]?\s*(?:are|is|include)?\s*[:\-]?\s*(.+)",
                    ]
                    for bp in before_patterns:
                        bm = re.search(bp, sent_clean, re.IGNORECASE)
                        if bm:
                            prereq = bm.group(bm.lastindex).strip()
                            if len(prereq) > 10 and prereq not in direct_step_texts:
                                direct_step_texts.append(prereq)
                                extracted_steps.append(TemporalEvent(
                                    description=prereq,
                                    order_index=len(extracted_steps) + 1,
                                    marker="before",
                                    evidence_chunk_id=chunk.chunk_id,
                                    document_name=chunk.document_name,
                                    raw_sentence=sent_clean
                                ))
                                break

        # Deduplicate direct steps while preserving order
        unique_steps = []
        seen = set()
        for st in direct_step_texts:
            norm = st.lower().strip()
            if norm not in seen:
                seen.add(norm)
                unique_steps.append(st)

        if unique_steps:
            result.evidence_found = True
            result.direct_answers = unique_steps[:6]  # Cap at top 6 distinct steps
            result.ordered_events = extracted_steps
            result.confidence = 0.85 if len(unique_steps) >= 2 else 0.70
            result.explanation = f"Found {len(result.direct_answers)} sequential step(s) with {relation} relationship."

        return result
