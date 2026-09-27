"""
app/rag/reasoning/comparator.py
Deterministic Entity & Feature Comparison Engine.
Produces structured comparison matrices across entities (A vs B, Feature X vs Feature Y),
aligning common attributes, contrast points, and source citations without guessing missing values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from app.rag.ingestion.models import DocumentChunk


@dataclass
class ComparisonAttributeRow:
    attribute_name: str
    entity_a_val: str
    entity_b_val: str
    evidence_citation: str = ""


@dataclass
class ComparisonResult:
    entity_a: str
    entity_b: str
    attribute_rows: List[ComparisonAttributeRow] = field(default_factory=list)
    summary_a: str = ""
    summary_b: str = ""
    sources: List[str] = field(default_factory=list)

    def to_formatted_answer(self) -> str:
        lines = [f"### Comparison: {self.entity_a.title()} vs. {self.entity_b.title()}\n"]
        
        if self.attribute_rows:
            lines.append(f"| Feature / Attribute | {self.entity_a.title()} | {self.entity_b.title()} |")
            lines.append("|---|---|---|")
            for row in self.attribute_rows:
                lines.append(f"| **{row.attribute_name}** | {row.entity_a_val} | {row.entity_b_val} |")
            lines.append("")
        else:
            lines.append(f"**1. {self.entity_a.title()}:**")
            lines.append(self.summary_a or "No specific details found in uploaded documents.")
            lines.append(f"\n**2. {self.entity_b.title()}:**")
            lines.append(self.summary_b or "No specific details found in uploaded documents.")
            lines.append("")

        if self.sources:
            unique_src = list(dict.fromkeys(self.sources))
            lines.append(f"*(Sources: {', '.join(unique_src)})*")

        return "\n".join(lines)


class DeterministicComparator:
    """
    Deterministic Comparison Engine.
    Structures multi-entity side-by-side attribute comparisons from retrieved evidence.
    """

    @classmethod
    def extract_key_value_pairs(cls, text: str) -> Dict[str, str]:
        """
        Extracts attribute-value pairs from text (e.g. 'Status: Active', 'Type: Pipeline', 'Price: $100').
        """
        pairs: Dict[str, str] = {}
        for line in text.split("\n"):
            line = line.strip()
            # Match "Key: Value" or "Key - Value"
            m = re.match(r"^([A-Za-z0-9\s]{3,25})\s*[:\-]\s*(.+)$", line)
            if m:
                key = m.group(1).strip().title()
                val = m.group(2).strip()
                if len(val) > 1 and len(key) > 2:
                    pairs[key] = val
        return pairs

    @classmethod
    def compare_entities(
        cls,
        entity_a: str,
        entity_b: str,
        chunks_a: List[DocumentChunk],
        chunks_b: List[DocumentChunk]
    ) -> str:
        """
        Builds a structured comparison between entity A and entity B from their respective retrieved chunks.
        """
        text_a = " ".join([c.text.strip() for c in chunks_a[:2]]) if chunks_a else ""
        text_b = " ".join([c.text.strip() for c in chunks_b[:2]]) if chunks_b else ""

        # Extract attributes from each side
        pairs_a = cls.extract_key_value_pairs(text_a)
        pairs_b = cls.extract_key_value_pairs(text_b)

        sources = []
        for c in chunks_a + chunks_b:
            if c.document_name and c.document_name not in sources:
                sources.append(c.document_name)

        all_keys = set(pairs_a.keys()).union(set(pairs_b.keys()))
        rows: List[ComparisonAttributeRow] = []

        for key in sorted(all_keys):
            val_a = pairs_a.get(key, "Not specified in document")
            val_b = pairs_b.get(key, "Not specified in document")
            rows.append(ComparisonAttributeRow(
                attribute_name=key,
                entity_a_val=val_a,
                entity_b_val=val_b,
                evidence_citation=", ".join(sources)
            ))

        # Build clean summary paragraphs if no explicit key-value rows were found
        summary_a = chunks_a[0].text.strip() if chunks_a else "No details found."
        summary_b = chunks_b[0].text.strip() if chunks_b else "No details found."

        result = ComparisonResult(
            entity_a=entity_a,
            entity_b=entity_b,
            attribute_rows=rows if len(rows) >= 2 else [],
            summary_a=summary_a,
            summary_b=summary_b,
            sources=sources
        )

        return result.to_formatted_answer()
