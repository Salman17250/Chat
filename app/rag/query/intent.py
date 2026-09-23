import re
from enum import Enum
from typing import Dict, List, Tuple

class QueryIntent(str, Enum):
    PROCEDURAL = "PROCEDURAL"       # Steps, process, how-to
    FACTUAL = "FACTUAL"             # Definitions, facts, concepts
    CAPABILITY = "CAPABILITY"       # Feasibility, support, rules
    GENERAL = "GENERAL"

# Generic linguistic patterns based on English interrogatives
INTENT_PATTERNS: Dict[QueryIntent, List[str]] = {
    QueryIntent.PROCEDURAL: [
        r"\bhow\s+(?:do|can|to|would|should|does)\b",
        r"\bsteps?\s+to\b",
        r"\bprocedure\s+(?:for|to)\b",
        r"\binstructions?\s+(?:for|to)\b",
        r"\bprocess\s+of\b",
        r"\bguide\s+(?:to|for)\b",
    ],
    QueryIntent.FACTUAL: [
        r"\bwhat\s+(?:is|are|does|means?)\b",
        r"\bexplain\b",
        r"\bdefinition\s+of\b",
        r"\bmeaning\s+of\b",
        r"\bdescribe\b",
        r"\boverview\s+of\b",
    ],
    QueryIntent.CAPABILITY: [
        r"\bcan\s+(?:i|we|users?|it|one)\b",
        r"\bis\s+it\s+possible\s+to\b",
        r"\bare\s+we\s+able\s+to\b",
        r"\bdoes\s+it\s+support\b",
        r"\bpermitted\s+to\b",
        r"\ballowed\s+to\b",
    ],
}

class RuleBasedIntentDetector:
    """Lightweight linguistic intent detector without domain or SaaS bias."""

    def __init__(self):
        self._compiled_patterns = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in INTENT_PATTERNS.items()
        }

    def detect_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """Detects linguistic query intent and returns (QueryIntent, confidence)."""
        q = query.strip()
        for intent, patterns in self._compiled_patterns.items():
            for pat in patterns:
                if pat.search(q):
                    return intent, 0.80

        return QueryIntent.GENERAL, 0.50

    def calculate_intent_alignment(self, intent: QueryIntent, chunk_text: str) -> float:
        """
        Evaluates structural alignment between question intent and chunk structure.
        For PROCEDURAL: looks for numbered sequences (e.g. 1., 2., Step 1).
        For FACTUAL: neutral baseline (semantic vectors handle topical matching).
        """
        if intent == QueryIntent.PROCEDURAL:
            # Check for structural step indicators in chunk text (e.g. "1.", "2.", "Step", bullet points)
            has_steps = bool(re.search(r'(?:^\s*\d+[\.\)]\s+|Step\s+\d+|^\s*[•\-\*]\s+)', chunk_text, re.MULTILINE))
            return 0.85 if has_steps else 0.50

        return 0.50
