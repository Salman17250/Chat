import re
from typing import Tuple, List, Optional
from app.rag.context.session_state import StructuredSessionState

class CoreferenceResolver:
    """
    Deterministic Coreference & Anaphora Resolution Engine (Zero LLM).
    Resolves conversational pronouns and demonstrative noun phrases:
      - 'it', 'this', 'that', 'they', 'them', 'its', 'their'
      - 'the above', 'same thing', 'this feature', 'that process', 'this step'
      - Elliptical references ('what about...', 'how about...')
    Uses active entity continuity, topic persistence, and syntactic substitution.
    """

    PRONOUN_PATTERNS = [
        # Standalone 'it' / 'its' / 'this' / 'that' / 'them' / 'they'
        (re.compile(r'\b(?:it|its)\b', re.IGNORECASE), "it"),
        (re.compile(r'\b(?:this|that)\b(?!\s+(?:is|was|are|were|can|will|should|would|has|have|document|feature|process|step|table|policy|pipeline))', re.IGNORECASE), "this"),
        (re.compile(r'\b(?:them|they|their)\b', re.IGNORECASE), "they"),
    ]

    ANAPHORIC_PHRASE_PATTERNS = [
        re.compile(r'\b(?:the\s+above|same\s+thing)\b', re.IGNORECASE),
        re.compile(r'\b(?:this|that)\s+(?:feature|process|step|module|pipeline|policy|component|service|concept|function)\b', re.IGNORECASE),
    ]

    ELLIPTICAL_PREFIXES = [
        re.compile(r'^(?:what\s+about|how\s+about|and\s+for|what\s+if)\s+', re.IGNORECASE),
        re.compile(r'^(?:is\s+it\s+the\s+same\s+for|does\s+that\s+apply\s+to)\s+', re.IGNORECASE),
    ]

    @classmethod
    def resolve_references(
        cls,
        query: str,
        session_state: Optional[StructuredSessionState] = None,
        context_entity: Optional[str] = None
    ) -> Tuple[str, bool, List[str]]:
        """
        Resolves coreferent expressions in query using session state or explicit context entity.
        Returns:
          (resolved_query, was_resolved, list_of_resolved_references)
        """
        raw_q = query.strip()
        target_entity = context_entity
        if not target_entity and session_state:
            target_entity = session_state.active_entity or session_state.active_topic

        if not target_entity:
            return raw_q, False, []

        resolved_q = raw_q
        resolved_refs: List[str] = []

        # 1. Resolve anaphoric phrases: e.g. "this feature" -> "Pipeline"
        for pat in cls.ANAPHORIC_PHRASE_PATTERNS:
            match = pat.search(resolved_q)
            if match:
                matched_text = match.group(0)
                resolved_refs.append(matched_text)
                resolved_q = pat.sub(target_entity, resolved_q)

        # 2. Resolve pronoun patterns: e.g. "configure it" -> "configure Pipeline"
        for pat, ref_type in cls.PRONOUN_PATTERNS:
            if pat.search(resolved_q):
                # Check that we don't replace "it" inside words or when it refers to dummy 'it' (e.g., "is it possible")
                # But for questions like "How do I configure it?", "What happens after creating it?", "What is it?"
                # replacement is desirable.
                # Exception: "is it possible to" -> preserve "is it possible"
                dummy_it = re.search(r'\b(?:is|was)\s+it\s+(?:possible|allowed|required|mandatory)\b', resolved_q, re.I)
                if dummy_it:
                    continue

                matches = pat.findall(resolved_q)
                if matches:
                    resolved_refs.extend(matches)
                    resolved_q = pat.sub(target_entity, resolved_q)

        # 3. Handle elliptical query framing: e.g. "What about stages?" -> "What about stages in Pipeline?"
        for el_pat in cls.ELLIPTICAL_PREFIXES:
            if el_pat.search(resolved_q):
                if target_entity.lower() not in resolved_q.lower():
                    resolved_q = f"{resolved_q} for {target_entity}"
                    resolved_refs.append("elliptical_topic")

        # Clean multiple spaces
        resolved_q = re.sub(r'\s{2,}', ' ', resolved_q).strip()
        was_resolved = bool(resolved_refs)

        return resolved_q, was_resolved, resolved_refs
