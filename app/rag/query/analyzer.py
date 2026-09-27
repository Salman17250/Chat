import re
from enum import Enum
from typing import List, Dict, Set, Optional, Tuple
from app.rag.query.normalizer import QueryNormalizer, STOP_WORDS, singularize
from app.rag.vocabulary.dynamic_vocabulary import DynamicCorpusVocabulary

class QueryProfile(str, Enum):
    EXACT_CODE = "EXACT_CODE"                   # Error codes, SKUs, camelCase, snake_case, IDs
    PROCEDURAL = "PROCEDURAL"                   # How to, steps to, instructions, creation
    TABLE_LOOKUP = "TABLE_LOOKUP"               # Pricing, specs, attributes, quantities, comparisons
    MULTI_PART = "MULTI_PART"                   # Compound questions with multiple distinct queries
    CONCEPTUAL_PARAPHRASE = "CONCEPTUAL"       # Explanatory, definitional, policies, natural questions
    WORKFLOW = "WORKFLOW"                       # Sequential operations, lifecycle stages, after-creation steps
    RELATIONSHIP = "RELATIONSHIP"               # Structure, prerequisites, containment, dependencies

class AnalyzedQuery:
    """Structured container for deep deterministic query intelligence."""
    def __init__(
        self,
        raw_query: str,
        normalized_query: str,
        profile: QueryProfile,
        sub_queries: List[str],
        primary_entity: Optional[str] = None,
        action_verb: Optional[str] = None,
        target_attributes: List[str] = None,
        expanded_acronyms: Dict[str, str] = None,
        is_elliptical: bool = False,
        core_query: Optional[str] = None,
        context_scope: Optional[str] = None
    ):
        self.raw_query = raw_query
        self.normalized_query = normalized_query
        self.profile = profile
        self.sub_queries = sub_queries if sub_queries else [raw_query]
        self.primary_entity = primary_entity
        self.action_verb = action_verb
        self.target_attributes = target_attributes or []
        self.expanded_acronyms = expanded_acronyms or {}
        self.is_elliptical = is_elliptical
        self.core_query = core_query
        self.context_scope = context_scope

class QueryAnalyzer:
    """
    Deterministic Query & Concept Intelligence Engine.
    Performs query profiling, multi-part clause splitting, grammatical role extraction,
    acronym expansion, and elliptical follow-up detection without an LLM.
    """

    # Multi-part conjunction splitter patterns
    MULTI_PART_PATTERN = re.compile(
        r'\s+\b(?:and|as\s+well\s+as|along\s+with)\s+(?=(?:how|what|when|where|who|which|why)\b)',
        re.IGNORECASE
    )

    # Procedural intent markers
    PROCEDURAL_PATTERNS = [
        r'\b(?:how\s+(?:to|can\s+i|do\s+i|can\s+we|to\s+get|do\s+we))\b',
        r'\b(?:steps?\s+to|procedure|process\s+(?:for|to)|guide\s+to|instructions?\s+for)\b',
        r'\b(?:how\s+do\s+you|way\s+to|directions?\s+to)\b'
    ]

    # Table/Attribute lookup markers
    TABLE_PATTERNS = [
        r'\b(?:table|matrix|column|row|grid|specification|specs)\b',
        r'\b(?:in\s+the\s+table|per\s+the\s+table|table\s+shows?)\b',
    ]

    # Elliptical / Follow-up markers
    ELLIPTICAL_PATTERNS = [
        r'^(?:what\s+about|how\s+about|and\s+for|what\s+if|and\s+what\s+about)\b',
        r'^(?:is\s+it\s+the\s+same\s+for|does\s+that\s+apply\s+to)\b',
        r'^(?:why\??|how\??|where\??)$'
    ]

    # Conversational prefix pattern (polite request framing)
    CONVERSATIONAL_PREFIX_PATTERN = re.compile(
        r'^(?:can\s+you\s+(?:please\s+)?(?:tell\s+me|explain|clarify|show\s+me|help\s+me\s+with|describe)|'
        r'could\s+you\s+(?:please\s+)?(?:tell\s+me|explain|clarify|show\s+me|describe)|'
        r'please\s+(?:tell\s+me|explain|clarify|show\s+me|describe)|'
        r'tell\s+me|i\s+(?:want|need|would\s+like)\s+to\s+(?:know|understand|find\s+out|see)|'
        r'kindly\s+(?:tell\s+me|explain|clarify|show\s+me))\s+',
        re.IGNORECASE
    )

    # Scoping preposition pattern (e.g. "what is [a] TARGET in/within/inside CONTEXT")
    SCOPING_PATTERN = re.compile(
        r'^((?:what\s+(?:is|are|does|do)|how\s+(?:does|do|can|to))\s+(?:a|an|the)?\s*([a-z0-9_\-\s]{2,30}?))\s+(?:in|within|inside)\s+([a-z0-9_\-\s]+)$',
        re.IGNORECASE
    )

    def __init__(self, normalizer: Optional[QueryNormalizer] = None):
        self.normalizer = normalizer or QueryNormalizer()

    def strip_conversational_prefix(self, query: str) -> str:
        """Strips conversational wrappers like 'can you please tell me' from a query."""
        q = query.strip()
        m = self.CONVERSATIONAL_PREFIX_PATTERN.match(q)
        if m:
            stripped = q[m.end():].strip()
            if len(stripped) >= 3:
                return stripped
        return q

    def extract_scoping_context(self, query: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Detects natural syntactic scoping: e.g. 'what is a source in pipeline'
        Returns (core_query, primary_entity, context_scope).
        """
        clean = self.strip_conversational_prefix(query).strip('?.,;:! ')
        m = self.SCOPING_PATTERN.match(clean)
        if m:
            core_query = m.group(1).strip()
            raw_target = m.group(2).strip()
            context_scope = m.group(3).strip()
            target_words = [w for w in raw_target.split() if w.lower() not in STOP_WORDS and len(w) >= 3]
            entity = target_words[0] if target_words else raw_target
            return core_query, entity, context_scope
        return None, None, None

    def split_multi_part(self, query: str) -> List[str]:
        """
        Splits compound questions into independent sub-queries.
        E.g. "How many vacation days do employees receive and how early must they request them?"
        -> ["How many vacation days do employees receive", "how early must they request them"]
        """
        splits = self.MULTI_PART_PATTERN.split(query)
        cleaned = [s.strip() for s in splits if len(s.strip()) >= 5]
        return cleaned if len(cleaned) > 1 else [query.strip()]

    def is_elliptical_query(self, query: str) -> bool:
        """Detects if query is an elliptical or follow-up question requiring conversational context."""
        q_lower = query.lower().strip()
        for p in self.ELLIPTICAL_PATTERNS:
            if re.search(p, q_lower):
                return True
        # If query has <= 3 words and lacks a standard subject/verb pattern
        words = q_lower.split()
        if len(words) <= 3 and words[0] in {"what", "how", "and", "why", "where"}:
            return True
        return False

    def extract_grammatical_roles(self, query: str, vocab: Optional[DynamicCorpusVocabulary] = None) -> Tuple[Optional[str], Optional[str], List[str]]:
        """
        Extracts action verb, primary entity, and target attributes deterministically
        without hardcoded business dictionaries.
        """
        norm_res = self.normalizer.normalize(query)
        q_norm = norm_res.normalized_text
        words = [w for w in q_norm.split() if len(w) >= 2]
        action_verb = None
        primary_entity = None
        target_attrs: List[str] = []

        # Interrogative lead: e.g. "how to [verb] [entity]"
        m_action = re.search(r'\b(?:how\s+to|how\s+can\s+i|steps\s+to)\s+([a-z]{3,})\b', query.lower())
        if m_action:
            action_verb = singularize(m_action.group(1))

        attribute_or_topic_nouns = {
            "technology", "technologies", "skill", "skills", "tool", "tools",
            "age", "education", "degree", "qualification", "profession", "job", "role",
            "price", "cost", "fee", "salary", "compensation", "period", "duration",
            "policy", "rule", "rules", "guideline", "guidelines", "process", "procedure",
            "details", "information", "name", "status", "deadline", "date", "time",
            "table", "row", "column", "list", "total", "difference", "comparison"
        }

        content_words = [w for w in words if w not in STOP_WORDS and len(w) >= 3]

        if content_words:
            cand = content_words[0]
            if cand.lower() not in attribute_or_topic_nouns:
                primary_entity = cand
            target_attrs = [w for w in content_words if w != primary_entity]

        return action_verb, primary_entity, target_attrs

    def classify_profile(self, query: str, sub_queries: List[str]) -> QueryProfile:
        """Deterministically classifies query profile for adaptive retrieval weighting."""
        # 1. Multi-part
        if len(sub_queries) > 1:
            return QueryProfile.MULTI_PART

        q_lower = query.lower().strip()

        # 2. Exact code / ID check
        # e.g., ERR_404, HTTP_500, CamelCase, contains digits mixed with letters
        tokens = query.strip().split()
        for t in tokens:
            t_clean = t.strip('.,;?!"\'')
            if re.match(r'^[A-Z0-9_\-]{4,}$', t_clean) and any(c.isdigit() for c in t_clean) and any(c.isalpha() for c in t_clean):
                return QueryProfile.EXACT_CODE
            if re.match(r'^[A-Z]{2,}\-[0-9]+', t_clean):
                return QueryProfile.EXACT_CODE

        # 3. Workflow / Lifecycle steps
        if re.search(r'\b(?:after\s+(?:creating|creation|setting\s+up|configuring|saving)|what\s+happens\s+after|next\s+steps?|workflow\s+of|lifecycle)\b', q_lower):
            return QueryProfile.WORKFLOW

        # 4. Procedural
        for p in self.PROCEDURAL_PATTERNS:
            if re.search(p, q_lower):
                return QueryProfile.PROCEDURAL

        # 5. Table / Attribute Lookup
        for p in self.TABLE_PATTERNS:
            if re.search(p, q_lower):
                return QueryProfile.TABLE_LOOKUP

        # 6. Relationship / Structure
        if re.search(r'\b(?:composed\s+of|consists\s+of|contains|has\s+stages?|prerequisites?|depends\s+on)\b', q_lower):
            return QueryProfile.RELATIONSHIP

        # 7. Default to Conceptual Paraphrase
        return QueryProfile.CONCEPTUAL_PARAPHRASE

    def analyze(self, query: str, vocab: Optional[DynamicCorpusVocabulary] = None) -> AnalyzedQuery:
        """
        Executes full deterministic analysis of an input query.
        """
        raw = query.strip()
        stripped_raw = self.strip_conversational_prefix(raw)
        norm_res = self.normalizer.normalize(stripped_raw if stripped_raw else raw)
        norm = norm_res.normalized_text
        sub_queries = self.split_multi_part(stripped_raw if stripped_raw else raw)
        profile = self.classify_profile(stripped_raw if stripped_raw else raw, sub_queries)
        is_elliptical = self.is_elliptical_query(raw)
        
        # Detect syntactic scoping (e.g. "what is a source in pipeline" -> core="what is a source", entity="source", scope="pipeline")
        core_query, scoping_entity, context_scope = self.extract_scoping_context(raw)
        
        analysis_target = core_query if core_query else (stripped_raw if stripped_raw else raw)
        action_verb, primary_entity, target_attrs = self.extract_grammatical_roles(analysis_target, vocab=vocab)
        if scoping_entity:
            primary_entity = scoping_entity

        # Acronym expansion if vocabulary is provided
        expanded_acronyms: Dict[str, str] = {}
        if vocab:
            tokens = re.findall(r'\b[A-Za-z0-9]+\b', raw)
            for t in tokens:
                expanded = vocab.expand_acronym(t)
                if expanded:
                    expanded_acronyms[t.upper()] = expanded

        return AnalyzedQuery(
            raw_query=raw,
            normalized_query=norm,
            profile=profile,
            sub_queries=sub_queries,
            primary_entity=primary_entity,
            action_verb=action_verb,
            target_attributes=target_attrs,
            expanded_acronyms=expanded_acronyms,
            is_elliptical=is_elliptical,
            core_query=core_query,
            context_scope=context_scope
        )
