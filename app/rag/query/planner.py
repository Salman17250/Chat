import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from app.rag.query.analyzer import QueryProfile, QueryAnalyzer, AnalyzedQuery
from app.rag.query.normalizer import QueryNormalizer, STOP_WORDS, singularize
from app.rag.query.decomposer import QueryDecomposer
from app.rag.query.coreference import CoreferenceResolver
from app.rag.vocabulary.dynamic_vocabulary import DynamicCorpusVocabulary

@dataclass
class QueryPlan:
    # Full Phase 1 & 7 Structured Specification
    normalized_query: str
    original_query: str
    language: str = "en"
    intent: str = "general"
    subject: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    action: Optional[str] = None
    qualifiers: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    temporal_context: Optional[str] = None
    comparison_targets: List[str] = field(default_factory=list)
    question_type: str = "what"
    document_scope: Optional[str] = None
    conversation_references: List[str] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    expanded_terms: List[str] = field(default_factory=list)
    retrieval_strategy: Dict[str, Any] = field(default_factory=dict)

    # Backward-compatible convenience accessors
    raw_query: str = ""
    clean_query: str = ""
    profile: QueryProfile = QueryProfile.CONCEPTUAL_PARAPHRASE
    keywords: List[str] = field(default_factory=list)
    numbers: List[float] = field(default_factory=list)
    currencies: List[str] = field(default_factory=list)
    percentages: List[float] = field(default_factory=list)
    target_attributes: List[str] = field(default_factory=list)
    can_bypass_embedding: bool = False
    requires_calculation: bool = False
    requires_comparison: bool = False
    requires_temporal: bool = False
    requires_table_lookup: bool = False
    active_branches: List[str] = field(default_factory=list)

class QueryPlanner:
    """
    Intelligent Deterministic Query Planning & Semantic Representation Engine (<0.2ms).
    Transforms natural-language questions into rich semantic representations:
      - Intent & entity role extraction
      - Action & qualifier identification (e.g. after_creation, next_step)
      - Dynamic question decomposition & reference tracking
      - Adaptive retrieval routing strategy with branch weights
    ZERO hardcoded answers, ZERO LLM on the query path.
    """

    # Interrogative classifiers
    QUESTION_TYPE_PATTERNS = [
        (re.compile(r'\bhow\s+to\b|\bhow\s+do\b|\bhow\s+can\b|\bhow\s+does\b', re.I), "how"),
        (re.compile(r'\bwhat\s+happens\b|\bwhat\s+is\b|\bwhat\s+are\b|\bwhat\b', re.I), "what"),
        (re.compile(r'\bwhy\b', re.I), "why"),
        (re.compile(r'\bwhen\b', re.I), "when"),
        (re.compile(r'\bwhere\b', re.I), "where"),
        (re.compile(r'\bwho\b', re.I), "who"),
        (re.compile(r'\bwhich\b', re.I), "which"),
        (re.compile(r'\b(?:can|could|is|are|does|do|will|should|would)\s+(?:i|we|it|a|the|user)\b', re.I), "boolean"),
    ]

    # Relational and action patterns
    ACTION_PATTERNS = [
        (re.compile(r'\bafter\s+(?:creating|creation|setting\s+up|setup|configuring|configuration|saving|submitting)\b', re.I), "after_creation"),
        (re.compile(r'\bbefore\s+(?:creating|creation|setting\s+up|configuring|deleting|submitting)\b', re.I), "before_action"),
        (re.compile(r'\b(?:how\s+to|how\s+do\s+i|steps\s+to)\s+(create|configure|setup|apply|delete|modify|update|cancel|add)\b', re.I), lambda m: f"how_to_{m.group(1).lower()}"),
        (re.compile(r'\b(?:used\s+for|purpose\s+of|role\s+of)\b', re.I), "purpose_usage"),
        (re.compile(r'\b(?:composed\s+of|consists\s+of|contains|has\s+stages?|structure\s+of)\b', re.I), "structure_containment"),
    ]

    QUALIFIER_KEYWORDS = {
        "after", "before", "during", "then", "next", "subsequently", "first", "finally", "last",
        "creating", "creation", "configured", "configuring", "mandatory", "optional", "default",
        "minimum", "maximum", "total", "average", "difference", "compare", "versus"
    }

    ATTRIBUTE_KEYWORDS = {
        "price", "cost", "fee", "salary", "compensation", "age", "education", "qualification",
        "skills", "technologies", "tools", "period", "duration", "validity", "deadline", "date",
        "department", "location", "status", "stage", "stages", "role", "profession", "discount"
    }

    COMMON_ACTION_VERBS = {
        "happen", "happens", "occur", "occurs", "work", "works", "take", "takes", "start", "starts",
        "do", "does", "did", "done", "see", "show", "tell", "give", "find", "get"
    }

    def __init__(self, analyzer: Optional[QueryAnalyzer] = None):
        self.analyzer = analyzer or QueryAnalyzer()
        self.normalizer = QueryNormalizer()
        self.decomposer = QueryDecomposer()

    def plan(
        self,
        query: str,
        vocab: Optional[DynamicCorpusVocabulary] = None,
        language: str = "en",
        context_entity: Optional[str] = None
    ) -> QueryPlan:
        original_q = query.strip()
        lower_q = original_q.lower()

        # 1. Normalize query
        norm_result = self.normalizer.normalize(original_q)
        normalized_q = norm_result.normalized_text
        tokens = norm_result.tokens

        # 2. Analyze query structure & syntactic scoping
        analyzed = self.analyzer.analyze(original_q, vocab=vocab)

        # 3. Detect Numbers, Percentages, Currencies
        numbers = [float(n.replace(",", "")) for n in re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', original_q)]
        percentages = [float(p.rstrip("%")) for p in re.findall(r'\b\d+(?:\.\d+)?\%', original_q)]
        currencies = re.findall(r'[\$₹€£]|USD|INR|EUR|GBP', original_q, re.IGNORECASE)

        # 4. Detect Question Type
        q_type = "what"
        for pat, t in self.QUESTION_TYPE_PATTERNS:
            if pat.search(lower_q):
                q_type = t
                break

        # 5. Detect Action and Qualifiers
        detected_action = None
        for pat, act in self.ACTION_PATTERNS:
            m = pat.search(lower_q)
            if m:
                detected_action = act(m) if callable(act) else act
                break

        qualifiers = [w for w in re.findall(r'\b\w+\b', lower_q) if w in self.QUALIFIER_KEYWORDS]

        # 6. Detect Comparison
        comp_targets = []
        is_comparison = False
        comp_match = re.search(r'\b(?:compare|difference\s+between|versus|vs\.?)\s+([a-zA-Z0-9_\s]+?)\s+(?:and|to|with)\s+([a-zA-Z0-9_\s]+)', lower_q)
        if comp_match:
            is_comparison = True
            comp_targets = [comp_match.group(1).strip(), comp_match.group(2).strip()]
        elif any(w in lower_q for w in ["compare", " vs ", "versus", "difference between"]):
            is_comparison = True

        # 7. Detect Calculation
        calc_words = ["calculate", "total", "sum", "discount", "gst", "tax", "vat", "net price", "final price", "average"]
        is_calculation = any(w in lower_q for w in calc_words) and (len(numbers) > 0 or len(percentages) > 0 or "gst" in lower_q or "discount" in lower_q)

        # 8. Detect Temporal Context
        temporal_words = ["days", "months", "years", "duration", "validity", "expires", "after", "before", "deadline", "period"]
        is_temporal = any(w in lower_q for w in temporal_words) and (any(c.isdigit() for c in lower_q) or "after" in lower_q or "before" in lower_q)
        temporal_ctx = None
        if "after" in lower_q:
            temporal_ctx = "after"
        elif "before" in lower_q:
            temporal_ctx = "before"
        elif "during" in lower_q:
            temporal_ctx = "during"

        # 9. Detect Table Lookup
        table_words = ["table", "column", "row", "price", "credit period", "specification", "sku", "cost", "fee"]
        is_table = (analyzed.profile == QueryProfile.TABLE_LOOKUP) or any(w in lower_q for w in table_words)

        # 10. Extract Entities & Subject
        entities: List[str] = []

        # Check for direct object of actions or prepositions (e.g. "after creating a [entity]", "after saving an [entity]", "purpose of [entity]")
        m_act_obj = re.search(r'\b(?:[a-z]{3,}ing|[a-z]{3,}tion|about|of|for|in|on|with|to|create|configure|setup|save|submit|apply|delete|manage)\s+(?:a|an|the)?\s*([a-zA-Z0-9_\-]{2,30})', lower_q)
        if m_act_obj:
            act_cand = m_act_obj.group(1).strip()
            if act_cand.lower() not in STOP_WORDS and act_cand.lower() not in self.QUALIFIER_KEYWORDS and act_cand.lower() not in self.ATTRIBUTE_KEYWORDS and act_cand.lower() not in self.COMMON_ACTION_VERBS:
                entities.append(act_cand.title())

        # Proper noun phrases
        prop_matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', original_q)
        for p in prop_matches:
            p_clean = p.strip()
            if (
                len(p_clean) >= 3
                and p_clean.lower() not in STOP_WORDS
                and p_clean.lower() not in self.ATTRIBUTE_KEYWORDS
                and p_clean.lower() not in self.QUALIFIER_KEYWORDS
                and p_clean.lower() not in self.COMMON_ACTION_VERBS
                and p_clean not in entities
            ):
                entities.append(p_clean)

        if analyzed.primary_entity:
            pe_clean = analyzed.primary_entity.strip()
            if (
                pe_clean.lower() not in self.ATTRIBUTE_KEYWORDS
                and pe_clean.lower() not in STOP_WORDS
                and pe_clean.lower() not in self.QUALIFIER_KEYWORDS
                and pe_clean.lower() not in self.COMMON_ACTION_VERBS
                and pe_clean not in entities
                and not any(pe_clean.lower() in e.lower() for e in entities)
            ):
                entities.append(pe_clean)

        # If context entity was supplied (e.g. from session state)
        if context_entity and context_entity not in entities:
            entities.append(context_entity)

        # Primary subject determination (guarded against action verbs)
        candidate_subj = entities[0] if entities else (analyzed.primary_entity or None)
        subject = candidate_subj if (candidate_subj and candidate_subj.lower() not in self.COMMON_ACTION_VERBS) else (entities[0] if entities else None)

        # 11. Extract Target Attributes
        attributes = list(analyzed.target_attributes or [])
        for w in tokens:
            w_sing = singularize(w)
            if (w in self.ATTRIBUTE_KEYWORDS or w_sing in self.ATTRIBUTE_KEYWORDS) and w not in attributes:
                attributes.append(w_sing)

        # 12. Relationships extraction
        relationships = []
        if "after" in lower_q or "next" in lower_q:
            relationships.append("next_step")
        if "before" in lower_q or "prerequisite" in lower_q:
            relationships.append("prerequisite")
        if any(w in lower_q for w in ["stage", "stages", "step", "steps"]):
            relationships.append("has_stage")
        if any(w in lower_q for w in ["used for", "purpose", "why"]):
            relationships.append("used_for")
        if any(w in lower_q for w in ["configure", "configuration", "setup"]):
            relationships.append("configured_by")

        # 13. Detect Intent Category
        if is_calculation:
            intent = "calculation"
        elif is_comparison:
            intent = "comparison"
        elif is_table:
            intent = "table_value"
        elif detected_action == "after_creation" or ("after" in lower_q and any(w in lower_q for w in ["create", "creating", "creation", "setup", "configure"])):
            intent = "workflow_explanation"
        elif analyzed.profile == QueryProfile.PROCEDURAL or "how" in q_type:
            intent = "procedure"
        elif any(w in lower_q for w in ["what is", "what are", "define", "meaning", "definition"]):
            intent = "definition"
        elif any(w in lower_q for w in ["explain", "overview", "describe", "details"]):
            intent = "explanation"
        elif relationships:
            intent = "relationship"
        elif is_temporal:
            intent = "temporal_information"
        elif entities and attributes:
            intent = "entity_information"
        else:
            intent = "general"

        # 14. Question Decomposition for compound queries
        sub_queries = self.decomposer.decompose(original_q)

        # 15. Dynamic Query Expansion via Corpus Vocabulary
        expanded_terms = []
        if vocab:
            for t in tokens:
                exp = vocab.expand_acronym(t)
                if exp:
                    expanded_terms.append(exp)
                morphs = vocab.get_morphological_variants(t)
                expanded_terms.extend(morphs[:2])

        # 16. Dynamic Retrieval Strategy Routing & Weights
        branches = ["exact", "entity", "bm25", "fuzzy"]
        branch_weights: Dict[str, float] = {
            "vector": 0.50,
            "bm25": 0.25,
            "exact": 0.15,
            "entity": 0.10,
            "table": 0.0,
            "fuzzy": 0.05,
            "parent_child": 0.10,
        }

        # Embedding bypass rule: Only exact single-token codes or standalone pure math bypass embedding
        is_exact_code = bool(re.search(r'^[A-Z0-9_\-]{4,}$', original_q.strip()))
        can_bypass_emb = is_exact_code or (
            is_calculation and len(numbers) >= 2 and not any(w in lower_q for w in ["what", "how", "why", "document", "plan", "policy", "pipeline"])
        )

        if not can_bypass_emb:
            branches.append("vector")

        if is_table:
            branches.append("table")
            branch_weights["table"] = 0.35
            branch_weights["exact"] = 0.25
            branch_weights["bm25"] = 0.20
            branch_weights["vector"] = 0.20

        if intent in ("workflow_explanation", "procedure"):
            branches.append("parent_child")
            branch_weights["vector"] = 0.40
            branch_weights["bm25"] = 0.25
            branch_weights["entity"] = 0.20
            branch_weights["parent_child"] = 0.30

        if intent == "definition":
            branch_weights["vector"] = 0.55
            branch_weights["bm25"] = 0.25
            branch_weights["entity"] = 0.20

        if is_comparison:
            branches.append("table")
            branch_weights["entity"] = 0.35
            branch_weights["vector"] = 0.35
            branch_weights["exact"] = 0.15
            branch_weights["table"] = 0.15

        retrieval_strategy = {
            "active_branches": branches,
            "weights": branch_weights,
            "can_bypass_embedding": can_bypass_emb,
            "require_parent_child": "parent_child" in branches
        }

        # Reference tracking
        conv_refs = []
        for ref_word in ["it", "this", "that", "they", "them", "the above"]:
            if re.search(r'\b' + ref_word + r'\b', lower_q):
                conv_refs.append(ref_word)

        return QueryPlan(
            normalized_query=normalized_q,
            original_query=original_q,
            language=language,
            intent=intent,
            subject=subject,
            entities=entities,
            action=detected_action,
            qualifiers=qualifiers,
            attributes=attributes,
            relationships=relationships,
            temporal_context=temporal_ctx,
            comparison_targets=comp_targets,
            question_type=q_type,
            document_scope=None,
            conversation_references=conv_refs,
            sub_queries=sub_queries,
            expanded_terms=list(set(expanded_terms)),
            retrieval_strategy=retrieval_strategy,
            # Backward-compatible fields:
            raw_query=original_q,
            clean_query=original_q,
            profile=analyzed.profile,
            keywords=[w for w in tokens if len(w) >= 3 and w not in STOP_WORDS],
            numbers=numbers,
            currencies=currencies,
            percentages=percentages,
            target_attributes=attributes,
            can_bypass_embedding=can_bypass_emb,
            requires_calculation=is_calculation,
            requires_comparison=is_comparison,
            requires_temporal=is_temporal,
            requires_table_lookup=is_table,
            active_branches=branches
        )
