import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from app.rag.query.analyzer import QueryProfile, QueryAnalyzer
from app.rag.query.normalizer import STOP_WORDS

@dataclass
class QueryPlan:
    raw_query: str
    clean_query: str
    intent: str
    profile: QueryProfile
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    numbers: List[float] = field(default_factory=list)
    currencies: List[str] = field(default_factory=list)
    percentages: List[float] = field(default_factory=list)
    comparison_targets: List[str] = field(default_factory=list)
    target_attributes: List[str] = field(default_factory=list)
    can_bypass_embedding: bool = False
    requires_calculation: bool = False
    requires_comparison: bool = False
    requires_temporal: bool = False
    requires_table_lookup: bool = False
    active_branches: List[str] = field(default_factory=list)

class QueryPlanner:
    """
    Lightweight Deterministic Query Planner (<0.1ms).
    Classifies queries, extracts structured targets (entities, attributes, math operations, comparisons),
    and produces an optimized retrieval execution plan WITHOUT calling an LLM.
    """
    def __init__(self, analyzer: Optional[QueryAnalyzer] = None):
        self.analyzer = analyzer or QueryAnalyzer()

    def plan(self, query: str) -> QueryPlan:
        clean_q = query.strip()
        lower_q = clean_q.lower()

        # 1. Analyze query via QueryAnalyzer
        analyzed = self.analyzer.analyze(clean_q)

        # 2. Extract numbers, percentages, currencies
        numbers = [float(n.replace(",", "")) for n in re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', clean_q)]
        percentages = [float(p.rstrip("%")) for p in re.findall(r'\b\d+(?:\.\d+)?\%', clean_q)]
        currencies = re.findall(r'[\$₹€£]|USD|INR|EUR|GBP', clean_q, re.IGNORECASE)

        # 3. Detect Comparison targets
        comp_targets = []
        is_comparison = False
        comp_match = re.search(r'\b(?:compare|difference\s+between|versus|vs\.?)\s+([a-zA-Z0-9_\s]+?)\s+(?:and|to|with)\s+([a-zA-Z0-9_\s]+)', lower_q)
        if comp_match:
            is_comparison = True
            comp_targets = [comp_match.group(1).strip(), comp_match.group(2).strip()]
        elif "compare" in lower_q or " vs " in lower_q or "versus" in lower_q:
            is_comparison = True

        # 4. Detect Calculation intent
        calc_words = ["calculate", "total", "sum", "discount", "gst", "tax", "vat", "net price", "final price", "average"]
        is_calculation = any(w in lower_q for w in calc_words) and (len(numbers) > 0 or len(percentages) > 0 or "gst" in lower_q or "discount" in lower_q)

        # 5. Detect Temporal intent
        temp_words = ["days", "months", "years", "duration", "validity", "expires", "after", "before", "deadline"]
        is_temporal = any(w in lower_q for w in temp_words) and any(c.isdigit() for c in lower_q)

        # 6. Detect Table Lookup intent
        table_words = ["table", "column", "row", "price", "credit period", "specification", "sku", "cost", "fee"]
        is_table = (analyzed.profile == QueryProfile.TABLE_LOOKUP) or any(w in lower_q for w in table_words)

        attribute_or_topic_nouns = {
            "technology", "technologies", "skill", "skills", "tool", "tools",
            "age", "education", "degree", "qualification", "profession", "job", "role",
            "price", "cost", "fee", "salary", "compensation", "period", "duration",
            "policy", "rule", "rules", "guideline", "guidelines", "process", "procedure",
            "details", "information", "name", "status", "deadline", "date", "time",
            "table", "row", "column", "list", "total", "difference", "comparison"
        }

        # 7. Extract Entities
        entities = []
        if analyzed.primary_entity:
            pe_clean = analyzed.primary_entity.strip()
            if pe_clean.lower() not in attribute_or_topic_nouns and pe_clean.lower() not in STOP_WORDS:
                entities.append(pe_clean)
        # Proper noun phrases
        prop_matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', clean_q)
        for p in prop_matches:
            p_clean = p.strip()
            if (
                len(p_clean) >= 3
                and p_clean.lower() not in STOP_WORDS
                and p_clean.lower() not in attribute_or_topic_nouns
                and p_clean not in entities
            ):
                entities.append(p_clean)

        # 8. Dynamic Target Attribute Detection
        target_attrs: List[str] = list(analyzed.target_attributes or [])
        attr_heuristics = [
            ("age", [r'\bage\b', r'\bhow\s+old\b', r'\byears?\s+old\b']),
            ("education", [r'\beducation\b', r'\bdegree\b', r'\bstudy\b', r'\bstudied\b', r'\bqualification\b', r'\bcollege\b', r'\buniversity\b']),
            ("skills", [r'\bskills?\b', r'\btechnolog(?:y|ies)\b', r'\btech(?:\s+stack)?\b', r'\btools?\b', r'\bframeworks?\b', r'\bprogramming\b']),
            ("profession", [r'\bprofession\b', r'\brole\b', r'\bjob\b', r'\bdesignation\b', r'\boccupation\b', r'\bwork(?:s)?\s+as\b', r'\bwho\s+is\b']),
            ("price", [r'\bprice\b', r'\bcost\b', r'\bfee\b', r'\bsubscription\b', r'\bhow\s+much\b', r'\bpricing\b']),
            ("period", [r'\bperiod\b', r'\bduration\b', r'\bvalidity\b', r'\bdeadline\b', r'\bhow\s+long\b', r'\bcancel(?:lation)?\b', r'\bnotice\b']),
            ("salary", [r'\bsalary\b', r'\bcompensation\b', r'\bpay\b', r'\bpackage\b', r'\bctc\b']),
            ("department", [r'\bdepartment\b', r'\bteam\b', r'\bdivision\b']),
            ("location", [r'\blocation\b', r'\bwhere\b', r'\baddress\b', r'\bcity\b', r'\bcountry\b', r'\bheadquarters\b'])
        ]
        for attr_key, patterns in attr_heuristics:
            if attr_key not in target_attrs:
                for pat in patterns:
                    if re.search(pat, lower_q):
                        target_attrs.append(attr_key)
                        break

        # 9. Extract clean keywords
        words = re.findall(r'\b\w+\b', lower_q)
        keywords = [w for w in words if len(w) >= 3 and w not in STOP_WORDS]

        # 10. Determine if embedding can be bypassed
        # Semantic search is MANDATORY for all conceptual, natural language, and attribute questions.
        # ONLY pure code lookups (e.g. raw invoice numbers INV-20394) or pure offline math bypass embedding.
        is_exact_code = bool(re.search(r'^[A-Z0-9_\-]{4,}$', clean_q))
        can_bypass = is_exact_code or (is_calculation and len(numbers) >= 2 and not any(w in lower_q for w in ["what", "how", "why", "document", "plan", "policy"]))

        # 11. Select Active Retrieval Branches
        branches = ["exact", "entity", "bm25", "fuzzy"]
        if is_table:
            branches.append("table")
        if not can_bypass:
            branches.append("vector")

        intent = "GENERAL"
        if is_calculation:
            intent = "CALCULATION"
        elif is_comparison:
            intent = "COMPARISON"
        elif is_table:
            intent = "TABLE_LOOKUP"
        elif is_temporal:
            intent = "TEMPORAL"
        elif analyzed.profile == QueryProfile.PROCEDURAL:
            intent = "PROCEDURAL"
        elif analyzed.profile == QueryProfile.EXACT_CODE:
            intent = "EXACT_CODE"

        return QueryPlan(
            raw_query=query,
            clean_query=clean_q,
            intent=intent,
            profile=analyzed.profile,
            entities=entities,
            keywords=keywords,
            numbers=numbers,
            currencies=currencies,
            percentages=percentages,
            comparison_targets=comp_targets,
            target_attributes=target_attrs,
            can_bypass_embedding=can_bypass,
            requires_calculation=is_calculation,
            requires_comparison=is_comparison,
            requires_temporal=is_temporal,
            requires_table_lookup=is_table,
            active_branches=branches
        )
