import re
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from app.rag.query.normalizer import STOP_WORDS

GENERIC_SECTION_WORDS = {
    "introduction", "overview", "general", "summary", "conclusion",
    "details", "information", "section", "chapter", "table", "notes",
    "about", "about me", "staff", "employee", "profile", "bio", "resume"
}

@dataclass
class ExtractedRelationship:
    subject: str
    predicate: str
    object_value: Any

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object_value
        }

class DynamicEntityExtractor:
    """
    Generalized, Domain-Agnostic Entity & Attribute Extraction Engine.
    Operates on ANY document (resumes, policies, manuals, product sheets, tables).
    ZERO hardcoded names, companies, prices, or document structures.
    """

    # Attribute extraction regexes
    ATTR_PATTERNS = [
        # Age
        (
            "age",
            re.compile(r'\b(?:am|is|aged)\s+(\d{1,3})\s*(?:years?\s+old|yr|yo)?\b', re.IGNORECASE),
            lambda m: m.group(1).strip()
        ),
        (
            "age",
            re.compile(r'\b(?:age|current\s+age)[:\s]+(\d{1,3})\b', re.IGNORECASE),
            lambda m: m.group(1).strip()
        ),
        # Profession / Role / Occupation
        (
            "profession",
            re.compile(r'\b(?:is|am|works\s+as)\s+(?:a|an)\s+([a-zA-Z0-9_\-\s]{3,35}?)(?:\.|\band\b|,|\n|$)', re.IGNORECASE),
            lambda m: m.group(1).strip()
        ),
        (
            "profession",
            re.compile(r'\b(?:profession|designation|title|job\s+title|role|occupation)[:\s]+([^\n,;]{2,40})', re.IGNORECASE),
            lambda m: m.group(1).strip()
        ),
        # Education / Degree
        (
            "education",
            re.compile(r'\b(?:completed|studied|holds|graduated\s+with)\s+(?:his|her|their|a|an)?\s*([A-Z]{2,6}|Bachelor[^\n,;]{0,30}|Master[^\n,;]{0,30}|Diploma[^\n,;]{0,30}|Ph\.?D\.?[^\n,;]{0,30})', re.IGNORECASE),
            lambda m: m.group(1).strip()
        ),
        (
            "education",
            re.compile(r'\b(?:education|degree|qualification)[:\s]+([^\n,;]{2,40})', re.IGNORECASE),
            lambda m: m.group(1).strip()
        ),
        # Skills / Technologies
        (
            "skills",
            re.compile(r'\b(?:works\s+with|technologies\s+include|tech\s+stack|skilled\s+in|skills\s+in)[:\s]+([^\n;]{2,100}?)(?:\.\s+[A-Z]|\.\s*$|\n|$)', re.IGNORECASE),
            lambda m: [s.strip() for s in re.split(r'[,/|]|(?:\band\b)', m.group(1)) if len(s.strip()) >= 2 and s.strip().lower() not in STOP_WORDS]
        ),
        (
            "skills",
            re.compile(r'\b(?:skills|technologies|tools)[:\s]+([^\n;]{2,100}?)(?:\.\s+[A-Z]|\.\s*$|\n|$)', re.IGNORECASE),
            lambda m: [s.strip() for s in re.split(r'[,/|]|(?:\band\b)', m.group(1)) if len(s.strip()) >= 2 and s.strip().lower() not in STOP_WORDS]
        ),
        # Price / Cost
        (
            "price",
            re.compile(r'\b(?:price|cost|fee|subscription|charge|pricing)[:\s]+([₹\$€£]?\s*\d+(?:,\d+)*(?:\.\d+)?(?:\s*(?:INR|USD|EUR|per\s+month|monthly|annually|per\s+year))?)', re.IGNORECASE),
            lambda m: m.group(1).strip()
        ),
        # Duration / Period / Validity
        (
            "period",
            re.compile(r'\b(?:within|period\s+of|duration\s+of|validity\s+of|notice\s+period\s+is|deadline\s+is)\s+(\d+\s*(?:days?|months?|years?|weeks?|hours?))\b', re.IGNORECASE),
            lambda m: m.group(1).strip()
        ),
        (
            "period",
            re.compile(r'\b(?:period|duration|validity|notice\s+period|credit\s+period|deadline)[:\s]+(\d+\s*(?:days?|months?|years?|weeks?|hours?))\b', re.IGNORECASE),
            lambda m: m.group(1).strip()
        ),
        # Salary / Compensation
        (
            "salary",
            re.compile(r'\b(?:salary|compensation|package|ctc)[:\s]+([₹\$€£]?\s*\d+(?:,\d+)*(?:\.\d+)?(?:\s*(?:LPA|per\s+annum|monthly|INR|USD))?)', re.IGNORECASE),
            lambda m: m.group(1).strip()
        ),
        # Department / Team
        (
            "department",
            re.compile(r'\b(?:department|team|division)[:\s]+([a-zA-Z\s]{2,30})', re.IGNORECASE),
            lambda m: m.group(1).strip()
        ),
        # Location
        (
            "location",
            re.compile(r'\b(?:location|address|based\s+in|located\s+in)[:\s]+([a-zA-Z\s,]{2,40})', re.IGNORECASE),
            lambda m: m.group(1).strip()
        )
    ]

    # Generic Key-Value pattern: e.g. "Status: Active", "Battery: 5000mAh"
    GENERIC_KV_PATTERN = re.compile(r'^([A-Z][a-zA-Z0-9_\s]{1,25}):\s*([^\n\r]{1,80})$', re.MULTILINE)

    @classmethod
    def extract_entities(cls, text: str, section: Optional[str] = None) -> List[str]:
        """
        Dynamically extracts candidate entities from text and section.
        Identifies proper noun sequences, capitalized names, and codes without hardcoded lists.
        """
        entities: List[str] = []
        seen = set()

        def add_entity(e: str):
            clean = e.strip(" .,:;!?'\"()[]{}")
            lower = clean.lower()
            if (
                len(clean) >= 3
                and lower not in STOP_WORDS
                and lower not in GENERIC_SECTION_WORDS
                and lower not in seen
            ):
                seen.add(lower)
                entities.append(clean)

        # 1. Check section title if it represents an entity
        if section and section.strip():
            sec_clean = section.strip()
            # If section contains " - " (e.g. "Profile - Firstname Lastname")
            if " - " in sec_clean:
                parts = sec_clean.split(" - ")
                for p in parts:
                    if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', p.strip()):
                        add_entity(p.strip())
            elif re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', sec_clean):
                add_entity(sec_clean)

        # 2. Extract Capitalized Name/Entity Sequences (2-4 capitalized words, e.g. "Full Name", "Standard Plan")
        for m in re.finditer(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b', text):
            cand = m.group(0).strip()
            words = [w.lower() for w in cand.split()]
            if not all(w in STOP_WORDS for w in words):
                add_entity(cand)

        # 3. Single capitalized name if introduced: "My name is X", "I am X", "Employee: X"
        intro_matches = re.findall(r'\b(?:name\s+is|i\s+am|meet|employee[:\s]+|customer[:\s]+)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text, re.IGNORECASE)
        for cand in intro_matches:
            add_entity(cand)

        # 4. Code-like identifiers: e.g. "INV-20394", "SKU-8891"
        for m in re.finditer(r'\b[A-Z0-9_\-]{3,}\b', text):
            token = m.group(0).strip()
            if any(c.isdigit() for c in token) or "-" in token or "_" in token:
                add_entity(token)

        return entities

    @classmethod
    def extract_attributes(cls, text: str) -> Dict[str, Any]:
        """
        Extracts typed attributes from prose and key-value lines dynamically.
        """
        attributes: Dict[str, Any] = {}

        # 1. Pattern-based attribute extraction
        for attr_name, pattern, parser in cls.ATTR_PATTERNS:
            for match in pattern.finditer(text):
                try:
                    val = parser(match)
                    if val and attr_name not in attributes:
                        attributes[attr_name] = val
                except Exception:
                    pass

        # 2. Generic Key-Value lines
        for m in cls.GENERIC_KV_PATTERN.finditer(text):
            k_raw, v_raw = m.group(1).strip(), m.group(2).strip()
            k_clean = k_raw.lower().replace(" ", "_")
            if k_clean not in STOP_WORDS and len(v_raw) > 0 and k_clean not in attributes:
                attributes[k_clean] = v_raw

        return attributes

    @classmethod
    def extract_relationships(
        cls,
        text: str,
        entities: List[str],
        attributes: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Forms knowledge graph triples (Subject, Predicate, Object) by associating
        extracted attributes with the primary entity of the chunk.
        """
        relationships: List[Dict[str, Any]] = []
        if not entities or not attributes:
            return relationships

        primary_subject = entities[0]
        for attr_key, attr_val in attributes.items():
            relationships.append({
                "subject": primary_subject,
                "predicate": attr_key,
                "object": attr_val
            })

        return relationships

    @classmethod
    def enrich_chunk(cls, chunk) -> None:
        """
        Enriches a DocumentChunk in-place with entities, attributes, and relationships.
        """
        text = chunk.text
        sec = chunk.section
        chunk_entities = cls.extract_entities(text, section=sec)
        chunk_attrs = cls.extract_attributes(text)
        
        # Merge typed_attributes from table rows if present
        if getattr(chunk, "typed_attributes", None):
            for k, v in chunk.typed_attributes.items():
                k_clean = str(k).lower().replace(" ", "_")
                if k_clean not in chunk_attrs:
                    chunk_attrs[k_clean] = v

        chunk_rels = cls.extract_relationships(text, chunk_entities, chunk_attrs)

        chunk.entities = chunk_entities
        chunk.attributes = chunk_attrs
        chunk.relationships = chunk_rels
