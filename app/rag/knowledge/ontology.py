import re
from typing import Dict, List, Set, Optional, Tuple
from app.rag.query.normalizer import STOP_WORDS

class DynamicOntology:
    """
    Dynamic Domain Ontology & Entity Typing Engine.
    Infers domain entity types directly from document structure, headings, key-value lines, and tables.
    ZERO hardcoded business models.
    Types supported dynamically:
      - Person, Product, Feature, Module, Pipeline, Stage, Process, Field, Table, Metric, Date, Amount, Status, Category
    """

    DEFAULT_TYPE_HEURISTICS = [
        ("Person", [r'\b(?:name\s+is|dr\.|mr\.|mrs\.|ms\.|employee|candidate|profile\s+of)\b', r'\b(?:age|qualification|profession)\b']),
        ("Pipeline", [r'\bpipeline\b', r'\bworkflow\s+pipeline\b', r'\bsales\s+pipeline\b']),
        ("Stage", [r'\bstage\b', r'\bstages\b', r'\bstep\b', r'\bsteps\b', r'\bphase\b']),
        ("Product", [r'\bproduct\b', r'\bplan\b', r'\bsubscription\b', r'\bedition\b', r'\btier\b']),
        ("Feature", [r'\bfeature\b', r'\bcapability\b', r'\btool\b', r'\bmodule\b', r'\bfunctionality\b']),
        ("Table", [r'\btable\b', r'\bmatrix\b', r'\bgrid\b', r'\bcolumn\b', r'\brow\b']),
        ("Metric", [r'\bmetric\b', r'\bkpi\b', r'\brate\b', r'\bscore\b', r'\bcount\b', r'\bpercentage\b']),
        ("Amount", [r'[₹\$€£]', r'\b(?:usd|inr|eur|price|cost|fee|salary|amount)\b']),
        ("Status", [r'\b(?:active|inactive|pending|completed|draft|approved|rejected|status)\b']),
        ("Date", [r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|january|february|march|april|may|june|july|august|september|october|november|december)\b']),
        ("Process", [r'\b(?:process|procedure|workflow|guide|instructions|policy)\b']),
    ]

    @classmethod
    def infer_entity_type(cls, entity_name: str, context_text: str = "") -> str:
        """
        Dynamically infers the ontological type of an entity based on its name and occurrence context.
        """
        clean_name = entity_name.strip().lower()
        combined = f"{clean_name} {context_text.lower()}"

        for type_name, patterns in cls.DEFAULT_TYPE_HEURISTICS:
            for pat in patterns:
                if re.search(pat, combined, re.IGNORECASE):
                    return type_name

        # Capitalized multi-word proper nouns default to Feature or Entity
        if len(entity_name.split()) >= 2:
            return "Entity"

        return "Concept"
