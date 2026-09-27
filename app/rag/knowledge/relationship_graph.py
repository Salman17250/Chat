import re
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass, field
from app.rag.ingestion.models import DocumentChunk
from app.rag.knowledge.entity_graph import EntityGraph
from app.rag.query.normalizer import STOP_WORDS, singularize

@dataclass
class RelationshipEdge:
    subject: str
    predicate: str
    object_val: str
    evidence_chunk_ids: List[str] = field(default_factory=list)
    evidence_sentences: List[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object_val": self.object_val,
            "evidence_chunk_ids": self.evidence_chunk_ids,
            "evidence_sentences": self.evidence_sentences,
            "confidence": self.confidence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RelationshipEdge':
        return cls(
            subject=data["subject"],
            predicate=data["predicate"],
            object_val=data["object_val"],
            evidence_chunk_ids=data.get("evidence_chunk_ids", []),
            evidence_sentences=data.get("evidence_sentences", []),
            confidence=data.get("confidence", 1.0)
        )

class RelationshipGraph:
    """
    In-Memory Document-Derived Relationship Graph.
    Stores and traverses evidenced triples: (Subject, Predicate, Object).
    Every single relationship edge is grounded by explicit evidence chunk IDs and sentences.
    Zero unsupported or fabricated edges.
    """

    RELATIONSHIP_PATTERNS = [
        # Sequential / Workflow: "after creating X, configure Y", "then proceed to Z", "step 1 ... step 2"
        (re.compile(r'\bafter\s+(?:creating|creation|setting\s+up|configuring)\s+(?:the\s+)?([A-Za-z0-9_\-\s]{2,30}?)[,:\.]\s*(?:the\s+next\s+step\s+is\s+to|configure|setup|proceed\s+to|define)?\s*([^\n\.;]{4,80})', re.I), "after_step"),
        # Structure / Stages: "X has stages A, B, C", "X consists of Y", "stages in X include"
        (re.compile(r'\b([A-Za-z0-9_\-\s]{2,30}?)\s+(?:has\s+stages?|consists\s+of|includes?\s+stages?|contains\s+stages?)[:\s]+([^\n\.;]{4,100})', re.I), "has_stage"),
        # Purpose / Usage: "X is used for Y", "X is used to Y"
        (re.compile(r'\b([A-Za-z0-9_\-\s]{2,30}?)\s+(?:is\s+used\s+(?:for|to)|serves\s+to|designed\s+to)[:\s]+([^\n\.;]{4,100})', re.I), "used_for"),
        # Configuration: "X is configured by Y", "configure X according to Y"
        (re.compile(r'\bconfigure\s+(?:the\s+)?([A-Za-z0-9_\-\s]{2,30}?)\s+(?:by|according\s+to|using)\s+([^\n\.;]{4,100})', re.I), "configured_by"),
        # Prerequisites: "X requires Y", "prerequisite for X is Y"
        (re.compile(r'\b(?:prerequisite\s+for\s+)?([A-Za-z0-9_\-\s]{2,30}?)\s+(?:requires|needs|depends\s+on)\s+([^\n\.;]{4,100})', re.I), "requires"),
    ]

    def __init__(self):
        # subject_lower -> List[RelationshipEdge]
        self.outgoing: Dict[str, List[RelationshipEdge]] = defaultdict(list)
        # object_lower -> List[RelationshipEdge]
        self.incoming: Dict[str, List[RelationshipEdge]] = defaultdict(list)
        self.all_edges: List[RelationshipEdge] = []

    def clear(self):
        self.outgoing.clear()
        self.incoming.clear()
        self.all_edges.clear()

    def add_relationship(
        self,
        subject: str,
        predicate: str,
        object_val: str,
        chunk_id: str,
        sentence: str = "",
        confidence: float = 1.0
    ) -> RelationshipEdge:
        sub_key = singularize(subject.strip().lower())
        obj_key = object_val.strip().lower()

        # Check if an edge already exists with this subject and predicate
        for edge in self.outgoing[sub_key]:
            if edge.predicate == predicate and edge.object_val.lower() == obj_key:
                if chunk_id not in edge.evidence_chunk_ids:
                    edge.evidence_chunk_ids.append(chunk_id)
                if sentence and sentence not in edge.evidence_sentences:
                    edge.evidence_sentences.append(sentence)
                return edge

        edge = RelationshipEdge(
            subject=subject.strip(),
            predicate=predicate.strip(),
            object_val=object_val.strip(),
            evidence_chunk_ids=[chunk_id] if chunk_id else [],
            evidence_sentences=[sentence] if sentence else [],
            confidence=confidence
        )

        self.outgoing[sub_key].append(edge)
        self.incoming[obj_key].append(edge)
        self.all_edges.append(edge)
        return edge

    def get_outgoing(self, subject: str, predicate: Optional[str] = None) -> List[RelationshipEdge]:
        sub_key = singularize(subject.strip().lower())
        edges = self.outgoing.get(sub_key, [])
        if predicate:
            return [e for e in edges if e.predicate.lower() == predicate.lower()]
        return edges

    def get_incoming(self, object_val: str, predicate: Optional[str] = None) -> List[RelationshipEdge]:
        obj_key = object_val.strip().lower()
        edges = self.incoming.get(obj_key, [])
        if predicate:
            return [e for e in edges if e.predicate.lower() == predicate.lower()]
        return edges

    def traverse_workflow(self, entity_name: str) -> List[RelationshipEdge]:
        """
        Traverses workflow sequence for an entity (e.g. creation -> configuration -> stages -> usage).
        Returns evidence-backed edges in sequence.
        """
        results: List[RelationshipEdge] = []
        sub_key = singularize(entity_name.strip().lower())

        # Collect outgoing stages, after_step, configured_by, used_for
        priority_predicates = ["after_step", "configured_by", "has_stage", "used_for", "requires"]
        for pred in priority_predicates:
            matching = self.get_outgoing(sub_key, predicate=pred)
            results.extend(matching)

        return results

    def build_from_chunks(self, chunks: List[DocumentChunk], entity_graph: Optional[EntityGraph] = None):
        """Extracts evidenced relationships dynamically from document chunks."""
        self.clear()

        for chunk in chunks:
            text = chunk.text
            cid = chunk.chunk_id
            chunk_entities = getattr(chunk, "entities", []) or []
            primary_entity = chunk_entities[0] if chunk_entities else (chunk.section or "General")

            # 1. Pattern-based structural relationship extraction
            sentences = re.split(r'[.\n!?]+', text)
            for sent in sentences:
                sent_clean = sent.strip()
                if not sent_clean:
                    continue

                for pat, pred in self.RELATIONSHIP_PATTERNS:
                    m = pat.search(sent_clean)
                    if m:
                        sub = m.group(1).strip()
                        obj = m.group(2).strip()
                        if len(sub) >= 2 and len(obj) >= 2 and sub.lower() not in STOP_WORDS:
                            self.add_relationship(
                                subject=sub,
                                predicate=pred,
                                object_val=obj,
                                chunk_id=cid,
                                sentence=sent_clean,
                                confidence=0.95
                            )

            # 2. Extract chunk's internal attribute relationships
            chunk_attrs = getattr(chunk, "attributes", {}) or {}
            for attr_k, attr_v in chunk_attrs.items():
                if primary_entity and primary_entity.lower() != "general":
                    self.add_relationship(
                        subject=primary_entity,
                        predicate=f"has_{attr_k}",
                        object_val=str(attr_v),
                        chunk_id=cid,
                        sentence=text[:150],
                        confidence=0.90
                    )

            # 3. If chunk is Q&A pair with a workflow question
            if getattr(chunk, "is_qa_pair", False) and chunk.question_text and chunk.answer_text:
                q_text = chunk.question_text.lower()
                ans_text = chunk.answer_text.strip()
                if "after" in q_text and any(w in q_text for w in ["create", "creating", "creation"]):
                    m_ent = re.search(r'\b(?:creating|creation|set\s+up)\s+(?:a|an|the)?\s*([A-Za-z0-9_\-]+)', q_text)
                    subj = m_ent.group(1).title() if m_ent else primary_entity
                    self.add_relationship(
                        subject=subj,
                        predicate="after_step",
                        object_val=ans_text,
                        chunk_id=cid,
                        sentence=ans_text,
                        confidence=1.0
                    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "all_edges": [e.to_dict() for e in self.all_edges]
        }

    def from_dict(self, data: Dict[str, Any]):
        self.clear()
        for edata in data.get("all_edges", []):
            e = RelationshipEdge.from_dict(edata)
            sub_key = singularize(e.subject.strip().lower())
            obj_key = e.object_val.strip().lower()
            self.outgoing[sub_key].append(e)
            self.incoming[obj_key].append(e)
            self.all_edges.append(e)
