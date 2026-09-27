from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from app.rag.ingestion.models import DocumentChunk
from app.rag.ranking.adaptive_scorer import ScoredChunk

@dataclass
class EvidenceNode:
    node_id: str
    node_type: str  # "claim", "chunk", "entity", "document"
    text: str
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    section: Optional[str] = None
    page: Optional[int] = None
    score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "text": self.text,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "section": self.section,
            "page": self.page,
            "score": round(self.score, 4)
        }

@dataclass
class EvidenceEdge:
    source_id: str
    target_id: str
    relation: str  # "supported_by", "contains_entity", "sub_part_of"

@dataclass
class ProvenanceRecord:
    claim_text: str
    chunk_id: str
    document_name: str
    document_id: str
    section: str
    page: int
    confidence_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim_text,
            "chunk_id": self.chunk_id,
            "document": self.document_name,
            "document_id": self.document_id,
            "section": self.section,
            "page": self.page,
            "score": round(self.confidence_score, 4)
        }

class EvidenceGraphBuilder:
    """
    Evidence Graph & Provenance Construction Engine.
    Maps every answer claim back to its exact grounded chunk node, document, section, and page.
    Guarantees full explainability and non-hallucinated lineage.
    """

    @classmethod
    def build_evidence_graph(
        cls,
        claims: List[Tuple[str, DocumentChunk, float]],
    ) -> Tuple[List[EvidenceNode], List[EvidenceEdge], List[ProvenanceRecord]]:
        """
        Builds an evidence graph connecting claim nodes to chunk and document nodes.
        """
        nodes: List[EvidenceNode] = []
        edges: List[EvidenceEdge] = []
        records: List[ProvenanceRecord] = []
        seen_nodes: Set[str] = set()

        for idx, (claim_text, chunk, score) in enumerate(claims):
            claim_node_id = f"claim_{idx}"
            chunk_node_id = f"chunk_{chunk.chunk_id}"
            doc_node_id = f"doc_{chunk.document_id}"

            # 1. Claim Node
            nodes.append(EvidenceNode(
                node_id=claim_node_id,
                node_type="claim",
                text=claim_text,
                score=score
            ))

            # 2. Chunk Node
            if chunk_node_id not in seen_nodes:
                seen_nodes.add(chunk_node_id)
                nodes.append(EvidenceNode(
                    node_id=chunk_node_id,
                    node_type="chunk",
                    text=chunk.text[:200],
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    section=chunk.section,
                    page=chunk.page,
                    score=score
                ))

            # 3. Document Node
            if doc_node_id not in seen_nodes:
                seen_nodes.add(doc_node_id)
                nodes.append(EvidenceNode(
                    node_id=doc_node_id,
                    node_type="document",
                    text=chunk.document_name,
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    score=1.0
                ))

            # Edges: Claim -> Supported By -> Chunk -> Part Of -> Document
            edges.append(EvidenceEdge(source_id=claim_node_id, target_id=chunk_node_id, relation="supported_by"))
            edges.append(EvidenceEdge(source_id=chunk_node_id, target_id=doc_node_id, relation="part_of"))

            # Provenance record
            records.append(ProvenanceRecord(
                claim_text=claim_text,
                chunk_id=chunk.chunk_id,
                document_name=chunk.document_name,
                document_id=chunk.document_id,
                section=chunk.section,
                page=chunk.page,
                confidence_score=score
            ))

        return nodes, edges, records
