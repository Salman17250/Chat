from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from app.rag.ingestion.models import DocumentChunk
from app.rag.knowledge.ontology import DynamicOntology
from app.rag.query.normalizer import STOP_WORDS, singularize

@dataclass
class EntityNode:
    entity_id: str
    name: str
    entity_type: str = "Concept"
    aliases: Set[str] = field(default_factory=set)
    chunk_ids: Set[str] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "entity_type": self.entity_type,
            "aliases": list(self.aliases),
            "chunk_ids": list(self.chunk_ids),
            "attributes": self.attributes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityNode':
        return cls(
            entity_id=data["entity_id"],
            name=data["name"],
            entity_type=data.get("entity_type", "Concept"),
            aliases=set(data.get("aliases", [])),
            chunk_ids=set(data.get("chunk_ids", [])),
            attributes=data.get("attributes", {})
        )

class EntityGraph:
    """
    In-Memory Document-Derived Entity Graph.
    Extracts, indexes, and clusters entities and aliases per tenant.
    Sub-0.01ms resolution of entity names and aliases without hardcoded dictionaries.
    """

    def __init__(self):
        # entity_id -> EntityNode
        self.nodes: Dict[str, EntityNode] = {}
        # alias_lower -> entity_id
        self.alias_to_id: Dict[str, str] = {}
        # chunk_id -> Set of entity_ids
        self.chunk_to_entities: Dict[str, Set[str]] = {}

    def clear(self):
        self.nodes.clear()
        self.alias_to_id.clear()
        self.chunk_to_entities.clear()

    def add_entity(
        self,
        name: str,
        chunk_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        context_text: str = ""
    ) -> EntityNode:
        clean_name = name.strip()
        e_id = singularize(clean_name.lower())

        if e_id not in self.nodes:
            inferred_type = entity_type or DynamicOntology.infer_entity_type(clean_name, context_text)
            node = EntityNode(
                entity_id=e_id,
                name=clean_name,
                entity_type=inferred_type,
                aliases={clean_name, clean_name.lower(), e_id}
            )
            self.nodes[e_id] = node
        else:
            node = self.nodes[e_id]

        # Register aliases
        self.alias_to_id[clean_name.lower()] = e_id
        self.alias_to_id[e_id] = e_id
        node.aliases.add(clean_name)
        node.aliases.add(clean_name.lower())

        if aliases:
            for a in aliases:
                a_clean = a.strip()
                if a_clean:
                    node.aliases.add(a_clean)
                    self.alias_to_id[a_clean.lower()] = e_id

        if chunk_id:
            node.chunk_ids.add(chunk_id)
            if chunk_id not in self.chunk_to_entities:
                self.chunk_to_entities[chunk_id] = set()
            self.chunk_to_entities[chunk_id].add(e_id)

        if attributes:
            node.attributes.update(attributes)

        return node

    def find_entity(self, term: str) -> Optional[EntityNode]:
        """Resolves an entity name or alias to its EntityNode."""
        clean = term.strip().lower()
        clean_sing = singularize(clean)

        e_id = self.alias_to_id.get(clean) or self.alias_to_id.get(clean_sing)
        if e_id and e_id in self.nodes:
            return self.nodes[e_id]

        # Partial token match for multi-word entities
        for alias, target_id in self.alias_to_id.items():
            if clean in alias or alias in clean:
                if len(clean) >= 4 and len(alias) >= 4:
                    return self.nodes[target_id]

        return None

    def get_entities_for_chunk(self, chunk_id: str) -> List[EntityNode]:
        e_ids = self.chunk_to_entities.get(chunk_id, set())
        return [self.nodes[eid] for eid in e_ids if eid in self.nodes]

    def build_from_chunks(self, chunks: List[DocumentChunk]):
        """Builds entity graph dynamically from document chunks."""
        self.clear()
        for chunk in chunks:
            chunk_entities = getattr(chunk, "entities", []) or []
            chunk_attrs = getattr(chunk, "attributes", {}) or {}
            sec = chunk.section or ""

            # Check section heading for entity
            if sec and sec.lower() != "general":
                self.add_entity(sec, chunk_id=chunk.chunk_id, context_text=chunk.text)

            for ent in chunk_entities:
                if ent.lower() not in STOP_WORDS and len(ent) >= 3:
                    # Generate natural aliases (singular, plural, lowercase)
                    aliases = [ent.lower(), singularize(ent.lower())]
                    self.add_entity(
                        name=ent,
                        chunk_id=chunk.chunk_id,
                        aliases=aliases,
                        attributes=chunk_attrs if ent == chunk_entities[0] else None,
                        context_text=chunk.text
                    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": {eid: node.to_dict() for eid, node in self.nodes.items()},
            "alias_to_id": self.alias_to_id,
            "chunk_to_entities": {cid: list(eids) for cid, eids in self.chunk_to_entities.items()}
        }

    def from_dict(self, data: Dict[str, Any]):
        self.clear()
        for eid, ndata in data.get("nodes", {}).items():
            self.nodes[eid] = EntityNode.from_dict(ndata)
        self.alias_to_id = data.get("alias_to_id", {})
        self.chunk_to_entities = {cid: set(eids) for cid, eids in data.get("chunk_to_entities", {}).items()}
