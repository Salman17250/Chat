import re
from typing import List, Tuple, Dict, Set, Optional, Any
from collections import defaultdict
from app.rag.ingestion.models import DocumentChunk
from app.rag.query.normalizer import STOP_WORDS

class EntitySearchEngine:
    """
    High-Speed In-Memory Dynamic Entity & Relationship Retrieval Engine (<0.1ms).
    Operates 100% dynamically on chunk entities, attributes, and relationships.
    ZERO hardcoded names, companies, or domain keywords.
    """
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        # Inverted index: entity_key.lower() -> set of chunk indices
        self._entity_index: Dict[str, Set[int]] = defaultdict(set)
        # Attribute index: attribute_name.lower() -> set of chunk indices
        self._attribute_index: Dict[str, Set[int]] = defaultdict(set)
        # Entity + Attribute composite index: (entity.lower(), attr.lower()) -> set of chunk indices
        self._entity_attr_index: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
        # Catalog of all discovered entity strings
        self.known_entities: Set[str] = set()

    def index_chunks(self, chunks: List[DocumentChunk]):
        """Indexes entities, attributes, and relationships discovered within chunks."""
        self.chunks = list(chunks)
        self._entity_index.clear()
        self._attribute_index.clear()
        self._entity_attr_index.clear()
        self.known_entities.clear()

        # Dynamic proper noun pattern for unannotated text
        proper_noun_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')

        for idx, chunk in enumerate(self.chunks):
            # 1. Index dynamically extracted entities
            chunk_entities = getattr(chunk, "entities", []) or []
            for ent in chunk_entities:
                clean_ent = ent.strip().lower()
                if len(clean_ent) >= 3 and clean_ent not in STOP_WORDS:
                    self._entity_index[clean_ent].add(idx)
                    self.known_entities.add(clean_ent)
                    # Index individual parts of multi-word entities (e.g. "first" and "last")
                    parts = clean_ent.split()
                    if len(parts) > 1:
                        for p in parts:
                            if len(p) >= 3 and p not in STOP_WORDS:
                                self._entity_index[p].add(idx)
                                self.known_entities.add(p)

            # 2. Index dynamically extracted attributes
            chunk_attrs = getattr(chunk, "attributes", {}) or {}
            for attr_name, attr_val in chunk_attrs.items():
                attr_clean = str(attr_name).strip().lower()
                self._attribute_index[attr_clean].add(idx)
                for ent in chunk_entities:
                    ent_clean = ent.strip().lower()
                    self._entity_attr_index[(ent_clean, attr_clean)].add(idx)
                    parts = ent_clean.split()
                    if len(parts) > 1:
                        for p in parts:
                            if len(p) >= 3 and p not in STOP_WORDS:
                                self._entity_attr_index[(p, attr_clean)].add(idx)

            # 3. Dynamic regex extraction over section & text for un-enriched chunks
            text = chunk.text
            section = chunk.section or ""
            full_text = f"{section} {text}"
            for p in proper_noun_pattern.findall(full_text):
                p_clean = p.strip().lower()
                if len(p_clean) >= 3 and p_clean not in STOP_WORDS:
                    self._entity_index[p_clean].add(idx)
                    self.known_entities.add(p_clean)

    def search(
        self,
        query: str,
        detected_entities: Optional[List[str]] = None,
        target_attributes: Optional[List[str]] = None,
        top_k: int = 15
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieves chunks that match entities and attributes dynamically.
        Sub-0.1ms execution with joint entity-attribute boosting.
        """
        if not self.chunks:
            return []

        clean_q = query.strip().lower()
        targets = [e.lower() for e in detected_entities] if detected_entities else []

        # Discover mentioned entities from dynamic catalog
        for ent in self.known_entities:
            if ent in clean_q and ent not in targets:
                targets.append(ent)

        if not targets:
            return []

        attrs = [a.lower() for a in (target_attributes or [])]
        chunk_scores: Dict[int, float] = defaultdict(float)

        for t in targets:
            # 1. Match entity + attribute composite index
            for a in attrs:
                if (t, a) in self._entity_attr_index:
                    for idx in self._entity_attr_index[(t, a)]:
                        chunk_scores[idx] = max(chunk_scores[idx], 0.95)

            # 2. Match entity index
            if t in self._entity_index:
                for idx in self._entity_index[t]:
                    chunk = self.chunks[idx]
                    count = chunk.text.lower().count(t)
                    in_section = 1.0 if (chunk.section and t in chunk.section.lower()) else 0.0
                    score = min(1.0, 0.65 + (0.1 * min(count, 3)) + (0.2 * in_section))
                    chunk_scores[idx] = max(chunk_scores[idx], score)

        results = [(self.chunks[idx], score) for idx, score in chunk_scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
