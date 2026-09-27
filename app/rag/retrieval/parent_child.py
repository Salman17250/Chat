from typing import List, Dict, Set, Optional, Tuple
from app.rag.ingestion.models import DocumentChunk

class ParentChildRetriever:
    """
    Hierarchical Document Structure & Neighbor Context Retrieval Engine.
    Traverses document hierarchy:
      Document -> Section -> Chunks -> Sub-chunks / Table Rows
    Features:
      1. Pulls parent section context and headings
      2. Expands neighboring chunks (previous and next in sequential order)
      3. Expands child chunks (e.g. detailed table rows, bullet sub-items)
      4. Preserves document sequence order without exploding context window.
    """

    @classmethod
    def expand_context_hierarchy(
        cls,
        seed_chunks: List[DocumentChunk],
        chunk_by_id: Dict[str, DocumentChunk],
        max_neighbors_per_seed: int = 1,
        include_parent: bool = True,
        max_total_chunks: int = 10
    ) -> List[DocumentChunk]:
        """
        Expands seed candidate chunks with parent and adjacent siblings strictly within
        the same document and section, preserving document reading order.
        """
        if not seed_chunks:
            return []

        expanded_chunks: List[DocumentChunk] = []
        seen_chunk_ids: Set[str] = set()

        for chunk in seed_chunks:
            if chunk.chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk.chunk_id)
                expanded_chunks.append(chunk)

            # 1. Expand Previous Neighbor in same section
            if max_neighbors_per_seed >= 1 and chunk.prev_chunk_id:
                prev_c = chunk_by_id.get(chunk.prev_chunk_id)
                if (
                    prev_c
                    and prev_c.chunk_id not in seen_chunk_ids
                    and prev_c.section == chunk.section
                    and prev_c.document_id == chunk.document_id
                    and not prev_c.is_table_row
                ):
                    seen_chunk_ids.add(prev_c.chunk_id)
                    expanded_chunks.append(prev_c)

            # 2. Expand Next Neighbor in same section
            if max_neighbors_per_seed >= 1 and chunk.next_chunk_id:
                next_c = chunk_by_id.get(chunk.next_chunk_id)
                if (
                    next_c
                    and next_c.chunk_id not in seen_chunk_ids
                    and next_c.section == chunk.section
                    and next_c.document_id == chunk.document_id
                    and not next_c.is_table_row
                ):
                    seen_chunk_ids.add(next_c.chunk_id)
                    expanded_chunks.append(next_c)

            # 3. Expand Parent Chunk (e.g. parent section or parent table for a row)
            if include_parent and chunk.parent_chunk_id:
                parent_c = chunk_by_id.get(chunk.parent_chunk_id)
                if parent_c and parent_c.chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(parent_c.chunk_id)
                    expanded_chunks.append(parent_c)

            if len(expanded_chunks) >= max_total_chunks:
                break

        # Sort expanded chunks by document and chunk sequence number if possible
        def sort_key(c: DocumentChunk):
            seq = 0
            if "_c" in c.chunk_id:
                try:
                    seq = int(c.chunk_id.rsplit("_c", 1)[1])
                except Exception:
                    seq = 0
            return (c.document_id, c.page, seq)

        expanded_chunks.sort(key=sort_key)
        return expanded_chunks[:max_total_chunks]

    @classmethod
    def get_section_flow(
        cls,
        section_name: str,
        document_id: str,
        chunks: List[DocumentChunk]
    ) -> List[DocumentChunk]:
        """
        Retrieves all consecutive non-table chunks in a section in document reading order.
        Crucial for complete multi-step procedures and workflows.
        """
        matching = [
            c for c in chunks
            if c.document_id == document_id
            and (c.section or "").lower() == section_name.lower()
            and not c.is_table_row
        ]

        def sort_key(c: DocumentChunk):
            seq = 0
            if "_c" in c.chunk_id:
                try:
                    seq = int(c.chunk_id.rsplit("_c", 1)[1])
                except Exception:
                    seq = 0
            return (c.page, seq)

        matching.sort(key=sort_key)
        return matching
