from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class DocumentBlock:
    """A structural block extracted from a document (heading, paragraph, table, list, question, answer)."""
    text: str
    page: int
    block_type: str  # "heading", "paragraph", "table", "list_item", "question", "answer"
    section_title: str = ""
    level: int = 1   # heading level 1, 2, 3
    is_qa_pair: bool = False
    question_text: Optional[str] = None
    answer_text: Optional[str] = None
    question_number: Optional[int] = None

@dataclass
class DocumentChunk:
    """Structure-aware chunk with complete metadata tracking."""
    document_id: str
    document_name: str
    page: int
    section: str
    chunk_id: str
    text: str
    word_count: int = 0
    tenant_id: str = "default"
    user_id: str = "default"
    embedding: Optional[List[float]] = None
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    category: str = "general"
    is_table_row: bool = False
    table_headers: List[str] = field(default_factory=list)
    table_row_data: Dict[str, Any] = field(default_factory=dict)
    typed_attributes: Dict[str, Any] = field(default_factory=dict)
    parent_chunk_id: Optional[str] = None
    is_qa_pair: bool = False
    question_text: Optional[str] = None
    answer_text: Optional[str] = None
    question_number: Optional[int] = None
    entities: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    relationships: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page": self.page,
            "section": self.section,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "word_count": self.word_count,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "embedding": self.embedding,
            "prev_chunk_id": self.prev_chunk_id,
            "next_chunk_id": self.next_chunk_id,
            "category": self.category,
            "is_table_row": self.is_table_row,
            "table_headers": self.table_headers,
            "table_row_data": self.table_row_data,
            "typed_attributes": self.typed_attributes,
            "parent_chunk_id": self.parent_chunk_id,
            "is_qa_pair": self.is_qa_pair,
            "question_text": self.question_text,
            "answer_text": self.answer_text,
            "question_number": self.question_number,
            "entities": self.entities,
            "attributes": self.attributes,
            "relationships": self.relationships,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentChunk':
        return cls(
            document_id=data.get("document_id", ""),
            document_name=data.get("document_name", ""),
            page=data.get("page", 1),
            section=data.get("section", ""),
            chunk_id=data.get("chunk_id", ""),
            text=data.get("text", ""),
            word_count=data.get("word_count", 0),
            tenant_id=data.get("tenant_id", "default"),
            user_id=data.get("user_id", "default"),
            embedding=data.get("embedding"),
            prev_chunk_id=data.get("prev_chunk_id"),
            next_chunk_id=data.get("next_chunk_id"),
            category=data.get("category", "general"),
            is_table_row=data.get("is_table_row", False),
            table_headers=data.get("table_headers", []),
            table_row_data=data.get("table_row_data", {}),
            typed_attributes=data.get("typed_attributes", {}),
            parent_chunk_id=data.get("parent_chunk_id"),
            is_qa_pair=data.get("is_qa_pair", False),
            question_text=data.get("question_text"),
            answer_text=data.get("answer_text"),
            question_number=data.get("question_number"),
            entities=data.get("entities", []),
            attributes=data.get("attributes", {}),
            relationships=data.get("relationships", []),
        )

@dataclass
class ParsedDocument:
    document_id: str
    document_name: str
    total_pages: int
    blocks: List[DocumentBlock]
    file_hash: str = ""
    tenant_id: str = "default"
    user_id: str = "default"
