from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class ChatRequest(BaseModel):
    model_config = {"extra": "ignore"}
    message: str = Field(..., description="The user question or query")
    session_id: Optional[str] = Field(default="default", description="Session ID for multi-turn conversation")
    history: Optional[List[Dict[str, str]]] = Field(default=None, description="Previous conversation turns")
    tenant_id: Optional[str] = Field(default=None, description="Tenant ID for tenant isolation")
    user_id: Optional[str] = Field(default=None, description="User ID for user-level isolation")
    contact_number: Optional[str] = Field(default=None, description="User contact number for unresolved query follow-up")
    contact_email: Optional[str] = Field(default=None, description="User contact email for follow-up")
    filter_doc_id: Optional[str] = Field(default=None, description="Optional document filter")
    filter_section: Optional[str] = Field(default=None, description="Optional section filter")
    include_debug: Optional[bool] = Field(default=False, description="Enable explainability debug trace")
    top_k: Optional[int] = Field(default=5, description="Number of results to retrieve")

class SourceCitation(BaseModel):
    document: Optional[str] = Field(default=None, description="Source document name")
    document_id: Optional[str] = Field(default=None, description="Source document ID")
    page: Optional[int] = Field(default=None, description="Page number in source document")
    section: Optional[str] = Field(default=None, description="Section or heading title")
    score: float = Field(..., description="Relevance / confidence score")
    text: str = Field(..., description="Excerpt from the source document")
    chunk_id: Optional[str] = Field(default=None, description="Unique chunk ID")

class ChatResponse(BaseModel):
    success: bool
    answer: str
    filename: Optional[str] = None
    confidence: Optional[str] = Field(default=None, description="Confidence category: HIGH, MEDIUM, LOW, VERY_LOW")
    confidence_score: Optional[float] = Field(default=None, description="Composite relevance score")
    sources: List[SourceCitation] = []
    debug_trace: Optional[Dict[str, Any]] = Field(default=None, description="Explainability debug trace")
    detected_language: Optional[str] = Field(default=None, description="Detected query language (en/hi/hinglish/gu)")
    english_query: Optional[str] = Field(default=None, description="Translated English query used for retrieval")
    escalation_status: Optional[str] = Field(default=None, description="Escalation state: None, ASK_CONTACT, ESCALATED")

class DocumentInfo(BaseModel):
    filename: str
    chunks: int
    pages: Optional[int] = None

class StatusResponse(BaseModel):
    success: bool
    documents_loaded: List[str]
    total_documents: int
    total_chunks: int
    documents: List[DocumentInfo]
    storage: str
    original_files: str
    retrieval: str
    generation: str

class UploadResponse(BaseModel):
    success: bool
    message: str
    filename: Optional[str] = None
    chunks: Optional[int] = None
    storage: Optional[str] = None
    error: Optional[str] = None

class ClearResponse(BaseModel):
    success: bool
    message: str
