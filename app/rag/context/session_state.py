import time
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, field

@dataclass
class ConversationTurnRecord:
    raw_query: str
    normalized_query: str
    semantic_query: str
    intent: str
    entities: List[str]
    subject: Optional[str] = None
    retrieved_doc_id: Optional[str] = None
    retrieved_doc_name: Optional[str] = None
    retrieved_section: Optional[str] = None
    answer_text: Optional[str] = None
    answer_entities: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

class StructuredSessionState:
    """
    Structured Session State Tracker for Deterministic Multi-Turn Reasoning.
    Maintains:
      - active_document: currently focused document ID & filename
      - active_entity: primary entity of focus
      - active_topic: current subject / thematic topic
      - recent_entities: recency-ordered list of mentioned entities
      - recent_topics: recency-ordered list of topics
      - recent_intents: recency-ordered list of intents
      - recent_queries: raw queries in sequence
      - last_answer_entities: entities present in the preceding answer
    Thread-safe and supports TTL expiration.
    """
    def __init__(
        self,
        session_id: str,
        tenant_id: str = "default",
        user_id: str = "default",
        max_history: int = 5,
        ttl_seconds: int = 900
    ):
        self.session_id = session_id
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.max_history = max_history
        self.ttl_seconds = ttl_seconds

        self.active_document_id: Optional[str] = None
        self.active_document_name: Optional[str] = None
        self.active_section: Optional[str] = None
        self.active_entity: Optional[str] = None
        self.active_topic: Optional[str] = None

        self.recent_entities: List[str] = []
        self.recent_topics: List[str] = []
        self.recent_intents: List[str] = []
        self.recent_queries: List[str] = []
        self.last_answer_entities: List[str] = []

        self.turns: List[ConversationTurnRecord] = []
        self.last_active: float = time.time()
        self.pending_unresolved_query: Optional[str] = None
        self.pending_unresolved_time: Optional[float] = None

    def is_expired(self, current_time: Optional[float] = None) -> bool:
        now = current_time or time.time()
        return (now - self.last_active) > self.ttl_seconds

    def touch(self):
        self.last_active = time.time()

    def record_turn(
        self,
        raw_query: str,
        normalized_query: str,
        semantic_query: str,
        intent: str,
        entities: List[str],
        subject: Optional[str] = None,
        doc_id: Optional[str] = None,
        doc_name: Optional[str] = None,
        section: Optional[str] = None,
        answer_text: Optional[str] = None,
        answer_entities: Optional[List[str]] = None
    ):
        """Updates session state with a completed turn."""
        self.touch()
        ans_ents = answer_entities or []

        # 1. Update active anchors
        if entities:
            self.active_entity = entities[0]
        elif subject:
            self.active_entity = subject

        if subject:
            self.active_topic = subject
        elif section and section.lower() != "general":
            self.active_topic = section

        if doc_id:
            self.active_document_id = doc_id
        if doc_name:
            self.active_document_name = doc_name
        if section:
            self.active_section = section

        # 2. Update recency lists (avoid duplicates, keep most recent at front)
        for ent in reversed(entities):
            if ent in self.recent_entities:
                self.recent_entities.remove(ent)
            self.recent_entities.insert(0, ent)
        self.recent_entities = self.recent_entities[:self.max_history * 2]

        if self.active_topic:
            if self.active_topic in self.recent_topics:
                self.recent_topics.remove(self.active_topic)
            self.recent_topics.insert(0, self.active_topic)
        self.recent_topics = self.recent_topics[:self.max_history]

        if intent:
            if intent in self.recent_intents:
                self.recent_intents.remove(intent)
            self.recent_intents.insert(0, intent)
        self.recent_intents = self.recent_intents[:self.max_history]

        self.recent_queries.insert(0, raw_query)
        self.recent_queries = self.recent_queries[:self.max_history]

        self.last_answer_entities = list(ans_ents)

        # 3. Append to turn history
        turn = ConversationTurnRecord(
            raw_query=raw_query,
            normalized_query=normalized_query,
            semantic_query=semantic_query,
            intent=intent,
            entities=entities,
            subject=subject,
            retrieved_doc_id=doc_id,
            retrieved_doc_name=doc_name,
            retrieved_section=section,
            answer_text=answer_text,
            answer_entities=ans_ents,
            timestamp=self.last_active
        )
        self.turns.append(turn)
        if len(self.turns) > self.max_history:
            self.turns.pop(0)

    def get_last_turn(self) -> Optional[ConversationTurnRecord]:
        return self.turns[-1] if self.turns else None
