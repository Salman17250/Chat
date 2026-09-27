import time
import threading
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict
from dataclasses import dataclass, field
from app.rag.query.analyzer import AnalyzedQuery
from app.rag.context.session_state import StructuredSessionState, ConversationTurnRecord
from app.rag.query.coreference import CoreferenceResolver

# Backward-compatible alias
ConversationTurn = ConversationTurnRecord
SessionContext = StructuredSessionState

class ConversationalContextManager:
    """
    100% Deterministic Conversational Context & Coreference Engine.
    Handles:
      - Structured session state (active entity, topic, document, recency queues)
      - Coreference and anaphora resolution ('it', 'this', 'that', 'they', 'the above')
      - Topic & entity carry-forward for elliptical queries
      - Ambiguity detection
      - TTL expiration (900s) and thread-safe high concurrency.
    """

    def __init__(self, ttl_seconds: int = 900, max_sessions: int = 1000):
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self.lock = threading.Lock()
        # LRU cache of sessions: session_key -> StructuredSessionState
        self.sessions: OrderedDict[str, StructuredSessionState] = OrderedDict()

    def _make_key(self, tenant_id: str, user_id: str, session_id: str) -> str:
        return f"{tenant_id}:{user_id}:{session_id}"

    def get_session(self, tenant_id: str, user_id: str, session_id: str) -> Optional[StructuredSessionState]:
        """Retrieves active session if not expired."""
        with self.lock:
            key = self._make_key(tenant_id, user_id, session_id)
            if key not in self.sessions:
                return None

            session = self.sessions[key]
            now = time.time()
            if session.is_expired(now):
                del self.sessions[key]
                return None

            # Move to end for LRU
            self.sessions.move_to_end(key)
            return session

    def get_or_create_session(self, tenant_id: str, user_id: str, session_id: str) -> StructuredSessionState:
        with self.lock:
            key = self._make_key(tenant_id, user_id, session_id)
            if key in self.sessions:
                session = self.sessions[key]
                now = time.time()
                if not session.is_expired(now):
                    self.sessions.move_to_end(key)
                    return session
                del self.sessions[key]

            if len(self.sessions) >= self.max_sessions:
                self.sessions.popitem(last=False)
            session = StructuredSessionState(
                session_id=session_id,
                tenant_id=tenant_id,
                user_id=user_id,
                ttl_seconds=self.ttl_seconds
            )
            self.sessions[key] = session
            return session

    def enrich_query(
        self,
        tenant_id: str,
        user_id: str,
        session_id: Optional[str],
        analyzed_query: AnalyzedQuery
    ) -> Tuple[str, bool, Optional[str]]:
        """
        Deterministically resolves conversational context and coreference for queries.
        Returns:
          (search_query, was_enriched, clarification_message)
        """
        raw_q = analyzed_query.raw_query.strip()
        if not session_id:
            return raw_q, False, None

        session = self.get_session(tenant_id, user_id, session_id)
        if not session:
            return raw_q, False, None

        # 1. Coreference Resolution (e.g. "How do I configure it?" -> "How do I configure Pipeline?")
        resolved_q, was_resolved, resolved_refs = CoreferenceResolver.resolve_references(
            query=raw_q,
            session_state=session
        )
        if was_resolved:
            return resolved_q, True, None

        # 2. Elliptical Query Resolution (e.g. "What about stages?", "Why?")
        if analyzed_query.is_elliptical:
            if not session.turns and not session.active_entity:
                if len(raw_q.split()) <= 3:
                    return raw_q, False, "Could you please specify which topic or document you are referring to?"
                return raw_q, False, None

            prev_entity = session.active_entity or ""
            prev_doc = session.active_document_name or ""
            prev_sec = session.active_section or ""

            context_parts = []
            if prev_entity and prev_entity.lower() not in raw_q.lower():
                context_parts.append(prev_entity)
            if prev_sec and prev_sec.lower() != "general" and prev_sec.lower() not in raw_q.lower():
                context_parts.append(prev_sec)
            if prev_doc:
                clean_doc = prev_doc.rsplit('.', 1)[0].replace('_', ' ')
                if clean_doc.lower() not in raw_q.lower():
                    context_parts.append(clean_doc)

            context_str = " ".join(context_parts).strip()
            if context_str:
                enriched_q = f"{raw_q} {context_str}"
                return enriched_q, True, None

        return raw_q, False, None

    def record_turn(
        self,
        tenant_id: str,
        user_id: str,
        session_id: Optional[str],
        raw_query: str,
        analyzed_query: AnalyzedQuery,
        retrieved_doc_id: Optional[str] = None,
        retrieved_doc_name: Optional[str] = None,
        retrieved_section: Optional[str] = None,
        retrieved_text_snippet: Optional[str] = None,
        answer_text: Optional[str] = None,
        answer_entities: Optional[List[str]] = None
    ):
        """Records completed turn into session memory and updates active anchors."""
        if not session_id:
            return

        session = self.get_or_create_session(tenant_id, user_id, session_id)
        
        entities = []
        if analyzed_query.primary_entity:
            entities.append(analyzed_query.primary_entity)
        if getattr(analyzed_query, "target_attributes", None):
            for attr in analyzed_query.target_attributes:
                if attr not in entities:
                    entities.append(attr)

        session.record_turn(
            raw_query=raw_query,
            normalized_query=analyzed_query.normalized_query,
            semantic_query=analyzed_query.core_query or raw_query,
            intent=analyzed_query.profile.value if hasattr(analyzed_query, "profile") else "GENERAL",
            entities=entities,
            subject=analyzed_query.primary_entity,
            doc_id=retrieved_doc_id,
            doc_name=retrieved_doc_name,
            section=retrieved_section,
            answer_text=answer_text or retrieved_text_snippet,
            answer_entities=answer_entities
        )
