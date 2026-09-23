import time
import threading
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict
from dataclasses import dataclass, field
from app.rag.query.analyzer import AnalyzedQuery

@dataclass
class ConversationTurn:
    raw_query: str
    analyzed_query: AnalyzedQuery
    retrieved_doc_id: Optional[str] = None
    retrieved_doc_name: Optional[str] = None
    retrieved_section: Optional[str] = None
    retrieved_text_snippet: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

class SessionContext:
    """Tracks last N conversational turns for a specific user session."""
    def __init__(self, session_id: str, tenant_id: str, user_id: str, max_turns: int = 3):
        self.session_id = session_id
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.max_turns = max_turns
        self.turns: List[ConversationTurn] = []
        self.last_active: float = time.time()
        self.pending_unresolved_query: Optional[str] = None
        self.pending_unresolved_time: Optional[float] = None

    def add_turn(self, turn: ConversationTurn):
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)
        self.last_active = time.time()

    def get_last_turn(self) -> Optional[ConversationTurn]:
        return self.turns[-1] if self.turns else None

class ConversationalContextManager:
    """
    100% Deterministic Conversational Context Engine.
    Handles session state, topic/entity carry-forward, reference resolution on elliptical queries,
    TTL context expiration, and ambiguity detection without any LLM.
    Thread-safe for high concurrency.
    """

    def __init__(self, ttl_seconds: int = 900, max_sessions: int = 1000):
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self.lock = threading.Lock()
        # LRU cache of sessions: session_key -> SessionContext
        self.sessions: OrderedDict[str, SessionContext] = OrderedDict()

    def _make_key(self, tenant_id: str, user_id: str, session_id: str) -> str:
        return f"{tenant_id}:{user_id}:{session_id}"

    def get_session(self, tenant_id: str, user_id: str, session_id: str) -> Optional[SessionContext]:
        """Retrieves active session if not expired."""
        with self.lock:
            key = self._make_key(tenant_id, user_id, session_id)
            if key not in self.sessions:
                return None

            session = self.sessions[key]
            now = time.time()
            # Check TTL expiration
            if now - session.last_active > self.ttl_seconds:
                del self.sessions[key]
                return None

            # Move to end for LRU
            self.sessions.move_to_end(key)
            return session

    def get_or_create_session(self, tenant_id: str, user_id: str, session_id: str) -> SessionContext:
        with self.lock:
            key = self._make_key(tenant_id, user_id, session_id)
            if key in self.sessions:
                session = self.sessions[key]
                now = time.time()
                if now - session.last_active <= self.ttl_seconds:
                    self.sessions.move_to_end(key)
                    return session
                del self.sessions[key]

            if len(self.sessions) >= self.max_sessions:
                self.sessions.popitem(last=False)
            session = SessionContext(session_id, tenant_id, user_id)
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
        Deterministically resolves conversational context for follow-up queries.
        Returns:
          (search_query, was_enriched, clarification_message)
        """
        raw_q = analyzed_query.raw_query.strip()
        if not session_id:
            return raw_q, False, None

        session = self.get_session(tenant_id, user_id, session_id)

        # If query is elliptical (e.g. "What about international customers?", "Why?")
        if analyzed_query.is_elliptical:
            if not session or not session.turns:
                # Ambiguity detection: query requires prior context, but none exists
                if len(raw_q.split()) <= 3:
                    return raw_q, False, "Could you please specify which topic or document you are referring to?"
                return raw_q, False, None

            last_turn = session.get_last_turn()
            if not last_turn:
                return raw_q, False, None

            # Carry forward primary entity, document, and section
            prev_entity = last_turn.analyzed_query.primary_entity or ""
            prev_doc = last_turn.retrieved_doc_name or ""
            prev_sec = last_turn.retrieved_section or ""

            context_parts = []
            if prev_entity:
                context_parts.append(prev_entity)
            if prev_sec and prev_sec != "General":
                context_parts.append(prev_sec)
            if prev_doc:
                # Strip extension for clean search term
                clean_doc = prev_doc.rsplit('.', 1)[0].replace('_', ' ')
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
        retrieved_text_snippet: Optional[str] = None
    ):
        """Records completed turn into session memory."""
        if not session_id:
            return

        session = self.get_or_create_session(tenant_id, user_id, session_id)
        turn = ConversationTurn(
            raw_query=raw_query,
            analyzed_query=analyzed_query,
            retrieved_doc_id=retrieved_doc_id,
            retrieved_doc_name=retrieved_doc_name,
            retrieved_section=retrieved_section,
            retrieved_text_snippet=retrieved_text_snippet
        )
        session.add_turn(turn)
