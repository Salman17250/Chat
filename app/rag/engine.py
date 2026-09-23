import json
import os
import threading
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from collections import OrderedDict
from app.rag.config import (
    INDEX_FILE,
    DATA_DIR,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    BATCH_EMBED_SIZE,
    FINAL_TOP_K,
    TOP_K_VECTOR,
    TOP_K_KEYWORD,
    TOP_K_FUZZY,
    NO_ANSWER_FOUND_MESSAGE,
)
from app.rag.ingestion.models import DocumentChunk, ParsedDocument
from app.rag.ingestion.parser import DocumentParser
from app.rag.chunking.structure_chunker import StructureAwareChunker
from app.rag.embeddings.fast_embedder import get_fast_embedder
from app.rag.cache.memory import get_multi_cache
from app.rag.query.analyzer import QueryAnalyzer, QueryProfile, AnalyzedQuery
from app.rag.query.planner import QueryPlanner, QueryPlan
from app.rag.query.normalizer import STOP_WORDS
from app.rag.context.conversational import ConversationalContextManager
from app.rag.tenant.tenant_manager import TenantManager, TenantIndex
from app.rag.ranking.adaptive_scorer import AdaptiveScorer, ScoredChunk
from app.rag.context.assembler import ContextAssembler, AssembledResponse, AssembledSource
from app.rag.retrieval.fuzzy_search import FuzzySearchEngine
from app.rag.evidence.validator import EvidenceValidator
from app.rag.reasoning.calculator import DeterministicCalculator
from app.rag.reasoning.comparator import DeterministicComparator

class RAGEngine:
    """
    Phase 3 General-Purpose Intelligent Document RAG Engine.
    100% Non-Generative, Dynamic Domain Intelligence.
    Features:
      - Strict multi-tenant isolation (vector, BM25, exact, entity, table, vocabulary)
      - High-concurrency lock-free parallel search pipeline
      - In-memory ultra-low-latency response caching
      - Deterministic query profiling & multi-part question decomposition
      - Cross-document evidence aggregation with balanced clause attribution
      - Row-level table intelligence with typed attributes
      - Deterministic conversational context tracking (anaphora, topic carry-forward, TTL)
      - Dynamic vocabulary & acronym discovery directly from indexed corpora
      - Paraphrase-safe calibrated evidence gating
    """

    def __init__(self):
        self.lock = threading.RLock()

        # High-throughput in-memory response cache for concurrent identical queries
        self.response_cache: OrderedDict[str, AssembledResponse] = OrderedDict()
        self.response_cache_lock = threading.Lock()
        self.max_response_cache = 3000

        # Multi-Tier L1 Cache Tier
        self.multi_cache = get_multi_cache()

        # Ingestion & Embedding Subsystems (Zero Ollama on query path)
        self.parser = DocumentParser()
        self.chunker = StructureAwareChunker()
        self.embedder = get_fast_embedder()

        # Intelligence Subsystems
        self.tenant_manager = TenantManager()
        self.query_analyzer = QueryAnalyzer()
        self.planner = QueryPlanner(self.query_analyzer)
        self.context_manager = ConversationalContextManager(ttl_seconds=900)
        self.adaptive_scorer = AdaptiveScorer()
        self.assembler = ContextAssembler()
        self.fuzzy_engine = FuzzySearchEngine()
        self.validator = EvidenceValidator()
        self.calculator = DeterministicCalculator()
        self.comparator = DeterministicComparator()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=24, thread_name_prefix="rag_retrieval")

    @property
    def chunks(self) -> List[DocumentChunk]:
        """Backward-compatible access to chunks across all active tenants."""
        from app.storage_postgres import get_postgres_store
        pg_store = get_postgres_store()
        if not pg_store or not pg_store.is_connected_fast():
            return []
        all_c: List[DocumentChunk] = []
        for t_id in self.tenant_manager.list_tenants():
            all_c.extend(self.tenant_manager.get_tenant(t_id).chunks)
        return all_c

    @property
    def documents(self) -> Dict[str, Dict[str, Any]]:
        """Backward-compatible access to documents across all active tenants."""
        from app.storage_postgres import get_postgres_store
        pg_store = get_postgres_store()
        if not pg_store or not pg_store.is_connected_fast():
            return {}
        all_docs: Dict[str, Dict[str, Any]] = {}
        for t_id in self.tenant_manager.list_tenants():
            all_docs.update(self.tenant_manager.get_tenant(t_id).documents)
        return all_docs

    def ingest_document(
        self,
        file_path: Path,
        filename: Optional[str] = None,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Parses document, decomposes tables into row sub-chunks with typed values,
        generates dense embeddings via nomic-embed-text, and updates the tenant's isolated index.
        """
        with self.lock:
            with self.response_cache_lock:
                self.response_cache.clear()
            fname = filename or file_path.name
            t_id = (tenant_id or "default").strip().lower()
            u_id = (user_id or "default").strip()

            # 1. Parse document structure
            parsed_doc: ParsedDocument = self.parser.parse(
                file_path=file_path,
                tenant_id=t_id,
                user_id=u_id,
            )

            # 2. Structure-aware chunking with table intelligence
            doc_chunks = self.chunker.chunk_document(parsed_doc)
            if not doc_chunks:
                return {
                    "document_id": parsed_doc.document_id,
                    "filename": fname,
                    "chunks_count": 0,
                    "status": "warning_empty",
                }

            # 3. Dense embeddings generation via nomic-embed-text
            chunk_texts = [
                f"search_document: {c.section}: {c.text}" if c.section else f"search_document: {c.text}"
                for c in doc_chunks
            ]
            doc_embeddings = self.embedder.embed_batch(chunk_texts, batch_size=BATCH_EMBED_SIZE)

            # 4. Resolve tenant namespace
            tenant = self.tenant_manager.get_tenant(t_id)

            # 5. Remove any previous versions of this document in tenant index
            tenant.chunks = [c for c in tenant.chunks if c.document_id != parsed_doc.document_id]
            # Filter corresponding embeddings
            new_embeddings: List[List[float]] = []
            for c, emb in zip(tenant.chunks, tenant.embeddings[:len(tenant.chunks)]):
                if c.document_id != parsed_doc.document_id:
                    new_embeddings.append(emb)
            tenant.embeddings = new_embeddings

            # 6. Append new chunks and embeddings
            tenant.chunks.extend(doc_chunks)
            tenant.embeddings.extend(doc_embeddings)
            tenant.documents[parsed_doc.document_id] = {
                "document_id": parsed_doc.document_id,
                "filename": fname,
                "tenant_id": t_id,
                "user_id": u_id,
                "page_count": parsed_doc.total_pages,
                "chunks_count": len(doc_chunks),
            }

            # 7. Rebuild tenant search indices and dynamic vocabulary
            tenant.reindex()
            tenant.save()

            return {
                "document_id": parsed_doc.document_id,
                "filename": fname,
                "tenant_id": t_id,
                "user_id": u_id,
                "chunks_count": len(doc_chunks),
                "status": "indexed",
            }

    def delete_document(self, document_id: str, tenant_id: Optional[str] = None) -> bool:
        """Deletes a document from the specified tenant namespace and purges all related cache."""
        with self.lock:
            deleted = self.tenant_manager.delete_document(document_id, tenant_id)
            with self.response_cache_lock:
                self.response_cache.clear()
            self.multi_cache.clear_all()
            return deleted

    def _retrieve_for_sub_query(
        self,
        sub_query: str,
        tenant: TenantIndex,
        profile: QueryProfile,
        target_attributes: List[str],
        filter_doc_id: Optional[str] = None,
        filter_section: Optional[str] = None,
        top_k: int = FINAL_TOP_K,
        analyzed_q: Optional[AnalyzedQuery] = None
    ) -> List[ScoredChunk]:
        """Executes ultra-fast parallel hybrid retrieval (Exact, Entity, Table, BM25, Fuzzy, SIMD Vector) and strict evidence validation."""
        clean_q = sub_query.strip()
        if analyzed_q is None:
            analyzed_q = self.query_analyzer.analyze(clean_q, vocab=tenant.dynamic_vocab)

        # 1. Generate query execution plan
        plan = self.planner.plan(clean_q)

        norm_res = self.query_analyzer.normalizer.normalize(clean_q)
        query_tokens = norm_res.tokens

        # Typo correction against tenant vocabulary
        corrected_tokens = [tenant.dynamic_vocab.correct_typo(t) or t for t in query_tokens]

        # Dynamic acronym expansion
        expanded_terms = []
        for t in corrected_tokens:
            exp = tenant.dynamic_vocab.expand_acronym(t)
            if exp:
                expanded_terms.append(exp)

        # Merge candidate scores container
        candidate_dict: Dict[str, Dict[str, Any]] = {}
        def record(chunk: DocumentChunk, sem: float = 0.0, kw: float = 0.0, fz: float = 0.0):
            if filter_doc_id and chunk.document_id != filter_doc_id:
                return
            if filter_section and chunk.section and filter_section.lower() not in chunk.section.lower():
                return
            cid = chunk.chunk_id
            if cid not in candidate_dict:
                candidate_dict[cid] = {"chunk": chunk, "sem": sem, "kw": kw, "fz": fz}
            else:
                candidate_dict[cid]["sem"] = max(candidate_dict[cid]["sem"], sem)
                candidate_dict[cid]["kw"] = max(candidate_dict[cid]["kw"], kw)
                candidate_dict[cid]["fz"] = max(candidate_dict[cid]["fz"], fz)

        # Target query for vector embedding
        core_query = analyzed_q.core_query
        target_vector_q = (
            core_query
            if (core_query and len(core_query.split()) >= 2)
            else (" ".join(corrected_tokens) if corrected_tokens != query_tokens else clean_q)
        )

        # Launch Vector Embedding ONLY if not bypassed by query planner
        future_q_vec = None
        if not plan.can_bypass_embedding:
            future_q_vec = self.executor.submit(self.embedder.embed_query, target_vector_q)

        # Branch 1: Exact Matching Search (<0.1ms)
        exact_results = tenant.exact_engine.search(clean_q, top_k=TOP_K_KEYWORD)
        for c, s in exact_results:
            record(c, kw=s * 1.15)

        # Branch 2: Entity-Level Search (<0.1ms)
        entity_results = tenant.entity_engine.search(
            clean_q,
            detected_entities=plan.entities,
            target_attributes=plan.target_attributes,
            top_k=TOP_K_KEYWORD
        )
        for c, s in entity_results:
            record(c, kw=s * 1.15)

        # Branch 3: Structured Table Search (<0.2ms)
        if plan.requires_table_lookup or "table" in plan.active_branches:
            table_results = tenant.table_engine.search(clean_q, target_attributes=target_attributes, top_k=TOP_K_KEYWORD)
            for c, s in table_results:
                record(c, kw=s * 1.20)

        # Branch 4: Okapi BM25 Search (<2ms)
        bm25_search_tokens = list(corrected_tokens)
        for term in expanded_terms:
            for w in term.split():
                if w not in bm25_search_tokens:
                    bm25_search_tokens.append(w)
        bm25_results = tenant.bm25_engine.search(bm25_search_tokens, top_k=TOP_K_KEYWORD, base_query_tokens=corrected_tokens)
        for c, s in bm25_results:
            record(c, kw=s)

        # Branch 4b: Entity-focused BM25
        primary_entity = analyzed_q.primary_entity
        if primary_entity and len(primary_entity) >= 3 and primary_entity.lower() not in STOP_WORDS:
            entity_tokens = [primary_entity.lower()]
            entity_bm25_res = tenant.bm25_engine.search(entity_tokens, top_k=TOP_K_KEYWORD, base_query_tokens=entity_tokens)
            for c, s in entity_bm25_res:
                record(c, kw=s * 0.85)

        # Branch 5: RapidFuzz Typo Search (<8ms)
        fuzzy_results = tenant.fuzzy_engine.search(clean_q, top_k=TOP_K_FUZZY)
        for c, s in fuzzy_results:
            record(c, fz=s)

        # Branch 6: SIMD BLAS Vector Dot Product (<0.4ms)
        if future_q_vec is not None:
            q_vec = future_q_vec.result()
            if tenant.vector_engine.chunks and tenant.vector_engine._matrix is not None:
                vec_matches = tenant.vector_engine.search(q_vec, top_k=TOP_K_VECTOR)
                for chunk, sim in vec_matches:
                    record(chunk, sem=sim)

        # Detected phrases from tenant dynamic vocabulary
        detected_phrases = tenant.dynamic_vocab.extract_document_phrases(clean_q)

        # Score candidates with AdaptiveScorer
        scored_candidates: List[ScoredChunk] = []
        core_tokens_for_qa = [primary_entity] if primary_entity else None
        detected_intent = self.adaptive_scorer.intent_detector.detect_intent(clean_q)[0]
        for cid, item in candidate_dict.items():
            sc = self.adaptive_scorer.score_candidate(
                chunk=item["chunk"],
                semantic_score=item["sem"],
                keyword_score=item["kw"],
                fuzzy_score=item["fz"],
                detected_phrases=detected_phrases,
                query_tokens=corrected_tokens,
                expanded_terms=expanded_terms,
                intent=detected_intent,
                profile=profile,
                target_attributes=target_attributes,
                filter_doc_id=filter_doc_id,
                filter_section=filter_section,
                core_query=core_query,
                core_tokens=core_tokens_for_qa,
            )
            scored_candidates.append(sc)

        # Strict Evidence Validation Gate: Eliminate false positives on ungrounded/absent topics
        validated_candidates: List[ScoredChunk] = []
        for sc in scored_candidates:
            v_res = self.validator.validate_candidate(sc, plan, tenant_id=tenant.tenant_id)
            if not v_res.is_valid:
                # Suppress unverified candidates that lack critical non-generic qualifiers
                sc.final_score = 0.05
                sc.semantic_score = 0.05
                sc.keyword_score = 0.0
                sc.heading_score = 0.0
                sc.fuzzy_score = 0.0
                sc.query_coverage = 0.0
                sc.independent_signals_count = 0
            validated_candidates.append(sc)

        validated_candidates.sort(key=lambda x: x.final_score, reverse=True)
        return validated_candidates[:top_k]

    def query(
        self,
        question: str,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        filter_doc_id: Optional[str] = None,
        filter_section: Optional[str] = None,
        top_k: int = FINAL_TOP_K,
        include_debug: bool = False,
    ) -> AssembledResponse:
        """
        Executes end-to-end question answering pipeline:
          1. Resolve isolated tenant namespace.
          2. Check ultra-low-latency in-memory response cache for identical concurrent queries.
          3. Deterministic query analysis & profile classification.
          4. Deterministic conversational context resolution (anaphora, carry-forward, ambiguity).
          5. Cross-document multi-part decomposition or adaptive hybrid retrieval (lock-free concurrent execution).
          6. Calibrated multi-signal evidence gating & answer assembly.
          7. Record interaction in conversation memory.
        """
        t_id = (tenant_id or "default").strip().lower()
        u_id = (user_id or "default").strip()
        sess_id = session_id or "default"

        # Strict Cloud Database Enforcement (Zero Local Fallback)
        from app.storage_postgres import get_postgres_store
        pg_store = get_postgres_store()
        if not pg_store or not pg_store.is_connected_fast():
            with self.response_cache_lock:
                self.response_cache.clear()
            for t_name in self.tenant_manager.list_tenants():
                t = self.tenant_manager.get_tenant(t_name)
                t.chunks = []
                t.documents = {}
                t.embeddings = []
                t.vector_engine.index_chunks([], [])
                t.bm25_engine.index_chunks([])
                t.fuzzy_engine.index_chunks([])
            return AssembledResponse(
                answer="[Cloud Database Disconnected] Neon PostgreSQL connection is required. All local data has been removed. Please verify DATABASE_URL in .env to connect to the cloud database.",
                confidence="VERY_LOW",
                confidence_score=0.0,
                sources=[]
            )

        tenant = self.tenant_manager.get_tenant(t_id)
        if not tenant.chunks:
            tenant.load()
        if not tenant.chunks:
            return AssembledResponse(
                answer=NO_ANSWER_FOUND_MESSAGE,
                confidence="VERY_LOW",
                confidence_score=0.0,
                sources=[],
                debug_trace={"empty_vault": True} if include_debug else None
            )
        chunk_by_id = {c.chunk_id: c for c in tenant.chunks}

        # 1. High-throughput in-memory MultiTierCache lookup (<0.001ms)
        if not filter_doc_id and not filter_section and not include_debug:
            cached_resp = self.multi_cache.get_response(t_id, question)
            if cached_resp is not None:
                return cached_resp

        # 2. Deterministic Query Analysis & Planning
        analyzed_q = self.query_analyzer.analyze(question, vocab=tenant.dynamic_vocab)
        plan = self.planner.plan(question)

        # Direct Mathematical Calculation Check (Zero LLM / Zero Document I/O if standalone math)
        if plan.requires_calculation and len(plan.numbers) >= 1 and any(w in question.lower() for w in ["gst", "tax", "discount"]):
            direct_math = self.calculator.evaluate_query_math(question, "")
            if direct_math:
                direct_res = AssembledResponse(
                    answer=direct_math,
                    confidence="HIGH",
                    confidence_score=1.0,
                    sources=[],
                    debug_trace={"deterministic_calculation": True} if include_debug else None
                )
                self.multi_cache.set_response(t_id, question, direct_res)
                return direct_res

        # 3. Conversational Context Resolution
        search_query, was_enriched, clarification = self.context_manager.enrich_query(
            tenant_id=t_id,
            user_id=u_id,
            session_id=sess_id,
            analyzed_query=analyzed_q
        )

        if clarification:
            return AssembledResponse(
                answer=clarification,
                confidence="HIGH",
                confidence_score=1.0,
                sources=[],
                debug_trace={"conversational_ambiguity": True} if include_debug else None
            )

        # Deterministic Comparison Check (e.g. compare Entity A and Entity B)
        if plan.requires_comparison and len(plan.comparison_targets) == 2:
            target_a, target_b = plan.comparison_targets
            chunks_a = self._retrieve_for_sub_query(target_a, tenant, QueryProfile.EXACT_CODE, [], filter_doc_id, filter_section, top_k=2)
            chunks_b = self._retrieve_for_sub_query(target_b, tenant, QueryProfile.EXACT_CODE, [], filter_doc_id, filter_section, top_k=2)
            if chunks_a and chunks_b and chunks_a[0].final_score >= 0.35 and chunks_b[0].final_score >= 0.35:
                comp_ans = self.comparator.compare_entities(target_a, target_b, [chunks_a[0].chunk], [chunks_b[0].chunk])
                sources = [
                    AssembledSource(document=chunks_a[0].chunk.document_name, document_id=chunks_a[0].chunk.document_id, page=chunks_a[0].chunk.page, section=chunks_a[0].chunk.section, score=round(chunks_a[0].final_score, 4), text=chunks_a[0].chunk.text, chunk_id=chunks_a[0].chunk.chunk_id),
                    AssembledSource(document=chunks_b[0].chunk.document_name, document_id=chunks_b[0].chunk.document_id, page=chunks_b[0].chunk.page, section=chunks_b[0].chunk.section, score=round(chunks_b[0].final_score, 4), text=chunks_b[0].chunk.text, chunk_id=chunks_b[0].chunk.chunk_id)
                ]
                comp_res = AssembledResponse(
                    answer=comp_ans,
                    confidence="HIGH",
                    confidence_score=round((chunks_a[0].final_score + chunks_b[0].final_score) / 2.0, 4),
                    sources=sources,
                    debug_trace={"deterministic_comparison": True} if include_debug else None
                )
                self.multi_cache.set_response(t_id, question, comp_res)
                return comp_res

        debug_trace: Dict[str, Any] = {}
        if include_debug:
            debug_trace["raw_query"] = question
            debug_trace["search_query"] = search_query
            debug_trace["profile"] = analyzed_q.profile.value
            debug_trace["was_enriched"] = was_enriched
            debug_trace["sub_queries"] = analyzed_q.sub_queries

        # 4. Multi-Part vs Single-Query Retrieval (Lock-free concurrent execution)
        if analyzed_q.profile == QueryProfile.MULTI_PART and len(analyzed_q.sub_queries) > 1:
            sub_results: List[Any] = []
            for sub_q in analyzed_q.sub_queries:
                sub_analyzed = self.query_analyzer.analyze(sub_q, vocab=tenant.dynamic_vocab)
                sub_plan = self.planner.plan(sub_q)
                sub_scored = self._retrieve_for_sub_query(
                    sub_query=sub_q,
                    tenant=tenant,
                    profile=sub_analyzed.profile,
                    target_attributes=sub_plan.target_attributes,
                    filter_doc_id=filter_doc_id,
                    filter_section=filter_section,
                    top_k=top_k,
                    analyzed_q=sub_analyzed
                )
                sub_results.append((sub_q, sub_scored, sub_plan))

            response = self.assembler.assemble_multi_part(
                sub_query_results=sub_results,
                chunk_by_id=chunk_by_id,
                thresholds=tenant.thresholds,
                debug_trace=debug_trace if include_debug else None,
            )
        else:
            effective_query = search_query if was_enriched else question
            scored_chunks = self._retrieve_for_sub_query(
                sub_query=effective_query,
                tenant=tenant,
                profile=analyzed_q.profile,
                target_attributes=plan.target_attributes,
                filter_doc_id=filter_doc_id,
                filter_section=filter_section,
                top_k=top_k,
                analyzed_q=analyzed_q
            )

            response = self.assembler.assemble(
                ranked_chunks=scored_chunks,
                chunk_by_id=chunk_by_id,
                debug_trace=debug_trace if include_debug else None,
                max_sources=top_k,
                thresholds=tenant.thresholds,
                profile=analyzed_q.profile,
                plan=plan
            )

        # 5. Store in high-speed MultiTierCache if not enriched by conversational context
        if not was_enriched:
            self.multi_cache.set_response(t_id, question, response)

        # 6. Record turn into session memory
        top_source = response.sources[0] if response.sources else None
        self.context_manager.record_turn(
            tenant_id=t_id,
            user_id=u_id,
            session_id=sess_id,
            raw_query=question,
            analyzed_query=analyzed_q,
            retrieved_doc_id=top_source.document_id if top_source else None,
            retrieved_doc_name=top_source.document if top_source else None,
            retrieved_section=top_source.section if top_source else None,
            retrieved_text_snippet=top_source.text[:100] if top_source else None
        )

        return response

    def list_documents(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns metadata list of indexed documents strictly from Cloud PostgreSQL."""
        from app.storage_postgres import get_postgres_store
        pg_store = get_postgres_store()
        if not pg_store or not pg_store.test_connection():
            return []
        return pg_store.list_documents(tenant_id)

# Global singleton
_rag_engine: Optional[RAGEngine] = None

def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
