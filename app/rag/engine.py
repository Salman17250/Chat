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
import time
from app.rag.context.conversational import ConversationalContextManager
from app.rag.tenant.tenant_manager import TenantManager, TenantIndex
from app.rag.ranking.adaptive_scorer import AdaptiveScorer, ScoredChunk
from app.rag.context.assembler import ContextAssembler, AssembledResponse, AssembledSource
from app.rag.retrieval.fuzzy_search import FuzzySearchEngine
from app.rag.retrieval.parent_child import ParentChildRetriever
from app.rag.ranking.fusion import CandidateFusionEngine
from app.rag.query.decomposer import QueryDecomposer
from app.rag.query.expansion import QueryExpander
from app.rag.evidence.validator import EvidenceValidator
from app.rag.evidence.provenance import EvidenceGraphBuilder
from app.rag.evidence.coverage import EvidenceCoverageChecker
from app.rag.reasoning.calculator import DeterministicCalculator
from app.rag.reasoning.comparator import DeterministicComparator
from app.rag.reasoning.temporal import DeterministicTemporalReasoner
from app.rag.reasoning.logical import DeterministicLogicalReasoner
from app.rag.reasoning.multi_hop import MultiHopReasoningEngine
from app.rag.response.planner import ResponsePlanner, AnswerPlan
from app.rag.response.templates import AnswerTemplates
from app.rag.response.assembler import ResponseAssembler

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
        self.decomposer = QueryDecomposer()
        self.expander = QueryExpander()
        self.context_manager = ConversationalContextManager(ttl_seconds=900)
        self.adaptive_scorer = AdaptiveScorer()
        self.assembler = ContextAssembler()
        self.parent_child_retriever = ParentChildRetriever()
        self.fusion_engine = CandidateFusionEngine()
        self.fuzzy_engine = FuzzySearchEngine()
        self.validator = EvidenceValidator()
        self.evidence_graph_builder = EvidenceGraphBuilder()
        self.calculator = DeterministicCalculator()
        self.comparator = DeterministicComparator()
        self.temporal_reasoner = DeterministicTemporalReasoner()
        self.logical_reasoner = DeterministicLogicalReasoner()
        self.multi_hop_engine = MultiHopReasoningEngine()
        self.response_planner = ResponsePlanner()
        self.response_assembler = ResponseAssembler()
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
        analyzed_q: Optional[AnalyzedQuery] = None,
        timing_breakdown: Optional[Dict[str, float]] = None
    ) -> List[ScoredChunk]:
        """Executes ultra-fast parallel hybrid retrieval (Exact, Entity, Table, BM25, Fuzzy, SIMD Vector), candidate fusion, and strict evidence validation."""
        clean_q = sub_query.strip()
        if analyzed_q is None:
            analyzed_q = self.query_analyzer.analyze(clean_q, vocab=tenant.dynamic_vocab)

        # 1. Generate query execution plan
        plan = self.planner.plan(clean_q)

        norm_res = self.query_analyzer.normalizer.normalize(clean_q)
        query_tokens = norm_res.tokens

        # Typo correction against tenant vocabulary
        corrected_tokens = [tenant.dynamic_vocab.correct_typo(t) or t for t in query_tokens]

        # Dynamic acronym & query expansion
        t_exp = time.perf_counter()
        expanded_terms = self.expander.expand(clean_q, tenant.dynamic_vocab)
        if timing_breakdown is not None and "expansion_ms" not in timing_breakdown:
            timing_breakdown["expansion_ms"] = round((time.perf_counter() - t_exp) * 1000, 3)

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
        t_emb = time.perf_counter()
        if not plan.can_bypass_embedding:
            future_q_vec = self.executor.submit(self.embedder.embed_query, target_vector_q)
            if timing_breakdown is not None and "embedding_ms" not in timing_breakdown:
                timing_breakdown["embedding_ms"] = round((time.perf_counter() - t_emb) * 1000, 3)
        elif timing_breakdown is not None and "embedding_ms" not in timing_breakdown:
            timing_breakdown["embedding_ms"] = 0.0

        # Branch 1: Exact Matching Search (<0.1ms)
        t_ex = time.perf_counter()
        exact_results = tenant.exact_engine.search(clean_q, top_k=TOP_K_KEYWORD)
        if timing_breakdown is not None and "exact_search_ms" not in timing_breakdown:
            timing_breakdown["exact_search_ms"] = round((time.perf_counter() - t_ex) * 1000, 3)
        for c, s in exact_results:
            record(c, kw=s * 1.15)

        # Branch 2: Entity-Level Search (<0.1ms)
        t_ent = time.perf_counter()
        entity_results = tenant.entity_engine.search(
            clean_q,
            detected_entities=plan.entities,
            target_attributes=plan.target_attributes,
            top_k=TOP_K_KEYWORD
        )
        if timing_breakdown is not None and "entity_search_ms" not in timing_breakdown:
            timing_breakdown["entity_search_ms"] = round((time.perf_counter() - t_ent) * 1000, 3)
        for c, s in entity_results:
            record(c, kw=s * 1.15)

        # Branch 3: Structured Table Search (<0.2ms)
        t_tbl = time.perf_counter()
        table_results = []
        if plan.requires_table_lookup or "table" in plan.active_branches:
            table_results = tenant.table_engine.search(clean_q, target_attributes=target_attributes, top_k=TOP_K_KEYWORD)
            for c, s in table_results:
                record(c, kw=s * 1.20)
        if timing_breakdown is not None and "table_search_ms" not in timing_breakdown:
            timing_breakdown["table_search_ms"] = round((time.perf_counter() - t_tbl) * 1000, 3)

        # Branch 4: Okapi BM25 Search (<2ms)
        t_bm = time.perf_counter()
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
        if timing_breakdown is not None and "bm25_ms" not in timing_breakdown:
            timing_breakdown["bm25_ms"] = round((time.perf_counter() - t_bm) * 1000, 3)

        # Branch 5: RapidFuzz Typo Search (<8ms)
        t_fz = time.perf_counter()
        fuzzy_results = tenant.fuzzy_engine.search(clean_q, top_k=TOP_K_FUZZY)
        for c, s in fuzzy_results:
            record(c, fz=s)
        if timing_breakdown is not None and "fuzzy_search_ms" not in timing_breakdown:
            timing_breakdown["fuzzy_search_ms"] = round((time.perf_counter() - t_fz) * 1000, 3)

        # Branch 6: SIMD BLAS Vector Dot Product (<0.4ms)
        t_vec = time.perf_counter()
        vec_matches = []
        if future_q_vec is not None:
            q_vec = future_q_vec.result()
            if tenant.vector_engine.chunks and tenant.vector_engine._matrix is not None:
                vec_matches = tenant.vector_engine.search(q_vec, top_k=TOP_K_VECTOR)
                for chunk, sim in vec_matches:
                    record(chunk, sem=sim)
        if timing_breakdown is not None and "vector_search_ms" not in timing_breakdown:
            timing_breakdown["vector_search_ms"] = round((time.perf_counter() - t_vec) * 1000, 3)

        # Branch Fusion
        t_fuse = time.perf_counter()
        branch_dict = {
            "exact": exact_results,
            "entity": entity_results,
            "table": table_results,
            "bm25": bm25_results,
            "fuzzy": fuzzy_results,
            "vector": vec_matches
        }
        fused_candidates = self.fusion_engine.fuse_candidates(branch_dict, plan, top_k=top_k * 2)
        if timing_breakdown is not None and "fusion_ms" not in timing_breakdown:
            timing_breakdown["fusion_ms"] = round((time.perf_counter() - t_fuse) * 1000, 3)

        # Detected phrases from tenant dynamic vocabulary
        detected_phrases = tenant.dynamic_vocab.extract_document_phrases(clean_q)

        # Score candidates with AdaptiveScorer
        t_rerank = time.perf_counter()
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
        if timing_breakdown is not None and "reranking_ms" not in timing_breakdown:
            timing_breakdown["reranking_ms"] = round((time.perf_counter() - t_rerank) * 1000, 3)

        # Strict Evidence Validation Gate: Eliminate false positives on ungrounded/absent topics
        t_val = time.perf_counter()
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
        if timing_breakdown is not None and "validation_ms" not in timing_breakdown:
            timing_breakdown["validation_ms"] = round((time.perf_counter() - t_val) * 1000, 3)

        validated_candidates.sort(key=lambda x: x.final_score, reverse=True)

        # Parent / Child / Neighbor Context Expansion
        if validated_candidates and validated_candidates[0].final_score >= 0.20:
            chunk_by_id = {c.chunk_id: c for c in tenant.chunks}
            seed_chunks = [vc.chunk for vc in validated_candidates[:top_k] if vc.final_score >= 0.20]
            expanded = self.parent_child_retriever.expand_context_hierarchy(seed_chunks, chunk_by_id, max_neighbors_per_seed=1)
            existing_cids = {vc.chunk.chunk_id for vc in validated_candidates}
            for ec in expanded:
                if ec.chunk_id not in existing_cids:
                    existing_cids.add(ec.chunk_id)
                    validated_candidates.append(ScoredChunk(
                        chunk=ec,
                        final_score=0.35,
                        semantic_score=0.35,
                        keyword_score=0.35,
                        fuzzy_score=0.0,
                        heading_score=0.0,
                        query_coverage=0.5,
                        independent_signals_count=2,
                        profile=profile
                    ))

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
        Executes end-to-end intelligent RAG reasoning pipeline:
          1. Resolve isolated tenant namespace.
          2. Check ultra-low-latency in-memory response cache for identical concurrent queries.
          3. Query normalization and language analysis.
          4. Structured conversational context & coreference resolution.
          5. Intent & Entity dynamic extraction.
          6. Deterministic reasoning branches (Calculator, Comparator, Temporal, Logical, Multi-Hop).
          7. Hybrid multi-branch retrieval (Vector, BM25, Exact, Entity, Table, Fuzzy).
          8. Candidate fusion & hierarchical parent/child expansion.
          9. Calibrated evidence validation & answer plan composition.
          10. Structured observability timing & conversation session memory.
        """
        t_start = time.perf_counter()
        timing_breakdown: Dict[str, float] = {
            "query_normalization_ms": 0.0,
            "context_resolution_ms": 0.0,
            "intent_detection_ms": 0.0,
            "entity_detection_ms": 0.0,
            "decomposition_ms": 0.0,
            "expansion_ms": 0.0,
            "embedding_ms": 0.0,
            "vector_search_ms": 0.0,
            "bm25_ms": 0.0,
            "exact_search_ms": 0.0,
            "entity_search_ms": 0.0,
            "table_search_ms": 0.0,
            "fuzzy_search_ms": 0.0,
            "fusion_ms": 0.0,
            "reranking_ms": 0.0,
            "graph_build_ms": 0.0,
            "validation_ms": 0.0,
            "reasoning_ms": 0.0,
            "answer_composition_ms": 0.0,
            "total_ms": 0.0
        }

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
            timing_breakdown["total_ms"] = round((time.perf_counter() - t_start) * 1000, 3)
            return AssembledResponse(
                answer="[Cloud Database Disconnected] Neon PostgreSQL connection is required. All local data has been removed. Please verify DATABASE_URL in .env to connect to the cloud database.",
                confidence="VERY_LOW",
                confidence_score=0.0,
                sources=[],
                timing_breakdown=timing_breakdown
            )

        tenant = self.tenant_manager.get_tenant(t_id)
        if not tenant.chunks:
            tenant.load()
        if not tenant.chunks:
            timing_breakdown["total_ms"] = round((time.perf_counter() - t_start) * 1000, 3)
            return AssembledResponse(
                answer=NO_ANSWER_FOUND_MESSAGE,
                confidence="VERY_LOW",
                confidence_score=0.0,
                sources=[],
                debug_trace={"empty_vault": True} if include_debug else None,
                timing_breakdown=timing_breakdown
            )
        chunk_by_id = {c.chunk_id: c for c in tenant.chunks}

        # 1. High-throughput in-memory MultiTierCache lookup (<0.001ms)
        if not filter_doc_id and not filter_section and not include_debug:
            cached_resp = self.multi_cache.get_response(t_id, question)
            if cached_resp is not None:
                return cached_resp

        # 2. Query Normalization (<0.1ms)
        t_norm = time.perf_counter()
        norm_res = self.query_analyzer.normalizer.normalize(question)
        timing_breakdown["query_normalization_ms"] = round((time.perf_counter() - t_norm) * 1000, 3)

        # 3. Conversational Context & Coreference Resolution (<0.1ms)
        t_ctx = time.perf_counter()
        analyzed_q_pre = self.query_analyzer.analyze(question, vocab=tenant.dynamic_vocab)
        search_query, was_enriched, clarification = self.context_manager.enrich_query(
            tenant_id=t_id,
            user_id=u_id,
            session_id=sess_id,
            analyzed_query=analyzed_q_pre
        )
        timing_breakdown["context_resolution_ms"] = round((time.perf_counter() - t_ctx) * 1000, 3)

        if clarification:
            timing_breakdown["total_ms"] = round((time.perf_counter() - t_start) * 1000, 3)
            return AssembledResponse(
                answer=clarification,
                confidence="HIGH",
                confidence_score=1.0,
                sources=[],
                debug_trace={"conversational_ambiguity": True} if include_debug else None,
                timing_breakdown=timing_breakdown
            )

        # 4. Intent & Entity Detection (<0.2ms)
        t_intent = time.perf_counter()
        analyzed_q = self.query_analyzer.analyze(search_query, vocab=tenant.dynamic_vocab)
        timing_breakdown["intent_detection_ms"] = round((time.perf_counter() - t_intent) * 1000, 3)

        t_entity = time.perf_counter()
        plan = self.planner.plan(search_query)
        timing_breakdown["entity_detection_ms"] = round((time.perf_counter() - t_entity) * 1000, 3)

        # 5. Direct Deterministic Math Calculation (Zero LLM / Zero Document I/O)
        t_math = time.perf_counter()
        if plan.requires_calculation or any(w in question.lower() for w in ["gst", "tax", "discount", "average", "ratio", "+", "-", "*", "/"]):
            direct_math = self.calculator.evaluate_query_math(question, "")
            if direct_math:
                timing_breakdown["reasoning_ms"] = round((time.perf_counter() - t_math) * 1000, 3)
                timing_breakdown["total_ms"] = round((time.perf_counter() - t_start) * 1000, 3)
                direct_res = AssembledResponse(
                    answer=direct_math,
                    confidence="HIGH",
                    confidence_score=1.0,
                    sources=[],
                    answer_type="calculation",
                    reasoning={"type": "calculation", "formula": "deterministic_arithmetic"},
                    debug_trace={"deterministic_calculation": True, "timing_breakdown": timing_breakdown} if include_debug else {"timing_breakdown": timing_breakdown},
                    timing_breakdown=timing_breakdown
                )
                self.multi_cache.set_response(t_id, question, direct_res)
                return direct_res

        # 6. Direct Deterministic Comparison (A vs B)
        if plan.requires_comparison and len(plan.comparison_targets) == 2:
            t_comp = time.perf_counter()
            target_a, target_b = plan.comparison_targets
            chunks_a = self._retrieve_for_sub_query(target_a, tenant, QueryProfile.EXACT_CODE, [], filter_doc_id, filter_section, top_k=2, timing_breakdown=timing_breakdown)
            chunks_b = self._retrieve_for_sub_query(target_b, tenant, QueryProfile.EXACT_CODE, [], filter_doc_id, filter_section, top_k=2, timing_breakdown=timing_breakdown)
            if chunks_a and chunks_b and chunks_a[0].final_score >= 0.30 and chunks_b[0].final_score >= 0.30:
                comp_ans = self.comparator.compare_entities(target_a, target_b, [chunks_a[0].chunk], [chunks_b[0].chunk])
                sources = [
                    AssembledSource(document=chunks_a[0].chunk.document_name, document_id=chunks_a[0].chunk.document_id, page=chunks_a[0].chunk.page, section=chunks_a[0].chunk.section, score=round(chunks_a[0].final_score, 4), text=chunks_a[0].chunk.text, chunk_id=chunks_a[0].chunk.chunk_id),
                    AssembledSource(document=chunks_b[0].chunk.document_name, document_id=chunks_b[0].chunk.document_id, page=chunks_b[0].chunk.page, section=chunks_b[0].chunk.section, score=round(chunks_b[0].final_score, 4), text=chunks_b[0].chunk.text, chunk_id=chunks_b[0].chunk.chunk_id)
                ]
                timing_breakdown["reasoning_ms"] = round((time.perf_counter() - t_comp) * 1000, 3)
                timing_breakdown["total_ms"] = round((time.perf_counter() - t_start) * 1000, 3)
                comp_res = AssembledResponse(
                    answer=comp_ans,
                    confidence="HIGH",
                    confidence_score=round((chunks_a[0].final_score + chunks_b[0].final_score) / 2.0, 4),
                    sources=sources,
                    answer_type="comparison",
                    reasoning={"type": "comparison", "target_a": target_a, "target_b": target_b},
                    debug_trace={"deterministic_comparison": True, "timing_breakdown": timing_breakdown} if include_debug else {"timing_breakdown": timing_breakdown},
                    timing_breakdown=timing_breakdown
                )
                self.multi_cache.set_response(t_id, question, comp_res)
                return comp_res

        # 7. Deterministic Temporal & Sequential Reasoning
        t_temp = time.perf_counter()
        temp_intent = self.temporal_reasoner.detect_temporal_intent(search_query)
        if temp_intent:
            relation, target_subject = temp_intent
            temp_scored = self._retrieve_for_sub_query(
                sub_query=target_subject,
                tenant=tenant,
                profile=QueryProfile.EXACT_CODE,
                target_attributes=[],
                filter_doc_id=filter_doc_id,
                filter_section=filter_section,
                top_k=top_k,
                timing_breakdown=timing_breakdown
            )
            candidate_chunks = [sc.chunk for sc in temp_scored if sc.final_score >= 0.20]
            if candidate_chunks:
                temp_res = self.temporal_reasoner.reason_sequence(target_subject, relation, candidate_chunks)
                if temp_res.evidence_found:
                    formatted_temp = temp_res.to_formatted_answer()
                    temp_sources = []
                    seen_cids = set()
                    for ev in temp_res.ordered_events:
                        if ev.evidence_chunk_id and ev.evidence_chunk_id in chunk_by_id and ev.evidence_chunk_id not in seen_cids:
                            seen_cids.add(ev.evidence_chunk_id)
                            c = chunk_by_id[ev.evidence_chunk_id]
                            temp_sources.append(AssembledSource(
                                document=c.document_name,
                                document_id=c.document_id,
                                page=c.page,
                                section=c.section,
                                score=0.85,
                                text=ev.raw_sentence or c.text,
                                chunk_id=c.chunk_id
                            ))
                    timing_breakdown["reasoning_ms"] = round((time.perf_counter() - t_temp) * 1000, 3)
                    timing_breakdown["total_ms"] = round((time.perf_counter() - t_start) * 1000, 3)
                    temp_response = AssembledResponse(
                        answer=formatted_temp,
                        confidence="HIGH",
                        confidence_score=temp_res.confidence,
                        sources=temp_sources,
                        answer_type="workflow",
                        reasoning={"type": "temporal", "relation": relation, "steps": len(temp_res.direct_answers)},
                        debug_trace={"temporal_reasoning": True, "timing_breakdown": timing_breakdown} if include_debug else {"timing_breakdown": timing_breakdown},
                        timing_breakdown=timing_breakdown
                    )
                    self.multi_cache.set_response(t_id, question, temp_response)
                    return temp_response

        # 8. Deterministic Logical Reasoning
        t_logic = time.perf_counter()
        logic_intent = self.logical_reasoner.detect_logical_intent(search_query)
        if logic_intent:
            log_scored = self._retrieve_for_sub_query(
                sub_query=search_query,
                tenant=tenant,
                profile=analyzed_q.profile,
                target_attributes=plan.target_attributes,
                filter_doc_id=filter_doc_id,
                filter_section=filter_section,
                top_k=top_k,
                timing_breakdown=timing_breakdown
            )
            candidate_chunks = [sc.chunk for sc in log_scored if sc.final_score >= 0.20]
            if candidate_chunks:
                logic_res = self.logical_reasoner.evaluate_logic(search_query, candidate_chunks)
                if logic_res.supported:
                    log_sources = []
                    for cl in logic_res.evidence_claims:
                        for c in candidate_chunks:
                            if cl in c.text:
                                log_sources.append(AssembledSource(
                                    document=c.document_name,
                                    document_id=c.document_id,
                                    page=c.page,
                                    section=c.section,
                                    score=0.85,
                                    text=cl,
                                    chunk_id=c.chunk_id
                                ))
                                break
                    timing_breakdown["reasoning_ms"] = round((time.perf_counter() - t_logic) * 1000, 3)
                    timing_breakdown["total_ms"] = round((time.perf_counter() - t_start) * 1000, 3)
                    logic_response = AssembledResponse(
                        answer=logic_res.to_formatted_answer(),
                        confidence="HIGH",
                        confidence_score=0.85,
                        sources=log_sources or [AssembledSource(
                            document=candidate_chunks[0].document_name,
                            document_id=candidate_chunks[0].document_id,
                            page=candidate_chunks[0].page,
                            section=candidate_chunks[0].section,
                            score=0.85,
                            text=candidate_chunks[0].text,
                            chunk_id=candidate_chunks[0].chunk_id
                        )],
                        answer_type="explanation",
                        reasoning={"type": "logical", "verdict": logic_res.verdict},
                        debug_trace={"logical_reasoning": True, "timing_breakdown": timing_breakdown} if include_debug else {"timing_breakdown": timing_breakdown},
                        timing_breakdown=timing_breakdown
                    )
                    self.multi_cache.set_response(t_id, question, logic_response)
                    return logic_response

        debug_trace: Dict[str, Any] = {}
        if include_debug:
            debug_trace["raw_query"] = question
            debug_trace["search_query"] = search_query
            debug_trace["profile"] = analyzed_q.profile.value
            debug_trace["was_enriched"] = was_enriched
            debug_trace["sub_queries"] = analyzed_q.sub_queries

        # 9. Multi-Part vs Single-Query Retrieval
        if analyzed_q.profile == QueryProfile.MULTI_PART and len(analyzed_q.sub_queries) > 1:
            t_decomp = time.perf_counter()
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
                    analyzed_q=sub_analyzed,
                    timing_breakdown=timing_breakdown
                )
                sub_results.append((sub_q, sub_scored, sub_plan))

            timing_breakdown["decomposition_ms"] = round((time.perf_counter() - t_decomp) * 1000, 3)

            t_comp = time.perf_counter()
            response = self.assembler.assemble_multi_part(
                sub_query_results=sub_results,
                chunk_by_id=chunk_by_id,
                thresholds=tenant.thresholds,
                debug_trace=debug_trace if include_debug else None,
            )
            timing_breakdown["answer_composition_ms"] = round((time.perf_counter() - t_comp) * 1000, 3)
            response.answer_type = "multi_part"
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
                analyzed_q=analyzed_q,
                timing_breakdown=timing_breakdown
            )

            t_comp = time.perf_counter()
            response = self.assembler.assemble(
                ranked_chunks=scored_chunks,
                chunk_by_id=chunk_by_id,
                debug_trace=debug_trace if include_debug else None,
                max_sources=top_k,
                thresholds=tenant.thresholds,
                profile=analyzed_q.profile,
                plan=plan
            )
            timing_breakdown["answer_composition_ms"] = round((time.perf_counter() - t_comp) * 1000, 3)
            response.answer_type = "explanation" if plan.intent not in ("definition", "workflow") else plan.intent

        # 10. Evidence Graph Construction (<0.1ms)
        t_graph = time.perf_counter()
        if response.sources and response.confidence != "VERY_LOW":
            claim_tuples = []
            for s in response.sources[:3]:
                if s.chunk_id in chunk_by_id:
                    claim_tuples.append((response.answer[:80], chunk_by_id[s.chunk_id], s.score))
            if claim_tuples:
                self.evidence_graph_builder.build_evidence_graph(claim_tuples)
        timing_breakdown["graph_build_ms"] = round((time.perf_counter() - t_graph) * 1000, 3)

        # 11. Finalize timing and rich metadata
        timing_breakdown["total_ms"] = round((time.perf_counter() - t_start) * 1000, 3)
        response.timing_breakdown = timing_breakdown
        if response.debug_trace is None:
            response.debug_trace = {}
        response.debug_trace["timing_breakdown"] = timing_breakdown

        # 12. Store in high-speed MultiTierCache if not enriched by conversational context
        if not was_enriched and response.confidence != "VERY_LOW":
            self.multi_cache.set_response(t_id, question, response)

        # 13. Record turn into session memory
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
