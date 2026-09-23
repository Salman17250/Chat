from typing import List, Dict, Any, Optional, Tuple
from app.rag.ingestion.models import DocumentChunk
from app.rag.embeddings.ollama_embedder import OllamaEmbedder
from app.rag.vocabulary.dynamic_vocabulary import DynamicCorpusVocabulary
from app.rag.query.normalizer import QueryNormalizer
from app.rag.query.expansion import SynonymExpander
from app.rag.query.phrase_detector import PhraseDetector
from app.rag.query.decomposer import QueryDecomposer
from app.rag.query.intent import RuleBasedIntentDetector, QueryIntent
from app.rag.retrieval.vector_search import VectorSearchEngine
from app.rag.retrieval.bm25_search import BM25SearchEngine
from app.rag.retrieval.fuzzy_search import FuzzySearchEngine
from app.rag.ranking.deterministic_scorer import DeterministicScorer, ScoredChunk
from app.rag.config import (
    TOP_K_VECTOR,
    TOP_K_KEYWORD,
    TOP_K_FUZZY,
    INITIAL_CANDIDATE_POOL,
    FINAL_TOP_K,
)

class HybridRetrievalEngine:
    """
    Advanced Multi-Stage Hybrid Retrieval Engine.
    Supports dynamic corpus vocabulary, multi-part query decomposition,
    strict score normalization, and comprehensive explainability traces.
    """

    def __init__(
        self,
        embedder: Optional[OllamaEmbedder] = None,
        normalizer: Optional[QueryNormalizer] = None,
        vocab: Optional[DynamicCorpusVocabulary] = None,
        expander: Optional[SynonymExpander] = None,
        phrase_detector: Optional[PhraseDetector] = None,
        decomposer: Optional[QueryDecomposer] = None,
        intent_detector: Optional[RuleBasedIntentDetector] = None,
        scorer: Optional[DeterministicScorer] = None,
    ):
        self.embedder = embedder or OllamaEmbedder()
        self.normalizer = normalizer or QueryNormalizer()
        self.vocab = vocab or DynamicCorpusVocabulary()
        self.expander = expander or SynonymExpander(vocab=self.vocab)
        self.phrase_detector = phrase_detector or PhraseDetector(vocab=self.vocab)
        self.decomposer = decomposer or QueryDecomposer()
        self.intent_detector = intent_detector or RuleBasedIntentDetector()
        self.scorer = scorer or DeterministicScorer()

        self.vector_engine = VectorSearchEngine()
        self.bm25_engine = BM25SearchEngine(normalizer=self.normalizer)
        self.fuzzy_engine = FuzzySearchEngine(normalizer=self.normalizer)

        self.all_chunks: List[DocumentChunk] = []
        self.chunk_by_id: Dict[str, DocumentChunk] = {}

    def index_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        """Builds all vector, lexical, fuzzy, and dynamic vocabulary search indexes."""
        self.all_chunks = list(chunks)
        self.chunk_by_id = {c.chunk_id: c for c in self.all_chunks}

        # 1. Build dynamic document vocabulary from chunks
        self.vocab.build_from_chunks(self.all_chunks)
        self.expander.set_vocabulary(self.vocab)
        self.phrase_detector.set_vocabulary(self.vocab)

        # 2. Build search engines
        self.vector_engine.index_chunks(self.all_chunks, embeddings)
        self.bm25_engine.index_chunks(self.all_chunks)
        self.fuzzy_engine.index_chunks(self.all_chunks)

    def _retrieve_single_query(
        self,
        query: str,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        filter_doc_id: Optional[str] = None,
        filter_section: Optional[str] = None,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """Retrieves and merges candidates for a single query clause."""
        norm_res = self.normalizer.normalize(query)
        clean_query = norm_res.normalized_text
        query_tokens = norm_res.tokens

        # Check for typo correction against corpus vocabulary
        corrected_tokens = []
        for tok in query_tokens:
            fix = self.vocab.correct_typo(tok)
            corrected_tokens.append(fix if fix else tok)

        # Synonym / Term Expansion using Dynamic Document Vocabulary
        expanded_query, expanded_terms = self.expander.expand_query(clean_query)

        # Dynamic Phrase Detection from corpus vocabulary & quotes
        detected_phrases = self.phrase_detector.detect_phrases(query, tokens=corrected_tokens)

        # Intent Detection
        intent, intent_conf = self.intent_detector.detect_intent(query)

        # 1. Dense Semantic Vector search
        query_vector = self.embedder.embed_text(clean_query)
        vector_results = self.vector_engine.search(query_vector, top_k=TOP_K_VECTOR)

        # 2. BM25 Lexical Keyword search
        bm25_search_tokens = list(corrected_tokens)
        for term in expanded_terms:
            for word in term.split():
                if word not in bm25_search_tokens:
                    bm25_search_tokens.append(word)
        bm25_results = self.bm25_engine.search(
            bm25_search_tokens,
            top_k=TOP_K_KEYWORD,
            base_query_tokens=corrected_tokens
        )

        # 3. Fuzzy search for spelling resilience
        fuzzy_results = self.fuzzy_engine.search(query, top_k=TOP_K_FUZZY)

        # Candidate Merging with Tenant/User Isolation
        candidate_dict: Dict[str, Dict[str, Any]] = {}

        def record(chunk: DocumentChunk, sem: float = 0.0, kw: float = 0.0, fz: float = 0.0):
            if tenant_id and chunk.tenant_id and chunk.tenant_id != tenant_id:
                return
            if user_id and chunk.user_id and chunk.user_id != user_id:
                return
            if filter_doc_id and chunk.document_id != filter_doc_id:
                return
            if filter_section and chunk.section and filter_section.lower() not in chunk.section.lower():
                return

            cid = chunk.chunk_id
            if cid not in candidate_dict:
                candidate_dict[cid] = {
                    "chunk": chunk,
                    "semantic_score": sem,
                    "keyword_score": kw,
                    "fuzzy_score": fz,
                }
            else:
                candidate_dict[cid]["semantic_score"] = max(candidate_dict[cid]["semantic_score"], sem)
                candidate_dict[cid]["keyword_score"] = max(candidate_dict[cid]["keyword_score"], kw)
                candidate_dict[cid]["fuzzy_score"] = max(candidate_dict[cid]["fuzzy_score"], fz)

        for c, s in vector_results:
            record(c, sem=s)
        for c, s in bm25_results:
            record(c, kw=s)
        for c, s in fuzzy_results:
            record(c, fz=s)

        clause_meta = {
            "query": query,
            "normalized_query": clean_query,
            "query_tokens": corrected_tokens,
            "expanded_terms": expanded_terms,
            "detected_phrases": detected_phrases,
            "intent": intent,
            "intent_confidence": intent_conf,
        }

        return candidate_dict, clause_meta

    def retrieve(
        self,
        query: str,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        filter_doc_id: Optional[str] = None,
        filter_section: Optional[str] = None,
        top_k: int = FINAL_TOP_K,
        include_debug: bool = False,
    ) -> Tuple[List[ScoredChunk], Dict[str, Any]]:
        """
        Executes multi-stage hybrid retrieval.
        Supports query decomposition for multi-part questions, merging evidence across clauses.
        """
        debug_trace: Dict[str, Any] = {}

        # 1. Query Decomposition (Requirement 4)
        sub_queries = self.decomposer.decompose(query)
        is_multi_part = len(sub_queries) > 1

        all_candidate_dicts: List[Dict[str, Dict[str, Any]]] = []
        clause_metas: List[Dict[str, Any]] = []

        for sub_q in sub_queries:
            c_dict, c_meta = self._retrieve_single_query(
                query=sub_q,
                tenant_id=tenant_id,
                user_id=user_id,
                filter_doc_id=filter_doc_id,
                filter_section=filter_section,
            )
            all_candidate_dicts.append(c_dict)
            clause_metas.append(c_meta)

        # Merge candidates across sub-queries
        merged_candidates: Dict[str, Dict[str, Any]] = {}
        for sub_idx, c_dict in enumerate(all_candidate_dicts):
            for cid, c_data in c_dict.items():
                if cid not in merged_candidates:
                    merged_candidates[cid] = {
                        "chunk": c_data["chunk"],
                        "semantic_score": c_data["semantic_score"],
                        "keyword_score": c_data["keyword_score"],
                        "fuzzy_score": c_data["fuzzy_score"],
                        "sub_query_matches": {sub_idx},
                    }
                else:
                    merged_candidates[cid]["semantic_score"] = max(
                        merged_candidates[cid]["semantic_score"], c_data["semantic_score"]
                    )
                    merged_candidates[cid]["keyword_score"] = max(
                        merged_candidates[cid]["keyword_score"], c_data["keyword_score"]
                    )
                    merged_candidates[cid]["fuzzy_score"] = max(
                        merged_candidates[cid]["fuzzy_score"], c_data["fuzzy_score"]
                    )
                    merged_candidates[cid]["sub_query_matches"].add(sub_idx)

        # Combine meta information across sub-clauses
        combined_tokens = []
        combined_expanded = []
        combined_phrases = []
        for m in clause_metas:
            combined_tokens.extend(m["query_tokens"])
            combined_expanded.extend(m["expanded_terms"])
            combined_phrases.extend(m["detected_phrases"])

        primary_intent = clause_metas[0]["intent"]

        # Deterministic Reranking
        scored_candidates: List[ScoredChunk] = []
        for cid, item in merged_candidates.items():
            scored = self.scorer.score_candidate(
                chunk=item["chunk"],
                semantic_score=item["semantic_score"],
                keyword_score=item["keyword_score"],
                fuzzy_score=item["fuzzy_score"],
                detected_phrases=combined_phrases,
                query_tokens=combined_tokens,
                expanded_terms=combined_expanded,
                intent=primary_intent,
                filter_doc_id=filter_doc_id,
                filter_section=filter_section,
            )

            # Boost chunk if it addresses multiple parts of a multi-part question
            if is_multi_part and len(item["sub_query_matches"]) > 1:
                scored.final_score = min(1.0, scored.final_score * 1.08)

            scored_candidates.append(scored)

        # Sort descending by composite final_score
        scored_candidates.sort(key=lambda x: x.final_score, reverse=True)
        selected_chunks = scored_candidates[:top_k]

        if include_debug:
            debug_trace["query"] = query
            debug_trace["is_multi_part"] = is_multi_part
            debug_trace["sub_queries"] = sub_queries
            debug_trace["normalized_query"] = " ".join(combined_tokens)
            debug_trace["detected_phrases"] = combined_phrases
            debug_trace["intent"] = primary_intent.value
            debug_trace["results"] = [
                {
                    "document_name": sc.chunk.document_name,
                    "section": sc.chunk.section,
                    "page": sc.chunk.page,
                    "chunk_id": sc.chunk.chunk_id,
                    "vector_score": round(sc.semantic_score, 4),
                    "bm25_score": round(sc.keyword_score, 4),
                    "fuzzy_score": round(sc.fuzzy_score, 4),
                    "phrase_score": round(sc.phrase_score, 4),
                    "heading_score": round(sc.heading_score, 4),
                    "intent_score": round(sc.intent_score, 4),
                    "metadata_score": round(sc.metadata_score, 4),
                    "final_score": round(sc.final_score, 4),
                    "independent_signals": sc.independent_signals_count,
                    "matched_terms": sc.matched_terms,
                    "matched_phrases": sc.matched_phrases,
                }
                for sc in selected_chunks
            ]

        return selected_chunks, debug_trace
