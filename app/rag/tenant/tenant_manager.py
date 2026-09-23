import os
import json
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from app.rag.config import DATA_DIR, OLLAMA_EMBED_MODEL
from app.rag.ingestion.models import DocumentChunk
from app.rag.retrieval.vector_search import VectorSearchEngine
from app.rag.retrieval.bm25_search import BM25SearchEngine
from app.rag.retrieval.fuzzy_search import FuzzySearchEngine
from app.rag.retrieval.exact_search import ExactSearchEngine
from app.rag.retrieval.entity_search import EntitySearchEngine
from app.rag.retrieval.table_search import TableSearchEngine
from app.rag.vocabulary.dynamic_vocabulary import DynamicCorpusVocabulary

TENANTS_DIR = DATA_DIR / "tenants"

# Calibrated defaults when no specific calibration file exists yet
DEFAULT_CALIBRATED_THRESHOLDS: Dict[str, float] = {
    "min_semantic_pass": 0.65,          # High semantic similarity guarantees grounding even with 0 BM25
    "min_semantic_cutoff": 0.38,        # Below this, passage is unlikely relevant
    "min_bm25_cutoff": 0.10,            # Minimal lexical signal required if semantic is weak
    "min_concept_coverage": 0.50,       # At least 50% of query concepts matched
    "score_margin_boost": 0.08          # Top candidate significantly ahead of second rank
}

class TenantIndex:
    """Isolated storage, indices, and vocabulary for a single tenant."""

    def __init__(self, tenant_id: str, tenant_dir: Path):
        self.tenant_id = tenant_id
        self.tenant_dir = tenant_dir
        self.index_file = tenant_dir / "index.json"
        self.thresholds_file = tenant_dir / "calibrated_thresholds.json"

        self.documents: Dict[str, Dict[str, Any]] = {}
        self.chunks: List[DocumentChunk] = []
        self.embeddings: List[List[float]] = []

        # Isolated engines per tenant (Zero IDF or vector contamination)
        self.vector_engine = VectorSearchEngine()
        self.bm25_engine = BM25SearchEngine()
        self.fuzzy_engine = FuzzySearchEngine()
        self.exact_engine = ExactSearchEngine()
        self.entity_engine = EntitySearchEngine()
        self.table_engine = TableSearchEngine()
        self.dynamic_vocab = DynamicCorpusVocabulary()
        self.thresholds = dict(DEFAULT_CALIBRATED_THRESHOLDS)

    def load(self):
        """Loads tenant data strictly from Cloud PostgreSQL. If cloud is disconnected, no chunks are loaded."""
        if self.thresholds_file.exists():
            try:
                with open(self.thresholds_file, "r", encoding="utf-8") as f:
                    self.thresholds.update(json.load(f))
            except Exception as e:
                print(f"[TenantIndex:{self.tenant_id}] Error loading thresholds: {e}")

        # Cloud-First Loading: Load documents, chunks, and vectors directly from Neon Cloud PostgreSQL
        try:
            import numpy as np
            from app.storage_postgres import get_postgres_store
            pg_store = get_postgres_store()
            if pg_store and pg_store.is_connected_fast():
                cloud_docs, raw_chunks, _ = pg_store.load_tenant_data(self.tenant_id)
                self.documents = cloud_docs
                self.chunks = [DocumentChunk.from_dict(c) for c in raw_chunks]

                # Synchronize dense vector embeddings (1ms RAM load from local projection, backed by Neon Cloud)
                cache_file = Path("data/cloud_vector_cache.npz")
                loaded_vectors = False
                if cache_file.exists() and len(self.chunks) > 0:
                    try:
                        cdata = np.load(cache_file)
                        cached_mat = cdata["embeddings"]
                        cached_ids = cdata["chunk_ids"].tolist() if hasattr(cdata["chunk_ids"], "tolist") else list(cdata["chunk_ids"])
                        current_ids = [c.chunk_id for c in self.chunks]
                        if len(cached_mat) == len(self.chunks) and cached_ids == current_ids:
                            self.embeddings = cached_mat
                            loaded_vectors = True
                    except Exception as ex:
                        print(f"[TenantIndex:{self.tenant_id}] Vector cache notice: {ex}")

                if not loaded_vectors and len(self.chunks) > 0:
                    cloud_mat, cloud_ids = pg_store.get_tenant_vectors(self.tenant_id)
                    current_ids = [c.chunk_id for c in self.chunks]
                    if cloud_mat is not None and cloud_ids == current_ids and len(cloud_mat) == len(self.chunks):
                        self.embeddings = cloud_mat
                        loaded_vectors = True
                        try:
                            cache_file.parent.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(cache_file, embeddings=np.asarray(self.embeddings, dtype=np.float32), chunk_ids=current_ids)
                        except Exception:
                            pass

                if not loaded_vectors and len(self.chunks) > 0:
                    try:
                        from app.rag.embeddings.ollama_embedder import get_embedding_engine
                        embedder = get_embedding_engine()
                        texts = [f"search_document: {c.section}: {c.text}" if c.section else f"search_document: {c.text}" for c in self.chunks]
                        self.embeddings = embedder.embed_batch(texts)
                        current_ids = [c.chunk_id for c in self.chunks]
                        pg_store.save_tenant_vectors(self.tenant_id, current_ids, self.embeddings)
                        try:
                            cache_file.parent.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(cache_file, embeddings=np.asarray(self.embeddings, dtype=np.float32), chunk_ids=current_ids)
                        except Exception:
                            pass
                    except Exception as ex:
                        print(f"[TenantIndex:{self.tenant_id}] Vector generation notice: {ex}")

                # Rebuild in-memory indices and dynamic vocabulary from cloud data
                self.vector_engine.index_chunks(self.chunks, self.embeddings)
                self.bm25_engine.index_chunks(self.chunks)
                self.fuzzy_engine.index_chunks(self.chunks)
                self.exact_engine.index_chunks(self.chunks)
                self.entity_engine.index_chunks(self.chunks)
                self.table_engine.index_chunks(self.chunks)
                self.dynamic_vocab.build_from_chunks(self.chunks)

                print(f"[TenantIndex:{self.tenant_id}] Loaded {len(self.chunks)} chunks across {len(self.documents)} documents from Neon Cloud PostgreSQL (100% in RAM).")
                return
            else:
                print(f"[TenantIndex:{self.tenant_id}] Cloud database is disconnected. Purging all runtime data.")
                cache_file = Path("data/cloud_vector_cache.npz")
                if cache_file.exists():
                    try:
                        cache_file.unlink()
                    except Exception:
                        pass
                self.documents = {}
                self.chunks = []
                self.embeddings = []
                self.vector_engine.index_chunks([], [])
                self.bm25_engine.index_chunks([])
                self.fuzzy_engine.index_chunks([])
                self.exact_engine.index_chunks([])
                self.entity_engine.index_chunks([])
                self.table_engine.index_chunks([])
                return
        except Exception as e:
            print(f"[TenantIndex:{self.tenant_id}] Cloud database notice: {e}. Purging all runtime data.")
            cache_file = Path("data/cloud_vector_cache.npz")
            if cache_file.exists():
                try:
                    cache_file.unlink()
                except Exception:
                    pass
            self.documents = {}
            self.chunks = []
            self.embeddings = []
            self.vector_engine.index_chunks([], [])
            self.bm25_engine.index_chunks([])
            self.fuzzy_engine.index_chunks([])
            self.exact_engine.index_chunks([])
            self.entity_engine.index_chunks([])
            self.table_engine.index_chunks([])
            return

    def save(self):
        """Syncs document chunks directly to Cloud PostgreSQL."""
        self.tenant_dir.mkdir(parents=True, exist_ok=True)
        # Save thresholds
        with open(self.thresholds_file, "w", encoding="utf-8") as f:
            json.dump(self.thresholds, f, indent=2)

        # Sync to Cloud PostgreSQL
        try:
            from app.storage_postgres import get_postgres_store
            pg_store = get_postgres_store()
            has_embs = self.embeddings is not None and len(self.embeddings) > 0
            if pg_store and len(self.chunks) > 0:
                # 1. Insert/Update documents and chunks in Cloud PostgreSQL
                chunks_by_doc = {}
                for idx, c in enumerate(self.chunks):
                    emb = self.embeddings[idx] if (has_embs and idx < len(self.embeddings)) else None
                    if emb is not None:
                        c_dict = c.to_dict()
                        c_dict["embedding"] = emb.tolist() if hasattr(emb, "tolist") else emb
                        chunks_by_doc.setdefault(c.document_id, []).append(c_dict)

                for did, dmeta in self.documents.items():
                    d_chunks = chunks_by_doc.get(did, [])
                    if d_chunks:
                        pg_store.insert_document_and_chunks(
                            doc_id=did,
                            filename=dmeta.get("filename", did),
                            tenant_id=self.tenant_id,
                            page_count=dmeta.get("page_count", 1),
                            chunks=d_chunks
                        )

                # 2. Save compressed vector matrix to Cloud PostgreSQL
                if has_embs:
                    current_ids = [c.chunk_id for c in self.chunks]
                    pg_store.save_tenant_vectors(self.tenant_id, current_ids, self.embeddings)
        except Exception as pg_err:
            print(f"[TenantIndex:{self.tenant_id}] Warning: Cloud PostgreSQL sync: {pg_err}")

    def reindex(self):
        """Rebuilds in-memory search structures."""
        self.vector_engine.index_chunks(self.chunks, self.embeddings)
        self.bm25_engine.index_chunks(self.chunks)
        self.fuzzy_engine.index_chunks(self.chunks)
        self.exact_engine.index_chunks(self.chunks)
        self.entity_engine.index_chunks(self.chunks)
        self.table_engine.index_chunks(self.chunks)
        self.dynamic_vocab.build_from_chunks(self.chunks)

    def delete_document(self, identifier: str) -> bool:
        """
        Permanently deletes a document from this tenant in RAM.
        Matches by doc_id (exact or case-insensitive), filename, or filename stem.
        """
        import numpy as np
        from pathlib import Path
        clean_id = identifier.strip().lower()
        clean_stem = Path(clean_id).stem.lower()

        target_doc_ids = set()
        for did, dmeta in list(self.documents.items()):
            fn = dmeta.get("filename", "").lower()
            fn_stem = Path(fn).stem.lower()
            if (
                did.lower() == clean_id
                or did.lower() == clean_stem
                or fn == clean_id
                or fn_stem == clean_stem
                or fn.replace("_", " ") == clean_id.replace("_", " ")
                or fn.replace(" ", "_") == clean_id.replace(" ", "_")
                or fn_stem.replace("_", " ") == clean_stem.replace("_", " ")
                or fn_stem.replace(" ", "_") == clean_stem.replace(" ", "_")
            ):
                target_doc_ids.add(did)

        # Fallback: check chunks for any matching document_id
        if not target_doc_ids:
            for c in self.chunks:
                if c.document_id and (
                    c.document_id.lower() == clean_id 
                    or c.document_id.lower() == clean_stem
                    or (c.document_name and c.document_name.lower() == clean_id)
                ):
                    target_doc_ids.add(c.document_id)

        if not target_doc_ids:
            return False

        # 1. Remove from self.documents
        for did in target_doc_ids:
            self.documents.pop(did, None)

        # 2. Filter self.chunks and self.embeddings in lockstep
        valid_indices = [i for i, c in enumerate(self.chunks) if c.document_id not in target_doc_ids]
        self.chunks = [self.chunks[i] for i in valid_indices]

        if self.embeddings is not None and len(self.embeddings) > 0:
            if isinstance(self.embeddings, np.ndarray):
                self.embeddings = self.embeddings[valid_indices] if valid_indices else np.empty((0, 768), dtype=np.float32)
            else:
                self.embeddings = [self.embeddings[i] for i in valid_indices if i < len(self.embeddings)]
        else:
            self.embeddings = []

        # 3. Rebuild in-memory search engines (all engines will be clean/empty if chunks is empty)
        self.reindex()

        # 4. Invalidate local file vector cache
        try:
            cache_file = Path("data/cloud_vector_cache.npz")
            if cache_file.exists():
                cache_file.unlink()
        except Exception:
            pass

        return True


class TenantManager:
    """
    Multi-tenant isolation coordinator.
    Ensures that every tenant's dense vectors, BM25 inverted indices, term frequencies,
    acronyms, and vocabulary are hard-partitioned with zero cross-tenant contamination.
    """

    def __init__(self, tenants_dir: Path = TENANTS_DIR):
        self.tenants_dir = tenants_dir
        self.lock = threading.RLock()
        self.tenants: Dict[str, TenantIndex] = {}
        self._init_storage()

    def _init_storage(self):
        """Initializes tenants and loads partitions from Cloud PostgreSQL."""
        self.tenants_dir.mkdir(parents=True, exist_ok=True)
        default_dir = self.tenants_dir / "default"
        default_dir.mkdir(parents=True, exist_ok=True)

        with self.lock:
            # Ensure default tenant is loaded from Cloud PostgreSQL
            t_index = TenantIndex(tenant_id="default", tenant_dir=default_dir)
            t_index.load()
            self.tenants["default"] = t_index

    def get_tenant(self, tenant_id: Optional[str] = None) -> TenantIndex:
        """Retrieves or creates isolated tenant namespace."""
        t_id = (tenant_id or "default").strip().lower()
        with self.lock:
            if t_id not in self.tenants:
                t_dir = self.tenants_dir / t_id
                t_dir.mkdir(parents=True, exist_ok=True)
                t_index = TenantIndex(tenant_id=t_id, tenant_dir=t_dir)
                t_index.load()
                self.tenants[t_id] = t_index
            return self.tenants[t_id]

    def list_tenants(self) -> List[str]:
        with self.lock:
            return list(self.tenants.keys())

    def delete_document(self, identifier: str, tenant_id: Optional[str] = None) -> bool:
        """Deletes a document from the specified tenant (or all active tenants if not specified)."""
        with self.lock:
            if tenant_id:
                t_id = tenant_id.strip().lower()
                if t_id in self.tenants:
                    return self.tenants[t_id].delete_document(identifier)
                return False
            deleted_any = False
            for t_id in list(self.tenants.keys()):
                if self.tenants[t_id].delete_document(identifier):
                    deleted_any = True
            return deleted_any
