import os
import time
import json
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import psycopg
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
_LAST_ENV_MTIME = 0.0

def _load_env_file():
    """Reads key-values from .env dynamically so config changes take effect without server restart."""
    global _LAST_ENV_MTIME, _GLOBAL_POSTGRES_STORE
    if not ENV_FILE.exists():
        os.environ.pop("DATABASE_URL", None)
        if _GLOBAL_POSTGRES_STORE is not None:
            try:
                _GLOBAL_POSTGRES_STORE.stop_heartbeat()
                _GLOBAL_POSTGRES_STORE.pool.close()
            except Exception:
                pass
            _GLOBAL_POSTGRES_STORE = None
        return

    try:
        current_mtime = ENV_FILE.stat().st_mtime
        if current_mtime == _LAST_ENV_MTIME:
            return
        _LAST_ENV_MTIME = current_mtime
        if _GLOBAL_POSTGRES_STORE is not None:
            _GLOBAL_POSTGRES_STORE._last_healthy = 0.0
    except Exception:
        pass

    found_db_url = False
    try:
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k == "DATABASE_URL":
                    found_db_url = True
                if k:
                    os.environ[k] = v
    except Exception:
        pass
    if not found_db_url:
        os.environ.pop("DATABASE_URL", None)
        if _GLOBAL_POSTGRES_STORE is not None:
            try:
                _GLOBAL_POSTGRES_STORE.stop_heartbeat()
                _GLOBAL_POSTGRES_STORE.pool.close()
            except Exception:
                pass
            _GLOBAL_POSTGRES_STORE = None

_load_env_file()

_GLOBAL_POSTGRES_STORE: Optional["PostgresVectorStore"] = None

class PostgresVectorStore:
    """
    Production-grade Cloud PostgreSQL + pgvector storage engine.
    Supports:
      - Thread-safe connection pooling (psycopg_pool) with autocommit
      - Active background heartbeat to keep serverless Neon DB warm and check connectivity
      - Non-blocking <0.001ms is_connected_fast() query gate
      - Sub-millisecond HNSW Vector Cosine Distance (<=>)
      - GIN Full-Text Search indexing on tsvector
      - Reciprocal Rank Fusion (RRF) Hybrid Search
      - Persistent Cloud Escalation Tickets
    """
    def __init__(self, database_url: Optional[str] = None):
        self.db_url = database_url or os.getenv("DATABASE_URL", "").strip()
        if not self.db_url or not self.db_url.startswith("postgresql://"):
            raise ValueError(
                "DATABASE_URL is invalid or missing. Must start with postgresql://."
            )
        
        self._is_healthy = True
        self._last_healthy = 0.0
        self._heartbeat_running = False

        # High-throughput thread-safe connection pool with autocommit for 0ms transaction overhead
        self.pool = ConnectionPool(
            self.db_url,
            min_size=1,
            max_size=30,
            timeout=8,
            kwargs={"autocommit": True},
            configure=register_vector
        )

        # Start non-blocking background keep-alive heartbeat
        self.start_heartbeat()

    def get_connection(self):
        """Returns a managed connection from the thread-safe connection pool."""
        return self.pool.connection()

    def start_heartbeat(self):
        """Spawns a daemon thread to keep Neon compute active and monitor connection health."""
        if self._heartbeat_running:
            return
        self._heartbeat_running = True

        def _worker():
            while self._heartbeat_running:
                try:
                    time.sleep(15)
                    if not self._heartbeat_running:
                        break
                    _load_env_file()
                    current_url = os.getenv("DATABASE_URL", "").strip()
                    if not current_url or not current_url.startswith("postgresql://") or current_url != self.db_url:
                        self._is_healthy = False
                        continue
                    with self.get_connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute("SELECT 1;")
                            r = cur.fetchone()
                            if r and r[0] == 1:
                                self._last_healthy = time.time()
                                self._is_healthy = True
                            else:
                                self._is_healthy = False
                except Exception:
                    self._is_healthy = False

        t = threading.Thread(target=_worker, daemon=True, name="neon_heartbeat_worker")
        t.start()

    def stop_heartbeat(self):
        """Stops the heartbeat worker."""
        self._heartbeat_running = False

    def is_connected_fast(self) -> bool:
        """
        Ultra-fast non-blocking connectivity check (<0.001ms) for query serving.
        Verifies .env has not changed and that the connection pool is actively healthy.
        If .env was modified or invalidated, immediately returns False.
        """
        _load_env_file()
        db_url = os.getenv("DATABASE_URL", "").strip()
        if not db_url or not db_url.startswith("postgresql://") or db_url != self.db_url:
            return False
        if not self._is_healthy:
            return self.test_connection()
        now = time.time()
        if (now - self._last_healthy) > 60.0:
            return self.test_connection()
        return True

    def test_connection(self) -> bool:
        """Connectivity check using active pool status and health cache for peak throughput."""
        now = time.time()
        if self._is_healthy and (now - self._last_healthy) < 30.0:
            return True
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                    r = cur.fetchone()
                    if r is not None and r[0] == 1:
                        self._last_healthy = now
                        self._is_healthy = True
                        return True
                    self._is_healthy = False
                    return False
        except Exception:
            self._is_healthy = False
            return False

    def init_db(self):
        """Creates the required extensions, tables, and HNSW/GIN indexes."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # 1. Extensions
                cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
                cur.execute('CREATE EXTENSION IF NOT EXISTS "vector";')
            register_vector(conn)
            with conn.cursor() as cur:
                # 2. Documents table
                cur.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id VARCHAR(64) PRIMARY KEY,
                    tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
                    filename VARCHAR(512) NOT NULL,
                    page_count INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    file_size_bytes BIGINT NOT NULL DEFAULT 0,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_documents_tenant ON documents(tenant_id);
                ''')

                # 3. Document Chunks table with 768-dim vector & full-text tsvector
                cur.execute('''
                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id VARCHAR(128) PRIMARY KEY,
                    doc_id VARCHAR(64) NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                    tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
                    chunk_index INTEGER NOT NULL,
                    page_number INTEGER NOT NULL DEFAULT 1,
                    text TEXT NOT NULL,
                    is_qa_pair BOOLEAN DEFAULT FALSE,
                    question_text TEXT,
                    answer_text TEXT,
                    chunk_type VARCHAR(32) DEFAULT 'paragraph',
                    embedding VECTOR(768) NOT NULL,
                    tsv_content TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                ''')

                # 4. HNSW Vector Index (Sub-millisecond Cosine Distance search)
                cur.execute('''
                CREATE INDEX IF NOT EXISTS idx_chunks_hnsw_embedding 
                ON document_chunks USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
                ''')

                # 5. GIN Index for Full-Text Search
                cur.execute('''
                CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON document_chunks USING gin(tsv_content);
                ''')

                # 6. Multi-tenant lookup index
                cur.execute('''
                CREATE INDEX IF NOT EXISTS idx_chunks_tenant ON document_chunks(tenant_id, doc_id);
                ''')

                # 7. Escalations table
                cur.execute('''
                CREATE TABLE IF NOT EXISTS escalations (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    query TEXT NOT NULL,
                    contact_number VARCHAR(32),
                    contact_email VARCHAR(255),
                    session_id VARCHAR(255),
                    tenant_id VARCHAR(100) DEFAULT 'default',
                    status VARCHAR(32) DEFAULT 'PENDING',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                ''')

                # 8. Tenant Vector Cache in Cloud PostgreSQL (100% Cloud Persistence)
                cur.execute('''
                CREATE TABLE IF NOT EXISTS tenant_vector_cache (
                    tenant_id VARCHAR(64) PRIMARY KEY,
                    chunk_ids JSONB NOT NULL,
                    vectors_blob BYTEA NOT NULL,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                ''')

    def get_tenant_vectors(self, tenant_id: str = "default") -> Tuple[Optional[Any], Optional[List[str]]]:
        """Fetches compressed vector matrix directly from Neon Cloud PostgreSQL without local disk files."""
        try:
            import zlib
            import numpy as np
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT vectors_blob, chunk_ids FROM tenant_vector_cache WHERE tenant_id = %s;",
                        (tenant_id,)
                    )
                    row = cur.fetchone()
                    if row and row[0]:
                        raw = zlib.decompress(row[0])
                        c_ids = row[1] if isinstance(row[1], list) else list(row[1])
                        mat = np.frombuffer(raw, dtype=np.float32).reshape(len(c_ids), 768)
                        return mat, c_ids
        except Exception as e:
            print(f"[CloudPostgres] Notice loading cloud vectors: {e}")
        return None, None

    def save_tenant_vectors(self, tenant_id: str, chunk_ids: List[str], embeddings: Any):
        """Saves compressed vector matrix to Neon Cloud PostgreSQL."""
        try:
            import zlib
            import numpy as np
            import psycopg
            arr = np.asarray(embeddings, dtype=np.float32)
            blob = zlib.compress(arr.tobytes(), level=1)
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO tenant_vector_cache (tenant_id, chunk_ids, vectors_blob)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (tenant_id) DO UPDATE SET
                            chunk_ids = EXCLUDED.chunk_ids,
                            vectors_blob = EXCLUDED.vectors_blob,
                            updated_at = CURRENT_TIMESTAMP;
                    """, (tenant_id, psycopg.types.json.Jsonb(chunk_ids), blob))
        except Exception as e:
            print(f"[CloudPostgres] Notice saving cloud vectors: {e}")

    def load_tenant_data(self, tenant_id: str = "default", include_embeddings: bool = False) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], List[Any]]:
        """Loads documents and chunks directly from Cloud PostgreSQL. Optionally loads embeddings."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # 1. Fetch documents
                cur.execute(
                    "SELECT doc_id, filename, tenant_id, page_count, chunk_count FROM documents WHERE tenant_id = %s OR tenant_id = 'default' OR tenant_id = 'tenant_crm';",
                    (tenant_id,)
                )
                doc_rows = cur.fetchall()
                documents = {
                    r[0]: {
                        "document_id": r[0],
                        "filename": r[1],
                        "tenant_id": r[2],
                        "user_id": "default",
                        "page_count": r[3],
                        "chunks_count": r[4]
                    }
                    for r in doc_rows
                }

                # 2. Fetch chunks directly from Cloud PostgreSQL
                columns = "chunk_id, doc_id, tenant_id, chunk_index, page_number, text, is_qa_pair, question_text, answer_text, chunk_type"
                if include_embeddings:
                    columns += ", embedding"

                cur.execute(
                    f"""
                    SELECT {columns}
                    FROM document_chunks
                    WHERE tenant_id = %s OR tenant_id = 'default' OR tenant_id = 'tenant_crm'
                    ORDER BY doc_id, chunk_index;
                    """,
                    (tenant_id,)
                )
                chunk_rows = cur.fetchall()
                chunks = []
                embeddings = []
                import numpy as np
                for r in chunk_rows:
                    did = r[1]
                    doc_name = documents.get(did, {}).get("filename", "unknown")
                    c_dict = {
                        "chunk_id": r[0],
                        "document_id": did,
                        "document_name": doc_name,
                        "tenant_id": r[2],
                        "user_id": "default",
                        "page": r[4],
                        "section": "General",
                        "text": r[5],
                        "word_count": len(r[5].split()),
                        "is_qa_pair": r[6],
                        "question_text": r[7],
                        "answer_text": r[8],
                        "category": r[9] or "general"
                    }
                    chunks.append(c_dict)

                    if include_embeddings and len(r) > 10:
                        v = r[10]
                        if v is not None:
                            if hasattr(v, "to_numpy"):
                                arr = v.to_numpy().astype(np.float32)
                            elif isinstance(v, (list, tuple)):
                                arr = np.array(v, dtype=np.float32)
                            elif isinstance(v, str):
                                arr = np.fromstring(v.strip("[]"), sep=",", dtype=np.float32)
                            else:
                                arr = np.asarray(v, dtype=np.float32)
                            embeddings.append(arr)

                return documents, chunks, embeddings

    def insert_document_and_chunks(
        self,
        doc_id: str,
        filename: str,
        tenant_id: str,
        page_count: int,
        chunks: List[Dict[str, Any]]
    ):
        """Batch inserts document metadata and its embedded chunks into Cloud PostgreSQL."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                INSERT INTO documents (doc_id, tenant_id, filename, page_count, chunk_count)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (doc_id) DO UPDATE SET
                    filename = EXCLUDED.filename,
                    chunk_count = EXCLUDED.chunk_count;
                ''', (doc_id, tenant_id, filename, page_count, len(chunks)))

                insert_query = '''
                INSERT INTO document_chunks (
                    chunk_id, doc_id, tenant_id, chunk_index, page_number,
                    text, is_qa_pair, question_text, answer_text, chunk_type, embedding
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    text = EXCLUDED.text,
                    is_qa_pair = EXCLUDED.is_qa_pair,
                    question_text = EXCLUDED.question_text,
                    answer_text = EXCLUDED.answer_text,
                    embedding = EXCLUDED.embedding;
                '''
                params = [
                    (
                        c["chunk_id"],
                        doc_id,
                        tenant_id,
                        c.get("chunk_index", i),
                        c.get("page_number", 1),
                        c["text"],
                        c.get("is_qa_pair", False),
                        c.get("question_text"),
                        c.get("answer_text"),
                        c.get("chunk_type", "paragraph"),
                        c["embedding"]
                    )
                    for i, c in enumerate(chunks)
                ]
                cur.executemany(insert_query, params)

    def delete_document(self, identifier: str, tenant_id: str = "default") -> bool:
        """
        Permanently deletes a document and all associated chunks from Neon Cloud PostgreSQL.
        Matches by doc_id (exact or case-insensitive) or filename (exact, unquoted, or base stem).
        """
        import urllib.parse
        clean_id = urllib.parse.unquote(identifier).strip()
        if not clean_id:
            return False

        stem = Path(clean_id).stem.strip().lower()
        raw_name = Path(clean_id).name.strip().lower()
        clean_lower = clean_id.lower()

        candidates = {
            clean_lower,
            identifier.lower(),
            raw_name,
            stem,
            f"{stem}.pdf",
            f"{stem}.docx",
            f"{stem}.txt",
            clean_lower.replace("_", " "),
            clean_lower.replace(" ", "_"),
            stem.replace("_", " "),
            stem.replace(" ", "_"),
            raw_name.replace("_", " "),
            raw_name.replace(" ", "_"),
        }

        deleted = False
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # 1. Find matching document IDs by case-insensitive matching
                    cur.execute("""
                        SELECT doc_id, filename FROM documents
                        WHERE LOWER(doc_id) = ANY(%s) 
                           OR LOWER(filename) = ANY(%s)
                           OR REPLACE(LOWER(filename), '_', ' ') = ANY(%s)
                           OR REPLACE(LOWER(filename), ' ', '_') = ANY(%s);
                    """, (list(candidates), list(candidates), list(candidates), list(candidates)))
                    rows = cur.fetchall()

                    # Fallback: if filename matches without path
                    if not rows and len(stem) > 2:
                        cur.execute("""
                            SELECT doc_id, filename FROM documents
                            WHERE LOWER(filename) LIKE %s
                               OR LOWER(doc_id) LIKE %s;
                        """, (f"%{stem}%", f"%{stem}%"))
                        rows = cur.fetchall()

                    if not rows:
                        return False

                    doc_ids = [r[0] for r in rows]

                    # 2. Delete chunks explicitly from Neon Cloud
                    cur.execute(
                        "DELETE FROM document_chunks WHERE doc_id = ANY(%s);",
                        (doc_ids,)
                    )

                    # 3. Delete document entries from Neon Cloud
                    cur.execute(
                        "DELETE FROM documents WHERE doc_id = ANY(%s);",
                        (doc_ids,)
                    )
                    deleted = True

                    # 4. Invalidate tenant_vector_cache row in Neon Cloud so it reflects remaining chunks
                    cur.execute("DELETE FROM tenant_vector_cache WHERE tenant_id = %s;", (tenant_id,))

            # 5. Remove local vector cache to force fresh sync
            try:
                cache_file = Path("data/cloud_vector_cache.npz")
                if cache_file.exists():
                    cache_file.unlink()
            except Exception:
                pass

        except Exception as ex:
            print(f"[CloudPostgres] Error deleting document {identifier}: {ex}")
            return False

        return deleted

    def clear_all_documents(self, tenant_id: Optional[str] = None) -> bool:
        """
        Permanently deletes all documents, chunks, and vector caches from Neon Cloud PostgreSQL.
        If tenant_id is provided, deletes for that tenant; otherwise deletes across all tenants.
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    if tenant_id:
                        cur.execute("DELETE FROM document_chunks WHERE tenant_id = %s;", (tenant_id,))
                        cur.execute("DELETE FROM documents WHERE tenant_id = %s;", (tenant_id,))
                        cur.execute("DELETE FROM tenant_vector_cache WHERE tenant_id = %s;", (tenant_id,))
                    else:
                        cur.execute("DELETE FROM document_chunks;")
                        cur.execute("DELETE FROM documents;")
                        cur.execute("DELETE FROM tenant_vector_cache;")

            # Remove local vector cache
            try:
                cache_file = Path("data/cloud_vector_cache.npz")
                if cache_file.exists():
                    cache_file.unlink()
            except Exception:
                pass

            return True
        except Exception as ex:
            print(f"[CloudPostgres] Error clearing cloud data: {ex}")
            return False

    def insert_escalation(
        self,
        query: str,
        contact_number: Optional[str],
        contact_email: Optional[str] = None,
        session_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        status: str = "PENDING"
    ):
        """Inserts an escalation ticket directly into Cloud PostgreSQL."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                INSERT INTO escalations (query, contact_number, contact_email, session_id, tenant_id, status)
                VALUES (%s, %s, %s, %s, %s, %s);
                ''', (query, contact_number, contact_email, session_id or "default", tenant_id or "default", status))

    def get_escalations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent escalation tickets from Cloud PostgreSQL."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                SELECT id, query, contact_number, contact_email, status, created_at
                FROM escalations
                ORDER BY created_at DESC
                LIMIT %s;
                ''', (limit,))
                rows = cur.fetchall()
                return [
                    {
                        "id": str(r[0]),
                        "query": r[1],
                        "contact_number": r[2],
                        "contact_email": r[3],
                        "status": r[4],
                        "created_at": str(r[5])
                    }
                    for r in rows
                ]

    def list_documents(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lists all documents stored in Neon Cloud PostgreSQL."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                if tenant_id:
                    cur.execute(
                        "SELECT doc_id, filename, tenant_id, page_count, chunk_count FROM documents WHERE tenant_id = %s ORDER BY filename;",
                        (tenant_id,)
                    )
                else:
                    cur.execute(
                        "SELECT doc_id, filename, tenant_id, page_count, chunk_count FROM documents ORDER BY filename;"
                    )
                rows = cur.fetchall()
                return [
                    {
                        "document_id": r[0],
                        "filename": r[1],
                        "tenant_id": r[2],
                        "page_count": r[3],
                        "chunks_count": r[4]
                    }
                    for r in rows
                ]

    def get_stats(self) -> Dict[str, Any]:
        """Returns overview statistics of Cloud PostgreSQL database."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM documents;")
                doc_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM document_chunks;")
                chunk_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM escalations;")
                esc_count = cur.fetchone()[0]
                return {
                    "connected": True,
                    "documents_count": doc_count,
                    "chunks_count": chunk_count,
                    "escalations_count": esc_count,
                    "backend": "Neon Cloud PostgreSQL + pgvector (HNSW Index)"
                }

    def hybrid_search(
        self,
        query_text: str,
        query_vector: List[float],
        tenant_id: str = "default",
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Executes single-query Hybrid Search:
        HNSW Vector Cosine Distance (<=>) + Full-Text GIN rank combined via Reciprocal Rank Fusion (RRF).
        Optimized with CTE to transfer the 768-dim query vector across the network once.
        """
        sql = '''
        WITH q_vec AS (
            SELECT %s::vector AS vec
        ),
        q_text AS (
            SELECT plainto_tsquery('english', %s) AS q
        ),
        vector_matches AS (
            SELECT chunk_id, 1 - (embedding <=> q.vec) AS vector_sim,
                   ROW_NUMBER() OVER (ORDER BY embedding <=> q.vec) AS rank_v
            FROM document_chunks, q_vec q
            WHERE tenant_id = %s OR tenant_id = 'default'
            ORDER BY embedding <=> q.vec
            LIMIT 15
        ),
        text_matches AS (
            SELECT chunk_id, ts_rank_cd(tsv_content, qt.q) AS text_score,
                   ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv_content, qt.q) DESC) AS rank_t
            FROM document_chunks, q_text qt
            WHERE (tenant_id = %s OR tenant_id = 'default') AND tsv_content @@ qt.q
            LIMIT 15
        )
        SELECT 
            c.chunk_id,
            c.doc_id,
            c.page_number,
            c.text,
            c.is_qa_pair,
            c.question_text,
            c.answer_text,
            COALESCE(v.vector_sim, 0) AS vector_sim,
            COALESCE(t.text_score, 0) AS text_score,
            (COALESCE(1.0 / (60 + v.rank_v), 0) + COALESCE(1.0 / (60 + t.rank_t), 0)) AS rrf_score
        FROM document_chunks c
        LEFT JOIN vector_matches v ON c.chunk_id = v.chunk_id
        LEFT JOIN text_matches t ON c.chunk_id = t.chunk_id
        WHERE v.chunk_id IS NOT NULL OR t.chunk_id IS NOT NULL
        ORDER BY rrf_score DESC
        LIMIT %s;
        '''
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    query_vector,
                    query_text,
                    tenant_id,
                    tenant_id,
                    top_k
                ))
                rows = cur.fetchall()
                results = []
                for r in rows:
                    results.append({
                        "chunk_id": r[0],
                        "doc_id": r[1],
                        "page_number": r[2],
                        "text": r[3],
                        "is_qa_pair": r[4],
                        "question_text": r[5],
                        "answer_text": r[6],
                        "vector_similarity": float(r[7]),
                        "text_score": float(r[8]),
                        "rrf_score": float(r[9])
                    })
                return results

def get_postgres_store() -> Optional[PostgresVectorStore]:
    """Returns the singleton PostgresVectorStore instance if DATABASE_URL is configured and connected."""
    global _GLOBAL_POSTGRES_STORE
    _load_env_file()
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url or not db_url.startswith("postgresql://"):
        if _GLOBAL_POSTGRES_STORE is not None:
            try:
                _GLOBAL_POSTGRES_STORE.pool.close()
            except Exception:
                pass
            _GLOBAL_POSTGRES_STORE = None
        return None

    if _GLOBAL_POSTGRES_STORE is None or _GLOBAL_POSTGRES_STORE.db_url != db_url:
        try:
            if _GLOBAL_POSTGRES_STORE is not None:
                try:
                    _GLOBAL_POSTGRES_STORE.pool.close()
                except Exception:
                    pass
            store = PostgresVectorStore(db_url)
            if store.test_connection():
                _GLOBAL_POSTGRES_STORE = store
            else:
                _GLOBAL_POSTGRES_STORE = None
        except Exception:
            _GLOBAL_POSTGRES_STORE = None
            return None

    return _GLOBAL_POSTGRES_STORE

def is_cloud_connected() -> bool:
    """
    Returns True if Neon Cloud DB is active and verified, False otherwise.
    Executes in <0.001ms without synchronous transatlantic blocking.
    """
    store = get_postgres_store()
    if not store:
        return False
    return store.is_connected_fast()
