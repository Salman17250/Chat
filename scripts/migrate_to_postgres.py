import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.storage_postgres import PostgresVectorStore

def load_env_file():
    env_file = ROOT_DIR / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip("'\"")

def migrate():
    load_env_file()
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        print("[ERROR] DATABASE_URL is not set in .env!")
        return False

    print("[*] Connecting to Neon Cloud PostgreSQL...")
    store = PostgresVectorStore(db_url)

    print("[*] Ensuring database schema, pgvector, HNSW & GIN indexes...")
    store.init_db()
    print("[+] Database schema and HNSW indexes verified.")

    # 1. Migrate documents and chunks from rag_data/tenants/*/index.json
    tenants_dir = ROOT_DIR / "rag_data" / "tenants"
    total_docs = 0
    total_chunks = 0
    start_time = time.time()

    if tenants_dir.exists():
        for t_dir in tenants_dir.iterdir():
            if not t_dir.is_dir():
                continue
            t_idx = t_dir / "index.json"
            if not t_idx.exists():
                continue

            tenant_id = t_dir.name
            print(f"\n[*] Processing tenant '{tenant_id}' from {t_idx}...")
            try:
                with open(t_idx, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[-] Error reading {t_idx}: {e}")
                continue

            raw_docs = data.get("documents", {})
            raw_chunks = data.get("chunks", [])
            raw_embeddings = data.get("embeddings", [])

            # Map embeddings to chunks if separate list
            if raw_embeddings and len(raw_embeddings) == len(raw_chunks):
                for idx, emb in enumerate(raw_embeddings):
                    if raw_chunks[idx].get("embedding") is None:
                        raw_chunks[idx]["embedding"] = emb

            # Group chunks by doc_id
            chunks_by_doc: Dict[str, List[Dict[str, Any]]] = {}
            for i, c in enumerate(raw_chunks):
                did = c.get("document_id") or c.get("doc_id") or "default_doc"
                c_dict = {
                    "chunk_id": c.get("chunk_id", f"{did}_c{i}"),
                    "chunk_index": i,
                    "page_number": c.get("page", 1),
                    "text": c.get("text", ""),
                    "is_qa_pair": bool(c.get("is_qa_pair", False)),
                    "question_text": c.get("question_text"),
                    "answer_text": c.get("answer_text"),
                    "chunk_type": c.get("category", "paragraph"),
                    "embedding": c.get("embedding"),
                }
                chunks_by_doc.setdefault(did, []).append(c_dict)

            # Process documents dictionary
            if isinstance(raw_docs, dict):
                doc_items = raw_docs.items()
            elif isinstance(raw_docs, list):
                doc_items = [(d.get("document_id") or d.get("filename", f"doc_{i}"), d) for i, d in enumerate(raw_docs)]
            else:
                doc_items = []

            for did, dmeta in doc_items:
                doc_chunks = chunks_by_doc.get(did, [])
                # Filter out chunks without embeddings
                valid_chunks = [c for c in doc_chunks if c.get("embedding") is not None and len(c.get("embedding", [])) == 768]
                if not valid_chunks:
                    print(f"    [-] Skipping {did}: No valid 768-dim embeddings found ({len(doc_chunks)} chunks).")
                    continue

                filename = dmeta.get("filename", did)
                page_count = dmeta.get("page_count", 1)
                store.insert_document_and_chunks(
                    doc_id=did,
                    filename=filename,
                    tenant_id=tenant_id,
                    page_count=page_count,
                    chunks=valid_chunks
                )
                total_docs += 1
                total_chunks += len(valid_chunks)
                print(f"    [+] Document '{filename}' ({did}): Migrated {len(valid_chunks)} chunks to Cloud Postgres.")

    # 2. Migrate escalations from rag_data/escalations.json
    esc_file = ROOT_DIR / "rag_data" / "escalations.json"
    total_escalations = 0
    if esc_file.exists():
        try:
            with open(esc_file, "r", encoding="utf-8") as f:
                esc_data = json.load(f)
            if isinstance(esc_data, list):
                print(f"\n[*] Migrating {len(esc_data)} escalation records to Cloud Postgres...")
                with store.get_connection() as conn:
                    with conn.cursor() as cur:
                        for item in esc_data:
                            cur.execute('''
                            INSERT INTO escalations (query, contact_number, contact_email, status, created_at)
                            VALUES (%s, %s, %s, %s, TO_TIMESTAMP(%s))
                            ON CONFLICT DO NOTHING;
                            ''', (
                                item.get("query", ""),
                                item.get("contact_number"),
                                item.get("contact_email"),
                                item.get("status", "PENDING"),
                                item.get("epoch", time.time())
                            ))
                            total_escalations += 1
                print(f"[+] Migrated {total_escalations} escalation tickets.")
        except Exception as e:
            print(f"[-] Escalation migration warning: {e}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"CLOUD MIGRATION COMPLETED SUCCESSFULLY in {elapsed:.2f}s")
    print(f"Total Documents in Cloud DB:    {total_docs}")
    print(f"Total Chunks in Cloud DB:       {total_chunks}")
    print(f"Total Escalations in Cloud DB:  {total_escalations}")
    print("=" * 60)

    # 3. Verification Hybrid Search Query directly against Neon Cloud PostgreSQL
    print("\n[*] Running Verification Hybrid Query against Neon Cloud Database...")
    test_query = "What is the lead module?"
    from app.rag.embeddings.ollama_embedder import OllamaEmbedder
    from app.rag.config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL
    embedder = OllamaEmbedder(base_url=OLLAMA_BASE_URL, model_name=OLLAMA_EMBED_MODEL)
    q_vec = embedder.embed_text(test_query)

    t0 = time.time()
    results = store.hybrid_search(
        query_text=test_query,
        query_vector=q_vec,
        tenant_id="default",
        top_k=3
    )
    t_search = (time.time() - t0) * 1000

    print(f"[+] Hybrid Search executed in {t_search:.2f} ms! Returned {len(results)} results:")
    for r in results:
        print(f"    - [{r['chunk_id']}] (RRF Score: {r['rrf_score']:.4f}, Vec Sim: {r['vector_similarity']:.4f}, Doc: {r['doc_id']})")
        print(f"      Text excerpt: {r['text'][:120]}...")

    return True

if __name__ == "__main__":
    success = migrate()
    if not success:
        sys.exit(1)
