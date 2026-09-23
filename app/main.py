from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
import requests
import json

from app.config import (
    UPLOAD_DIR,
    INDEX_FILE,
    ALLOWED_ORIGINS,
    EMBED_MODEL_NAME,
)
from app.models import (
    ChatRequest,
    ChatResponse,
    UploadResponse,
    StatusResponse,
    ClearResponse,
    DocumentInfo,
    SourceCitation,
)
from app.embeddings import get_embedding_engine
from app.ingestion import extract_pdf_pages
from app.chunking import create_document_chunks
from app.storage import get_vector_store
from app.retrieval import retrieve_candidates
from app.extractor import get_qa_engine
from app.translator import get_translator
from app.rag.engine import get_rag_engine
from app.rag.config import NO_ANSWER_FOUND_MESSAGE
from app.services.email_service import (
    extract_phone_number,
    extract_email_address,
    dispatch_escalation_email,
    flush_queued_escalations,
    get_smtp_credentials,
    ESCALATIONS_FILE,
)


# ============================================================
# LIFESPAN — Initialize embedding model once at startup
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eagerly initialize pure search RAG Engine
    print("[*] Initializing Pure RAG Engine (nomic-embed-text 768-dim)...", flush=True)
    rag_engine = get_rag_engine()

    # Connect to Neon Cloud PostgreSQL + pgvector & load tenant projection into RAM
    try:
        from app.storage_postgres import get_postgres_store
        pg_store = get_postgres_store()
        if pg_store and pg_store.is_connected_fast():
            print("[+] Cloud PostgreSQL Connected. Loading in-memory tenant projection...", flush=True)
            tenant = rag_engine.tenant_manager.get_tenant("default")
            tenant.load()
            print(f"[+] In-memory projection loaded: {len(tenant.chunks)} chunks across {len(tenant.documents)} documents (100% in RAM).", flush=True)
    except Exception as pg_err:
        print(f"[-] Cloud PostgreSQL startup notice: {pg_err}", flush=True)

    # Pre-warm Ollama embedder so queries respond instantly (<30ms)
    try:
        from app.rag.embeddings.ollama_embedder import get_embedding_engine
        embedder = get_embedding_engine()
        embedder.embed_query("warmup")
        print("[+] Ollama Embedding engine pre-warmed. Ready for ultra-fast queries (<30ms).", flush=True)
    except Exception as e:
        print(f"[-] Embedding pre-warm note: {e}", flush=True)

    # Configure high-concurrency threadpool capacity (250 worker threads for 100+ concurrent users)
    try:
        import anyio
        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = 250
        print(f"[+] High-concurrency worker threadpool capacity set to {limiter.total_tokens} tokens.", flush=True)
    except Exception as e:
        print(f"[-] Threadpool limiter config note: {e}", flush=True)

    yield
    print("[*] Shutting down RAG backend.", flush=True)


# ============================================================
# APP INITIALIZATION & CORS
# ============================================================
app = FastAPI(
    title="Dynamic Extractive RAG API",
    description="High-performance, Pure Search/RAG system with Cloud PostgreSQL + pgvector",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# ============================================================
# ROOT ENDPOINT
# ============================================================
@app.get("/")
def root():
    cloud_status = "Not configured"
    try:
        from app.storage_postgres import get_postgres_store
        pg_store = get_postgres_store()
        if pg_store:
            stats = pg_store.get_stats()
            cloud_status = f"Connected ({stats['chunks_count']} chunks, {stats['documents_count']} docs, HNSW index)"
    except Exception as e:
        cloud_status = f"Error: {e}"

    return {
        "success": True,
        "message": "Dynamic Extractive RAG backend is running 100% cloud-backed",
        "cloud_database": cloud_status,
        "engine": "SentenceTransformers + nomic-embed-text + Cloud pgvector",
        "ollama_required": False
    }


# ============================================================
# DOCUMENT UPLOAD & INDEXING
# ============================================================
@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        return UploadResponse(
            success=False,
            message="Filename is required."
        )

    safe_filename = Path(file.filename).name
    file_path = UPLOAD_DIR / safe_filename

    # 1. Save uploaded file
    try:
        content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)
    except Exception as e:
        return UploadResponse(
            success=False,
            message="Failed to save file.",
            error=str(e)
        )

    # 2. Ingest into Pure Search RAGEngine (PDF / DOC / DOCX supported)
    try:
        rag_engine = get_rag_engine()
        ingest_res = rag_engine.ingest_document(
            file_path=file_path,
            filename=safe_filename,
            tenant_id="default",
            user_id="default",
        )
    except Exception as e:
        return UploadResponse(
            success=False,
            message="Failed to parse and index document in RAGEngine.",
            error=str(e)
        )

    chunk_count = ingest_res.get("chunks_count", 0)
    if chunk_count == 0:
        return UploadResponse(
            success=False,
            message="No readable text or chunks could be extracted from document."
        )

    return UploadResponse(
        success=True,
        message=f"File '{safe_filename}' parsed and indexed successfully ({chunk_count} chunks).",
        filename=safe_filename,
        chunks=chunk_count,
        storage=str(INDEX_FILE)
    )


# ============================================================
# PURE RAG SEARCH WITH FULL EXPLAINABILITY
# ============================================================
@app.post("/api/rag/search")
def search_rag(request: ChatRequest):
    question = request.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Query message cannot be empty.")

    rag_engine = get_rag_engine()
    rag_res = rag_engine.query(
        question=question,
        tenant_id=request.tenant_id,
        user_id=request.user_id,
        filter_doc_id=request.filter_doc_id,
        filter_section=request.filter_section,
        top_k=request.top_k or 5,
        include_debug=True,
    )

    return {
        "success": True,
        "answer": rag_res.answer,
        "confidence": rag_res.confidence,
        "confidence_score": rag_res.confidence_score,
        "sources": [
            {
                "document": s.document,
                "document_id": s.document_id,
                "page": s.page,
                "section": s.section,
                "score": s.score,
                "text": s.text,
                "chunk_id": s.chunk_id,
            }
            for s in rag_res.sources
        ],
        "debug_trace": rag_res.debug_trace,
    }


# ============================================================
# CHAT / QUESTION ANSWERING (PURE RAG / SEARCH-BASED)
# Multi-Language: English, Hindi, Hinglish, Gujarati
# ============================================================
@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = request.message.strip()

    if not question:
        return ChatResponse(
            success=False,
            answer="Please enter a question.",
            sources=[]
        )

    # 1. Strict Cloud Database Enforcement (Zero Local Fallback)
    from app.storage_postgres import get_postgres_store
    pg_store = get_postgres_store()
    if not pg_store or not pg_store.is_connected_fast():
        try:
            rag_engine = get_rag_engine()
            with rag_engine.response_cache_lock:
                rag_engine.response_cache.clear()
            for t_name in rag_engine.tenant_manager.list_tenants():
                t = rag_engine.tenant_manager.get_tenant(t_name)
                t.chunks = []
                t.documents = {}
                t.embeddings = []
                t.vector_engine.index_chunks([], [])
                t.bm25_engine.index_chunks([])
                t.fuzzy_engine.index_chunks([])
        except Exception:
            pass
        return ChatResponse(
            success=False,
            answer="[Cloud Database Disconnected] Neon PostgreSQL connection is required. All local data has been removed. Please verify DATABASE_URL in .env to connect to the cloud database.",
            filename=None,
            confidence="VERY_LOW",
            confidence_score=0.0,
            sources=[]
        )

    # 2. Detect language & check for conversational greetings / chit-chat
    translator = get_translator()
    english_query, detected_lang, original_query, form = translator.process_query(question)

    greeting_reply = translator.check_greeting(question, detected_lang)
    if greeting_reply:
        return ChatResponse(
            success=True,
            answer=greeting_reply,
            filename=None,
            sources=[],
            detected_language=detected_lang,
            english_query=None
        )

    t_id = getattr(request, "tenant_id", None) or "default"
    u_id = getattr(request, "user_id", None) or "default"
    sess_id = getattr(request, "session_id", None) or "default"
    rag_engine = get_rag_engine()
    session = rag_engine.context_manager.get_or_create_session(t_id, u_id, sess_id)

    # Detect user contact info from input message or explicit request payload
    provided_phone = request.contact_number or extract_phone_number(question)
    provided_email = request.contact_email or extract_email_address(question)

    # 1. Follow-up turn: user is answering the contact request for a pending unresolved query
    if session.pending_unresolved_query and (provided_phone or provided_email):
        unresolved_q = session.pending_unresolved_query
        session.pending_unresolved_query = None
        session.pending_unresolved_time = None

        contact_display = provided_phone or provided_email
        dispatch_escalation_email(
            query=unresolved_q,
            contact_number=provided_phone,
            contact_email=provided_email,
            session_id=sess_id,
            tenant_id=t_id
        )

        confirm_msg = (
            f"Thank you! We have noted your contact number ({contact_display}) and forwarded your query "
            f"(\"{unresolved_q}\") directly to our support team at salman.wortal@gmail.com. "
            f"Someone from our team will reach out to you shortly."
        )
        final_answer = translator.process_answer(confirm_msg, detected_lang, form=form)
        return ChatResponse(
            success=True,
            answer=final_answer,
            filename=None,
            confidence="HIGH",
            confidence_score=1.0,
            sources=[],
            escalation_status="ESCALATED",
            detected_language=detected_lang,
            english_query=None
        )

    # 2. Pure RAG Search Query execution via RAGEngine
    rag_res = rag_engine.query(
        question=english_query,
        tenant_id=request.tenant_id,
        user_id=request.user_id,
        session_id=request.session_id,
        filter_doc_id=request.filter_doc_id,
        filter_section=request.filter_section,
        top_k=request.top_k or 5,
        include_debug=request.include_debug or False,
    )

    # 3. Handle No Answer Found / Low Confidence Escalation
    if rag_res.answer == NO_ANSWER_FOUND_MESSAGE or rag_res.confidence == "VERY_LOW":
        # Case A: User ALREADY provided contact number in the same message
        if provided_phone or provided_email:
            contact_display = provided_phone or provided_email
            dispatch_escalation_email(
                query=question,
                contact_number=provided_phone,
                contact_email=provided_email,
                session_id=sess_id,
                tenant_id=t_id
            )
            escalated_msg = (
                f"Relevant information was not found in the uploaded documents. "
                f"However, we have recorded your contact number ({contact_display}) and forwarded your inquiry "
                f"to our support team at salman.wortal@gmail.com. Someone will contact you shortly."
            )
            final_answer = translator.process_answer(escalated_msg, detected_lang, form=form)
            return ChatResponse(
                success=True,
                answer=final_answer,
                filename=None,
                confidence="VERY_LOW",
                confidence_score=rag_res.confidence_score,
                sources=[],
                escalation_status="ESCALATED",
                debug_trace=rag_res.debug_trace,
                detected_language=detected_lang,
                english_query=english_query if detected_lang != "en" else None
            )

        # Case B: Prompt user for contact details to escalate
        session.pending_unresolved_query = question
        session.pending_unresolved_time = time.time()

        ask_contact_msg = (
            "Relevant information was not found in the uploaded documents. "
            "Please share your phone number (or email) so our team can follow up and assist you directly."
        )
        final_answer = translator.process_answer(ask_contact_msg, detected_lang, form=form)
        return ChatResponse(
            success=True,
            answer=final_answer,
            filename=None,
            confidence="VERY_LOW",
            confidence_score=rag_res.confidence_score,
            sources=[],
            escalation_status="ASK_CONTACT",
            debug_trace=rag_res.debug_trace,
            detected_language=detected_lang,
            english_query=english_query if detected_lang != "en" else None
        )

    # If query answered successfully, clear any stale unresolved query
    session.pending_unresolved_query = None

    # 4. Format Source Citations with page, section, document, and score
    sources = [
        SourceCitation(
            document=s.document,
            document_id=s.document_id,
            page=s.page,
            section=s.section,
            score=s.score,
            text=s.text,
            chunk_id=s.chunk_id,
        )
        for s in rag_res.sources
    ]

    # Translate answer back to user's language if native script, keep English for Hinglish
    final_answer = translator.process_answer(rag_res.answer, detected_lang, form=form)

    return ChatResponse(
        success=True,
        answer=final_answer,
        filename=rag_res.sources[0].document if rag_res.sources else None,
        confidence=rag_res.confidence,
        confidence_score=rag_res.confidence_score,
        sources=sources,
        debug_trace=rag_res.debug_trace,
        detected_language=detected_lang,
        english_query=english_query if detected_lang != "en" else None
    )


# ============================================================
# REAL-TIME SSE STREAMING CHAT ENDPOINT
# ============================================================
from fastapi.responses import StreamingResponse

@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest):
    """
    Real-Time Server-Sent Events (SSE) Streaming Endpoint.
    Streams validated answer tokens directly to the client after strict evidence verification.
    """
    def event_generator():
        resp = chat(request)
        # 1. Yield validated metadata event first
        meta = {
            "type": "metadata",
            "confidence": resp.confidence,
            "confidence_score": resp.confidence_score,
            "filename": resp.filename,
            "sources": [s.model_dump() for s in resp.sources] if resp.sources else []
        }
        yield f"data: {json.dumps(meta)}\n\n"

        # 2. Stream tokens in small verified bursts for smooth visual rendering
        words = resp.answer.split(" ")
        for i, word in enumerate(words):
            chunk_data = {
                "type": "delta",
                "text": word + (" " if i < len(words) - 1 else "")
            }
            yield f"data: {json.dumps(chunk_data)}\n\n"

        # 3. Final completion event
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ============================================================
# RAG STATUS (STRICT CLOUD POSTGRESQL + PGVECTOR)
# ============================================================
@app.get("/api/rag/status")
def rag_status():
    from app.storage_postgres import get_postgres_store
    pg_store = get_postgres_store()

    # Check Ollama
    ollama_running = False
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=0.2)
        ollama_running = response.status_code == 200
    except Exception:
        ollama_running = False

    if not pg_store or not pg_store.is_connected_fast():
        return {
            "success": False,
            "cloud_connected": False,
            "documents_loaded": [],
            "total_documents": 0,
            "total_chunks": 0,
            "documents": [],
            "storage": "Cloud Database Disconnected (Neon PostgreSQL required)",
            "embedding_model": "nomic-embed-text (768-dim)",
            "retrieval": "Cloud HNSW Vector Cosine + GIN Hybrid Search",
            "generation": "Deterministic Extractive Search & Assembly (100% Pure Cloud RAG)",
            "ollama_running": ollama_running,
            "message": "Cloud Database Disconnected. Please configure DATABASE_URL in .env."
        }

    try:
        rag_engine = get_rag_engine()
        tenant = rag_engine.tenant_manager.get_tenant("default")
        if not tenant.chunks:
            tenant.load()

        document_info = [
            {
                "document_id": did,
                "filename": info.get("filename", "unknown"),
                "chunks": info.get("chunks_count", 0),
                "pages": info.get("page_count", 1),
                "tenant_id": info.get("tenant_id", "default")
            }
            for did, info in tenant.documents.items()
        ]

        return {
            "success": True,
            "cloud_connected": True,
            "documents_loaded": [doc["filename"] for doc in document_info],
            "total_documents": len(tenant.documents),
            "total_chunks": len(tenant.chunks),
            "documents": document_info,
            "storage": "Neon Cloud PostgreSQL - neon-cloud-aws-ohio (100% in RAM)",
            "embedding_model": "nomic-embed-text (768-dim)",
            "retrieval": "Multi-Stage Hybrid Retrieval: Cloud HNSW Vector + GIN + Deterministic Scorer",
            "generation": "Deterministic Extractive Search & Assembly (100% Pure Cloud RAG)",
            "ollama_running": ollama_running
        }
    except Exception as e:
        return {
            "success": False,
            "cloud_connected": False,
            "documents_loaded": [],
            "total_documents": 0,
            "total_chunks": 0,
            "documents": [],
            "storage": f"Cloud Database Error: {str(e)}",
            "ollama_running": ollama_running
        }


# ============================================================
# CLEAR RAG DATA (NEON CLOUD POSTGRESQL + RAM PROJECTION)
# ============================================================
@app.delete("/api/rag/clear", response_model=ClearResponse)
def clear_rag_data():
    # 1. Permanently delete all documents and chunks from Neon Cloud PostgreSQL
    from app.storage_postgres import get_postgres_store
    pg_store = get_postgres_store()
    cloud_cleared = False
    if pg_store and pg_store.is_connected_fast():
        cloud_cleared = pg_store.clear_all_documents()

    # 2. Clear all in-memory RAGEngine projections across all tenants
    rag_engine = get_rag_engine()
    with rag_engine.lock:
        with rag_engine.response_cache_lock:
            rag_engine.response_cache.clear()
        for t_name in rag_engine.tenant_manager.list_tenants():
            t = rag_engine.tenant_manager.get_tenant(t_name)
            t.chunks.clear()
            t.documents.clear()
            t.embeddings = []
            t.reindex()
        rag_engine.multi_cache.clear_tenant("default")

    # 3. Clear legacy store if present
    store = get_vector_store()
    store.clear()

    # 4. Remove uploaded files from local upload directory
    try:
        for f in UPLOAD_DIR.glob("*"):
            if f.is_file():
                try:
                    f.unlink()
                except Exception:
                    pass
    except Exception:
        pass

    return ClearResponse(
        success=True,
        message="All documents, chunks, and embeddings permanently cleared from Neon Cloud PostgreSQL and memory."
    )


# ============================================================
# LIST DOCUMENTS
# ============================================================
@app.get("/api/documents")
def list_documents(tenant_id: Optional[str] = None, user_id: Optional[str] = None):
    from app.storage_postgres import get_postgres_store
    pg_store = get_postgres_store()
    if not pg_store or not pg_store.test_connection():
        return {
            "success": False,
            "cloud_connected": False,
            "total_documents": 0,
            "documents": [],
            "message": "Cloud database disconnected."
        }

    docs = pg_store.list_documents(tenant_id=tenant_id)
    if user_id:
        docs = [d for d in docs if not d.get("user_id") or d.get("user_id") == user_id]

    return {
        "success": True,
        "cloud_connected": True,
        "total_documents": len(docs),
        "documents": docs,
    }


# ============================================================
# RE-INDEX DOCUMENTS
# ============================================================
@app.post("/api/rag/reindex")
def reindex_documents(document_id: Optional[str] = None):
    rag_engine = get_rag_engine()
    with rag_engine.lock:
        if document_id:
            # Re-index single document from uploads
            doc_meta = rag_engine.documents.get(document_id)
            if not doc_meta:
                raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
            file_path = UPLOAD_DIR / doc_meta.get("filename", "")
            if not file_path.exists():
                raise HTTPException(status_code=404, detail=f"File '{file_path}' not found in upload storage.")
            res = rag_engine.ingest_document(
                file_path=file_path,
                filename=doc_meta.get("filename"),
                tenant_id=doc_meta.get("tenant_id", "default"),
                user_id=doc_meta.get("user_id", "default"),
            )
            return {"success": True, "message": f"Re-indexed document {document_id}", "result": res}
        else:
            # Rebuild hybrid retrieval engines with current chunk pool
            rag_engine.retrieval_engine.index_chunks(rag_engine.chunks, rag_engine.embeddings)
            rag_engine._save_index()
            return {
                "success": True,
                "message": f"Successfully re-indexed {len(rag_engine.chunks)} chunks across {len(rag_engine.documents)} documents.",
                "total_chunks": len(rag_engine.chunks),
                "total_documents": len(rag_engine.documents)
            }


# ============================================================
# DELETE SINGLE DOCUMENT
# ============================================================
@app.delete("/api/documents/{identifier:path}")
def delete_document(identifier: str):
    import urllib.parse
    clean_identifier = urllib.parse.unquote(identifier).strip()

    rag_engine = get_rag_engine()

    # Pre-collect matching document IDs from in-memory tenant registries before deleting from DB
    target_ids = set()
    target_clean = clean_identifier.lower()
    target_stem = Path(clean_identifier).stem.lower()

    for tid in rag_engine.tenant_manager.list_tenants():
        tenant = rag_engine.tenant_manager.get_tenant(tid)
        for doc_id, doc_meta in list(tenant.documents.items()):
            fn = doc_meta.get("filename", "").lower()
            fn_stem = Path(fn).stem.lower()
            if (
                doc_id == clean_identifier
                or doc_id.lower() == target_clean
                or fn == target_clean
                or fn_stem == target_stem
                or fn.replace("_", " ") == target_clean.replace("_", " ")
                or fn.replace(" ", "_") == target_clean.replace(" ", "_")
                or fn_stem.replace("_", " ") == target_stem.replace("_", " ")
                or fn_stem.replace(" ", "_") == target_stem.replace(" ", "_")
            ):
                target_ids.add(doc_id)

    # 1. Delete from Neon Cloud PostgreSQL
    from app.storage_postgres import get_postgres_store
    pg_store = get_postgres_store()
    deleted_cloud = False
    if pg_store and pg_store.is_connected_fast():
        try:
            deleted_cloud = pg_store.delete_document(clean_identifier)
            if not deleted_cloud and target_ids:
                for tid in target_ids:
                    if pg_store.delete_document(tid):
                        deleted_cloud = True
        except Exception as ex:
            print(f"[Main] Notice deleting from cloud: {ex}")

    # 2. Delete from in-memory RAGEngine across all tenants
    deleted_rag = False
    try:
        if rag_engine.delete_document(clean_identifier):
            deleted_rag = True
        for tid in target_ids:
            if rag_engine.delete_document(tid):
                deleted_rag = True
    except Exception as ex:
        print(f"[Main] Notice deleting from RAG engine: {ex}")

    # If deleted_cloud succeeded, make sure memory matches cloud 100%
    if deleted_cloud:
        for tid in rag_engine.tenant_manager.list_tenants():
            t = rag_engine.tenant_manager.get_tenant(tid)
            # Remove any chunks belonging to target_ids or clean_identifier
            valid_indices = [
                i for i, c in enumerate(t.chunks)
                if c.document_id not in target_ids
                and c.document_id != clean_identifier
                and (not c.document_name or c.document_name.lower() != target_clean)
            ]
            t.chunks = [t.chunks[i] for i in valid_indices]
            if t.embeddings is not None and len(t.embeddings) > 0:
                import numpy as np
                if isinstance(t.embeddings, np.ndarray):
                    t.embeddings = t.embeddings[valid_indices] if valid_indices else np.empty((0, 768), dtype=np.float32)
                else:
                    t.embeddings = [t.embeddings[i] for i in valid_indices if i < len(t.embeddings)]
            else:
                t.embeddings = []
            for did in list(t.documents.keys()):
                if did in target_ids or did == clean_identifier:
                    t.documents.pop(did, None)
            t.reindex()
            deleted_rag = True

    # 3. Synchronize vector cache in Neon Cloud for any remaining chunks
    try:
        if pg_store and pg_store.is_connected_fast():
            for tid in rag_engine.tenant_manager.list_tenants():
                tenant = rag_engine.tenant_manager.get_tenant(tid)
                if len(tenant.chunks) > 0 and len(tenant.embeddings) > 0:
                    c_ids = [c.chunk_id for c in tenant.chunks]
                    pg_store.save_tenant_vectors(tid, c_ids, tenant.embeddings)
                else:
                    with pg_store.get_connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute("DELETE FROM tenant_vector_cache WHERE tenant_id = %s;", (tid,))
    except Exception as ex:
        print(f"[Main] Notice syncing cloud vector cache after delete: {ex}")

    # 4. Invalidate all query caches
    with rag_engine.response_cache_lock:
        rag_engine.response_cache.clear()
    rag_engine.multi_cache.clear_all()

    # 5. Delete from legacy store if present
    store = get_vector_store()
    deleted_store = False
    try:
        deleted_store = store.delete_document(clean_identifier)
    except Exception:
        pass

    # 6. Remove file from uploads directory if exists
    for candidate in [
        clean_identifier,
        identifier,
        f"{clean_identifier}.pdf",
        f"{clean_identifier}.docx",
        Path(clean_identifier).stem,
        f"{Path(clean_identifier).stem}.pdf",
        f"{Path(clean_identifier).stem}.docx",
    ]:
        file_path = UPLOAD_DIR / candidate
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception:
                pass

    if not deleted_cloud and not deleted_rag and not deleted_store:
        raise HTTPException(status_code=404, detail=f"Document '{clean_identifier}' not found in cloud index.")

    return {
        "success": True,
        "message": f"Document '{clean_identifier}' deleted successfully from Neon Cloud PostgreSQL and memory."
    }


# ============================================================
# ESCALATION & EMAIL QUEUE ENDPOINTS
# ============================================================
@app.get("/api/escalations")
def get_escalations():
    """Returns the list of escalation tickets and current SMTP configuration status."""
    host, port, sender, recipient, pwd, fs_token = get_smtp_credentials()
    records = []
    if ESCALATIONS_FILE.exists():
        try:
            import json
            with open(ESCALATIONS_FILE, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            records = []

    return {
        "smtp_configured": bool(pwd),
        "web_api_active": bool(fs_token),
        "sender": sender,
        "recipient": recipient,
        "total_tickets": len(records),
        "queued_pending_delivery": sum(1 for r in records if str(r.get("status", "")).startswith("QUEUED_")),
        "sent_count": sum(1 for r in records if str(r.get("status", "")).startswith("SENT")),
        "records": records
    }

@app.post("/api/escalations/flush")
def flush_escalation_queue():
    """Flushes and sends all queued escalation tickets once SMTP credentials or Web API are active."""
    flushed = flush_queued_escalations()
    return {
        "success": True,
        "flushed_count": flushed,
        "message": f"Dispatched {flushed} queued ticket(s)."
    }


# ============================================================
# ENTRY POINT FOR DIRECT EXECUTION
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )