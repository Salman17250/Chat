import numpy as np
from typing import List, Dict, Any, Tuple
from app.config import CANDIDATE_POOL_SIZE, MIN_SIMILARITY, RELATIVE_SCORE_MARGIN
from app.storage import get_vector_store
from app.embeddings import get_embedding_engine
from app.typo import get_cached_vocabulary, correct_query_for_retrieval

def retrieve_candidates(query: str, pool_size: int = CANDIDATE_POOL_SIZE) -> Tuple[List[Dict[str, Any]], str]:
    """
    Dynamically retrieves candidate chunks across all indexed documents.
    Returns (candidate_chunks, corrected_query).
    """
    store = get_vector_store()
    documents = store.get_all_documents()

    if not documents:
        return [], query

    # 1. Algorithmic typo correction based on document vocabulary (cached)
    vocabulary = get_cached_vocabulary(documents)
    corrected_query = correct_query_for_retrieval(query, vocabulary)

    # 2. Query embedding
    embedder = get_embedding_engine()
    q_vec = np.array(embedder.embed_query(corrected_query), dtype=np.float32)

    # 3. Global candidate evaluation
    all_candidates = []

    for doc in documents:
        filename = doc.get("filename", "")
        chunks = doc.get("chunks", [])

        for chunk in chunks:
            emb = chunk.get("embedding")
            if not emb:
                continue

            c_vec = np.array(emb, dtype=np.float32)
            score = float(np.dot(q_vec, c_vec))

            all_candidates.append({
                "filename": filename,
                "chunk_index": chunk.get("index", 0),
                "page": chunk.get("page", 1),
                "text": chunk.get("text", ""),
                "sentences": chunk.get("sentences", []),
                "score": round(score, 4),
            })

    if not all_candidates:
        return [], corrected_query

    # Sort globally by score descending
    all_candidates.sort(key=lambda x: x["score"], reverse=True)

    top_score = all_candidates[0]["score"]
    if top_score < MIN_SIMILARITY:
        return [], corrected_query

    # 4. Dynamic thresholding: select candidates within relative margin of top score
    selected_candidates = []
    min_dynamic_threshold = max(MIN_SIMILARITY, top_score * (1.0 - RELATIVE_SCORE_MARGIN))

    for cand in all_candidates[:pool_size]:
        if cand["score"] >= min_dynamic_threshold:
            selected_candidates.append(cand)

    return selected_candidates, corrected_query
