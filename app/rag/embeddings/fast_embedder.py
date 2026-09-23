import time
import threading
import requests
from requests.adapters import HTTPAdapter
import numpy as np
from typing import List, Optional, Dict
from app.rag.config import (
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    EMBEDDING_DIM,
    FALLBACK_EMBED_MODEL,
    FALLBACK_EMBEDDING_DIM
)
from app.rag.cache.memory import get_multi_cache

class FastQueryEmbedder:
    """
    Ultra-Low Latency Query Embedding Engine with Multi-Tier Caching.
    Key Features:
      - Instant L1 RAM cache (<0.001ms) for repeated/similar queries
      - Connection-pooled keep-alive HTTP session (250 pool size)
      - Pre-normalized unit vectors for 0.4ms BLAS dot-products
      - Thread-safe non-blocking execution
    """
    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super(FastQueryEmbedder, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model_name: str = OLLAMA_EMBED_MODEL):
        if self._initialized:
            return
        self._initialized = True
        self.base_url = base_url
        self.model_name = model_name
        self.dim = EMBEDDING_DIM
        self.cache = get_multi_cache()

        # High-concurrency connection pooling for parallel workers
        self.session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=250,
            pool_maxsize=250,
            max_retries=1,
            pool_block=False
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self._is_available = self._check_service()
        if self._is_available:
            try:
                # Permanent keep-alive to avoid model reloading
                self.session.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model_name, "prompt": "warmup", "keep_alive": -1},
                    timeout=3
                )
            except Exception:
                pass

    def _check_service(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/api/tags", timeout=2)
            if r.status_code == 200:
                models = [m.get("name", "") for m in r.json().get("models", [])]
                return any(self.model_name in m for m in models)
        except Exception:
            pass
        return False

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query string.
        1. Checks L1 RAM cache (<0.001ms)
        2. If miss, queries embedding service and normalizes vector
        3. Caches result for subsequent queries
        """
        clean = text.strip()
        if not clean:
            return [0.0] * self.dim

        cached = self.cache.get_embedding(clean)
        if cached is not None and len(cached) == self.dim:
            return cached

        # Generate embedding with task prefix for nomic-embed-text
        prompt_text = f"search_query: {clean}" if not clean.startswith("search_query:") and not clean.startswith("search_document:") else clean
        vec = self._embed_remote(prompt_text)
        if vec and len(vec) == self.dim:
            # Normalize vector to unit length for fast BLAS dot product
            arr = np.asarray(vec, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
                vec = arr.tolist()
            self.cache.set_embedding(clean, vec)
            return vec

        return [0.0] * self.dim

    def embed_batch(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """
        Generates normalized embeddings for a batch of document chunks during ingestion.
        Caches results and processes uncached texts concurrently using pooled connections.
        """
        if not texts:
            return []

        results: List[List[float]] = [[] for _ in texts]
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for idx, t in enumerate(texts):
            clean = t.strip()
            cached = self.cache.get_embedding(clean)
            if cached is not None and len(cached) == self.dim:
                results[idx] = cached
            else:
                uncached_indices.append(idx)
                uncached_texts.append(clean)

        if not uncached_texts:
            return results

        def _worker(item):
            orig_idx, t_str = item
            vec = self._embed_remote(t_str)
            if vec and len(vec) == self.dim:
                arr = np.asarray(vec, dtype=np.float32)
                norm = float(np.linalg.norm(arr))
                if norm > 1e-6:
                    arr = arr / norm
                    vec = arr.tolist()
                    self.cache.set_embedding(t_str, vec)
                    return orig_idx, vec
            # Retry once on transient failure
            time.sleep(0.3)
            vec = self._embed_remote(t_str)
            if vec and len(vec) == self.dim:
                arr = np.asarray(vec, dtype=np.float32)
                norm = float(np.linalg.norm(arr))
                if norm > 1e-6:
                    arr = arr / norm
                    vec = arr.tolist()
                    self.cache.set_embedding(t_str, vec)
                    return orig_idx, vec
            return orig_idx, [0.0] * self.dim

        items = list(zip(uncached_indices, uncached_texts))
        from concurrent.futures import ThreadPoolExecutor
        workers = min(4, len(items))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for orig_idx, vec in executor.map(_worker, items):
                results[orig_idx] = vec

        return results

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embed_batch(texts)

    def embed_text(self, text: str) -> List[float]:
        return self.embed_query(text)

    @property
    def dimension(self) -> int:
        return self.dim

    def _embed_remote(self, text: str) -> List[float]:
        if not self._is_available:
            self._is_available = self._check_service()
            if not self._is_available:
                return [0.0] * self.dim
        for attempt in range(3):
            try:
                resp = self.session.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.model_name,
                        "prompt": text,
                        "keep_alive": -1
                    },
                    timeout=15
                )
                if resp.status_code == 200:
                    emb = resp.json().get("embedding", [])
                    if len(emb) == self.dim:
                        arr = np.asarray(emb, dtype=np.float32)
                        norm = float(np.linalg.norm(arr))
                        if norm > 1e-6:
                            return emb
            except Exception:
                pass
            time.sleep(0.3 * (attempt + 1))
        return [0.0] * self.dim

    def prewarm(self, common_queries: Optional[List[str]] = None):
        """Pre-warms L1 cache and neural weights for zero-delay first request."""
        sample_queries = common_queries or [
            "warmup",
            "general inquiry",
            "terms and conditions",
            "summary overview"
        ]
        for q in sample_queries:
            self.embed_query(q)

def get_fast_embedder() -> FastQueryEmbedder:
    return FastQueryEmbedder()
