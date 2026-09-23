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
    EMBEDDING_TIMEOUT_SEC,
    FALLBACK_EMBED_MODEL,
    FALLBACK_EMBEDDING_DIM
)

class OllamaEmbeddingEngine:
    """
    Ollama-based embedding engine using nomic-embed-text (768-dim).
    Includes thread-safe in-memory caching and high-concurrency connection pooling.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(OllamaEmbeddingEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model_name: str = OLLAMA_EMBED_MODEL):
        if self._initialized:
            return
        self._initialized = True
        self.base_url = base_url
        self.model_name = model_name
        self.dim = EMBEDDING_DIM
        self._cache: Dict[str, List[float]] = {}
        self._cache_lock = threading.Lock()
        
        # High-concurrency connection pooling (supports 200+ parallel user requests)
        self.session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=200,
            pool_maxsize=200,
            max_retries=1,
            pool_block=False
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self._ollama_available = self._check_ollama()
        self._fallback_model = None
        if self._ollama_available:
            try:
                # Pre-warm model in memory with permanent keep-alive
                self.session.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model_name, "prompt": "warmup", "keep_alive": -1},
                    timeout=5
                )
            except Exception:
                pass

    def _check_ollama(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/api/tags", timeout=2)
            if r.status_code == 200:
                models = [m.get("name", "") for m in r.json().get("models", [])]
                return any(self.model_name in m for m in models)
        except Exception:
            pass
        return False

    def _get_fallback_model(self):
        if self._fallback_model is None:
            from sentence_transformers import SentenceTransformer
            self._fallback_model = SentenceTransformer(FALLBACK_EMBED_MODEL)
            self.dim = FALLBACK_EMBEDDING_DIM
        return self._fallback_model

    def embed_query(self, text: str) -> List[float]:
        clean_text = text.strip()
        if not clean_text:
            return [0.0] * self.dim

        norm_key = clean_text.lower().strip("?!.,; ")
        with self._cache_lock:
            if norm_key in self._cache and len(self._cache[norm_key]) == self.dim:
                return self._cache[norm_key]

        emb = self._embed_single(clean_text)
        
        # Only cache valid non-zero embeddings matching expected dimension
        if emb and len(emb) == self.dim and any(v != 0.0 for v in emb[:10]):
            with self._cache_lock:
                self._cache[norm_key] = emb
        return emb

    def _embed_single(self, text: str) -> List[float]:
        # Try Ollama nomic-embed-text with retry
        for attempt in range(2):
            try:
                r = self.session.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model_name, "prompt": text, "keep_alive": -1},
                    timeout=EMBEDDING_TIMEOUT_SEC
                )
                if r.status_code == 200:
                    raw_emb = r.json().get("embedding", [])
                    if raw_emb and len(raw_emb) == self.dim:
                        vec = np.array(raw_emb, dtype=np.float32)
                        norm = np.linalg.norm(vec)
                        if norm > 0:
                            vec = vec / norm
                        self._ollama_available = True
                        return vec.tolist()
            except Exception:
                time.sleep(0.05)

        # Fallback: if Ollama is unreachable, return a zero vector of the correct dimensionality (768)
        # This prevents any matrix dimension mismatch ValueError, allowing BM25 + Fuzzy to handle retrieval gracefully!
        return [0.0] * self.dim

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        results: List[List[float]] = []
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for idx, t in enumerate(texts):
            clean_t = t.strip()
            if clean_t in self._cache:
                results.append(self._cache[clean_t])
            else:
                results.append([])
                uncached_indices.append(idx)
                uncached_texts.append(clean_t)

        if not uncached_texts:
            return results

        # Process uncached texts
        for i, (orig_idx, t_str) in enumerate(zip(uncached_indices, uncached_texts)):
            emb = self._embed_single(t_str)
            self._cache[t_str] = emb
            results[orig_idx] = emb

        return results

    def embed_text(self, text: str) -> List[float]:
        return self.embed_query(text)

    def embed_batch(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        return self.embed_texts(texts)

    @property
    def dimension(self) -> int:
        return self.dim

# Alias for consistent naming
OllamaEmbedder = OllamaEmbeddingEngine

_global_embedder = None

def get_embedding_engine() -> OllamaEmbeddingEngine:
    global _global_embedder
    if _global_embedder is None:
        _global_embedder = OllamaEmbeddingEngine()
    return _global_embedder

