import threading
import time
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple, List

class ThreadSafeLRU:
    """Thread-safe generic LRU cache with TTL support."""
    def __init__(self, maxsize: int = 5000, ttl_seconds: float = 3600.0):
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                return None
            ts, val = self._cache[key]
            if time.time() - ts > self.ttl_seconds:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return val

    def set(self, key: str, value: Any):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (time.time(), value)
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)

    def clear(self):
        with self._lock:
            self._cache.clear()

    def __len__(self):
        with self._lock:
            return len(self._cache)

class MultiTierCache:
    """
    High-throughput in-process L1 Cache Tier.
    Provides isolated sub-caches for:
      - Query Embeddings (eliminates repeated vector inference)
      - Query Plans & Classifications
      - Verified Responses (sub-millisecond instant return)
    """
    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super(MultiTierCache, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        # L1 Query Embedding Cache (text -> normalized vector)
        self.embedding_cache = ThreadSafeLRU(maxsize=10000, ttl_seconds=86400.0)
        # L1 Response Cache ((tenant_id:query) -> response)
        self.response_cache = ThreadSafeLRU(maxsize=5000, ttl_seconds=1800.0)
        # L1 Query Plan Cache (query -> QueryPlan)
        self.plan_cache = ThreadSafeLRU(maxsize=5000, ttl_seconds=3600.0)

    def get_embedding(self, text: str) -> Optional[List[float]]:
        clean = text.strip().lower()
        return self.embedding_cache.get(clean)

    def set_embedding(self, text: str, vector: List[float]):
        if vector and any(v != 0.0 for v in vector[:10]):
            clean = text.strip().lower()
            self.embedding_cache.set(clean, vector)

    def get_response(self, tenant_id: str, query: str) -> Optional[Any]:
        key = f"{tenant_id}:{query.strip().lower()}"
        return self.response_cache.get(key)

    def set_response(self, tenant_id: str, query: str, response: Any):
        key = f"{tenant_id}:{query.strip().lower()}"
        self.response_cache.set(key, response)

    def clear_tenant(self, tenant_id: str):
        # Clears all responses for a specific tenant upon document update
        prefix = f"{tenant_id}:"
        with self.response_cache._lock:
            keys_to_del = [k for k in self.response_cache._cache if k.startswith(prefix)]
            for k in keys_to_del:
                del self.response_cache._cache[k]

    def clear_all(self):
        # Clears all cached responses and plans
        self.response_cache.clear()
        self.plan_cache.clear()

    def stats(self) -> Dict[str, int]:
        return {
            "embedding_cache_size": len(self.embedding_cache),
            "response_cache_size": len(self.response_cache),
            "plan_cache_size": len(self.plan_cache),
        }

def get_multi_cache() -> MultiTierCache:
    return MultiTierCache()
