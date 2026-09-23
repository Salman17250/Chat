import json
import hashlib
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.config import INDEX_FILE, DATA_DIR

class BaseVectorStore(ABC):
    @abstractmethod
    def add_document(self, filename: str, chunks: List[Dict[str, Any]], file_hash: Optional[str] = None) -> bool:
        pass

    @abstractmethod
    def get_all_documents(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_document(self, filename: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def delete_document(self, filename: str) -> bool:
        pass

    @abstractmethod
    def clear(self) -> bool:
        pass


class JSONVectorStore(BaseVectorStore):
    def __init__(self, index_path: Path = INDEX_FILE):
        self.index_path = index_path
        self._lock = threading.Lock()
        self._cache: Optional[Dict[str, Any]] = None
        self._load()

    def _load(self) -> Dict[str, Any]:
        with self._lock:
            if not self.index_path.exists():
                self._cache = {"documents": []}
                return self._cache
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if not isinstance(data, dict) or "documents" not in data:
                        data = {"documents": []}
                    self._cache = data
                    return self._cache
            except Exception:
                self._cache = {"documents": []}
                return self._cache

    def _save(self) -> bool:
        with self._lock:
            if self._cache is None:
                return False
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            tmp_path = DATA_DIR / f"{self.index_path.stem}.tmp"
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(self._cache, f, ensure_ascii=False, indent=2)
                tmp_path.replace(self.index_path)
                return True
            except Exception:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
                return False

    def add_document(self, filename: str, chunks: List[Dict[str, Any]], file_hash: Optional[str] = None) -> bool:
        with self._lock:
            if self._cache is None:
                self._load()

            documents = self._cache.get("documents", [])
            # Remove existing document with same filename
            documents = [doc for doc in documents if doc.get("filename") != filename]

            # Calculate pages
            pages = set()
            for chunk in chunks:
                if "page" in chunk:
                    pages.add(chunk["page"])

            doc_entry = {
                "filename": filename,
                "file_hash": file_hash or "",
                "total_chunks": len(chunks),
                "total_pages": len(pages) if pages else 1,
                "chunks": chunks
            }
            documents.append(doc_entry)
            self._cache["documents"] = documents

        return self._save()

    def get_all_documents(self) -> List[Dict[str, Any]]:
        with self._lock:
            if self._cache is None:
                self._load()
            return self._cache.get("documents", [])

    def get_document(self, filename: str) -> Optional[Dict[str, Any]]:
        docs = self.get_all_documents()
        for doc in docs:
            if doc.get("filename") == filename:
                return doc
        return None

    def delete_document(self, filename: str) -> bool:
        with self._lock:
            if self._cache is None:
                self._load()
            documents = self._cache.get("documents", [])
            target = filename.strip().lower()
            if not target:
                return False

            matched = False
            new_docs = []
            for doc in documents:
                doc_filename = doc.get("filename", "")
                doc_lower = doc_filename.lower()
                doc_stem = Path(doc_filename).stem.lower()

                # Match exact, or stem (without .pdf), or with .pdf appended
                if (
                    doc_lower == target
                    or doc_stem == target
                    or doc_lower == f"{target}.pdf"
                    or doc_stem == Path(target).stem
                ):
                    matched = True
                else:
                    new_docs.append(doc)

            if not matched:
                return False

            self._cache["documents"] = new_docs
        return self._save()

    def clear(self) -> bool:
        with self._lock:
            self._cache = {"documents": []}
        return self._save()

# Global store instance
_global_store: Optional[JSONVectorStore] = None
_store_lock = threading.Lock()

def get_vector_store() -> JSONVectorStore:
    global _global_store
    if _global_store is None:
        with _store_lock:
            if _global_store is None:
                _global_store = JSONVectorStore()
    return _global_store
