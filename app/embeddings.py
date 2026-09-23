import threading
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from app.config import EMBED_MODEL_NAME

CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class EmbeddingEngine:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(EmbeddingEngine, cls).__new__(cls)
                    cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        self.model = SentenceTransformer(EMBED_MODEL_NAME)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        text = text.strip()
        if not text:
            return [0.0] * 384
        embedding = self.model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embedding[0].tolist()

    def embed_matrix(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return np.array(embeddings, dtype=np.float32)


class CrossEncoderEngine:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(CrossEncoderEngine, cls).__new__(cls)
                    cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        self.model = CrossEncoder(CROSS_ENCODER_MODEL_NAME)

    def predict(self, pairs: List[List[str]]) -> np.ndarray:
        if not pairs:
            return np.array([], dtype=np.float32)
        scores = self.model.predict(pairs, show_progress_bar=False)
        return np.array(scores, dtype=np.float32)


def get_embedding_engine() -> EmbeddingEngine:
    return EmbeddingEngine()

def get_cross_encoder_engine() -> CrossEncoderEngine:
    return CrossEncoderEngine()
