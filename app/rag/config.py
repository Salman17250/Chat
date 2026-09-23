from pathlib import Path
from typing import Dict, Any

# ============================================================
# DIRECTORIES & PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "rag_data"
INDEX_FILE = DATA_DIR / "index_v2.json"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# OLLAMA EMBEDDING CONFIG
# ============================================================
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768
EMBEDDING_TIMEOUT_SEC = 30
BATCH_EMBED_SIZE = 16

# Fallback sentence-transformer if Ollama is unavailable
FALLBACK_EMBED_MODEL = "all-MiniLM-L6-v2"
FALLBACK_EMBEDDING_DIM = 384

# ============================================================
# CHUNKING CONFIG
# ============================================================
MIN_CHUNK_WORDS = 35
MAX_CHUNK_WORDS = 250
CHUNK_OVERLAP_WORDS = 35

# ============================================================
# RETRIEVAL CANDIDATE POOL SIZES
# ============================================================
TOP_K_VECTOR = 25
TOP_K_KEYWORD = 25
TOP_K_FUZZY = 15
INITIAL_CANDIDATE_POOL = 40
FINAL_TOP_K = 5

# ============================================================
# DETERMINISTIC RERANKING WEIGHTS
# Formula:
# final_score = (semantic * W_SEM) + (keyword * W_KW) + (phrase * W_PHRASE)
#             + (fuzzy * W_FUZZ) + (heading * W_HEAD) + (intent * W_INTENT)
#             + (metadata * W_META)
# ============================================================
SEMANTIC_WEIGHT = 0.38
KEYWORD_WEIGHT = 0.20
HEADING_WEIGHT = 0.16
PHRASE_WEIGHT = 0.12
INTENT_WEIGHT = 0.06
FUZZY_WEIGHT = 0.05
METADATA_WEIGHT = 0.03

# ============================================================
# CONFIDENCE THRESHOLDS
# ============================================================
MIN_CONFIDENCE = 0.32
CONFIDENCE_HIGH = 0.72
CONFIDENCE_MEDIUM = 0.55
CONFIDENCE_LOW = 0.40

# Adjacent Chunk Context Expansion Threshold
ADJACENT_EXPANSION_CONFIDENCE_THRESHOLD = 0.65

# Duplicate suppression overlap threshold
DUPLICATE_OVERLAP_THRESHOLD = 0.85

# ============================================================
# NO ANSWER FOUND STRING
# ============================================================
NO_ANSWER_FOUND_MESSAGE = "Relevant information was not found in the uploaded document."
