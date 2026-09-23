from pathlib import Path

# ============================================================
# DIRECTORIES & PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "rag_data"
INDEX_FILE = DATA_DIR / "index.json"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# EMBEDDING CONFIG
# ============================================================
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# ============================================================
# OLLAMA SMART STRUCTURING CONFIG (INGESTION-ONLY)
# ============================================================
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_STRUCTURING_MODEL = "qwen2.5:1.5b"
ENABLE_LLM_STRUCTURING = False

# ============================================================
# CHUNKING CONFIG
# ============================================================
CHUNK_SIZE_WORDS = 150
CHUNK_OVERLAP_WORDS = 40

# ============================================================
# RETRIEVAL & QA CONFIG
# ============================================================
# Candidate pool dynamically fetched before sentence-level re-ranking
CANDIDATE_POOL_SIZE = 6

# Minimum chunk similarity threshold to consider evidence
MIN_SIMILARITY = 0.20

# Overall confidence threshold required to return an answer vs "not found"
CONFIDENCE_THRESHOLD = 0.25

# Score margin relative to top candidate for dynamic candidate filtering
RELATIVE_SCORE_MARGIN = 0.35

# ============================================================
# FUZZY TYPO CORRECTION CONFIG
# ============================================================
# Purely algorithmic Levenshtein correction using document vocabulary
FUZZY_MAX_EDIT_DISTANCE = 1
FUZZY_MIN_WORD_LEN = 4

# ============================================================
# CORS
# ============================================================
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*"
]
