import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
MODELS_DIR = BASE_DIR / "models"

# Embedding model (runs locally, ~90MB)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Ollama model name — run `ollama pull gemma2:2b` before using
LLM_MODEL = "gemma2:2b"
LLM_BASE_URL = "http://localhost:11434"

# Chunking
CHUNK_SIZE = 400       # tokens
CHUNK_OVERLAP = 60

# Retrieval
TOP_K = 5
SIMILARITY_THRESHOLD = 0.3

# FAISS index path
FAISS_INDEX_PATH = EMBEDDINGS_DIR / "index.faiss"
METADATA_PATH = EMBEDDINGS_DIR / "metadata.json"

# Supported file types
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c", ".html"}

DATA_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
