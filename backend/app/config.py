"""
Complete environment configuration for demo backend.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent.absolute()

# Ensure environment variables are loaded from the absolute path of .env
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

# Database Configuration
# No hardcoded defaults for credentials — must be set in .env
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_USER = os.getenv('DB_USER', '')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')
DB_NAME = os.getenv('DB_NAME', '')

# DB credentials are optional at startup — users supply them via the frontend UI
# per-request. The DB_* vars here are only used by the legacy /search endpoint.

# Connection Pool
DB_POOL_MIN = int(os.getenv('DB_POOL_MIN', '2'))
DB_POOL_MAX = int(os.getenv('DB_POOL_MAX', '10'))



# Application paths — relative paths are resolved against BASE_DIR
def _resolve(env_key: str, default: Path) -> Path:
    val = os.getenv(env_key)
    if not val:
        return default
    p = Path(val)
    return p if p.is_absolute() else BASE_DIR / p

INDEX_DIR = str(_resolve('INDEX_DIR', BASE_DIR / "faiss_indexes"))
CONFIGS_DIR = _resolve('CONFIGS_DIR', BASE_DIR / "configs")
HNSW_M = int(os.getenv('HNSW_M', '32'))  # HNSW graph connectivity

# Server Configuration
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', '8000'))
RELOAD = os.getenv('RELOAD', 'false').lower() == 'true'

# CORS (for React frontend)
CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:3001,http://localhost:5173').split(',')

# Embedding models available in demo
AVAILABLE_EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",  
    "intfloat/e5-large-v2",            
    "BAAI/bge-m3",                             
    "sentence-transformers/all-mpnet-base-v2", 
]

# FAISS index types available in demo
AVAILABLE_INDEX_TYPES = [
    "hnsw",  
    "flat", 
    "ivf",   
]

# Model dimension mapping (needed for index creation)
MODEL_DIMENSIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "intfloat/e5-large-v2": 1024,
    "BAAI/bge-m3": 1024,
    "sentence-transformers/all-mpnet-base-v2": 768,
}

# Model short names for file naming
MODEL_SHORT_NAMES = {
    "sentence-transformers/all-MiniLM-L6-v2": "minilm",
    "intfloat/e5-large-v2": "e5",
    "BAAI/bge-m3": "bge",
    "sentence-transformers/all-mpnet-base-v2": "mpnet",
}

# Default model and embedding dimension
DEFAULT_MODEL_NAME = os.getenv('MODEL_NAME', 'BAAI/bge-m3')
DEFAULT_EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', '1024'))

# Index maintenance check interval in seconds (for listener script)
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', '60'))