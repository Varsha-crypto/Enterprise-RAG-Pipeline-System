import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import asyncio
import psycopg2
import json
import logging
import uuid
import time
import re
import tempfile
import io
import numpy as np
import uvicorn

# Suppress Hugging Face Hub progress bars to avoid Windows UnicodeEncodeError crashes
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Prevent deadlocks on Windows


from sentence_transformers import SentenceTransformer
from logging import config as logging_config
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query, Request, UploadFile, File, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from psycopg2 import sql as pgsql
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from .config import MODEL_SHORT_NAMES
from .pipeline.ingestion import DatabaseSource, DataIngestion
from .pipeline.chunking import TextChunking
from .pipeline.embedding import EmbeddingGeneration
from .pipeline.indexing import IndexBuilding
from .services.faiss_manager import faiss_manager
from .utils.progress_stream import progress_manager  # pub/sub SSE queues

logger = logging.getLogger(__name__)

from .pipeline.orchestrator import PipelineProgress, PipelineOrchestrator, orchestrator

# ============================================================================

from .config import (
    DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME,
    CORS_ORIGINS, INDEX_DIR, CONFIGS_DIR, BASE_DIR,
    AVAILABLE_EMBEDDING_MODELS,
    AVAILABLE_INDEX_TYPES,
    MODEL_DIMENSIONS,
    HOST, PORT, RELOAD
)

from .services.embedding_service import EmbeddingService
from .services.llm_service import llm_service  # local HuggingFace inference
from .db.dynamic_connection import DynamicDatabaseConnection
from .workflows.import_config import import_workflow
from .workflows.build_pipeline import build_workflow

print(f"BASE_DIR: {BASE_DIR}")
print(f"CONFIGS_DIR: {CONFIGS_DIR}")

# Configure logging
LOG_FILE = os.path.join(BASE_DIR, "backend.log")
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# File Handler
file_handler = logging.FileHandler(LOG_FILE, mode='a')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
root_logger.addHandler(file_handler)

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
root_logger.addHandler(console_handler)

logger.info(f"Logging initialized. Output signals: Console & {LOG_FILE}")

# ── Sensitive data log filter ─────────────────────────────────────────────────
class _SensitiveDataFilter(logging.Filter):
    _PATTERNS = [
        (re.compile(r'(password\s*[=:]\s*)[^\s,\'"]+', re.I), r'\1[REDACTED]'),
        (re.compile(r'(token\s*[=:]\s*)[^\s,\'"]+', re.I),    r'\1[REDACTED]'),
        (re.compile(r'(api.?key\s*[=:]\s*)[^\s,\'"]+', re.I), r'\1[REDACTED]'),
    ]
    def filter(self, record: logging.LogRecord) -> bool:
        msg = str(record.getMessage())
        for pat, repl in self._PATTERNS:
            msg = pat.sub(repl, msg)
        record.msg = msg
        record.args = ()
        return True

for _h in root_logger.handlers:
    _h.addFilter(_SensitiveDataFilter())

# ── Rate limiter ─────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

# Initialize FastAPI
app = FastAPI(
    title="Semantic Search API - Complete",
    description="RAG pipeline with multi-model FAISS semantic search",
    version="2.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ─────────────────────────────────────────────────────────────────────
# Restrict to explicit origins from env; never wildcard with credentials=True
_ALLOWED_ORIGINS = [o.strip() for o in CORS_ORIGINS if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)

# ── API Key authentication ────────────────────────────────────────────────────
_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_API_KEY = os.environ.get("SHAKTIDB_API_KEY", "")

async def _require_api_key(key: str = Depends(_API_KEY_HEADER)):
    """Require X-API-Key header on all non-health endpoints."""
    if not _API_KEY:
        # API key not configured — allow access but warn loudly
        logger.warning("SHAKTIDB_API_KEY not set. Running without authentication!")
        return
    if key != _API_KEY:
        logger.warning(f"Rejected request with invalid API key")
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# ── Security headers + request logging middleware ────────────────────────────
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    logger.info(f"INCOMING REQUEST: {request.method} {request.url.path} from {request.client.host if request.client else 'unknown'}")
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    # Only add HSTS if served over HTTPS (don't break local HTTP dev)
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    if response.status_code >= 400:
        logger.warning(f"RESPONSE STATUS: {response.status_code} {request.method} {request.url.path}")
    else:
        logger.info(f"RESPONSE STATUS: {response.status_code}")
    return response

_embedding_service = None

@app.on_event("startup")
async def startup_event():
    global _embedding_service
    logger.info("Starting semantic search backend...")

    # faiss_manager is now the shared global instance
    if _embedding_service is None:
        _embedding_service = EmbeddingService()

    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
    logger.info("Backend startup complete (on-demand indexing active)")

    # Indexes will load on-demand during search calls
    logger.info("Backend startup: ready (lazy indexing enabled)")

    logger.info("=== REGISTERED ROUTES ===")
    for route in app.routes:
        logger.info(f"{route.path} [{route.name}]")
    logger.info("========================")
    logger.info("Startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down semantic search backend...")

    # Unload LLM models (iterate over copy — unload_model mutates the dict)
    for model_name in list(llm_service.loaded_models.keys()):
        llm_service.unload_model(model_name)
        logger.info(f"LLM unloaded: {model_name}")

    # Unload embedding models
    for model_name in list(embedding_service.models.keys()):
        del embedding_service.models[model_name]
        logger.info(f"Embedding model unloaded: {model_name}")
    embedding_service.current_model_name = None

    # Final GPU memory release
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA cache cleared")

    logger.info("Shutdown complete.")



def get_safe_config_name(name: str) -> str:
    """Sanitize config name and guard against path traversal."""
    if not name or not isinstance(name, str):
        raise ValueError("Config name must be a non-empty string")
    safe = re.sub(r'[^\w\-]', '_', name)
    if len(safe) > 200:
        safe = safe[:200]
    # Resolve and verify it stays inside CONFIGS_DIR
    candidate = (CONFIGS_DIR / safe).resolve()
    if not str(candidate).startswith(str(CONFIGS_DIR.resolve())):
        raise ValueError(f"Config name '{name}' would escape the configs directory")
    return safe


# ============================================================================
# BASIC ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "Semantic Search API - System-Agnostic RAG Pipeline",
        "status": "running",
        "available_models": AVAILABLE_EMBEDDING_MODELS,
        "available_index_types": AVAILABLE_INDEX_TYPES,
        "indexed_tables": faiss_manager.get_indexed_tables(),
        "version": "2.0.0"
    }


@app.get("/api/config-options")
async def get_config_options():
    """Get configuration options for frontend (pipeline choice dropdowns)."""
    return {
        "embedding_models": AVAILABLE_EMBEDDING_MODELS,
        "index_types": AVAILABLE_INDEX_TYPES,
        "chunking_strategies": ["fixed_size", "sentence_based"]
    }


@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.post("/api/check-table-exists")
async def check_table_exists(
    db_host: str,
    db_port: int,
    db_user: str,
    db_password: str,
    db_name: str,
    table_name: str
):
    """Check if a table exists in a specific database."""
    try:
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            dbname=db_name,
            connect_timeout=5
        )
        cursor = conn.cursor()
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s
            )
        """, (table_name,))
        exists = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return {"exists": exists}
    except Exception as e:
        logger.error(f"Table check failed: {e}")
        # If DB doesn't exist yet, the table definitely doesn't exist
        if "database" in str(e).lower() and "does not exist" in str(e).lower():
            return {"exists": False}
        raise HTTPException(500, f"Database check failed: {str(e)}")


@app.post("/api/inspect-file")
async def inspect_file(file: UploadFile = File(...)):
    """
    Inspect a structured file (CSV, Excel, Parquet, JSON) and return:
    - Column names
    - Potential embedding columns (by name heuristics or value type)
    - A sample row
    - File metadata
    """
    supported_structured = ['.csv', '.xlsx', '.parquet', '.json']
    suffix = Path(file.filename).suffix.lower() if file.filename else ''

    if suffix not in supported_structured:
        # Non-structured files (txt, docx, etc.) have no schema to inspect
        return {
            "has_structure": False,
            "columns": [],
            "potential_embeddings": [],
            "sample": None,
            "filename": file.filename,
            "format": suffix.lstrip('.').upper() or 'TXT'
        }

    try:
        content = await file.read()
        from .pipeline.ingestion import UploadedFileSource
        source = UploadedFileSource(content, file.filename or "unknown")
        inspection = source.inspect()

        return {
            "has_structure": inspection.get("has_structure", False),
            "columns": inspection.get("columns", []),
            "potential_embeddings": inspection.get("potential_embeddings", []),
            "sample": inspection.get("sample"),
            "filename": file.filename,
            "format": suffix.lstrip('.').upper(),
            "size_bytes": len(content),
            "error": inspection.get("error")
        }
    except Exception as e:
        logger.error(f"File inspection failed: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to inspect file: {str(e)}")


@app.get("/health")
async def health_check():
    """Public liveness probe — returns minimal info only."""
    return {"status": "ok"}


@app.get("/debug/health", dependencies=[Depends(_require_api_key)])
async def debug_health():
    """Detailed health — requires valid API key."""
    global _embedding_service
    return {
        "embedding_service_initialized": _embedding_service is not None,
        "faiss_indexed_tables": faiss_manager.get_indexed_tables(),
    }


# ============================================================================
# SEARCH ENDPOINTS
# ============================================================================

@app.get("/search")
async def search(
    query: str = Query(..., description="Search query text"),
    embedding_model: str = Query(..., description="Embedding model to use"),
    index_type: str = Query(..., description="FAISS index type to use"),
    db_table: str = Query("documents", description="Optional table name"),
    top_k: int = Query(5, ge=1, le=100, description="Number of results to return")
):
    """Perform semantic search with user-selected model and index type."""
    start_time = time.time()

    if embedding_model not in AVAILABLE_EMBEDDING_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid embedding model. Available: {AVAILABLE_EMBEDDING_MODELS}")
    if index_type not in AVAILABLE_INDEX_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid index type. Available: {AVAILABLE_INDEX_TYPES}")

    model_short = MODEL_SHORT_NAMES.get(embedding_model, embedding_model.split('/')[-1].lower())
    # Note: /search endpoint uses default DB_NAME as it doesn't have a config context
    table_name = f"{DB_NAME}__{db_table}__{model_short}_{index_type}"

    if not faiss_manager.is_table_indexed(table_name):
        available = faiss_manager.get_indexed_tables()
        raise HTTPException(status_code=404, detail=f"Index '{table_name}' not found. Available: {available}")

    try:
        encode_start = time.time()
        loop = asyncio.get_running_loop()
        query_embedding = await loop.run_in_executor(
            None,
            lambda: _embedding_service.encode(query, model_name=embedding_model, normalize=True) # type: ignore
        )
        encode_time = time.time() - encode_start

        faiss_start = time.time()
        loop = asyncio.get_running_loop()
        distances, indices = await loop.run_in_executor(
            None,
            lambda: faiss_manager.search(table_name, query_embedding, top_k)
        )
        faiss_time = time.time() - faiss_start

        db_ids = faiss_manager.get_database_ids(table_name, indices[0])

        if not db_ids:
            return {"results": [], "total": 0, "query": query}

        # Map each DB id → its FAISS distance before the DB fetch.
        # FAISS may return -1 for missing slots; get_database_ids already filters those,
        # so valid_positions tracks which slots in distances[0] correspond to db_ids.
        valid_positions = [j for j, idx in enumerate(indices[0]) if idx != -1]
        id_to_dist = {db_ids[k]: float(distances[0][valid_positions[k]]) for k in range(len(db_ids))}

        db_start = time.time()
        loop = asyncio.get_running_loop()

        def _fetch_search():
            conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, dbname=DB_NAME)
            try:
                cursor = conn.cursor()
                query = pgsql.SQL(
                    "SELECT id, chunks, embedding_model FROM {table} "
                    "WHERE id IN ({placeholders}) AND embedding_model = %s"
                ).format(
                    table=pgsql.Identifier(db_table),
                    placeholders=pgsql.SQL(',').join(pgsql.Placeholder() * len(db_ids)),
                )
                cursor.execute(query, (*db_ids, embedding_model))
                rows = cursor.fetchall()
                cursor.close()
                return rows
            finally:
                conn.close()

        rows = await loop.run_in_executor(None, _fetch_search)
        db_time = time.time() - db_start

        # Sort rows by distance ascending (best match first) and assign rank.
        results = sorted(
            [
                {
                    "id": row[0],
                    "text": row[1],
                    "content": row[1],           # alias for frontend compatibility
                    "embedding_model": row[2],
                    "distance": id_to_dist.get(row[0], 0.0),
                    "similarity_score": round(max(0.0, 1.0 - id_to_dist.get(row[0], 0.0) / 2.0), 4),
                    "score": round(max(0.0, 1.0 - id_to_dist.get(row[0], 0.0) / 2.0), 4),
                    "metadata": {"source": row[2]}
                }
                for row in rows
            ],
            key=lambda r: r["distance"]
        )
        for i, r in enumerate(results):
            r["rank"] = i + 1
        total_time = time.time() - start_time
        return {
            "results": results, "total": len(results), "query": query,
            "embedding_model": embedding_model, "index_type": index_type,
            "timings": {"encoding_ms": round(encode_time * 1000, 2), "faiss_ms": round(faiss_time * 1000, 2),
                        "db_ms": round(db_time * 1000, 2), "total_ms": round(total_time * 1000, 2)}
        }
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unified-search")
@limiter.limit("30/minute")
async def unified_search(
    request: Request,
    query: str,
    config_name: str,
    top_k: int = 5,
):
    """
    Universal search endpoint — FAISS lookup + DB fetch only, no LLM summary.
    Call /generate-summary separately to get the AI summary without blocking results.
    Works for imported configs (Workflow A) and pipeline-built configs (Workflow B).
    """
    start_time = time.time()

    safe_config_name = get_safe_config_name(config_name)
    config_path = CONFIGS_DIR / f"{safe_config_name}.json"
    if not os.path.exists(config_path):
        raise HTTPException(404, f"Configuration '{config_name}' not found")

    with open(config_path, 'r') as f:
        cfg = json.load(f)

    mode = cfg.get('mode', 'unknown')
    if mode == 'needs_pipeline':
        raise HTTPException(400, f"Pipeline not started for config '{config_name}'. Please execute pipeline first.")

    is_partial = (mode == 'partially_ready')
    batches_completed = cfg.get('batches_completed', 0)
    total_batches = cfg.get('total_batches', 1)

    embedding_model = cfg['pipeline']['embedding']['model']
    index_type = cfg['pipeline']['indexing']['type']
    db_config = cfg['database']
    storage_config = cfg['storage']
    search_top_k = top_k  # always honour the caller's top_k

    try:
        encode_start = time.time()
        # Offload encoding to thread pool to avoid blocking the event loop and potential deadlocks on Windows
        loop = asyncio.get_running_loop()
        query_embedding = await loop.run_in_executor(
            None, 
            lambda: _embedding_service.encode(query, model_name=embedding_model, normalize=True) # type: ignore
        )
        # Pad to max dimension (1024) for system-wide consistency
        max_dim = max(MODEL_DIMENSIONS.values())
        if len(query_embedding) < max_dim:
            padded = np.zeros(max_dim, dtype=np.float32)
            padded[:len(query_embedding)] = query_embedding
            query_embedding = padded
        encode_time = time.time() - encode_start

        model_short = MODEL_SHORT_NAMES.get(embedding_model, embedding_model.split('/')[-1].lower())
        table = db_config.get('table', 'documents')
        dbname = db_config.get('dbname', 'unknown_db')
        table_name = f"{dbname}__{table}__{model_short}_{index_type}"
        logger.info(f"Searching table: {table_name}")

        if not faiss_manager.is_table_indexed(table_name):
            stored_index_dir = storage_config['index_path']
            # Fall back to local faiss_indexes dir when stored path doesn't exist (e.g. cross-OS migration)
            resolved_dir = Path(stored_index_dir) if Path(stored_index_dir).exists() else BASE_DIR / "faiss_indexes" / f"{safe_config_name}.index"
            index_path = str(resolved_dir / f"{table_name}.index")
            ids_path = str(resolved_dir / f"{table_name}_ids.pkl")
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    lambda: faiss_manager.load_index_from_path(index_path, ids_path, table_name)
                )
            except Exception as e:
                raise HTTPException(500, f"Failed to load index '{table_name}': {e}")

        faiss_start = time.time()
        logger.info(f"[Search] Querying FAISS for {table_name} (top_k={search_top_k})...")
        # Offload FAISS search as it can be CPU intensive
        loop = asyncio.get_running_loop()
        distances, indices = await loop.run_in_executor(
            None,
            lambda: faiss_manager.search(table_name, query_embedding, search_top_k)
        )
        faiss_time = time.time() - faiss_start
        logger.info(f"[Search] FAISS search complete in {faiss_time:.4f}s")

        db_ids = faiss_manager.get_database_ids(table_name, indices[0])
        logger.info(f"[Search] Found {len(db_ids)} DB IDs: {db_ids}")

        if not db_ids:
            logger.info("[Search] No IDs found, returning empty results.")
            return {"summary": None, "results": [], "total": 0, "query": query, "config_used": config_name,
                    "is_partial": is_partial}

        # Map each DB id → its FAISS distance before the DB fetch.
        # PostgreSQL may return rows in any order; keying by id keeps scores correct.
        valid_positions = [j for j, idx in enumerate(indices[0]) if idx != -1]
        id_to_dist = {db_ids[k]: float(distances[0][valid_positions[k]]) for k in range(len(db_ids))}

        db_start = time.time()
        logger.info(f"[Search] Fetching {len(db_ids)} documents from DB...")
        db_conn = DynamicDatabaseConnection(db_config)
        loop = asyncio.get_running_loop()
        rows = await loop.run_in_executor(
            None,
            lambda: db_conn.fetch_documents(db_ids)
        )
        db_time = time.time() - db_start
        logger.info(f"[Search] DB fetch complete in {db_time:.4f}s. Rows fetched: {len(rows)}")

        # Sort by distance ascending (best match first) and assign rank.
        results = sorted(
            [
                {
                    "id": row[0],
                    "text": row[1],
                    "content": row[1],           # alias for frontend compatibility
                    "embedding_model": row[2],
                    "distance": id_to_dist.get(row[0], 0.0),
                    "similarity_score": round(max(0.0, 1.0 - id_to_dist.get(row[0], 0.0) / 2.0), 4),
                    "score": round(max(0.0, 1.0 - id_to_dist.get(row[0], 0.0) / 2.0), 4),
                    "metadata": {"source": row[2]}
                }
                for row in rows
            ],
            key=lambda r: r["distance"]
        )
        for i, r in enumerate(results):
            r["rank"] = i + 1
        total_time = time.time() - start_time
        logger.info(f"[Search] SEARCH SUCCESS: Total time {total_time:.4f}s")

        response = {
            "summary": None,
            "results": results, "total": len(results), "query": query,
            "config_used": config_name, "config_type": mode, "is_partial": is_partial,
            "database_accessed": f"{db_config['host']}:{db_config['port']}/{db_config['dbname']}",
            "timings": {"encoding_ms": round(encode_time * 1000, 2), "faiss_ms": round(faiss_time * 1000, 2),
                        "db_ms": round(db_time * 1000, 2), "total_ms": round(total_time * 1000, 2)},
            "metadata": {"embedding_model": embedding_model, "index_type": index_type, "top_k": search_top_k}
        }
        if is_partial:
            response["batches_info"] = {"completed": batches_completed, "total": total_batches,
                                        "percentage": round((batches_completed / total_batches) * 100, 2)}
            response["warning"] = f"Partial results: {batches_completed}/{total_batches} batches complete."
        return response
    except Exception as e:
        logger.error(f"Unified search error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/generate-summary")
@limiter.limit("10/minute")
async def generate_summary(
    request: Request,
    query: str,
    chunks: List[str],
    model_name: str = "qwen3-0.6b",
    system_prompt: Optional[str] = Query(default=None),
    max_tokens: int = 1200,
):
    """
    Generate an AI summary from already-retrieved document chunks.
    Called separately from /unified-search so results are never blocked by LLM latency.
    """
    if not chunks:
        return {"summary": None}

    try:
        loop = asyncio.get_running_loop()
        sum_res = await loop.run_in_executor(
            None,
            lambda: llm_service.generate_summary(
                query=query,
                retrieved_chunks=chunks,
                model_name=model_name,
                max_new_tokens=max_tokens,
                system_prompt=system_prompt or None,
            )
        )
        return {"summary": sum_res.get("summary")}
    except RuntimeError as e:
        logger.warning(f"LLM disabled: {e}")
        return {"summary": None}
    except Exception as e:
        logger.error(f"Summary generation error: {e}", exc_info=True)
        return {"summary": None}


@app.post("/generate-summary-stream")
@limiter.limit("10/minute")
async def generate_summary_stream(
    request: Request,
    query: str,
    chunks: List[str],
    model_name: str = "qwen3-0.6b",
    system_prompt: Optional[str] = Query(default=None),
    max_tokens: int = 1200,
):
    """Stream AI summary tokens via SSE as they are generated (true token-by-token)."""
    import json as _json
    import threading as _threading

    async def _stream():
        if not chunks:
            yield 'data: {"type":"done"}\n\n'
            return
        try:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()
            _SENTINEL = object()

            def _run_in_thread():
                try:
                    for tok_type, tok_text in llm_service.stream_summary(
                        query=query,
                        retrieved_chunks=chunks,
                        model_name=model_name,
                        max_new_tokens=max_tokens,
                        system_prompt=system_prompt or None,
                    ):
                        asyncio.run_coroutine_threadsafe(queue.put((tok_type, tok_text)), loop)
                except Exception as exc:
                    asyncio.run_coroutine_threadsafe(queue.put(("error", str(exc))), loop)
                finally:
                    asyncio.run_coroutine_threadsafe(queue.put(_SENTINEL), loop)

            _threading.Thread(target=_run_in_thread, daemon=True).start()

            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                tok_type, tok_text = item
                if tok_type == "error":
                    yield f'data: {_json.dumps({"type": "error", "message": tok_text})}\n\n'
                    break
                if tok_text:
                    yield f'data: {_json.dumps({"type": tok_type, "token": tok_text})}\n\n'

            yield 'data: {"type":"done"}\n\n'
        except Exception as e:
            logger.error(f"Summary stream error: {e}", exc_info=True)
            yield f'data: {_json.dumps({"type": "error", "message": str(e)})}\n\n'

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ============================================================================
# WORKFLOW A: IMPORT CONFIG & SEARCH
# ============================================================================

@app.post("/import-config")
async def import_config(file: UploadFile):
    """Import and validate RAG configuration from uploaded JSON file."""
    try:
        content = await file.read()
        cfg = json.loads(content)
        validation_result = await import_workflow.validate_config(cfg)
        
        config_name = cfg.get('config_name', 'imported_config')
        safe_name = get_safe_config_name(config_name)
        
        # Always return the config so the frontend can populate fields even if validation fails
        validation_result['config'] = cfg
        
        if validation_result['ready']:
            CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
            with open(CONFIGS_DIR / f"{safe_name}.json", 'w') as f:
                json.dump(cfg, f, indent=2)
            logger.info(f"Imported config: {config_name}")
        else:
            logger.warning(f"Imported config {config_name} failed validation: {validation_result.get('errors')}")
            
        return validation_result
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON file")
    except Exception as e:
        logger.error(f"Import failed: {e}", exc_info=True)
        raise HTTPException(500, f"Import process failed: {str(e)}")


# ============================================================================
# WORKFLOW B: FILE UPLOAD & PIPELINE EXECUTION
# ============================================================================

@app.post("/create-pipeline-tracker")
async def create_pipeline_tracker():
    """
    Pre-allocate a progress_id BEFORE execution starts.
    Lets the frontend open SSE before triggering execution (eliminates race condition).
    """
    progress_id = orchestrator.create_pipeline()
    logger.info(f"[SSE] Pre-allocated pipeline tracker: {progress_id}")
    return {
        "success": True,
        "progress_id": progress_id,
        "stream_endpoint": f"/pipeline-progress-stream/{progress_id}"
    }


@app.post("/api/upload-file-for-pipeline")
@limiter.limit("5/minute")
async def upload_file_for_pipeline(
    request: Request,
    file: UploadFile = File(...),
    db_host: Optional[str] = Query(default=""),
    db_port: Optional[str] = Query(default=""),
    db_user: Optional[str] = Query(default=""),
    db_password: Optional[str] = Query(default=""),
    db_name: Optional[str] = Query(default=""),
    db_table: Optional[str] = Query(default="documents"),
    index_path: str = Query(default="./faiss_indexes"),
    embedding_model: str = Query(default="BAAI/bge-m3"),
    index_type: str = Query(default="hnsw"),
    chunking_strategy: str = Query(default="fixed_size"),
    chunk_size: Optional[int] = Query(default=500),
    chunk_overlap: Optional[int] = Query(default=50),
    incremental_mode: Optional[bool] = Query(default=False),
    batch_size: Optional[int] = Query(default=50),
    config_name: Optional[str] = None,
    vector_dim: Optional[int] = Query(default=None),
    hf_token: Optional[str] = Query(default=None),
    text_column: Optional[str] = Query(default=None),
    embedding_column: Optional[str] = Query(default=None),
    schema_json: Optional[str] = Query(None, alias="schema")
):
    """Upload file and create pipeline configuration."""
    if not chunking_strategy or chunking_strategy.strip() == "":
        chunking_strategy = "fixed_size"
    if not embedding_model or embedding_model.strip() == "":
        embedding_model = "BAAI/bge-m3"
    if not index_type or index_type.strip() == "":
        index_type = "hnsw"
    if chunking_strategy not in ['fixed_size', 'sentence_based']:
        raise HTTPException(400, "Invalid chunking strategy. Must be: fixed_size or sentence_based")
    _ALLOWED_EXTENSIONS = {'.txt', '.csv', '.json', '.parquet', '.xlsx', '.docx'}
    _ALLOWED_MIMES = {
        'text/plain', 'text/csv', 'application/csv',
        'application/json', 'application/x-json',
        'application/octet-stream',                              # parquet
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/zip',                                       # xlsx/docx are zip-based
    }

    if not file.filename:
        raise HTTPException(400, "Filename is required")
    _, ext = os.path.splitext(file.filename.lower())
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported extension '{ext}'. Allowed: {', '.join(_ALLOWED_EXTENSIONS)}")

    # Get dynamic vector dimension
    if vector_dim is None:
        vector_dim = MODEL_DIMENSIONS.get(embedding_model, 1024)

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(400, "File is empty")
        max_size = 100 * 1024 * 1024
        if len(content) > max_size:
            raise HTTPException(400, f"File too large. Max: {max_size / 1024 / 1024}MB")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to read file: {e}")

    # MIME type validation using file content (not just extension)
    try:
        import magic as _magic
        detected_mime = _magic.from_buffer(content, mime=True)
        if detected_mime not in _ALLOWED_MIMES:
            raise HTTPException(400, f"File content type '{detected_mime}' is not allowed.")
    except ImportError:
        logger.warning("python-magic not installed — skipping MIME content validation. Run: pip install python-magic")
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"MIME detection failed (non-fatal): {e}")

    _, ext = os.path.splitext(file.filename) if file.filename else ("", ".txt")
    temp_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix=ext)
    temp_file.write(content)
    temp_file.close()

    if not index_path:
        resolved_index_dir = BASE_DIR / "faiss_indexes"
    else:
        resolved_index_dir = Path(index_path)
        if not resolved_index_dir.is_absolute():
            resolved_index_dir = BASE_DIR / resolved_index_dir
    resolved_index_dir.mkdir(parents=True, exist_ok=True)

    db_port_int = 5433
    if db_host and db_port and db_user and db_password and db_name:
        db_port_int = int(db_port)
        try:
            admin_conn = psycopg2.connect(host=db_host, port=db_port_int,
                                          user=db_user, password=db_password,
                                          dbname=db_name, connect_timeout=5)
            admin_conn.autocommit = True
            cursor = admin_conn.cursor()
            
            # Process custom schema if provided
            extra_cols_sql = ""
            custom_schema = {}
            ALLOWED_PG_TYPES = {
                'text', 'varchar', 'character varying', 'char', 'bpchar',
                'integer', 'int', 'int4', 'int8', 'bigint', 'smallint', 'int2',
                'float', 'float4', 'float8', 'real', 'double precision', 'numeric', 'decimal',
                'boolean', 'bool',
                'date', 'timestamp', 'timestamptz', 'timestamp with time zone',
                'timestamp without time zone', 'time', 'timetz',
                'json', 'jsonb', 'uuid',
            }
            if schema_json:
                try:
                    custom_schema = json.loads(schema_json)
                    for col_name, col_type in custom_schema.items():
                        # Avoid duplicating standard columns
                        if col_name.lower() in ['id', 'chunks', 'vector', 'embedding_model', 'source', 'batch_number']:
                            continue
                        # Sanitize column name
                        safe_col = re.sub(r'[^\w]', '_', col_name)
                        # Validate column type against allowlist to prevent SQL injection
                        safe_type = col_type.strip().lower()
                        if safe_type not in ALLOWED_PG_TYPES:
                            logger.warning(f"Rejected invalid column type '{col_type}' for column '{col_name}'")
                            continue
                        extra_cols_sql += f", {safe_col} {safe_type}"
                except Exception as ex:
                    logger.warning(f"Failed to parse custom schema: {ex}")

            # Check if table exists first
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                )
            """, (db_table,))
            table_exists = cursor.fetchone()[0]

            if not table_exists:
                # Table does not exist - automatically CREATE IT!
                logger.info(f"Table {db_table} does not exist, creating it now with dim={vector_dim}...")
                create_sql = pgsql.SQL("""
                    CREATE TABLE {table} (
                        id SERIAL PRIMARY KEY,
                        chunks TEXT NOT NULL,
                        vector VECTOR(%s),
                        embedding_model TEXT NOT NULL,
                        source TEXT DEFAULT 'pipeline',
                        batch_number INTEGER DEFAULT NULL
                    )
                """).format(table=pgsql.Identifier(db_table))
                cursor.execute(create_sql, (vector_dim,))
                # Append validated extra columns one at a time (types already allowlisted above)
                for col_name, col_type in custom_schema.items():
                    if col_name.lower() in ['id', 'chunks', 'vector', 'embedding_model', 'source', 'batch_number']:
                        continue
                    safe_col = re.sub(r'[^\w]', '_', col_name)
                    safe_type = col_type.strip().lower()
                    if safe_type not in ALLOWED_PG_TYPES:
                        continue
                    cursor.execute(pgsql.SQL("ALTER TABLE {table} ADD COLUMN {col} " + safe_type).format(
                        table=pgsql.Identifier(db_table),
                        col=pgsql.Identifier(safe_col),
                    ))
                logger.info(f"Created table {db_table} automatically")
            
            cursor.close()
            admin_conn.close()
            
        except psycopg2.OperationalError as e:
            error_message = str(e)
            if "database" in error_message.lower() and "does not exist" in error_message.lower():
                # Database does not exist - create it
                logger.info(f"Database {db_name} does not exist, creating it now...")
                admin_conn = psycopg2.connect(host=db_host, port=db_port_int,
                                              user=db_user, password=db_password,
                                              dbname="postgres")
                admin_conn.autocommit = True
                cursor = admin_conn.cursor()
                cursor.execute(pgsql.SQL("CREATE DATABASE {}").format(pgsql.Identifier(db_name)))
                cursor.close()
                admin_conn.close()
                logger.info(f"Created database {db_name} automatically")

                # Now connect again to create table
                new_conn = psycopg2.connect(host=db_host, port=db_port_int,
                                            user=db_user, password=db_password,
                                            dbname=db_name)
                cursor = new_conn.cursor()
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                create_sql = pgsql.SQL("""
                    CREATE TABLE {table} (
                        id SERIAL PRIMARY KEY,
                        chunks TEXT NOT NULL,
                        vector VECTOR(%s),
                        embedding_model TEXT NOT NULL,
                        source TEXT DEFAULT 'pipeline',
                        batch_number INTEGER DEFAULT NULL
                    )
                """).format(table=pgsql.Identifier(db_table))
                cursor.execute(create_sql, (vector_dim,))
                for col_name, col_type in custom_schema.items():
                    if col_name.lower() in ['id', 'chunks', 'vector', 'embedding_model', 'source', 'batch_number']:
                        continue
                    safe_col = re.sub(r'[^\w]', '_', col_name)
                    safe_type = col_type.strip().lower()
                    if safe_type not in ALLOWED_PG_TYPES:
                        continue
                    cursor.execute(pgsql.SQL("ALTER TABLE {table} ADD COLUMN {col} " + safe_type).format(
                        table=pgsql.Identifier(db_table),
                        col=pgsql.Identifier(safe_col),
                    ))
                cursor.close()
                new_conn.commit()
                new_conn.close()
                logger.info(f"Created table {db_table} automatically")
                
            else:
                raise HTTPException(400, f"Database connection failed: {error_message}")
    else:
        # No DB credentials provided — require them from environment
        if not DB_HOST or not DB_USER or not DB_PASSWORD or not DB_NAME:
            raise HTTPException(
                400,
                "No database credentials provided and environment defaults (DB_HOST, DB_USER, "
                "DB_PASSWORD, DB_NAME) are not configured. Please provide database connection details."
            )
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        db_host = DB_HOST
        db_user = DB_USER
        db_password = DB_PASSWORD
        db_name = f"rag_db_{timestamp}"
        db_table = f"rag_table_{timestamp}"
        admin_conn = psycopg2.connect(host=db_host, port=db_port_int, user=db_user,
                                      password=db_password, dbname="postgres")
        admin_conn.autocommit = True
        cursor = admin_conn.cursor()
        cursor.execute(pgsql.SQL("CREATE DATABASE {}").format(pgsql.Identifier(db_name)))
        cursor.close()
        admin_conn.close()
        new_conn = psycopg2.connect(host=db_host, port=db_port_int, user=db_user,
                                    password=db_password, dbname=db_name)
        cursor = new_conn.cursor()
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cursor.execute(pgsql.SQL("""
            CREATE TABLE {table} (
                id SERIAL PRIMARY KEY,
                chunks TEXT NOT NULL,
                vector VECTOR(%s),
                embedding_model TEXT NOT NULL,
                source TEXT DEFAULT 'pipeline',
                batch_number INTEGER DEFAULT NULL
            )
        """).format(table=pgsql.Identifier(db_table)), (vector_dim,))
        new_conn.commit()
        cursor.close()
        new_conn.close()

    if not config_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if db_table and db_table != "documents":
            # If user provided a specific table name, use it as part of the config name
            safe_table = re.sub(r'[^\w\-_]', '_', db_table)
            config_name = f"upload_{safe_table}_{timestamp}"
        else:
            safe_filename = re.sub(r'[^\w\-_]', '_', file.filename.replace('.txt', ''))  # type: ignore
            config_name = f"upload_{safe_filename}_{timestamp}"

    cfg = {
        "config_name": config_name, "config_version": "1.0", "mode": "needs_pipeline",
        "database": {"host": db_host, "port": db_port_int, "user": db_user,
                     "password": db_password, "dbname": db_name, "table": db_table},
        "pipeline": {
            "chunking": {"strategy": chunking_strategy, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
            "embedding": {"model": embedding_model, "normalize": True, "batch_size": 32},
            "indexing": {"type": index_type, "parameters": {"M": 32} if index_type == "hnsw" else {}},
            "llm": {"model": "None", "max_tokens": 0, "temperature": 0}
        },
        "search": {"top_k": 5, "similarity_metric": "cosine", "rerank": False},
        "storage": {"index_path": str(resolved_index_dir / f"{config_name}.index"), "index_files": []},
        "source": {
            "type": "file", "filename": file.filename, "temp_path": temp_file.name, "size_bytes": len(content),
            "text_column": text_column,
            "embedding_column": embedding_column
        },
        "hf_token": hf_token,
        "incremental_mode": incremental_mode,
        "batch_size": batch_size if incremental_mode else len(content),
        "batches_completed": 0, "total_batches": 0,
        "created_at": datetime.now().isoformat(), "pipeline_completed": False
    }

    # If manual vector_dim provided, ensure it's recorded in config for future reference
    if vector_dim:
        cfg["pipeline"]["embedding"]["vector_dim"] = vector_dim


    safe_name = get_safe_config_name(config_name)
    CONFIGS_DIR.mkdir(exist_ok=True)
    with open(CONFIGS_DIR / f"{safe_name}.json", 'w') as f:
        json.dump(cfg, f, indent=2)

    estimated_chunks = len(content) // chunk_size  # type: ignore
    return {
        "success": True, "config_name": config_name, "config": cfg,
        "file_info": {"filename": file.filename, "size_kb": round(len(content) / 1024, 2), "estimated_chunks": estimated_chunks},
        "message": "File uploaded. Ready to execute pipeline.",
        "next_step": f"POST /execute-pipeline-from-file with config_name='{config_name}'"
    }


@app.post("/execute-pipeline-from-file")
async def execute_pipeline_from_file(
    config_name: str,
    background_tasks: BackgroundTasks,
    progress_id: Optional[str] = Query(default=None)
):
    """Execute pipeline for uploaded file."""
    safe_config_name = get_safe_config_name(config_name)
    config_path = CONFIGS_DIR / f"{safe_config_name}.json"
    if not os.path.exists(config_path):
        raise HTTPException(404, f"Configuration '{config_name}' not found")

    with open(config_path, 'r') as f:
        cfg = json.load(f)

    if cfg.get('source', {}).get('type') != 'file':
        raise HTTPException(400, "This endpoint is for file-based configs only")

    temp_file_path = cfg['source']['temp_path']
    if not os.path.exists(temp_file_path):
        raise HTTPException(404, "Uploaded file not found")

    if not progress_id:
        progress_id = orchestrator.create_pipeline()
        logger.info(f"[SSE] Created new pipeline tracker on execute: {progress_id}")
    else:
        if progress_id not in orchestrator.active_pipelines:
            logger.warning(f"[SSE] Pre-allocated progress_id {progress_id} not found, creating new slot")
            orchestrator.active_pipelines[progress_id] = PipelineProgress(progress_id)
        logger.info(f"[SSE] Reusing pre-allocated pipeline tracker: {progress_id}")

    async def run_pipeline(pid: str):
        try:
            result = await orchestrator.execute(
                config=cfg, progress_id=pid, source_type='file',
                source_data=temp_file_path, resume_from=None, config_name=config_name
            )
            if 'config' in result:
                with open(config_path, 'w') as f:
                    json.dump(result['config'], f, indent=2)
            try:
                os.remove(temp_file_path)
            except:
                pass
        except asyncio.CancelledError:
            logger.info(f"Pipeline task {pid} was cancelled.")
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
        finally:
            # Always clean up temp file regardless of success/failure
            try:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
            except Exception:
                pass
            orchestrator.unregister_task(pid)

    # Use asyncio.create_task instead of BackgroundTasks for manual cancellation control
    task = asyncio.create_task(run_pipeline(progress_id))
    orchestrator.register_task(progress_id, task)
    return {
        "success": True, "progress_id": progress_id,
        "message": "Pipeline execution started", "config_name": config_name,
        "poll_endpoint": f"/pipeline-progress/{progress_id}"
    }


@app.get("/pipeline-progress/{progress_id}")
async def get_pipeline_progress(progress_id: str):
    """Get pipeline execution progress."""
    progress = orchestrator.get_progress(progress_id)
    if not progress:
        raise HTTPException(404, f"Progress ID '{progress_id}' not found")
    return progress


@app.get("/pipeline-progress-stream/{progress_id}")
async def stream_progress(progress_id: str):
    """Stream real-time progress updates via SSE."""

    async def event_generator():
        queue = asyncio.Queue()
        progress_manager.queues[progress_id].add(queue)
        logger.info(f"[SSE] Client connected to stream for {progress_id}")

        try:
            current_status = orchestrator.get_progress(progress_id)
            if current_status:
                current_status["progress"] = current_status.get("overall_progress", 0)
                logger.info(f"[SSE] Sending initial status for {progress_id}: progress={current_status['progress']}")
                yield f"data: {json.dumps(current_status)}\n\n"
                if current_status.get('pipeline_completed') is True:
                    logger.info(f"[SSE] Pipeline {progress_id} was already completed on connect")
                    return

            HEARTBEAT_INTERVAL = 10
            last_completed = False

            while not last_completed:
                try:
                    update = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_INTERVAL)
                    update["progress"] = update.get("overall_progress", update.get("progress", 0))
                    payload = json.dumps(update)
                    logger.info(f"[SSE] Emitting progress={update.get('progress')} for {progress_id}")
                    yield f"data: {payload}\n\n"
                    await asyncio.sleep(0)
                    if update.get('pipeline_completed') is True:
                        logger.info(f"[SSE] Pipeline {progress_id} completed - closing stream")
                        last_completed = True
                    await asyncio.sleep(0.1)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    logger.debug(f"[SSE] Heartbeat sent for {progress_id}")
                    current = orchestrator.get_progress(progress_id)
                    if current and current.get('pipeline_completed') is True:
                        current["progress"] = current.get("overall_progress", 0)
                        yield f"data: {json.dumps(current)}\n\n"
                        last_completed = True
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logger.info(f"[SSE] Client disconnected from {progress_id}")
        except Exception as e:
            logger.error(f"[SSE] Unexpected error in generator for {progress_id}: {e}", exc_info=True)
        finally:
            progress_manager.queues[progress_id].discard(queue)
            if not progress_manager.queues[progress_id]:
                del progress_manager.queues[progress_id]
            logger.info(f"[SSE] Stream closed for {progress_id}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no", "Content-Type": "text/event-stream"}
    )


# ============================================================================
# CONFIG MANAGEMENT (BOTH WORKFLOWS)
# ============================================================================

@app.post("/api/cancel-pipeline")
async def cancel_pipeline(config_name: str = Query(..., description="Name of the configuration to cancel")):
    """Cancel a running pipeline. Responsibility for cleanup shifts to the orchestrator loop."""
    logger.info(f"[main.py] Cancellation endpoint hit for config: {config_name}")
    try:
        # 1. Look for active pipeline
        progress_id = orchestrator.get_progress_id_by_config(config_name)
        
        if progress_id:
            logger.info(f"[main.py] Found active progress_id {progress_id}. Signaling cancellation...")
            progress = orchestrator.active_pipelines[progress_id]
            
            # If it's already in a terminal state (completed/failed), clean it up NOW.
            # This ensures the frontend doesn't hang at 'Cleaning' if it was already DONE.
            if progress.status in ["completed", "failed", "initialized"]:
                logger.info(f"[main.py] Pipeline {progress_id} is in state '{progress.status}'. Triggering cleanup manually.")
                asyncio.create_task(orchestrator.handle_cancellation(progress_id, progress.config_payload or {}, "Manual Cleanup Request"))
            else:
                # Normal running state — set flag for loop to catch it at batch boundary
                orchestrator.request_cancellation(progress_id)
            
            return {"success": True, "message": "Cancellation/Cleanup signal sent"}
        
        # 2. If no active pipeline, it might be an orphaned run or a finished one. 
        # We can perform a "manual" cleanup here if a config file exists.
        safe_name = get_safe_config_name(config_name)
        config_path = CONFIGS_DIR / f"{safe_name}.json"
        
        if config_path.exists():
            logger.info(f"[main.py] No active pipeline, but config file found. Triggering manual cleanup fallback.")
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            
            # Create a dummy progress_id for the terminal signal if none existed
            # This ensures the frontend (which might still be listening) gets the signal
            fallback_id = "orphaned-cleanup" 
            
            # Use a background task to perform cleanup and EMIT SIGNAL
            # Note: We use handle_cancellation because it emits the 'cancelled' signal
            asyncio.create_task(orchestrator.handle_cancellation(fallback_id, cfg, "Manual Fallback Cleanup"))
            return {"success": True, "message": "Manual cleanup triggered for orphaned pipeline data", "progress_id": fallback_id}

        return {"success": True, "message": "No active pipeline or config found to cancel"}
    except Exception as e:
        logger.error(f"Error in cancel_pipeline endpoint: {e}")
        return {"success": False, "error": str(e)}




@app.post("/export-config")
async def export_config(config_name: str):
    """Export configuration as a downloadable ZIP containing config and index files."""
    import zipfile
    
    safe_config_name = get_safe_config_name(config_name)
    config_path = CONFIGS_DIR / f"{safe_config_name}.json"
    if not os.path.exists(config_path):
        raise HTTPException(404, f"Configuration '{config_name}' not found")
        
    with open(config_path, 'r') as f:
        cfg = json.load(f)
        
    cfg['last_used'] = datetime.now().isoformat()
    json_str = json.dumps(cfg, indent=2)
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        zip_file.writestr(f"{config_name}.json", json_str)
        
        storage_cfg = cfg.get('storage', {})
        index_files = storage_cfg.get('index_files', [])
        index_dir = Path(storage_cfg.get('index_path', ''))
        # index_path is typically the directory itself or a file path in some older configs,
        # but indexing.py incremental mode sets index_path to the directory.
        if index_dir.suffix: 
            index_dir = index_dir.parent

        for file_path_str in index_files:
            file_path = Path(file_path_str)
            if not file_path.is_absolute():
                file_path = index_dir / file_path
                
            # Try multiple resolution paths in case config holds an outdated absolute path
            # but the actual files were generated into the local 'indexes/' directory
            possible_paths = [
                file_path,
                Path('indexes') / Path(file_path_str).name,
                Path('indexes') / f"{config_name}.index" / Path(file_path_str).name,
                Path('indexes') / config_name / Path(file_path_str).name,
                index_dir / Path(file_path_str).name
            ]
            
            found_path = None
            for p in possible_paths:
                if p.exists():
                    found_path = p
                    break
                    
            if found_path:
                zip_file.write(found_path, arcname=found_path.name)
            else:
                logger.warning(f"Index file NOT FOUND: {file_path}. Original string: '{file_path_str}', Base Dir: '{index_dir}'")

    return StreamingResponse(
        iter([zip_buffer.getvalue()]),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={config_name}.zip"}
    )


@app.get("/list-configs")
async def list_configs():
    """List all saved configurations."""
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    configs = []
    for file in CONFIGS_DIR.glob("*.json"):
        try:
            with open(file, 'r') as f:
                cfg = json.load(f)
                configs.append({
                    "config_name": cfg.get('config_name'),
                    "mode": cfg.get('mode'),
                    "pipeline_completed": cfg.get('pipeline_completed', False),
                    "created_at": cfg.get('created_at'),
                    "embedding_model": cfg['pipeline']['embedding']['model'],
                    "database": f"{cfg['database']['host']}:{cfg['database']['port']}"
                })
        except Exception as e:
            logger.error(f"Error reading config {file}: {e}")
    return {"configs": configs}


@app.delete("/config/{config_name}")
async def delete_config(config_name: str):
    """Delete a configuration."""
    safe_config_name = get_safe_config_name(config_name)
    config_path = CONFIGS_DIR / f"{safe_config_name}.json"
    if not os.path.exists(config_path):
        raise HTTPException(404, f"Configuration '{config_name}' not found")
    os.remove(config_path)
    return {"success": True, "message": f"Configuration '{config_name}' deleted"}


# ============================================================================
# DB-TO-DB PIPELINE (WORKFLOW C)
# ============================================================================

class _DbColumnsRequest(BaseModel):
    db_host: str
    db_port: int
    db_user: str
    db_password: str
    db_name: str
    db_table: str


@app.post("/get-source-db-columns")
async def get_source_db_columns(body: _DbColumnsRequest):
    """Get available columns from source database table.
    Credentials are passed in the JSON request body — never in query strings.
    """
    db_host, db_port, db_user = body.db_host, body.db_port, body.db_user
    db_name, db_table = body.db_name, body.db_table
    db_config = {"host": db_host, "port": db_port, "user": db_user,
                 "password": body.db_password, "dbname": db_name, "table": db_table}
    db_conn = DynamicDatabaseConnection(db_config)
    try:
        # Directly call get_connection() first instead of test_connection()
        # This throws the actual exception directly with original message
        test_conn = db_conn.get_connection()
        test_conn.close()
        
        logger.info(f"Database connection test successful: {db_host}:{db_port}")
        columns = db_conn.get_table_columns()
        row_count = db_conn.count_rows_in_table()
        return {"success": True, "database": f"{db_host}:{db_port}/{db_name}",
                "table": db_table, "columns": columns, "row_count": row_count,
                "message": "Select which column contains chunks and which is the ID column"}
    except psycopg2.OperationalError as e:
        error_message = str(e)
        logger.error(f"Database connection failed: {e}")
        
        if "connection refused" in error_message.lower() or "could not connect" in error_message.lower():
            raise HTTPException(400, f"❌ Connection Failed: Cannot reach host {db_host} on port {db_port}. Check if PostgreSQL is running.")
        elif "password authentication failed" in error_message.lower():
            raise HTTPException(400, "❌ Authentication Failed: Wrong username or password.")
        elif "database" in error_message.lower() and "does not exist" in error_message.lower():
            raise HTTPException(404, f"❌ Database '{db_name}' does not exist on this server.")
        else:
            raise HTTPException(400, f"❌ Database Connection Error: {error_message}")
            
    except psycopg2.Error as e:
        error_message = str(e)
        logger.error(f"Database Error: {e}")
        
        if "relation" in error_message.lower() and "does not exist" in error_message.lower():
            raise HTTPException(404, f"❌ Table '{db_table}' does not exist in database '{db_name}'.")
        else:
            raise HTTPException(400, f"❌ Database Error: {error_message}")
            
    except Exception as e:
        error_message = str(e)
        logger.error(f"General Error: {e}")
        raise HTTPException(400, f"❌ Error: {error_message}")


@app.post("/configure-db-source-pipeline")
async def configure_db_source_pipeline(
    source_db_host: str = Query(...), source_db_port: int = Query(...),
    source_db_user: str = Query(...), source_db_password: str = Query(...),
    source_db_name: str = Query(...), source_db_table: str = Query(...),
    source_chunk_column: str = Query(...), source_id_column: str = Query(...),
    target_db_host: Optional[str] = Query(default=""), target_db_port: Optional[str] = Query(default=""),
    target_db_user: Optional[str] = Query(default=""), target_db_password: Optional[str] = Query(default=""),
    target_db_name: Optional[str] = Query(default=""), target_db_table: Optional[str] = Query(default=""),
    target_index_path: Optional[str] = Query(default="./faiss_indexes"),
    embedding_model: str = Query(default="BAAI/bge-m3"), index_type: str = Query(default="hnsw"),
    chunking_strategy: str = Query(default="fixed_size"), chunk_size: int = Query(default=500),
    chunk_overlap: int = Query(default=50), incremental_mode: bool = Query(default=False),
    batch_size: Optional[int] = Query(default=1000),
    config_name: Optional[str] = None,
    vector_dim: Optional[int] = Query(default=None),
    hf_token: Optional[str] = Query(default=None),
    target_text_column: str = Query(default="chunks", description="Text column name in target table (default: chunks for auto-created tables)")
):
    """Configure pipeline with source database as data input."""
    source_db_config = {"host": source_db_host, "port": source_db_port, "user": source_db_user,
                        "password": source_db_password, "dbname": source_db_name, "table": source_db_table}
    try:
        source_conn = DynamicDatabaseConnection(source_db_config)
        if not source_conn.test_connection():
            raise HTTPException(400, "ERROR - Source DB connection failed")
        if not source_conn.validate_columns(source_chunk_column, source_id_column):
            raise HTTPException(400, "Columns not found in source table. Use /get-source-db-columns to see available columns.")
        row_count = source_conn.count_rows_in_table()
        logger.info(f"Source DB validated: {row_count} rows")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Source DB validation failed: {e}")

    # Get dynamic vector dimension
    if vector_dim is None:
        vector_dim = MODEL_DIMENSIONS.get(embedding_model, 1024)

    if target_db_host and target_db_port is not None and target_db_user and target_db_password and target_db_name:
        try:
            target_conn = DynamicDatabaseConnection(
                {"host": target_db_host, "port": target_db_port, "user": target_db_user,
                 "password": target_db_password, "dbname": target_db_name, "table": target_db_table or "documents"})
            if not target_conn.test_connection():
                raise HTTPException(400, "ERROR - Target DB connection failed")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"Target DB connection failed: {e}")
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Use the same DB as source — avoids needing CREATE DATABASE (superuser) privilege.
        # Just create a new table inside the existing database.
        target_db_host = source_db_host
        target_db_port = source_db_port
        target_db_user = source_db_user
        target_db_password = source_db_password
        target_db_name = source_db_name
        target_db_table = f"rag_{source_db_table}_{timestamp}"
        try:
            new_conn = psycopg2.connect(host=target_db_host, port=target_db_port,
                                        user=target_db_user, password=target_db_password, dbname=target_db_name)
            cursor = new_conn.cursor()
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(f"""
                CREATE TABLE {target_db_table} (
                    id SERIAL PRIMARY KEY,
                    chunks TEXT NOT NULL,
                    vector VECTOR({vector_dim}),
                    embedding_model TEXT NOT NULL,
                    source TEXT DEFAULT 'pipeline',
                    batch_number INTEGER DEFAULT NULL
                )
            """)
            new_conn.commit()
            cursor.close()
            new_conn.close()
            logger.info(f"Created target table '{target_db_table}' in '{target_db_name}'")
        except Exception as e:
            raise HTTPException(500, f"Failed to create target table in '{target_db_name}': {e}. "
                                    f"Ensure the user '{source_db_user}' has CREATE TABLE and CREATE EXTENSION privileges.")

    if not config_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = f"db_source_{source_db_name}_{timestamp}"

    cfg = {
        "config_name": config_name, "config_version": "1.0", "mode": "needs_pipeline",
        "database": {"host": target_db_host, "port": target_db_port, "user": target_db_user,
                     "password": target_db_password, "dbname": target_db_name, "table": target_db_table or "documents",
                     "text_column": target_text_column},
        "source_db": {
            "db_config": {"host": source_db_host, "port": source_db_port, "user": source_db_user,
                          "password": source_db_password, "dbname": source_db_name, "table": source_db_table},
            "chunk_column": source_chunk_column, "id_column": source_id_column, "row_count": row_count
        },
        "pipeline": {
            "chunking": {"strategy": chunking_strategy, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
            "embedding": {"model": embedding_model, "normalize": True, "batch_size": 32},
            "indexing": {"type": index_type, "parameters": {"M": 32} if index_type == "hnsw" else {}},
            "llm": {"model": "None", "max_tokens": 0, "temperature": 0}
        },
        "search": {"top_k": 10, "similarity_metric": "cosine", "rerank": False},
        "storage": {"index_path": str(BASE_DIR / "faiss_indexes" / f"{config_name}.index"), "index_files": []},
        "source": {"type": "source_db", "description": f"{source_db_name}.{source_db_table}"},
        "hf_token": hf_token,
        "incremental_mode": incremental_mode, "batch_size": batch_size,
        "batches_completed": 0, "total_batches": 0,
        "created_at": datetime.now().isoformat(), "pipeline_completed": False
    }

    # If manual vector_dim provided, ensure it's recorded in config for future reference
    if vector_dim:
        cfg["pipeline"]["embedding"]["vector_dim"] = vector_dim


    safe_name = get_safe_config_name(config_name)
    CONFIGS_DIR.mkdir(exist_ok=True)
    with open(CONFIGS_DIR / f"{safe_name}.json", 'w') as f:
        json.dump(cfg, f, indent=2)

    progress_id = orchestrator.create_pipeline()
    return {
        "success": True, "config_name": config_name, "progress_id": progress_id,
        "stream_endpoint": f"/pipeline-progress-stream/{progress_id}",
        "config": cfg,
        "source_info": {"database": f"{source_db_host}:{source_db_port}/{source_db_name}",
                        "table": source_db_table, "chunk_column": source_chunk_column,
                        "id_column": source_id_column, "row_count": row_count},
        "target_info": {"database": f"{target_db_host}:{target_db_port}/{target_db_name}",
                        "table": target_db_table},
        "message": "DB source pipeline configured. Ready to execute.",
        "next_step": f"POST /execute-pipeline-from-db with config_name='{config_name}'"
    }


@app.post("/execute-pipeline-from-db")
async def execute_pipeline_from_db(
    config_name: str,
    background_tasks: BackgroundTasks,
    progress_id: Optional[str] = Query(default=None)
):
    """Execute pipeline using database as source."""
    safe_config_name = get_safe_config_name(config_name)
    config_path = CONFIGS_DIR / f"{safe_config_name}.json"
    if not os.path.exists(config_path):
        raise HTTPException(404, f"Configuration '{config_name}' not found")
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    if cfg.get('source', {}).get('type') != 'source_db':
        raise HTTPException(400, "This endpoint is for DB-source configs only")
    if not progress_id:
        progress_id = orchestrator.create_pipeline()

    async def run_pipeline(pid: str):
        try:
            result = await orchestrator.execute(
                config=cfg, progress_id=pid, source_type='source_db',
                source_data={"db_config": cfg['source_db']['db_config'],
                             "chunk_column": cfg['source_db']['chunk_column'],
                             "id_column": cfg['source_db']['id_column']},
                resume_from=None, config_name=config_name
            )
            if 'config' in result:
                with open(config_path, 'w') as f:
                    json.dump(result['config'], f, indent=2)
        except asyncio.CancelledError:
            logger.info(f"DB Pipeline task {pid} was cancelled.")
        except Exception as e:
            logger.error(f"DB source pipeline failed: {e}", exc_info=True)
        finally:
            orchestrator.unregister_task(pid)

    # Use asyncio.create_task instead of BackgroundTasks for manual cancellation control
    task = asyncio.create_task(run_pipeline(progress_id))
    orchestrator.register_task(progress_id, task)
    return {
        "success": True, "progress_id": progress_id, "config_name": config_name,
        "source": cfg['source_db'], "message": "Pipeline started with DB as source",
        "poll_endpoint": f"/pipeline-progress/{progress_id}",
        "stream_endpoint": f"/pipeline-progress-stream/{progress_id}"
    }


# ============================================================================
# PRE-EMBEDDED PIPELINE (source table already contains vectors)
# ============================================================================

@app.post("/configure-preembedded-pipeline")
async def configure_preembedded_pipeline(
    source_db_host: str = Query(...), source_db_port: int = Query(...),
    source_db_user: str = Query(...), source_db_password: str = Query(...),
    source_db_name: str = Query(...), source_db_table: str = Query(...),
    text_column: str = Query(..., description="Column containing display text"),
    vector_column: str = Query(..., description="Column containing pre-computed vectors"),
    id_column: str = Query(default="id"),
    embedding_model: str = Query(default="BAAI/bge-m3", description="Model that was used to produce the vectors (needed for query encoding)"),
    index_type: str = Query(default="hnsw"),
    index_path: str = Query(default="./faiss_indexes"),
    config_name: Optional[str] = None,
    vector_dim: Optional[int] = Query(default=None),
):
    """Configure a pipeline for a source table that already has pre-computed embeddings.
    Skips chunking and re-embedding — only builds a FAISS index.
    """
    source_db_config = {
        "host": source_db_host, "port": source_db_port,
        "user": source_db_user, "password": source_db_password,
        "dbname": source_db_name, "table": source_db_table,
        "text_column": text_column,
    }

    # Validate connection and columns
    try:
        conn = DynamicDatabaseConnection(source_db_config)
        if not conn.test_connection():
            raise HTTPException(400, "Source DB connection failed")
        row_count = conn.count_rows_in_table()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Source DB validation failed: {e}")

    if not config_name:
        config_name = f"preembedded_{source_db_name}_{source_db_table}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if vector_dim is None:
        vector_dim = MODEL_DIMENSIONS.get(embedding_model, 1024)

    cfg = {
        "config_name": config_name,
        "config_version": "1.0",
        "mode": "needs_pipeline",
        "pipeline_type": "preembedded",
        # The database used for retrieval is the source itself.
        # id_column is required so fetch_documents uses the correct PK column.
        "database": {**source_db_config, "id_column": id_column},
        "source_db": {
            "db_config": source_db_config,
            "text_column": text_column,
            "vector_column": vector_column,
            "id_column": id_column,
            "row_count": row_count,
        },
        "pipeline": {
            "chunking": {"strategy": "none"},
            "embedding": {"model": embedding_model, "normalize": True, "batch_size": 32, "vector_dim": vector_dim},
            "indexing": {"type": index_type, "parameters": {"M": 32} if index_type == "hnsw" else {}},
            "llm": {"model": "llama-3.1-8b", "max_tokens": 300, "temperature": 0.7},
        },
        "search": {"top_k": 10, "similarity_metric": "cosine", "rerank": False},
        "storage": {"index_path": str(BASE_DIR / "faiss_indexes" / f"{config_name}.index"), "index_files": []},
        "source": {"type": "preembedded", "description": f"{source_db_name}.{source_db_table}"},
        "created_at": datetime.now().isoformat(),
        "pipeline_completed": False,
    }

    safe_name = get_safe_config_name(config_name)
    CONFIGS_DIR.mkdir(exist_ok=True)
    with open(CONFIGS_DIR / f"{safe_name}.json", 'w') as f:
        json.dump(cfg, f, indent=2)

    progress_id = orchestrator.create_pipeline()
    return {
        "success": True,
        "config_name": config_name,
        "progress_id": progress_id,
        "stream_endpoint": f"/pipeline-progress-stream/{progress_id}",
        "config": cfg,
        "source_info": {
            "table": f"{source_db_name}.{source_db_table}",
            "text_column": text_column,
            "vector_column": vector_column,
            "id_column": id_column,
            "row_count": row_count,
        },
        "message": "Pre-embedded pipeline configured. Ready to execute.",
        "next_step": f"POST /execute-preembedded-pipeline?config_name={config_name}&progress_id={progress_id}",
    }


@app.post("/execute-preembedded-pipeline")
async def execute_preembedded_pipeline(
    config_name: str,
    background_tasks: BackgroundTasks,
    progress_id: Optional[str] = Query(default=None),
):
    """Execute the pre-embedded pipeline — only builds a FAISS index, no chunking or re-embedding."""
    safe_name = get_safe_config_name(config_name)
    config_path = CONFIGS_DIR / f"{safe_name}.json"
    if not os.path.exists(config_path):
        raise HTTPException(404, f"Configuration '{config_name}' not found")

    with open(config_path, 'r') as f:
        cfg = json.load(f)

    if cfg.get('pipeline_type') != 'preembedded':
        raise HTTPException(400, "This endpoint is only for pre-embedded configs (pipeline_type=preembedded)")

    if not progress_id or progress_id not in orchestrator.active_pipelines:
        progress_id = orchestrator.create_pipeline()

    async def _run(pid: str):
        try:
            result = await orchestrator.execute_preembedded(config=cfg, progress_id=pid, config_name=config_name)
            if result.get('config'):
                with open(config_path, 'w') as f:
                    json.dump(result['config'], f, indent=2)
        except asyncio.CancelledError:
            logger.info(f"Pre-embedded pipeline {pid} was cancelled.")
        except Exception as e:
            logger.error(f"Pre-embedded pipeline failed: {e}", exc_info=True)
        finally:
            orchestrator.unregister_task(pid)

    task = asyncio.create_task(_run(progress_id))
    orchestrator.register_task(progress_id, task)

    return {
        "success": True,
        "progress_id": progress_id,
        "config_name": config_name,
        "message": "Pre-embedded pipeline started",
        "poll_endpoint": f"/pipeline-progress/{progress_id}",
        "stream_endpoint": f"/pipeline-progress-stream/{progress_id}",
    }


# ============================================================================
# FEATURE: LOG VIEWER ENDPOINT
# ============================================================================

@app.get("/api/logs")
async def get_logs(lines: int = Query(default=200, ge=1, le=2000)):
    """Return the last N lines of the backend log file for the UI log viewer."""
    try:
        if not os.path.exists(LOG_FILE):
            return {"logs": [], "file": LOG_FILE}
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        tail = all_lines[-lines:]
        return {"logs": [l.rstrip("\n") for l in tail], "total": len(all_lines)}
    except Exception as e:
        raise HTTPException(500, f"Could not read log file: {e}")


# ============================================================================
# FEATURE: RESULT FEEDBACK ENDPOINT
# ============================================================================

# In-memory feedback store (resets on restart — persisted in a JSON file for durability)
FEEDBACK_FILE = os.path.join(BASE_DIR, "feedback.json")
_feedback_lock = asyncio.Lock()

def _load_feedback() -> list:
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def _save_feedback(data: list):
    try:
        with open(FEEDBACK_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}")

class _FeedbackPayload(BaseModel):
    query: str
    result_id: str
    content_snippet: str
    vote: str  # "up" | "down"
    config_name: Optional[str] = None

@app.post("/api/feedback")
async def submit_feedback(payload: _FeedbackPayload):
    """Store thumbs-up / thumbs-down feedback for a search result."""
    async with _feedback_lock:
        all_feedback = _load_feedback()
        entry = {
            "id": f"fb_{len(all_feedback)+1}_{int(__import__('time').time())}",
            "timestamp": __import__('datetime').datetime.utcnow().isoformat(),
            "config_name": payload.config_name,
            "query": payload.query,
            "result_id": payload.result_id,
            "content_snippet": payload.content_snippet[:200],
            "vote": payload.vote,
        }
        all_feedback.append(entry)
        _save_feedback(all_feedback)
        logger.info(f"Feedback recorded: {payload.vote} for result '{payload.result_id}' on query '{payload.query[:60]}'")
    return {"success": True, "id": entry["id"]}

@app.get("/api/feedback")
async def get_feedback(config_name: Optional[str] = Query(default=None)):
    """Return all stored feedback, optionally filtered by config."""
    data = _load_feedback()
    if config_name:
        data = [f for f in data if f.get("config_name") == config_name]
    return {"feedback": data, "count": len(data)}


# ============================================================================
# FEATURE: CHUNK PREVIEW ENDPOINT
# ============================================================================

class _ChunkPreviewRequest(BaseModel):
    text: str
    chunk_size: int = 500
    chunk_overlap: int = 50
    strategy: str = "fixed_size"

@app.post("/api/chunk-preview")
async def chunk_preview(payload: _ChunkPreviewRequest):
    """Return a sample of how text will be split into chunks."""
    try:
        from .pipeline.chunking import FixedSizeChunking, SentenceBasedChunking

        strategy = payload.strategy.lower().replace("-", "_")
        if strategy in ("sentence", "sentence_based"):
            chunker = SentenceBasedChunking(chunk_size=payload.chunk_size, chunk_overlap=payload.chunk_overlap)
        else:
            chunker = FixedSizeChunking(chunk_size=payload.chunk_size, chunk_overlap=payload.chunk_overlap)

        chunks = await chunker.chunk(payload.text)
        sample = chunks[:10]
        return {
            "chunks": [{"index": i, "text": c, "length": len(c)} for i, c in enumerate(sample)],
            "total_chunks": len(chunks),
            "sample_count": len(sample),
        }
    except Exception as e:
        raise HTTPException(500, f"Chunk preview failed: {e}")


# ============================================================================
# FEATURE: DUPLICATE DETECTION ENDPOINT
# ============================================================================

class _DuplicateCheckRequest(BaseModel):
    db_host: str
    db_port: str
    db_user: str
    db_password: str
    db_name: str
    table_name: str

@app.post("/api/check-duplicates")
async def check_duplicates(payload: _DuplicateCheckRequest):
    """Check whether the target table already has data from a previous pipeline run."""
    try:
        conn_params = {
            "host": payload.db_host,
            "port": int(payload.db_port) if payload.db_port else 5432,
            "user": payload.db_user,
            "password": payload.db_password,
            "dbname": payload.db_name,
        }
        db = DynamicDatabaseConnection(conn_params)

        def _check():
            conn = db.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
                        (payload.table_name,)
                    )
                    table_exists = cur.fetchone()[0]
                    if not table_exists:
                        return {"exists": False, "row_count": 0}
                    cur.execute(f"SELECT COUNT(*) FROM {payload.table_name}")
                    count = cur.fetchone()[0]
                    return {"exists": True, "row_count": count}
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        result = await asyncio.get_running_loop().run_in_executor(None, _check)
        return result
    except Exception as e:
        logger.warning(f"Duplicate check failed: {e}")
        return {"exists": False, "row_count": 0, "error": str(e)}


# ============================================================================
# LEGACY ENDPOINTS (BACKWARD COMPATIBILITY)
# ============================================================================

@app.post("/load-config")
async def load_config(config_name: str):
    """Legacy config loading."""
    safe_name = get_safe_config_name(config_name)
    config_path = CONFIGS_DIR / f"{safe_name}.json"
    if not os.path.exists(config_path):
        raise HTTPException(404, "Config not found")
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    return cfg


@app.post("/reload-indexes")
async def reload_indexes():
    """Reload FAISS indexes from disk."""
    try:
        loop = asyncio.get_running_loop()
        num_loaded = await loop.run_in_executor(None, faiss_manager.reload_all_indexes)
        return {"success": True, "num_loaded": num_loaded}
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, reload=RELOAD)