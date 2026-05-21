from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime

from sympy import false, true


class ConfigMode(str, Enum):
    READY = "ready"  # Config points to existing setup
    PARTIALLY_READY = "partially_ready" # Config has some batches processed but not all
    NEEDS_PIPELINE = "needs_pipeline"  # Config needs pipeline execution


class ChunkingStrategy(str, Enum):
    FIXED_SIZE = "fixed_size"
    SENTENCE_BASED = "sentence_based"
    SEMANTIC = "semantic"
    ENTITY_BASED = "entity_based"
    HIERARCHICAL = "hierarchical"
    DOCUMENT_STRUCTURE = "document_structure"


class DatabaseConfig(BaseModel):
    host: str = Field(..., description="Database host")
    port: int = Field(..., description="Database port")
    user: str = Field(..., description="Database user")
    password: str = Field(..., description="Database password (should be encrypted)")
    dbname: str = Field(..., description="Database name")
    table: str = Field(default="documents", description="Table name")
    
    class Config:
        json_schema_extra = {
            "example": {
                "host": "localhost",
                "port": 5433,
                "user": "postgres",
                "password": "postgres",
                "dbname": "appdb",
                "table": "documents"
            }
        }


class ChunkingConfig(BaseModel):
    strategy: ChunkingStrategy
    chunk_size: int = Field(default=500, ge=100, le=2000)
    overlap: int = Field(default=50, ge=0, le=500)
    max_size: Optional[int] = Field(default=None, description="Max size for sentence-based")

    
    class Config:
        json_schema_extra = {
            "example": {
                "strategy": "fixed_size",
                "chunk_size": 500,
                "overlap": 50
            }
        }


class EmbeddingConfig(BaseModel):
    model: str = Field(..., description="Embedding model name")
    normalize: bool = Field(default=True, description="Normalize embeddings")
    batch_size: int = Field(default=32, ge=1, le=128)
    
    class Config:
        json_schema_extra = {
            "example": {
                "model": "BAAI/bge-m3",
                "normalize": True,
                "batch_size": 32
            }
        }


class IndexingConfig(BaseModel):
    type: str = Field(..., description="Index type: hnsw, flat, ivf")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "hnsw",
                "parameters": {"M": 32}
            }
        }

class LLMConfig(BaseModel):
    """LLM configuration for summarization."""
    model: str = Field(default="llama-3.2-3b", description="LLM model name")
    max_tokens: int = Field(default=300, ge=50, le=1000, description="Max tokens in summary")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model": "llama-3.2-3b",
                "max_tokens": 300,
                "temperature": 0.7
            }
        }


class PipelineConfig(BaseModel):
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    indexing: IndexingConfig
    llm: Optional[LLMConfig] = None


class SearchConfig(BaseModel):
    top_k: int = Field(default=5, ge=1, le=100)
    similarity_metric: str = Field(default="cosine")
    rerank: bool = Field(default=False)

        
    class Config:
        json_schema_extra = {
            "example": {
                "top_k": 5,
                "similarity_metric": "cosine",
                "rerank": False
            }
        }



class StorageConfig(BaseModel):
    index_path: str = Field(..., description="Path to index directory")
    index_files: List[str] = Field(default_factory=list, description="List of index files")
    
    class Config:
        json_schema_extra = {
            "example": {
                "index_path": "./indexes",
                "index_files": [
                    "documents_bge_hnsw.index",
                    "documents_bge_hnsw_ids.pkl"
                ]
            }
        }

class RagConfig(BaseModel):
    config_name: str = Field(..., description="Unique configuration name")
    config_version: str = Field(default="1.0", description="Config schema version")
    mode: ConfigMode = Field(default=ConfigMode.NEEDS_PIPELINE)
    
    # Core configurations
    database: DatabaseConfig
    pipeline: PipelineConfig
    search: SearchConfig
    storage: StorageConfig
    
    # Metadata
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    created_by: Optional[str] = None
    description: Optional[str] = None
    pipeline_completed: bool = Field(default=False)
    last_used: Optional[str] = None

    # Batch tracking for incremental processing
    incremental_mode: bool = Field(
        default=False,
        description="Whether pipeline runs incrementally (enables search during processing)"
    )
    batches_completed: int = Field(
        default=0,
        description="Number of batches completed (for incremental mode)"
    )
    total_batches: int = Field(
        default=1,
        description="Total number of batches (for incremental mode)"
    )
    batch_size: int = Field(
        default=1000,
        description="Number of documents per batch (for incremental mode)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "config_name": "doc_search_db",
                "config_version": "1.0",
                "mode": "ready",
                "incremental_mode": True,
                "batches_completed": 3,
                "total_batches": 10,
                "batch_size": 1000,
                "pipeline_completed": False,
                "database": {
                    "host": "doc_db_host",
                    "port": 5432,
                    "user": "rag_user",
                    "password": "password",
                    "dbname": "docs",
                    "table": "documents"
                },
                "pipeline": {
                    "chunking": {
                        "strategy": "fixed_size",
                        "chunk_size": 500,
                        "overlap": 50
                    },
                    "embedding": {
                        "model": "BAAI/bge-m3",
                        "normalize": True,
                        "batch_size": 32
                    },
                    "indexing": {
                        "type": "hnsw",
                        "parameters": {"M": 32}
                    },
                    "llm": {
                        "model": "llama-3.2-3b",
                        "max_tokens": 300,
                        "temperature": 0.7
                    }
                },
                "search": {
                    "top_k": 5,
                    "similarity_metric": "cosine",
                    "rerank": False
                },
                "storage": {
                    "index_path": "/data/indexes/docs",
                    "index_files": [
                        "documents_bge_hnsw.index",
                        "documents_bge_hnsw_ids.pkl"
                    ]
                },
                "created_at": "2026-01-29T12:00:00Z",
                "created_by": "admin_user",
                "description": "Config for document search",
                "pipeline_completed": True
            }
        }


class ConfigValidationResult(BaseModel):
    ready: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    message: str
    checks: Dict[str, bool] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "ready": True,
                "errors": [],
                "warnings": ["Index files are on remote path"],
                "message": "Configuration validated successfully",
                "checks": {
                    "database_accessible": True,
                    "embeddings_exist": True,
                    "indexes_exist": True,
                    "storage_accessible": True
                }
            }
        }


class PipelineStepStatus(str, Enum):
    """Status of individual pipeline step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineStep(BaseModel):
    status: PipelineStepStatus = PipelineStepStatus.PENDING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class PipelineProgressResponse(BaseModel):
    progress_id: str
    current_step: Optional[str] = None
    steps: Dict[str, PipelineStep]
    overall_progress: int = Field(ge=0, le=100)
    status: str  # "running", "completed", "failed"
    
    class Config:
        json_schema_extra = {
            "example": {
                "progress_id": "abc123",
                "current_step": "embed",
                "steps": {
                    "ingest": {
                        "status": "completed",
                        "started_at": "2026-01-29T12:00:00Z",
                        "completed_at": "2026-01-29T12:00:30Z"
                    },
                    "chunk": {
                        "status": "completed",
                        "started_at": "2026-01-29T12:00:30Z",
                        "completed_at": "2026-01-29T12:01:00Z"
                    },
                    "embed": {
                        "status": "running",
                        "started_at": "2026-01-29T12:01:00Z"
                    },
                    "index": {"status": "pending"},
                    "retrieval": {"status": "pending"}
                },
                "overall_progress": 40,
                "status": "running"
            }
        }