"""
Workflow B: Handle building new RAG pipeline from scratch.
User provides DB with raw data + configuration for chunking, embedding and xindexing.
"""

import logging
import psycopg2
from typing import Dict, Any, Optional
from datetime import datetime
from ..config import BASE_DIR

logger = logging.getLogger(__name__)


class BuildPipelineWorkflow:    
    def __init__(self):
        pass
    
    async def check_database_data(self, db_config: Dict[str, Any]) -> Dict[str, Any]:
        try:
            conn = psycopg2.connect(
                host=db_config['host'],
                port=db_config['port'],
                user=db_config['user'],
                password=db_config['password'],
                dbname=db_config['dbname']
            )
            cursor = conn.cursor()
            
            table = db_config.get('table', 'documents')
            
            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                )
            """, (table,))
            
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                cursor.close()
                conn.close()
                return {
                    "data_found": False,
                    "message": f"Table '{table}' does not exist",
                    "num_rows": 0
                }
            
            # Count rows
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            num_rows = cursor.fetchone()[0]
            
            # Check for text column
            cursor.execute(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = %s AND data_type IN ('text', 'character varying')
            """, (table,))
            
            text_columns = [row[0] for row in cursor.fetchall()]
            
            cursor.close()
            conn.close()
            
            if num_rows == 0:
                return {
                    "data_found": False,
                    "message": f"Table '{table}' is empty",
                    "num_rows": 0
                }
            
            return {
                "data_found": True,
                "message": f"Found {num_rows} rows in table '{table}'",
                "num_rows": num_rows,
                "text_columns": text_columns,
                "table": table
            }
            
        except Exception as e:
            logger.error(f"Database check failed: {e}")
            return {
                "data_found": False,
                "message": f"Error checking database: {str(e)}",
                "num_rows": 0
            }
    
    def validate_pipeline_config(self, pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
        errors = []
        
        # Check chunking config
        if 'chunking' not in pipeline_config:
            errors.append("Missing chunking configuration")
        else:
            chunking = pipeline_config['chunking']
            if 'strategy' not in chunking:
                errors.append("Missing chunking strategy")
            if chunking.get('chunk_size', 0) < 100:
                errors.append("Chunk size must be at least 100 characters")
        
        # Check embedding config
        if 'embedding' not in pipeline_config:
            errors.append("Missing embedding configuration")
        else:
            embedding = pipeline_config['embedding']
            if 'model' not in embedding:
                errors.append("Missing embedding model")
        
        # Check indexing config
        if 'indexing' not in pipeline_config:
            errors.append("Missing indexing configuration")
        else:
            indexing = pipeline_config['indexing']
            if 'type' not in indexing:
                errors.append("Missing index type")
            if indexing.get('type') not in ['hnsw', 'flat', 'ivf']:
                errors.append("Invalid index type. Must be: hnsw, flat, or ivf")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "message": "Configuration is valid" if len(errors) == 0 else "Configuration has errors"
        }
    
    def create_default_config(
        self,
        db_config: Dict[str, Any],
        embedding_model: str = "BAAI/bge-m3",
        index_type: str = "hnsw",
        chunking_strategy: str = "fixed_size"
    ) -> Dict[str, Any]:
        """
        Create a default configuration for new pipeline.
        """
        
        config_name = f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return {
            "config_name": config_name,
            "config_version": "1.0",
            "mode": "needs_pipeline",
            "database": db_config,
            "pipeline": {
                "chunking": {
                    "strategy": chunking_strategy,
                    "chunk_size": 500,
                    "overlap": 50
                },
                "embedding": {
                    "model": embedding_model,
                    "normalize": True,
                    "batch_size": 32
                },
                "indexing": {
                    "type": index_type,
                    "parameters": {"M": 32} if index_type == "hnsw" else {}
                }
            },
            "search": {
                "top_k": 5,
                "similarity_metric": "cosine",
                "rerank": False
            },
            "storage": {
                "index_path": str(BASE_DIR / "faiss_indexes" / f"{config_name}.index"),
                "index_files": []
            },
            "created_at": datetime.now().isoformat(),
            "pipeline_completed": False
        }
    
    def estimate_pipeline_time(self, num_rows: int, embedding_model: str) -> Dict[str, Any]:
        # Approximate estimates; tune based on observed performance for your hardware
        
        ingest_time = max(5, num_rows * 0.01)  # ~0.01s per row
        chunk_time = max(10, num_rows * 0.02)  # ~0.02s per row
        
        # Embedding time varies by model
        if "minilm" in embedding_model.lower():
            embed_time = max(30, num_rows * 0.5)  # Fast model
        elif "bge" in embedding_model.lower() or "e5" in embedding_model.lower():
            embed_time = max(60, num_rows * 1.0)  # Slower but better
        else:
            embed_time = max(45, num_rows * 0.75)  # Medium
        
        index_time = max(15, num_rows * 0.05)  # ~0.05s per vector
        retrieval_time = 5  # Setup time
        
        total_time = ingest_time + chunk_time + embed_time + index_time + retrieval_time
        
        return {
            "ingest_seconds": int(ingest_time),
            "chunk_seconds": int(chunk_time),
            "embed_seconds": int(embed_time),
            "index_seconds": int(index_time),
            "retrieval_seconds": int(retrieval_time),
            "total_seconds": int(total_time),
            "total_minutes": round(total_time / 60, 1)
        }
    
    def get_pipeline_summary(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "config_name": config.get('config_name'),
            "database": f"{config['database']['host']}:{config['database']['port']}/{config['database']['dbname']}",
            "chunking": f"{config['pipeline']['chunking']['strategy']} (size: {config['pipeline']['chunking']['chunk_size']})",
            "embedding": config['pipeline']['embedding']['model'],
            "indexing": config['pipeline']['indexing']['type'],
            "status": "Not started" if not config.get('pipeline_completed') else "Completed"
        }


# Global workflow instance
build_workflow = BuildPipelineWorkflow()