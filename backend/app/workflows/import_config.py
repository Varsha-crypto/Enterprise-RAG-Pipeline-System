"""
Workflow A: Handle importing existing configuration and validating readiness.
User has a config file pointing to existing setup (DB + embeddings + indexes).
"""

import logging
import psycopg2
from pathlib import Path
from typing import Dict, Any, List
import os

logger = logging.getLogger(__name__)


class ImportConfigWorkflow:
    """Handle config import and validation."""
    
    def __init__(self):
        pass
    
    async def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        errors = []
        warnings = []
        checks = {
            "database_accessible": False,
            "embeddings_exist": False,
            "indexes_exist": False,
            "storage_accessible": False
        }
        
        try:
            # Check 1: Database connectivity
            db_config = config.get('database')
            if db_config:
                logger.info("Checking database connectivity...")
                db_accessible = await self._check_database_connection(db_config)
                checks["database_accessible"] = db_accessible
                if not db_accessible:
                    errors.append("Cannot connect to database")
                else:
                    logger.info("Database connection successful")
            else:
                errors.append("Missing 'database' section in configuration")

            # Check 2: Embeddings exist
            pipeline_cfg = config.get('pipeline', {})
            embedding_cfg = pipeline_cfg.get('embedding', {})
            model_name = embedding_cfg.get('model')
            
            if checks["database_accessible"] and model_name:
                logger.info("Checking if embeddings exist in database...")
                embeddings_exist = await self._check_embeddings_exist(
                    db_config,
                    model_name
                )
                checks["embeddings_exist"] = embeddings_exist
                
                if not embeddings_exist:
                    errors.append(
                        f"Database missing embeddings for model: {model_name}"
                    )
                else:
                    logger.info("Embeddings found in database")
            elif not model_name:
                warnings.append("Embedding model not specified in configuration")

            # Check 3: Index files exist
            storage_cfg = config.get('storage')
            if storage_cfg:
                logger.info("Checking if index files exist...")
                indexes_exist = await self._check_index_files_exist(storage_cfg)
                checks["indexes_exist"] = indexes_exist
                
                if not indexes_exist:
                    errors.append("Index files not found at specified path")
                else:
                    logger.info("Index files found")
            else:
                errors.append("Missing 'storage' section in configuration")

            # Check 4: Storage accessible
            if storage_cfg:
                logger.info("Checking storage accessibility...")
                storage_accessible = await self._check_storage_accessible(storage_cfg)
                checks["storage_accessible"] = storage_accessible
                
                if not storage_accessible:
                    warnings.append("Index storage path may not be accessible")
                else:
                    logger.info("Storage accessible")
            
            # Determine overall readiness
            ready = (
                checks["database_accessible"] and
                checks.get("embeddings_exist", False) and
                checks.get("indexes_exist", False)
            )
            
            if ready:
                message = "Configuration validated successfully. Ready to search."
            else:
                message = "Configuration validation complete. Some components require attention before search."
            
            return {
                "ready": ready,
                "errors": errors,
                "warnings": warnings,
                "message": message,
                "checks": checks
            }
            
        except Exception as e:
            logger.error(f"Validation error: {e}", exc_info=True)
            return {
                "ready": False,
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": warnings,
                "message": "Configuration validation encountered an error",
                "checks": checks
            }
    
    async def _check_database_connection(self, db_config: Dict[str, Any]) -> bool:
        try:
            conn = psycopg2.connect(
                host=db_config['host'],
                port=db_config['port'],
                user=db_config['user'],
                password=db_config['password'],
                dbname=db_config['dbname'],
                connect_timeout=5
            )
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    async def _check_embeddings_exist(
        self, 
        db_config: Dict[str, Any],
        embedding_model: str
    ) -> bool:
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
                return False
            
            # Check if embeddings exist for this model
            cursor.execute(f"""
                SELECT COUNT(*) 
                FROM {table} 
                WHERE embedding_model = %s
            """, (embedding_model,))
            
            count = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            return count > 0
            
        except Exception as e:
            logger.error(f"Embeddings check failed: {e}")
            return False
    
    async def _check_index_files_exist(self, storage_config: Dict[str, Any]) -> bool:
        try:
            index_path = Path(storage_config['index_path'])
            index_files = storage_config.get('index_files', [])
            
            if not index_files:
                logger.warning("No index files specified in config")
                return False
            
            # Check each index file
            for filename in index_files:
                file_path = index_path / filename
                if not file_path.exists():
                    logger.warning(f"Index file not found: {file_path}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Index file check failed: {e}")
            return False
    
    async def _check_storage_accessible(self, storage_config: Dict[str, Any]) -> bool:
        try:
            index_path = Path(storage_config['index_path'])
            
            # Check if path exists
            if not index_path.exists():
                logger.warning(f"Storage path does not exist: {index_path}")
                return False
            
            # Check if readable
            if not os.access(index_path, os.R_OK):
                logger.warning(f"Storage path not readable: {index_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Storage accessibility check failed: {e}")
            return False
    
    def get_config_summary(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": config.get('config_name'),
            "database": f"{config['database']['host']}:{config['database']['port']}/{config['database']['dbname']}",
            "embedding_model": config['pipeline']['embedding']['model'],
            "index_type": config['pipeline']['indexing']['type'],
            "chunking_strategy": config['pipeline']['chunking']['strategy'],
            "index_location": config['storage']['index_path'],
            "created_at": config.get('created_at'),
            "pipeline_completed": config.get('pipeline_completed', False)
        }


# Global workflow instance
import_workflow = ImportConfigWorkflow()