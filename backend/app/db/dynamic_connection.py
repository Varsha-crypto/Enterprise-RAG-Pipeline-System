"""
Dynamic Database Connection - Connects to any database based on configuration.
"""

import psycopg2
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


class DynamicDatabaseConnection:    
    def __init__(self, db_config: Dict[str, Any]):
        self.host = db_config['host']
        self.port = db_config['port']
        self.user = db_config['user']
        self.password = db_config['password']
        self.dbname = db_config['dbname']
        self.table = db_config.get('table', 'documents')
        self.text_column = db_config.get('text_column', 'chunks')
        self.id_column = db_config.get('id_column', 'id')

        logger.info(f"Initialized connection to {self.host}:{self.port}/{self.dbname}")
    
    def get_connection(self):
        try:
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                dbname=self.dbname,
                connect_timeout=10
            )
            logger.debug(f"Connected to {self.host}:{self.port}/{self.dbname}")
            return conn
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise Exception(f"Cannot connect to database {self.host}:{self.port}/{self.dbname}: {e}")
    
    def test_connection(self) -> bool:
        try:
            conn = self.get_connection()
            conn.close()
            logger.info(f"Database connection test successful: {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
        
    def validate_columns(self, chunk_column: str, id_column: str) -> bool:
        """
        Validate that specified columns exist in the table.
        
        Args:
            chunk_column: Expected chunk column name
            id_column: Expected ID column name
            
        Returns:
            True if both columns exist
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = %s
                AND table_schema = 'public'
            """, (self.table,))
            
            existing_columns = {row[0] for row in cursor.fetchall()}
            
            if chunk_column not in existing_columns:
                logger.error(
                    f"chunk_column '{chunk_column}' not found. "
                    f"Available columns: {existing_columns}"
                )
                return False
            
            if id_column not in existing_columns:
                logger.error(
                    f"id_column '{id_column}' not found. "
                    f"Available columns: {existing_columns}"
                )
                return False
            
            logger.info(f"Columns validated: chunk='{chunk_column}', id='{id_column}'")
            return True
            
        except Exception as e:
            logger.error(f"Column validation failed: {e}")
            return False
        finally:
            cursor.close()
            conn.close()
    
    def fetch_documents(self, doc_ids: List[int]) -> List[Tuple]:
        if not doc_ids:
            return []

        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            placeholders = ','.join(['%s'] * len(doc_ids))
            query = f"""
                SELECT {self.id_column}, {self.text_column}, embedding_model
                FROM {self.table}
                WHERE {self.id_column} IN ({placeholders})
            """
            cursor.execute(query, doc_ids)
            rows = cursor.fetchall()
            logger.debug(f"Fetched {len(rows)} documents from {self.table}")
            return rows

        except Exception as e:
            # embedding_model column may not exist in pre-embedded source tables
            if "embedding_model" in str(e):
                conn.rollback()
                cursor.close()
                cursor = conn.cursor()
                placeholders = ','.join(['%s'] * len(doc_ids))
                query = f"""
                    SELECT {self.id_column}, {self.text_column}, NULL AS embedding_model
                    FROM {self.table}
                    WHERE {self.id_column} IN ({placeholders})
                """
                cursor.execute(query, doc_ids)
                rows = cursor.fetchall()
                logger.debug(f"Fetched {len(rows)} documents (no embedding_model col) from {self.table}")
                return rows
            logger.error(f"Document fetch failed: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def count_documents(self, embedding_model: Optional[str] = None) -> int:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if embedding_model:
                query = f"SELECT COUNT(*) FROM {self.table} WHERE embedding_model = %s"
                cursor.execute(query, (embedding_model,))
            else:
                query = f"SELECT COUNT(*) FROM {self.table}"
                cursor.execute(query)
            
            count = cursor.fetchone()[0] #type: ignore
            logger.debug(f"Document count: {count}")
            
            return count
            
        except Exception as e:
            logger.error(f"Count query failed: {e}")
            raise
        finally:
            cursor.close()
            conn.close()

    def fetch_chunks(self, chunk_column: str, id_column: str) -> List[str]:
        """
        Fetch all chunks from source table. This method is exclusively used for the case where data is ingested from source DB.
        
        Args:
            chunk_column: Column containing text chunks
            id_column: Column containing IDs (for ordering)
            
        Returns:
            List of text chunks
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(f"""
                SELECT {chunk_column}
                FROM {self.table}
                WHERE {chunk_column} IS NOT NULL
                AND {chunk_column} != ''
                ORDER BY {id_column}
            """)
            
            chunks = [row[0] for row in cursor.fetchall()]
            
            logger.info(f"Fetched {len(chunks)} chunks from {self.table}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to fetch chunks: {e}")
            raise
        finally:
            cursor.close()
            conn.close()

    def fetch_chunks_batched(
        self,
        chunk_column: str,
        id_column: str,
        batch_size: int
    ) -> List[str]:
        """
        Fetch all chunks using batched reads to avoid memory issues.
        
        Args:
            chunk_column: Column containing text chunks
            id_column: Column containing IDs
            batch_size: Number of rows per batch
            
        Returns:
            List of all text chunks
        """
        all_chunks = []
        offset = 0
        
        while True:
            batch = self.fetch_chunks_offset(
                chunk_column=chunk_column,
                id_column=id_column,
                offset=offset,
                limit=batch_size
            )
            
            if not batch:
                break
            
            all_chunks.extend(batch)
            offset += batch_size
            
            logger.info(f"Fetched {len(all_chunks)} chunks so far (batch at offset {offset})")
            
            if len(batch) < batch_size:
                # Last batch
                break
        
        logger.info(f"Fetched total {len(all_chunks)} chunks (batched)")
        return all_chunks


    def fetch_chunks_offset(
        self,
        chunk_column: str,
        id_column: str,
        offset: int,
        limit: int
    ) -> List[str]:
        """
        Fetch a specific page of chunks.
        
        Args:
            chunk_column: Column containing text chunks
            id_column: Column containing IDs
            offset: Starting row offset
            limit: Number of rows to fetch
            
        Returns:
            List of text chunks for this page
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(f"""
                SELECT {chunk_column}
                FROM {self.table}
                WHERE {chunk_column} IS NOT NULL
                AND {chunk_column} != ''
                ORDER BY {id_column}
                LIMIT %s OFFSET %s
            """, (limit, offset))
            
            chunks = [row[0] for row in cursor.fetchall()]
            
            logger.debug(f"Fetched {len(chunks)} chunks at offset {offset}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to fetch chunks at offset {offset}: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def fetch_all_chunks(self) -> List[str]:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            query = f"SELECT DISTINCT chunks FROM {self.table} ORDER BY chunks"
            cursor.execute(query)
            
            chunks = [row[0] for row in cursor.fetchall()]
            logger.info(f"Fetched {len(chunks)} unique chunks from {self.table}")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Chunk fetch failed: {e}")
            raise
        finally:
            cursor.close()
            conn.close()

    def count_rows_in_table(self) -> int:
        """
        Count total rows in the configured table.
        
        Returns:
            Total row count
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {self.table}")
            count = cursor.fetchone()[0] #type: ignore
            
            logger.info(f"Table {self.table} has {count} rows")
            return count
            
        except Exception as e:
            logger.error(f"Failed to count rows: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def insert_embeddings(
        self, 
        chunks: List[str], 
        vectors: List[List[float]], 
        embedding_model: str,
        batch_number: Optional[int] = None
    ):
        """
        Insert embeddings into database with optional batch tracking.
        
        Args:
            chunks: List of text chunks
            vectors: List of embedding vectors
            embedding_model: Model name used for embeddings
            batch_number: Optional batch number for incremental processing
        """
        if len(chunks) != len(vectors):
            raise ValueError("Chunks and vectors must have same length")
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Build INSERT query based on whether batch tracking is enabled
            if batch_number is not None:
                query = f"""
                    INSERT INTO {self.table} (chunks, vector, embedding_model, source, batch_number)
                    VALUES (%s, %s, %s, %s, %s)
                """
                logger.debug(f"Inserting with batch tracking (batch {batch_number})")
            else:
                query = f"""
                    INSERT INTO {self.table} (chunks, vector, embedding_model, source)
                    VALUES (%s, %s, %s, %s)
                """
                logger.debug("Inserting without batch tracking")
            
            # Insert each chunk
            for chunk, vector in zip(chunks, vectors):
                vector_str = '[' + ','.join(map(str, vector)) + ']'
                
                if batch_number is not None:
                    cursor.execute(query, (chunk, vector_str, embedding_model, 'pipeline', batch_number))
                else:
                    cursor.execute(query, (chunk, vector_str, embedding_model, 'pipeline'))
            
            conn.commit()
            
            if batch_number is not None:
                logger.info(f"Inserted {len(chunks)} embeddings for {embedding_model} (batch {batch_number})")
            else:
                logger.info(f"Inserted {len(chunks)} embeddings for {embedding_model}")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Embedding insertion failed: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def check_embeddings_exist(self, embedding_model: str) -> bool:
        try:
            count = self.count_documents(embedding_model)
            exists = count > 0
            
            if exists:
                logger.info(f"Found {count} embeddings for {embedding_model}")
            else:
                logger.info(f"No embeddings found for {embedding_model}")
            
            return exists
            
        except Exception as e:
            logger.error(f"Embeddings check failed: {e}")
            return False
    
    def get_table_info(self) -> Dict[str, Any]:

        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                )
            """, (self.table,))
            
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                return {
                    "exists": False,
                    "message": f"Table '{self.table}' does not exist"
                }
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {self.table}")
            row_count = cursor.fetchone()[0]
            
            # Get unique embedding models
            cursor.execute(f"""
                SELECT embedding_model, COUNT(*) 
                FROM {self.table} 
                GROUP BY embedding_model
            """)
            models = {row[0]: row[1] for row in cursor.fetchall()}
            
            return {
                "exists": True,
                "table": self.table,
                "row_count": row_count,
                "embedding_models": models
            }
            
        except Exception as e:
            logger.error(f"Table info query failed: {e}")
            raise
        finally:
            cursor.close()
            conn.close()

    
    def get_table_columns(self) -> List[str]:
        """
        Get all column names for the configured table.
        Useful for UI to show user available columns to choose from.
        
        Returns:
            List of column names
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = %s
                AND table_schema = 'public'
                ORDER BY ordinal_position
            """, (self.table,))
            
            columns = [
                {"name": row[0], "type": row[1]}
                for row in cursor.fetchall()
            ]
            
            logger.info(f"Table {self.table} has {len(columns)} columns")
            return columns
            
        except Exception as e:
            logger.error(f"Failed to get columns: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def get_batch_statistics(self, embedding_model: str) -> Dict[str, Any]:
        """
        Get statistics about batches for a given embedding model.
        
        Args:
            embedding_model: Model name to check
            
        Returns:
            Dictionary with batch statistics
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Get batch info
            cursor.execute(f"""
                SELECT 
                    batch_number,
                    COUNT(*) as chunk_count,
                    MIN(id) as min_id,
                    MAX(id) as max_id
                FROM {self.table}
                WHERE embedding_model = %s
                AND batch_number IS NOT NULL
                GROUP BY batch_number
                ORDER BY batch_number
            """, (embedding_model,))
            
            batches = []
            for row in cursor.fetchall():
                batches.append({
                    "batch_number": row[0],
                    "chunk_count": row[1],
                    "id_range": f"{row[2]}-{row[3]}"
                })
            
            # Get total count
            cursor.execute(f"""
                SELECT COUNT(*)
                FROM {self.table}
                WHERE embedding_model = %s
            """, (embedding_model,))
            
            total_count = cursor.fetchone()[0]
            
            return {
                "embedding_model": embedding_model,
                "total_chunks": total_count,
                "batches": batches,
                "num_batches": len(batches)
            }
            
        except Exception as e:
            logger.error(f"Failed to get batch statistics: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
