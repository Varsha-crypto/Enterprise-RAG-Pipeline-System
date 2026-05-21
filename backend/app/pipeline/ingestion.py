import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from abc import ABC, abstractmethod

from ..db.dynamic_connection import DynamicDatabaseConnection

logger = logging.getLogger(__name__)

class DataSource(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    async def read(self) -> List[str]:
        """Read and return raw text data."""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate data source is accessible."""
        pass


class TextFileSource(DataSource):
    """Read data from plain text file."""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        logger.info(f"Initialized TextFileSource: {file_path}")
    
    def validate(self) -> bool:
        if not self.file_path.exists():
            logger.error(f"File not found: {self.file_path}")
            return False
        
        if not self.file_path.is_file():
            logger.error(f"Path is not a file: {self.file_path}")
            return False
        
        if not self.file_path.suffix.lower() == '.txt':
            logger.warning(f"File is not .txt: {self.file_path}")
        
        return True
    
    async def read(self) -> List[str]:
        """Read text file and return as single document."""
        if not self.validate():
            raise ValueError(f"Invalid text file: {self.file_path}")
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"Read {len(content)} characters from {self.file_path.name}")
            
            # Return as list with single document
            return [content]
            
        except UnicodeDecodeError:
            # Try with different encoding
            logger.warning("UTF-8 failed, trying latin-1 encoding")
            with open(self.file_path, 'r', encoding='latin-1') as f:
                content = f.read()
            return [content]
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            raise

    def get_char_count(self) -> int:
        """
        Return total character count of the file without reading it into memory.
        """
        if not self.validate():
            raise ValueError(f"Invalid text file: {self.file_path}")
        return self.file_path.stat().st_size

    async def read_batch_chars(self, char_offset: int, char_length: int) -> List[str]:
        """
        Read a character-range slice of the file without loading the whole file.

        Args:
            char_offset: Starting character position (0-indexed)
            char_length: Number of characters to read

        Returns:
            List containing one string (the file slice), or [] at EOF
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                f.seek(char_offset)
                content = f.read(char_length)
        except UnicodeDecodeError:
            logger.warning("UTF-8 seek failed, falling back to latin-1")
            with open(self.file_path, 'r', encoding='latin-1') as f:
                f.seek(char_offset)
                content = f.read(char_length)

        if not content:
            return []

        # Extend to next whitespace boundary to avoid cutting mid-word,
        # unless already at EOF
        if len(content) == char_length:
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    f.seek(char_offset + char_length)
                    # Read until next whitespace (max 200 extra chars)
                    extra = f.read(200)
                    if extra:
                        boundary = next(
                            (i for i, c in enumerate(extra) if c in ' \n\t\r'),
                            len(extra)
                        )
                        content += extra[:boundary]
            except Exception:
                pass  # Non-fatal: boundary extension is best-effort

        logger.info(
            f"Read batch: {len(content)} chars "
            f"(offset={char_offset}, requested={char_length})"
        )
        return [content]


class DatabaseSource(DataSource):
    """
    Read chunks from any external database.
    
    Source DB may be completely different from target DB.
    User specifies which columns contain the chunks and IDs.
    """
    
    def __init__(
        self,
        db_config: Dict[str, Any],
        chunk_column: Optional[str] = "chunks",
        id_column: Optional[str] = "id",
        batch_size: Optional[int] = None,
        progress_id: Optional[str] = None
    ):
        """
        Args:
            db_config: Source database connection config
            chunk_column: Name of column containing text chunks
            id_column: Name of primary key / ID column
            batch_size: Optional batch size for large tables
            progress_id: Optional pipeline ID for cancellation
        """
        self.db_config = db_config
        self.chunk_column = chunk_column
        self.id_column = id_column
        self.batch_size = batch_size
        self.progress_id = progress_id
        
        logger.info(
            f"DatabaseSource initialized: "
            f"{db_config['host']}:{db_config['port']}/{db_config['dbname']} "
            f"(table={db_config['table']}, chunk_col={chunk_column}, id_col={id_column})"
        )
    
    def validate(self) -> bool:
        """
        Validate source DB connection and column names.
        """
        from ..db.dynamic_connection import DynamicDatabaseConnection
        
        try:
            db_conn = DynamicDatabaseConnection(self.db_config)
            
            # Test connection
            if not db_conn.test_connection():
                logger.error("Source DB connection test failed")
                return False
            
            # Validate columns exist
            if not db_conn.validate_columns(self.chunk_column, self.id_column): #type: ignore
                logger.error(
                    f"Columns not found: chunk_column='{self.chunk_column}', "
                    f"id_column='{self.id_column}'"
                )
                return False
            
            logger.info("Source database validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Database validation failed: {e}")
            return False
    
    async def read(self) -> List[str]:
        """
        Read all chunks from source database.
        
        Returns:
            List of text chunks
        """
        if not self.validate():
            raise ValueError("Source database validation failed")
        
        from ..db.dynamic_connection import DynamicDatabaseConnection
        
        # --- Check For Cancellation ---
        if self.progress_id:
            try:
                from .orchestrator import orchestrator
                if self.progress_id in orchestrator.active_pipelines and orchestrator.active_pipelines[self.progress_id].is_cancelled:
                    logger.warning(f"Database read aborted: Pipeline {self.progress_id} was cancelled.")
                    import asyncio
                    raise asyncio.CancelledError("Pipeline cancelled before database read")
            except Exception as e:
                import asyncio
                if isinstance(e, asyncio.CancelledError): raise
        # ------------------------------

        db_conn = DynamicDatabaseConnection(self.db_config)
        
        if self.batch_size:
            # Use batched reading for large datasets
            chunks = db_conn.fetch_chunks_batched(
                chunk_column=self.chunk_column, #type: ignore
                id_column=self.id_column,  #type: ignore
                batch_size=self.batch_size
            )
        else:
            # Read all at once for small datasets
            chunks = db_conn.fetch_chunks(
                chunk_column=self.chunk_column, #type: ignore
                id_column=self.id_column #type: ignore
            )
        
        logger.info(f"Read {len(chunks)} chunks from source database")
        return chunks
    
    async def read_batch(self, offset: int, limit: int) -> List[str]:
        """
        Read a specific batch from source database.
        
        Used by incremental pipeline processing.
        
        Args:
            offset: Starting row offset
            limit: Number of rows to fetch
            
        Returns:
            List of text chunks for this batch
        """
        from ..db.dynamic_connection import DynamicDatabaseConnection
        
        db_conn = DynamicDatabaseConnection(self.db_config)
        chunks = db_conn.fetch_chunks_offset(
            chunk_column=self.chunk_column, #type: ignore
            id_column=self.id_column,  #type: ignore
            offset=offset,
            limit=limit
        )
        
        logger.info(f"Read batch: {len(chunks)} chunks (offset={offset})")
        return chunks
    
    def count_rows(self) -> int:
        """
        Count total rows in source table.
        
        Used to calculate number of batches for incremental processing.
        """
        from ..db.dynamic_connection import DynamicDatabaseConnection
        
        db_conn = DynamicDatabaseConnection(self.db_config)
        return db_conn.count_rows_in_table()


class UploadedFileSource(DataSource):
    """Read data from uploaded file (bytes)."""
    
    def __init__(self, file_content: bytes, filename: str):
        self.file_content = file_content
        self.filename = filename
        logger.info(f"Initialized UploadedFileSource: {filename} ({len(file_content)} bytes)")
    
    def validate(self) -> bool:
        """Check if content is valid."""
        if not self.file_content:
            logger.error("Empty file content")
            return False
        return True
    
    async def read(self) -> List[str]:
        """Read uploaded file content."""
        if not self.validate():
            raise ValueError("Invalid file content")
        
        try:
            # Try UTF-8 decoding
            content = self.file_content.decode('utf-8')
        except UnicodeDecodeError:
            # Fallback to latin-1
            logger.warning("UTF-8 failed, using latin-1 encoding")
            content = self.file_content.decode('latin-1')
        
        logger.info(f"Read {len(content)} characters from uploaded file {self.filename}")
        return [content]


class DataIngestion:

    def __init__(self):
        self.supported_sources = {
            'text_file': TextFileSource,
            'database': DatabaseSource,
            'uploaded_file': UploadedFileSource,
        }
        logger.info("DataIngestion initialized")

    def get_file_source(self, file_path: str) -> TextFileSource:
        """
        Return a TextFileSource instance for a given path.
        Used by the incremental orchestrator to call get_char_count() and
        read_batch_chars() without going through the ingest_from_file()
        path (which does full ingestion instead).
        """
        return TextFileSource(file_path)

    async def ingest_from_database(self, db_config: Dict[str, Any]) -> List[str]:
        logger.info("-" * 35)
        logger.info("INGESTION: Reading from database")
        logger.info("-" * 35)
        
        source = DatabaseSource(db_config)
        documents = await source.read()
        
        logger.info(f"Ingested {len(documents)} documents from database")
        return documents
    
    async def ingest_from_source_db(
        self,
        source_db_config: Dict[str, Any],
        chunk_column: str,
        id_column: str,
        batch_size: Optional[int] = None
    ) -> List[str]:
        """
        Ingest data from external source database.
        
        Args:
            source_db_config: Source database connection config
            chunk_column: Column containing text chunks
            id_column: Column containing row IDs
            batch_size: Optional batch size for large datasets
            
        Returns:
            List of text chunks
        """
        logger.info("-" * 35)
        logger.info("INGESTION: Reading chunks from source database")
        logger.info("-" * 35)
        
        source = DatabaseSource(
            db_config=source_db_config,
            chunk_column=chunk_column,
            id_column=id_column,
            batch_size=batch_size
        )
    
        documents = await source.read()
        
        logger.info(f"Ingested {len(documents)} chunks from source database")
        return documents
    
    
    async def ingest_from_file(self, file_path: str) -> List[str]:
        logger.info("-" * 35)
        logger.info(f"INGESTION: Reading from file {file_path}")
        logger.info("-" * 35)
        
        source = TextFileSource(file_path)
        documents = await source.read()
        
        logger.info(f"Ingested {len(documents)} documents from file")
        return documents
    
    async def ingest_from_upload(self, file_content: bytes, filename: str) -> List[str]:
        logger.info("-" * 35)
        logger.info(f"INGESTION: Reading from uploaded file {filename}")
        logger.info("-" * 35)
        
        source = UploadedFileSource(file_content, filename)
        documents = await source.read()
        
        logger.info(f"Ingested {len(documents)} documents from upload")
        return documents
    
    def get_source_type(self, config: Dict[str, Any]) -> str:
        # Check for different source indicators in config
        if 'file_path' in config:
            return 'text_file'
        elif 'source_db' in config:
            return 'source_db'     
        elif 'database' in config:
            return 'database'
        elif 'uploaded_file' in config:
            return 'uploaded_file'
        else:
            raise ValueError("Unknown data source in configuration")
    
    async def ingest(self, config: Dict[str, Any], progress_id: Optional[str] = None) -> List[str]:
        """
        Generic ingestion - now includes source_db type.
        """
        source_type = self.get_source_type(config)
        
        if source_type == 'text_file':
            return await self.ingest_from_file(config['file_path'])
        
        elif source_type == 'database':
            # Target DB as source 
            # Update source to include progress_id
            source = DatabaseSource(config['database'], progress_id=progress_id)
            return await source.read()
        
        elif source_type == 'source_db':
            # External source DB 
            source_db = config['source_db']
            source = DatabaseSource(
                db_config=source_db['db_config'],
                chunk_column=source_db['chunk_column'],
                id_column=source_db['id_column'],
                batch_size=source_db.get('batch_size'),
                progress_id=progress_id
            )
            return await source.read()
        
        elif source_type == 'uploaded_file':
            return await self.ingest_from_upload(
                config['uploaded_file']['content'],
                config['uploaded_file']['filename']
            )
        
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
    
    def get_statistics(self, documents: List[str]) -> Dict[str, Any]:
        total_chars = sum(len(doc) for doc in documents)
        avg_chars = total_chars / len(documents) if documents else 0
        
        return {
            "num_documents": len(documents),
            "total_characters": total_chars,
            "avg_characters_per_doc": round(avg_chars, 2),
            "min_length": min(len(doc) for doc in documents) if documents else 0,
            "max_length": max(len(doc) for doc in documents) if documents else 0
        }