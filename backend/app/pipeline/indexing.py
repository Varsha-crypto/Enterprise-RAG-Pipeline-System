import logging
import numpy as np
import faiss
import pickle
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time
from ..db.dynamic_connection import DynamicDatabaseConnection
import asyncio
import os
import psycopg2
try:
    from ..config import MODEL_SHORT_NAMES
except ImportError:
    MODEL_SHORT_NAMES = {}  # Fallback

logger = logging.getLogger(__name__)


class IndexBuilding:
    
    def __init__(self):
        self.supported_index_types = ['hnsw', 'flat', 'ivf']
        # Limit FAISS threads on Windows to prevent potential deadlocks in background tasks
        try:
            faiss.omp_set_num_threads(1)
            logger.info("FAISS threads set to 1 for stability")
        except Exception as e:
            logger.warning(f"Could not set FAISS threads: {e}")
        logger.info("IndexBuilding initialized")
    
    async def build_indexes(
        self,
        db_config: Dict[str, Any],
        embedding_config: Dict[str, Any],
        indexing_config: Dict[str, Any],
        storage_config: Dict[str, Any],
        progress_id: Optional[str] = None
    ) -> List[str]:

        logger.info("-" * 35)
        logger.info("INDEXING: Building FAISS indexes")
        logger.info("-" * 35)
        
        # --- Check For Cancellation ---
        if progress_id:
            try:
                from .orchestrator import orchestrator
                if progress_id in orchestrator.active_pipelines and orchestrator.active_pipelines[progress_id].is_cancelled:
                    logger.warning(f"Indexing aborted (Start): Pipeline {progress_id} was cancelled.")
                    raise asyncio.CancelledError("Pipeline cancelled before indexing")
            except Exception as e:
                if isinstance(e, asyncio.CancelledError): raise
        # ------------------------------

        embedding_model = embedding_config['model']
        index_type = indexing_config['type']
        index_params = indexing_config.get('parameters', {})
        
        logger.info(f"Embedding model: {embedding_model}")
        logger.info(f"Index type: {index_type}")
        logger.info(f"Parameters: {index_params}")
        
        # Fetch vectors from database
        vectors, ids = await self._fetch_vectors_from_db(
            db_config=db_config,
            embedding_model=embedding_model,
            progress_id=progress_id
        )
        
        if len(vectors) == 0:
            raise ValueError(f"No vectors found for model {embedding_model}")
        
        logger.info(f"Fetched {len(vectors)} vectors (dimension: {vectors.shape[1]})")

        # --- Check For Cancellation ---
        if progress_id:
            try:
                from .orchestrator import orchestrator
                if progress_id in orchestrator.active_pipelines and orchestrator.active_pipelines[progress_id].is_cancelled:
                    logger.warning(f"Indexing aborted (Pre-Build): Pipeline {progress_id} was cancelled.")
                    raise asyncio.CancelledError("Pipeline cancelled after fetching vectors")
            except Exception as e:
                if isinstance(e, asyncio.CancelledError): raise
        # ------------------------------
        
        # Build index
        start_time = time.time()
        index = await self._build_index(
            vectors=vectors,
            index_type=index_type,
            index_params=index_params
        )
        build_time = time.time() - start_time
        
        logger.info(f"Built {index_type} index in {build_time:.2f}s")
        
        # Save index and IDs
        index_files = await self._save_index(
            index=index,
            ids=ids,
            db_config=db_config,
            embedding_model=embedding_model,
            index_type=index_type,
            storage_config=storage_config
        )
        
        logger.info(f"Saved index files: {index_files}")
        
        return index_files
    
    async def build_index_incremental(
        self,
        db_config: Dict[str, Any],
        embedding_config: Dict[str, Any],
        indexing_config: Dict[str, Any],
        storage_config: Dict[str, Any],
        batch_number: int,
        is_final_batch: bool = False,
        existing_index_obj: Optional[faiss.Index] = None  # NEW: optimize by passing existing obj
    ) -> Any: # Returns List[str] or Tuple[List[str], faiss.Index]
        logger.info("-" * 35)
        logger.info(f"INCREMENTAL INDEXING: Batch {batch_number}")
        logger.info("-" * 35)
        
        embedding_model = embedding_config['model']
        index_type = indexing_config['type']
        index_params = indexing_config.get('parameters', {})
        
        model_short = MODEL_SHORT_NAMES.get(embedding_model, embedding_model.split('/')[-1].lower())
        table = db_config.get('table', 'documents')
        dbname = db_config.get('dbname', 'unknown_db')
        base_name = f"{dbname}__{table}__{model_short}_{index_type}"

        index_path = Path(storage_config['index_path'])
        index_path.mkdir(parents=True, exist_ok=True)

        index_file = index_path / f"{base_name}.index"
        ids_file = index_path / f"{base_name}_ids.pkl"

        # Fetch vectors for current batch
        vectors, ids = await self._fetch_vectors_from_db(
            db_config=db_config,
            embedding_model=embedding_model,
            batch_number=batch_number
        )
        
        if len(vectors) == 0:
            logger.warning(f"No vectors found for batch {batch_number}")
            return []
        
        logger.info(f"Fetched {len(vectors)} vectors for batch {batch_number}")
        
        # Determine the existing index and IDs to merge into
        existing_index = None
        existing_ids = []

        if existing_index_obj:
            logger.info(f"Using hot in-memory index for batch {batch_number}")
            existing_index = existing_index_obj
            if ids_file.exists():
                with open(ids_file, 'rb') as f:
                    existing_ids = pickle.load(f)
        elif index_file.exists() and ids_file.exists():
            logger.info(f"Existing index found on disk, merging batch {batch_number}")
            loop = asyncio.get_event_loop()
            existing_index = await loop.run_in_executor(None, faiss.read_index, str(index_file))
            with open(ids_file, 'rb') as f:
                existing_ids = pickle.load(f)

        if existing_index:
            # --- OPTIMIZATION: Incremental Growth ---
            is_currently_flat = isinstance(existing_index, faiss.IndexFlatL2)
            will_be_hnsw = (index_type == 'hnsw')
            total_after = existing_index.ntotal + len(ids)
            
            if is_currently_flat and will_be_hnsw and total_after >= 500:
                logger.info(f"Transitioning from small-batch Flat index to HNSW index (count={total_after})")
                
                # Offload DB fetch to executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, 
                    lambda: asyncio.run(self._fetch_vectors_from_db(
                        db_config=db_config,
                        embedding_model=embedding_model
                    )) if asyncio.iscoroutinefunction(self._fetch_vectors_from_db) else 
                    self._fetch_vectors_from_db_sync(db_config, embedding_model)
                )
                all_vectors, all_ids_from_db = result
                
                # Safety checks
                if np.isnan(all_vectors).any() or np.isinf(all_vectors).any():
                    logger.warning("NaN or Inf detected in vectors! Cleaning...")
                    all_vectors = np.nan_to_num(all_vectors)
                
                all_vectors = np.ascontiguousarray(all_vectors.astype('float32'))
                
                params = index_params or {}
                M = params.get('M', 32)
                
                logger.info(f"Creating HNSW index (dim={all_vectors.shape[1]}, M={M})...")
                merged_index = faiss.IndexHNSWFlat(all_vectors.shape[1], M)
                merged_index.hnsw.efConstruction = params.get('efConstruction', 40)
                
                logger.info(f"Transitioning: adding {all_vectors.shape[0]} vectors to HNSW...")
                await loop.run_in_executor(None, merged_index.add, all_vectors)
                
                combined_ids = all_ids_from_db
                await asyncio.sleep(0)
            
            elif is_currently_flat and not will_be_hnsw:
                # If it's already Flat and we want Flat, just add
                logger.info(f"Adding {len(ids)} vectors to existing Flat index")
                existing_index.add(vectors)
                merged_index = existing_index
                combined_ids = existing_ids + ids
            
            else:
                # Fallback for other cases (IVF, HNSW->HNSW, etc.)
                new_index = await self._build_index(
                    vectors=vectors,
                    index_type=index_type,
                    index_params=index_params
                )
                merged_index = await self._merge_indexes(existing_index, new_index, index_type, index_params)
                combined_ids = existing_ids + ids
            
            logger.info(f"Merged index now has {merged_index.ntotal} vectors")
            
        else:
            logger.info(f"No existing index, creating new for batch {batch_number}")
            
            # Build new index
            merged_index = await self._build_index(
                vectors=vectors,
                index_type=index_type,
                index_params=index_params
            )
            
            combined_ids = ids
        
        # Save merged index
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, faiss.write_index, merged_index, str(index_file))
        
        def _save_ids():
            with open(ids_file, 'wb') as f:
                pickle.dump(combined_ids, f)
        
        await loop.run_in_executor(None, _save_ids)
        
        logger.info(f"Saved merged index: {index_file}")
        
        # Return both the files and the index object so the orchestrator can keep it "hot"
        return [f"{base_name}.index", f"{base_name}_ids.pkl"], merged_index
    
    def _fetch_vectors_from_db_sync(
        self,
        db_config: Dict[str, Any],
        embedding_model: str,
        batch_number: Optional[int] = None,
        progress_id: Optional[str] = None
    ) -> Tuple[np.ndarray, List[int]]:
        """Synchronous version for executor."""
        logger.info(f"Fetching vectors for model: {embedding_model} (sync)")
        db_conn = DynamicDatabaseConnection(db_config)
        conn = db_conn.get_connection()
        cursor = conn.cursor()
        try:
            table = db_config.get('table', 'documents')
            if batch_number:
                query = f"SELECT id, vector FROM {table} WHERE embedding_model = %s AND batch_number = %s ORDER BY id"
                cursor.execute(query, (embedding_model, batch_number))
            else:
                query = f"SELECT id, vector FROM {table} WHERE embedding_model = %s ORDER BY id"
                cursor.execute(query, (embedding_model,))
            
            rows = cursor.fetchall()
            if not rows: return np.array([], dtype='float32').reshape(0, 0), []
            
            ids, vectors = [], []
            for row in rows:
                ids.append(row[0])
                v_str = row[1]
                if isinstance(v_str, str):
                    v = [float(x) for x in v_str.strip('[]').split(',')]
                else: v = v_str
                vectors.append(v)
            
            return np.ascontiguousarray(np.array(vectors, dtype='float32')), ids
        finally:
            cursor.close()
            conn.close()

    async def _fetch_vectors_from_db(
        self,
        db_config: Dict[str, Any],
        embedding_model: str,
        batch_number: Optional[int] = None,
        progress_id: Optional[str] = None
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Fetch vectors from database, offloaded to executor.
        """

        # --- Check For Cancellation ---
        if progress_id:
            try:
                from .orchestrator import orchestrator
                if progress_id in orchestrator.active_pipelines and orchestrator.active_pipelines[progress_id].is_cancelled:
                    logger.warning(f"Vector fetch aborted: Pipeline {progress_id} was cancelled.")
                    raise asyncio.CancelledError("Pipeline cancelled before fetching vectors")
            except Exception as e:
                if isinstance(e, asyncio.CancelledError): raise
        # ------------------------------

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self._fetch_vectors_from_db_sync(db_config, embedding_model, batch_number, progress_id)
        )
    
    async def _build_index(
        self,
        vectors: np.ndarray,
        index_type: str,
        index_params: Dict[str, Any]
    ) -> faiss.Index:

        dimension = vectors.shape[1]
        num_vectors = vectors.shape[0]
        
        # Fallback for small vector counts or HNSW issues on Windows
        # Increased to 500 for better stability
        if index_type == 'hnsw' and num_vectors < 500:
            logger.info(f"Vector count {num_vectors} is small (<500). Overriding HNSW with 'flat' for stability and speed.")
            index_type = 'flat'

        logger.info(f"Building {index_type} index for {num_vectors} vectors (dim: {dimension})")
        
        if index_type == 'hnsw':
            M = index_params.get('M', 32)
            index = faiss.IndexHNSWFlat(dimension, M)
            index.hnsw.efConstruction = index_params.get('efConstruction', 40)
            index.hnsw.efSearch = index_params.get('efSearch', 16)
            
            logger.info(f"HNSW parameters: M={M}")
            logger.info(f"Vectors shape: {vectors.shape}, dtype: {vectors.dtype}")
            if vectors.size > 0:
                logger.info(f"Vector 0 sample (first 5): {vectors[0][:5]}")
            
            # Use try-except to catch any FAISS specific errors
            try:
                logger.info("Calling index.add(vectors)...")
                start_add = time.time()
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, index.add, vectors)
                logger.info(f"index.add(vectors) completed in {time.time() - start_add:.4f}s")
                await asyncio.sleep(0)
            except Exception as e:
                logger.error(f"FAISS index.add failed: {e}")
                raise
            
        elif index_type == 'flat':
            index = faiss.IndexFlatL2(dimension)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, index.add, vectors)
            await asyncio.sleep(0)
            
        elif index_type == 'ivf':
            nlist = index_params.get('nlist', min(100, num_vectors // 10))
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
            logger.info(f"IVF parameters: nlist={nlist}")
            logger.info("Training IVF index...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, index.train, vectors)
            await asyncio.sleep(0)
            await loop.run_in_executor(None, index.add, vectors)
            await asyncio.sleep(0)
            
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        logger.info(f"Index contains {index.ntotal} vectors")
        return index
    
    async def _merge_indexes(
        self,
        existing_index: faiss.Index,
        new_index: faiss.Index,
        index_type: str,
        index_params: Optional[Dict[str, Any]] = None
    ) -> faiss.Index:
        """
        Merge two FAISS indexes.
        
        Args:
            existing_index: Existing index with previous batches
            new_index: New index with current batch
            index_type: Type of index (hnsw, flat, ivf)
            index_params: Indexing parameters for fallback
            
        Returns:
            Merged FAISS index
        """
        logger.info(f"Merging indexes: {existing_index.ntotal} + {new_index.ntotal} vectors")
        
        if index_type == 'flat':
            # For Flat index, use merge_from
            existing_index.merge_from(new_index, 0)
            return existing_index
            
        elif index_type == 'hnsw':
            # HNSW doesn't support merge_from directly
            
            # Optimization: If both are actually Flat indices (staying flat below threshold), use faster merge
            if isinstance(existing_index, faiss.IndexFlatL2) and isinstance(new_index, faiss.IndexFlatL2):
                logger.info("Both indices are Flat (below HNSW threshold), using fast merge_from")
                existing_index.merge_from(new_index, 0)
                return existing_index

            # Extract all vectors and rebuild
            logger.warning("HNSW index requires rebuilding for merge")
            
            # Get all vectors from existing index
            existing_vectors = self._extract_vectors_from_index(existing_index)
            
            # Get all vectors from new index
            new_vectors = self._extract_vectors_from_index(new_index)
            
            # Combine
            all_vectors = np.vstack([existing_vectors, new_vectors])
            
            # Rebuild HNSW index with all vectors
            dimension = all_vectors.shape[1]
            
            # Robustly get HNSW parameters: existing_index might be IndexFlatL2 if small
            params = index_params or {}
            M = getattr(getattr(existing_index, 'hnsw', None), 'M', params.get('M', 32))
            efConstruction = getattr(getattr(existing_index, 'hnsw', None), 'efConstruction', params.get('efConstruction', 40))
            
            merged_index = faiss.IndexHNSWFlat(dimension, M)
            merged_index.hnsw.efConstruction = efConstruction
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, merged_index.add, all_vectors)
            await asyncio.sleep(0)
            
            logger.info("Rebuilt HNSW index with merged vectors")
            return merged_index
            
        elif index_type == 'ivf':
            # IVF indexes can't be merged — quantizers trained independently per batch
            # Rebuild from all vectors extracted from both indexes
            existing_vectors = self._extract_vectors_from_index(existing_index)
            new_vectors = self._extract_vectors_from_index(new_index)
            all_vectors = np.vstack([existing_vectors, new_vectors])
            
            dimension = all_vectors.shape[1]
            nlist = (index_params or {}).get('nlist', min(100, all_vectors.shape[0] // 10))
            quantizer = faiss.IndexFlatL2(dimension)
            merged_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, merged_index.train, all_vectors)
            await asyncio.sleep(0)
            await loop.run_in_executor(None, merged_index.add, all_vectors)
            await asyncio.sleep(0)
            
            logger.info(f"Rebuilt IVF index with {merged_index.ntotal} merged vectors")
            return merged_index
            
        else:
            raise ValueError(f"Unsupported index type for merging: {index_type}")
        
    def _extract_vectors_from_index(self, index: faiss.Index) -> np.ndarray:
        """Vectorized extraction of all vectors from a FAISS index."""
        n = index.ntotal
        d = index.d
        
        logger.info(f"Extracting {n} vectors from index (dim={d})...")
        if n == 0:
            return np.zeros((0, d), dtype=np.float32)
            
        # Optimization: use reconstruct_n for vectorized speed (O(N) in C++)
        # instead of O(N) slow Python loop calls.
        try:
            logger.info("Attempting reconstruct_n...")
            return index.reconstruct_n(0, n)
        except Exception as e:
            logger.warning(f"reconstruct_n failed, falling back to loop: {e}")
            vectors = np.zeros((n, d), dtype=np.float32)
            for i in range(n):
                if i % 100 == 0:
                    logger.info(f"Reconstruction progress: {i}/{n}")
                vectors[i] = index.reconstruct(i)
            return vectors
    
    
    async def _save_index(
        self,
        index: faiss.Index,
        ids: List[int],
        db_config: Dict[str, Any],
        embedding_model: str,
        index_type: str,
        storage_config: Dict[str, Any]
    ) -> List[str]:

        # Get model short name
        try:
            from ..config import MODEL_SHORT_NAMES
        except ImportError:
            pass
        model_short = MODEL_SHORT_NAMES.get( #type:ignore
            embedding_model,
            embedding_model.split('/')[-1].lower()
        )
        
        # Create table name — must match search lookup format
        table = db_config.get('table', 'documents')
        dbname = db_config.get('dbname', 'unknown_db')
        base_name = f"{dbname}__{table}__{model_short}_{index_type}"
        
        # Create storage directory
        index_path = Path(storage_config['index_path'])
        index_path.mkdir(parents=True, exist_ok=True)
        
        # Save index file
        index_file = f"{base_name}.index"
        index_filepath = index_path / index_file
        faiss.write_index(index, str(index_filepath))
        logger.info(f"Saved index: {index_filepath}")
        
        # Save IDs mapping
        ids_file = f"{base_name}_ids.pkl"
        ids_filepath = index_path / ids_file
        
        def _save_ids_final():
            with open(ids_filepath, 'wb') as f:
                pickle.dump(ids, f)
                
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _save_ids_final)
        logger.info(f"Saved IDs: {ids_filepath}")
        
        return [index_file, ids_file]
    
    async def build_indexes_preembedded(
        self,
        source_db_config: Dict[str, Any],
        vector_column: str,
        id_column: str,
        embedding_config: Dict[str, Any],
        indexing_config: Dict[str, Any],
        storage_config: Dict[str, Any],
    ) -> List[str]:
        """
        Build FAISS index from pre-existing vectors in the source table.
        No chunking or re-embedding — reads vector_column directly.
        """
        logger.info("PREEMBEDDED INDEXING: fetching vectors from source table")

        embedding_model = embedding_config['model']
        index_type = indexing_config['type']
        index_params = indexing_config.get('parameters', {})

        model_short = MODEL_SHORT_NAMES.get(embedding_model, embedding_model.split('/')[-1].lower())
        table = source_db_config.get('table', 'documents')
        dbname = source_db_config.get('dbname', 'unknown_db')
        base_name = f"{dbname}__{table}__{model_short}_{index_type}"

        index_path = Path(storage_config['index_path'])
        index_path.mkdir(parents=True, exist_ok=True)

        def _fetch():
            conn = psycopg2.connect(
                host=source_db_config['host'],
                port=source_db_config['port'],
                user=source_db_config['user'],
                password=source_db_config['password'],
                dbname=source_db_config['dbname'],
                connect_timeout=30,
            )
            try:
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT {id_column}, {vector_column} FROM {table} "
                    f"WHERE {vector_column} IS NOT NULL ORDER BY {id_column}"
                )
                rows = cursor.fetchall()
                cursor.close()
                return rows
            finally:
                conn.close()

        loop = asyncio.get_event_loop()
        rows = await loop.run_in_executor(None, _fetch)

        if not rows:
            raise ValueError(f"No vectors found in {table}.{vector_column}")

        ids, vectors = [], []
        for row in rows:
            ids.append(row[0])
            v = row[1]
            if isinstance(v, str):
                v = [float(x) for x in v.strip('[]').split(',')]
            elif hasattr(v, 'tolist'):
                v = v.tolist()
            vectors.append(v)

        vectors_np = np.ascontiguousarray(np.array(vectors, dtype='float32'))
        logger.info(f"Loaded {len(ids)} vectors (dim={vectors_np.shape[1]})")

        index = await self._build_index(vectors=vectors_np, index_type=index_type, index_params=index_params)

        index_file = f"{base_name}.index"
        ids_file = f"{base_name}_ids.pkl"

        await loop.run_in_executor(None, faiss.write_index, index, str(index_path / index_file))

        def _save_ids():
            with open(index_path / ids_file, 'wb') as f:
                pickle.dump(ids, f)

        await loop.run_in_executor(None, _save_ids)
        logger.info(f"Saved preembedded index: {index_path / index_file}")

        return [index_file, ids_file]

    def validate_index(
        self,
        index: faiss.Index,
        vectors: np.ndarray,
        top_k: int = 5
    ) -> bool:

        if index.ntotal == 0:
            logger.error("Index is empty")
            return False
        
        # Test search with first vector
        test_vector = vectors[0:1]
        
        try:
            distances, indices = index.search(test_vector, min(top_k, index.ntotal))
            
            # First result should be the query itself (distance ~ 0)
            if distances[0][0] > 0.01:
                logger.warning(f"First result distance unexpectedly high: {distances[0][0]}")
            
            logger.info(f"Index validation passed (found {len(indices[0])} results)")
            return True
            
        except Exception as e:
            logger.error(f"Index validation failed: {e}")
            return False
    
    def get_index_statistics(self, index: faiss.Index) -> Dict[str, Any]:
        stats = {
            "num_vectors": index.ntotal,
            "dimension": index.d,
            "index_type": type(index).__name__,
            "is_trained": index.is_trained if hasattr(index, 'is_trained') else True
        }
        
        # Add type-specific stats
        if isinstance(index, faiss.IndexHNSWFlat):
            stats['M'] = index.hnsw.M
            stats['efConstruction'] = index.hnsw.efConstruction
            stats['efSearch'] = index.hnsw.efSearch
        elif isinstance(index, faiss.IndexIVFFlat):
            stats['nlist'] = index.nlist
            stats['nprobe'] = index.nprobe
        
        return stats