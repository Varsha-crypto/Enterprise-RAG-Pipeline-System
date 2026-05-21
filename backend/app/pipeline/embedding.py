import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional
import time
from ..utils.progress_stream import progress_manager, _yield_control

logger = logging.getLogger(__name__)


class EmbeddingGeneration:
    
    def __init__(self):
        logger.info("EmbeddingGeneration initialized")
    
    async def generate_embeddings(
        self,
        chunks: List[str],
        embedding_config: Dict[str, Any],
        db_config: Dict[str, Any],
        progress_id: Optional[str] = None,
        batch_number: Optional[int] = None
    ) -> Dict[str, Any]:

        logger.info("-" * 35)
        logger.info("EMBEDDING: Generating vector embeddings")
        logger.info("-" * 35)
        
        model_name = embedding_config['model']
        normalize = embedding_config.get('normalize', True)
        batch_size = embedding_config.get('batch_size', 32)
        
        logger.info(f"Model: {model_name}")
        logger.info(f"Normalize: {normalize}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Total chunks: {len(chunks)}")
        
        # Import embedding service
        from ..services.embedding_service import embedding_service

        # Pre-load diagnostics
        # Emit 'loading model' status so SSE bar shows activity during model download
        if progress_id:
            try:
                from .orchestrator import orchestrator
                if progress_id in orchestrator.active_pipelines:
                    orchestrator.active_pipelines[progress_id].update_stats(model_loading=True)
                    await orchestrator._emit_progress(progress_id, {
                        "stage": f"Loading embedding model ({model_name})..."
                    })
            except Exception:
                pass

        logger.info(f"Pre-loading embedding model: {model_name}")
        try:
            # CRITICAL: load_model is synchronous (downloads + loads weights from HuggingFace).
            # Running it directly on the event loop would freeze asyncio for 30-300s,
            # preventing SSE heartbeats and causing the stale-stream watchdog to fire.
            # run_in_executor() offloads it to a ThreadPoolExecutor, keeping the event loop alive.
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, embedding_service.load_model, model_name)
            logger.info(f"Embedding model ready: {model_name}")
        except RuntimeError as model_err:
            logger.error(f"Model load failed: {model_err}")
            raise  # Propagate up - orchestrator will emit failure SSE and close stream

        # Mark model as loaded and emit 'generating' status
        if progress_id:
            try:
                from .orchestrator import orchestrator
                if progress_id in orchestrator.active_pipelines:
                    orchestrator.active_pipelines[progress_id].update_stats(model_loading=False)
                    await orchestrator._emit_progress(progress_id, {
                        "stage": f"Generating embeddings for {len(chunks)} chunks..."
                    })
            except Exception:
                pass

        # Yield before starting batch loop so SSE flush happens
        await asyncio.sleep(0)

        # Generate embeddings in batches
        all_embeddings = []
        start_time = time.time()
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
            
            # Generate embeddings for batch using vectorization
            # Offload heavy NumPy/SentenceTransformer computation to a thread pool
            # so the FastAPI event loop stays perfectly unblocked for real-time SSE.
            loop = asyncio.get_event_loop()
            batch_ndarrays = await loop.run_in_executor(
                None,
                lambda b=batch, m=model_name, n=normalize: embedding_service.encode_batch(
                    b, model_name=m, normalize=n
                )
            )
            # `encode_batch` returns a 2D numpy array. `list()` converts it into a list of 1D arrays.
            batch_embeddings = list(batch_ndarrays)
            
            all_embeddings.extend(batch_embeddings)
            
            # Update real-time progress after each embedding batch
            if progress_id:
                try:
                    from .orchestrator import orchestrator
                    if progress_id in orchestrator.active_pipelines:
                        orchestrator.active_pipelines[progress_id].update_stats(
                            embeddings=len(all_embeddings)
                        )
                        # Use _emit_progress for a complete, consistent payload
                        await orchestrator._emit_progress(progress_id, {
                            "stage": f"Embedding batch {batch_num}/{total_batches}",
                        })
                except Exception as e:
                    logger.debug(f"Could not update progress stats: {e}")

            # Yield control to event loop for SSE flush
            await asyncio.sleep(0)

            # --- Check For Cancellation ---
            if progress_id:
                try:
                    from .orchestrator import orchestrator
                    if progress_id in orchestrator.active_pipelines and orchestrator.active_pipelines[progress_id].is_cancelled:
                        logger.warning(f"Embedding generation aborted: Pipeline {progress_id} was cancelled.")
                        raise asyncio.CancelledError("Pipeline cancelled during embedding generation")
                except Exception as e:
                    if isinstance(e, asyncio.CancelledError):
                        raise
            # ------------------------------
            
            # Log progress
            if batch_num % 5 == 0:
                elapsed = time.time() - start_time
                rate = len(all_embeddings) / elapsed
                logger.info(f"Progress: {len(all_embeddings)}/{len(chunks)} chunks ({rate:.1f} chunks/sec)")
        
        embedding_time = time.time() - start_time
        logger.info(f"Generated {len(all_embeddings)} embeddings in {embedding_time:.2f}s")
        
        # Insert into database (system-agnostic)
        insert_time = await self._insert_embeddings(
            chunks=chunks,
            embeddings=all_embeddings,
            embedding_model=model_name,
            db_config=db_config,
            batch_number=batch_number
        )
        
        total_time = embedding_time + insert_time
        
        return {
            "num_chunks": len(chunks),
            "num_embeddings": len(all_embeddings),
            "embedding_model": model_name,
            "embedding_time_seconds": round(embedding_time, 2),
            "insert_time_seconds": round(insert_time, 2),
            "total_time_seconds": round(total_time, 2),
            "chunks_per_second": round(len(chunks) / total_time, 2)
        }
    
    async def _insert_embeddings(
        self,
        chunks: List[str],
        embeddings: List[np.ndarray],
        embedding_model: str,
        db_config: Dict[str, Any],
        batch_number: Optional[int] = None
    ) -> float:
        """
        Insert embeddings into database.
        
        System-agnostic: Uses dynamic database connection.
        
        Args:
            chunks: Text chunks
            embeddings: Embedding vectors
            embedding_model: Model name
            db_config: Database configuration
            
        Returns:
            Time taken for insertion
        """
        logger.info("Inserting embeddings into database...")
        start_time = time.time()
        
        from ..db.dynamic_connection import DynamicDatabaseConnection
        
        # Connect to configured database
        db_conn = DynamicDatabaseConnection(db_config)
        
        # Get max dimension from config
        from ..config import MODEL_DIMENSIONS
        max_dim = max(MODEL_DIMENSIONS.values())
        
        # Prepare vectors for insertion
        vectors = []
        for j, embedding in enumerate(embeddings):
            if j % 50 == 0:
                await asyncio.sleep(0)
            # Convert to numpy array if needed
            if hasattr(embedding, 'numpy'):
                embedding = embedding.numpy()
            elif not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            # Pad to max dimension
            if len(embedding) < max_dim:
                padded = np.zeros(max_dim, dtype=np.float32)
                padded[:len(embedding)] = embedding
                vectors.append(padded.tolist())
            else:
                vectors.append(embedding.tolist())
        
        # Insert in batches
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, 
                lambda: db_conn.insert_embeddings(chunks, vectors, embedding_model, batch_number)
            )
        except Exception as e:
            logger.error(f"Database insertion failed: {e}")
            raise
        
        insert_time = time.time() - start_time
        logger.info(f"Inserted {len(chunks)} embeddings in {insert_time:.2f}s")
        
        return insert_time
    
    def validate_embeddings(
        self,
        embeddings: List[np.ndarray],
        expected_dim: int
    ) -> bool:
        
        if not embeddings:
            logger.error("No embeddings generated")
            return False
        
        for i, emb in enumerate(embeddings):
            # Check dimension
            if len(emb) != expected_dim:
                logger.error(f"Embedding {i} has wrong dimension: {len(emb)} vs {expected_dim}")
                return False
            
            # Check for NaN or Inf
            if np.isnan(emb).any() or np.isinf(emb).any():
                logger.error(f"Embedding {i} contains NaN or Inf")
                return False
        
        logger.info(f"Validated {len(embeddings)} embeddings")
        return True
    
    def get_embedding_statistics(
        self,
        embeddings: List[np.ndarray]
    ) -> Dict[str, Any]:
        
        if not embeddings:
            return {}
        
        # Convert to numpy array
        emb_array = np.array(embeddings)
        
        return {
            "count": len(embeddings),
            "dimension": emb_array.shape[1],
            "mean_norm": float(np.mean(np.linalg.norm(emb_array, axis=1))),
            "std_norm": float(np.std(np.linalg.norm(emb_array, axis=1))),
            "min_value": float(np.min(emb_array)),
            "max_value": float(np.max(emb_array)),
            "mean_value": float(np.mean(emb_array)),
            "std_value": float(np.std(emb_array))
        }