import asyncio
import os
import psycopg2
import json
import numpy as np
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import re

from ..config import MODEL_SHORT_NAMES, CONFIGS_DIR, MODEL_DIMENSIONS, BASE_DIR
from .ingestion import DatabaseSource, DataIngestion
from .chunking import TextChunking
from .embedding import EmbeddingGeneration
from .indexing import IndexBuilding
from ..services.faiss_manager import faiss_manager
from ..services.llm_service import llm_service
from ..services.embedding_service import embedding_service
from ..db.dynamic_connection import DynamicDatabaseConnection
from ..utils.progress_stream import progress_manager  # pub/sub SSE queues

logger = logging.getLogger(__name__)


class PipelineProgress:
    """Track pipeline execution progress."""
    
    def __init__(self, progress_id: str):
        self.progress_id = progress_id
        from ..services.embedding_service import embedding_service
        self.device = embedding_service.device
        self.start_time = datetime.now().isoformat()
        self.steps = {
            "connect_db": {"status": "pending", "started_at": None, "completed_at": None, "error": None},
            "ingest":     {"status": "pending", "started_at": None, "completed_at": None, "error": None},
            "chunk":      {"status": "pending", "started_at": None, "completed_at": None, "error": None},
            "embed":      {"status": "pending", "started_at": None, "completed_at": None, "error": None},
            "index":      {"status": "pending", "started_at": None, "completed_at": None, "error": None},
            "retrieval":  {"status": "pending", "started_at": None, "completed_at": None, "error": None}
        }
        self.current_step = None
        self.status = "initialized"
        self.metadata = {}
        self.chunks_processed = 0
        self.embeddings_generated = 0
        self.total_batches = None
        self.batches_completed = 0
        self.model_loading = False  # True while embedding model is downloading
        self.cancellation_requested = False
        self.config_name = None
        self.config_payload = None
    
    @property
    def is_cancelled(self) -> bool:
        """Returns True if cancellation has been requested for this pipeline."""
        return self.cancellation_requested

    def start_step(self, step: str):
        """Mark step as started."""
        self.current_step = step
        self.steps[step]["status"] = "running"
        self.steps[step]["started_at"] = datetime.now().isoformat()
        logger.info(f"Pipeline {self.progress_id}: Starting step '{step}'")
    
    def complete_step(self, step: str, metadata: Optional[Dict] = None):
        """Mark step as completed with optional metadata."""
        self.steps[step]["status"] = "completed"
        self.steps[step]["completed_at"] = datetime.now().isoformat()
        if metadata:
            self.metadata[step] = metadata
        logger.info(f"Pipeline {self.progress_id}: Completed step '{step}'")
    
    def fail_step(self, step: str, error: str):
        """Mark step as failed."""
        self.steps[step]["status"] = "failed"
        self.steps[step]["completed_at"] = datetime.now().isoformat()
        self.steps[step]["error"] = error
        self.status = "failed"
        logger.error(f"Pipeline {self.progress_id}: Failed step '{step}' - {error}")
    
    def update_stats(self, chunks: Optional[int] = None, embeddings: Optional[int] = None, model_loading: Optional[bool] = None):
        """Update granular stats."""
        if chunks is not None:
            self.chunks_processed = chunks
        if embeddings is not None:
            self.embeddings_generated = embeddings
        if model_loading is not None:        
            self.model_loading = model_loading
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        step_keys = ["connect_db", "ingest", "chunk", "embed", "index", "retrieval"]
        total_steps = len(step_keys)
        
        completed_count = sum(1 for step in self.steps.values() if step["status"] in ("completed", "skipped"))
        base_progress = (completed_count / total_steps) * 100
        overall_progress = base_progress
        
        if self.current_step and self.steps[self.current_step]["status"] == "running":
            step_weight = 100 / total_steps
            partial = step_weight * 0.1  # default: just started
            
            if self.current_step == "embed":
                chunk_meta = self.metadata.get("chunk", {})
                total_chunks = chunk_meta.get("num_chunks", 0)
                if total_chunks > 0:
                    partial = step_weight * (self.embeddings_generated / total_chunks)
            
            elif self.current_step == "chunk":
                ingest_meta = self.metadata.get("ingest", {})
                total_docs = ingest_meta.get("count") or ingest_meta.get("num_documents") or 0
                if total_docs > 0:
                    pass  # placeholder for potential future chunk progress tracking based on documents processed
            
            overall_progress += partial

        # --- Incremental Mode Batch-based Progress Override ---
        # If we are in incremental mode, we override the step-based progress 
        # with high-fidelity batch progress to avoid jumping to 67% (4/6 steps) prematurely.
        if self.total_batches and self.total_batches > 1:
            total_b = self.total_batches
            done_b  = self.batches_completed
            
            # Base progress by completed batches
            batch_based_progress = (done_b / total_b) * 100
            
            # Add sub-batch nuance if we have a current batch running
            batch_meta = self.metadata.get("batch_progress", {})
            curr_b = batch_meta.get("current_batch", done_b + 1)
            
            if curr_b > done_b and curr_b <= total_b:
                # How far are we into the CURRENT batch?
                # We map the 0-100 base_progress (connect_db to retrieval)
                # to the single batch's slice.
                # However, in loop, we only do chunk/embed/index.
                # Let's just use the current base_progress normalized.
                slice_size = 100 / total_b
                # we don't want it to jump too far, so we cap base_progress influence
                # within the batch slice.
                sub_progress = (base_progress / 100) * slice_size
                overall_progress = ((curr_b - 1) / total_b) * 100 + sub_progress
            else:
                overall_progress = batch_based_progress

            # Cap at 99.9 until truly finished
            overall_progress = min(overall_progress, 99.9)
        
        # In any mode, if all steps are done, it's 100.
        if completed_count == total_steps:
            overall_progress = 100.0
            self.status = "completed"
        elif any(step["status"] == "failed" for step in self.steps.values()):
            self.status = "failed"
        else:
            self.status = "running"
        
        return {
            "progress_id":          self.progress_id,
            "current_step":         self.current_step,
            "steps":                self.steps,
            "overall_progress":     overall_progress,
            "status":               self.status,
            "metadata":             self.metadata,
            "chunks_processed":     self.chunks_processed,
            "embeddings_generated": self.embeddings_generated,
            "batches_completed":    self.batches_completed,
            "total_batches":        self.total_batches,
            "model_loading":        getattr(self, 'model_loading', False),  
            "cancellation_requested": self.cancellation_requested
        }


class PipelineOrchestrator:
    def __init__(self):
        self.active_pipelines: Dict[str, PipelineProgress] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}  # Track background tasks
    
    def register_task(self, progress_id: str, task: asyncio.Task):
        """Track an active asyncio task for a pipeline."""
        self.running_tasks[progress_id] = task
        logger.info(f"Registered task for pipeline: {progress_id}")

    def unregister_task(self, progress_id: str):
        """Remove a task from tracking (usually on completion/failure)."""
        if progress_id in self.running_tasks:
            del self.running_tasks[progress_id]
            logger.info(f"##### PIPELINE {progress_id} FULLY TERMINATED #####")
            logger.info(f"Unregistered task for pipeline: {progress_id}")

    def create_pipeline(self) -> str:
        """Create new pipeline and return progress ID."""
        progress_id = str(uuid.uuid4())
        self.active_pipelines[progress_id] = PipelineProgress(progress_id)
        logger.info(f"Created pipeline: {progress_id}")
        return progress_id
    
    def get_progress(self, progress_id: str) -> Optional[Dict[str, Any]]:
        """Get progress for specific pipeline."""
        if progress_id not in self.active_pipelines:
            return None
        return self.active_pipelines[progress_id].get_status()

    def get_progress_id_by_config(self, config_name: str) -> Optional[str]:
        """Find the active progress_id for a given config_name."""
        for pid, progress in self.active_pipelines.items():
            if progress.config_name == config_name:
                return pid
        return None

    async def _emit_progress(self, progress_id: str, config: dict):
        """
        Push a progress event to all SSE clients listening on this progress_id.

        Uses PipelineProgress as the single source of truth for step/percentage
        data; config provides the human-readable stage label and batch counters.

        The 0.1s sleep after publish is intentional: sleep(0) only yields the
        event loop once, which is not enough for the full SSE delivery chain
        (publish -> queue.get() -> generator yield -> ASGI send -> TCP flush).
        100ms overhead is invisible against steps that each take multiple seconds.
        """
        if not progress_id:
            return
        try:
            if progress_id not in self.active_pipelines:
                return

            state = self.active_pipelines[progress_id].get_status()

            payload = {
                "progress":             state.get("overall_progress", 0),
                "overall_progress":     state.get("overall_progress", 0),  # backward-compatability mirror
                "current_step":         state.get("current_step"),
                "status":               state.get("status", "running"),
                "chunks_processed":     state.get("chunks_processed", 0),
                "embeddings_generated": state.get("embeddings_generated", 0),
                "stage":                config.get("stage"),
                "mode":                 "batch" if (state.get("total_batches") or 0) > 1 else "single",
                "batches_completed":    state.get("batches_completed", 0),
                "total_batches":        state.get("total_batches"),
                "pipeline_completed":   config.get("pipeline_completed", False),
                "llm_summary":          config.get("llm_summary"),
                "partial_results":       state.get("partial_results") or config.get("partial_results")
            }

            await progress_manager.publish(progress_id, payload)
            await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"SSE publish failed: {e}")
    # -------------------------------------------------------------------------

    async def execute(
        self,
        config: Dict[str, Any],
        progress_id: str,
        source_type: str = 'database',
        source_data: Optional[Any] = None,
        resume_from: Optional[Dict[str, Any]] = None,
        config_name: Optional[str] = None, 
    ) -> Dict[str, Any]:
        """Execute pipeline (incremental or standard)."""

        if config.get('incremental_mode', False):
            logger.info("Starting INCREMENTAL pipeline execution")
            return await self.execute_incremental(
                config=config,
                progress_id=progress_id,
                source_type=source_type,
                source_data=source_data
            )

        # Standard (non-incremental) path
        logger.info("Starting STANDARD pipeline execution")

        if progress_id not in self.active_pipelines:
            return {"success": False, "error": "Invalid progress_id"}

        progress = self.active_pipelines[progress_id]
        progress.config_name = config_name or config.get('config_name') # type: ignore
        progress.config_payload = config # type: ignore
        
        # Standard execution counts as 1 total batch
        progress.total_batches = 1 #type: ignore
        progress.batches_completed = 0

        ingestion = DataIngestion()
        chunking  = TextChunking()
        embedding = EmbeddingGeneration()
        indexing  = IndexBuilding()

        # STEP 0: Connect DB
        progress.start_step("connect_db")
        config["stage"] = "Connecting to database..."
        await self._emit_progress(progress_id, config)
        await asyncio.sleep(0.05)
        progress.complete_step("connect_db")
        config["stage"] = "Database Connected"
        await self._emit_progress(progress_id, config)

        try:
            pipeline_config = config['pipeline']
            storage_config  = config['storage']
            db_config       = config['database']

            # Add hf_token to embedding config if present in root
            embedding_cfg = pipeline_config['embedding']
            if 'hf_token' in config:
                embedding_cfg['hf_token'] = config['hf_token']

            # STEP 1: Ingest
            progress.start_step("ingest")
            config["stage"] = "Ingesting data..."
            await self._emit_progress(progress_id, config)
            logger.info("-" * 35)
            logger.info("STEP 1/5: Data Ingestion")
            logger.info("-" * 35)

            if source_type == 'database':
                # Update ingest call to use progress_id
                raw_data = await ingestion.ingest(config, progress_id=progress_id)
            elif source_type == 'file' and source_data:
                raw_data = await ingestion.ingest_from_file(source_data)
            elif source_type == 'upload' and source_data:
                raw_data = await ingestion.ingest_from_upload(
                    source_data['content'], source_data['filename']
                )
            elif source_type == 'source_db' and source_data:
                # Update ingest call to use progress_id
                raw_data = await ingestion.ingest(config, progress_id=progress_id)
            else:
                raise ValueError(f"Unsupported source type: {source_type}")

            await asyncio.sleep(0)

            if not raw_data or len(raw_data) == 0:
                raise Exception("No data found in source")

            ingest_stats = ingestion.get_statistics(raw_data)
            logger.info(f"Ingested {len(raw_data)} documents | {ingest_stats}")
            progress.complete_step("ingest", ingest_stats)
            config["stage"] = "Data Ingestion Complete"
            await self._emit_progress(progress_id, config)

            # First cancellation request check after ingestion
            if progress.is_cancelled:
                return await self.handle_cancellation(progress_id, config, "Post-Ingestion")

            # STEP 2: Chunk
            # DB sources (source_db / database) are already row-level records —
            # re-chunking them would split meaningful units arbitrarily.
            # Skip chunking and treat each row as its own chunk directly.
            is_db_source = source_type in ('source_db', 'database')

            progress.start_step("chunk")
            if is_db_source:
                config["stage"] = "Using DB rows as chunks (skipping chunking)..."
                await self._emit_progress(progress_id, config)
                logger.info("-" * 35)
                logger.info("STEP 2/5: Chunking skipped (DB source — rows used directly)")
                logger.info("-" * 35)
                # Each row from the DB is already a standalone text unit
                chunks = [str(doc) if not isinstance(doc, str) else doc for doc in raw_data]
                chunks = [c for c in chunks if c and len(c) >= 10]
            else:
                config["stage"] = "Chunking documents..."
                await self._emit_progress(progress_id, config)
                logger.info("-" * 35)
                logger.info("STEP 2/5: Text Chunking")
                logger.info("-" * 35)
                chunking_config = pipeline_config['chunking']
                chunks = await chunking.chunk_documents(raw_data, chunking_config, progress_id=progress_id)
                chunks = chunking.validate_chunks(chunks, min_length=10)

            chunk_stats = chunking.get_statistics(chunks)
            logger.info(f"Using {len(chunks)} chunks | {chunk_stats}")
            progress.update_stats(chunks=len(chunks))
            progress.complete_step("chunk", chunk_stats)
            config["stage"] = "Chunking step complete"
            await self._emit_progress(progress_id, config)

            # STEP 3: Embed
            progress.start_step("embed")
            # Emit 'loading model' immediately - SentenceTransformer download can
            # take 30-300s; this gives the frontend a signal before the wait starts.
            progress.update_stats(model_loading=True)
            config["stage"] = "Loading embedding model..."
            await self._emit_progress(progress_id, config)
            logger.info("-" * 35)
            logger.info("STEP 3/5: Embedding Generation")
            logger.info("-" * 35)

            embedding_config = pipeline_config['embedding']
            embed_result = await embedding.generate_embeddings(
                chunks=chunks,
                embedding_config=embedding_config,
                db_config=db_config,
                progress_id=progress_id
            )

            await asyncio.sleep(0)

            logger.info(f"Generated {embed_result['num_embeddings']} embeddings | {embed_result}")
            progress.update_stats(embeddings=embed_result['num_embeddings'], model_loading=False)
            progress.complete_step("embed", embed_result)
            config["stage"] = "Embedding Generation Complete"
            await self._emit_progress(progress_id, config)

            # Second cancellation request check after embedding
            if progress.is_cancelled:
                return await self.handle_cancellation(progress_id, config, "Post-Embedding")

            # STEP 4: Index
            progress.start_step("index")
            config["stage"] = "Building FAISS index..."
            await self._emit_progress(progress_id, config)
            logger.info("-" * 35)
            logger.info("STEP 4/5: FAISS Index Building")
            logger.info("-" * 35)

            indexing_config = pipeline_config['indexing']
            index_files = await indexing.build_indexes(
                db_config=db_config,
                embedding_config=embedding_config,
                indexing_config=indexing_config,
                storage_config=storage_config,
                progress_id=progress_id
            )

            index_stats = {"index_files": index_files, "num_files": len(index_files)}
            logger.info(f"Built FAISS indexes: {index_files}")
            progress.complete_step("index", index_stats)
            config["stage"] = "FAISS Index Building Complete"
            await self._emit_progress(progress_id, config)
            await asyncio.sleep(0)

            # STEP 5: Retrieval Setup
            progress.start_step("retrieval")
            config["stage"] = "Setting up retrieval system..."
            await self._emit_progress(progress_id, config)
            logger.info("-" * 35)
            logger.info("STEP 5/5: Retrieval Configuration")
            logger.info("-" * 35)

            retrieval_stats = await self._setup_retrieval(
                index_files=index_files,
                storage_config=storage_config,
                embedding_model=embedding_config['model'],
                index_type=indexing_config['type'],
                db_config=db_config
            )

            logger.info("Retrieval system ready")
            progress.complete_step("retrieval", retrieval_stats)

            # --- Generate Final LLM Summary for standard mode ---
            try:
                logger.info("Generating final LLM summary...")
                config["stage"] = "Generating final summary..."
                await self._emit_progress(progress_id, config)
                
                insights = await self._generate_realtime_insights(config)
                config.update(insights)
                progress.metadata["summary"] = insights.get("summary")
            except Exception as e:
                logger.error(f"Failed to generate final summary: {e}")

            # Mark complete
            config['pipeline_completed']    = True
            config['pipeline_completed_at'] = datetime.now().isoformat()
            config['mode']                  = 'ready'
            config['storage']['index_files'] = index_files
            config["progress"] = 100
            config["stage"]    = "Pipeline Execution Complete"
            # Finalize batch count
            progress.batches_completed = 1

            # Persist config BEFORE final SSE so frontend sees 'completed' on disk
            config_name = config.get('config_name')
            if config_name:
                safe_name   = re.sub(r'[^\w\-_]', '_', config_name)
                config_path = CONFIGS_DIR / f"{safe_name}.json"
                persistent_config = {k: v for k, v in config.items()
                                    if k not in ('batches_completed', 'total_batches', 'stage', 'progress')}
                with open(config_path, 'w') as f:
                    json.dump(persistent_config, f, indent=2)
                logger.info(f"Config persisted to {config_path}")

            # Final SSE emission
            await self._emit_progress(progress_id, config)

            logger.info("-" * 35)
            logger.info("PIPELINE EXECUTION COMPLETE")
            logger.info("-" * 35)

            summary = {
                "documents_ingested":    len(raw_data),
                "chunks_created":        len(chunks),
                "embeddings_generated":  embed_result['num_embeddings'],
                "index_files_created":   len(index_files),
                "total_time_seconds":    embed_result.get('total_time_seconds', 0)
            }

            return {
                "success":     True,
                "message":     "Pipeline executed successfully",
                "config":      config,
                "progress_id": progress_id,
                "summary":     summary,
                "batches_completed": 1,
                "total_batches": 1
            }

        except asyncio.CancelledError:
            logger.info(f"Pipeline {progress_id} task was cancelled via asyncio.CancelledError. Handling cleanup...")
            return await self.handle_cancellation(progress_id, config, "Asyncio Cancellation")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Pipeline execution failed: {error_msg}", exc_info=True)

            if progress.current_step:
                progress.fail_step(progress.current_step, error_msg)

            # CRITICAL: pipeline_completed must be True so the SSE stream closes
            # and the frontend shows error UI instead of hanging at 'running'.
            config["pipeline_completed"] = True
            config["status"] = "failed"
            config["error"]  = error_msg
            await self._emit_progress(progress_id, config)

            return {"success": False, "error": error_msg, "progress_id": progress_id, "config": config}

    async def execute_incremental(
        self,
        config: Dict[str, Any],
        progress_id: str,
        source_type: str = 'database',
        source_data: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Execute pipeline incrementally in batch cycles without loading the full
        dataset into memory first.

        For source_db: batches by DB row offset - never loads all rows at once.
        For file/upload: batches by character range using TextFileSource.read_batch_chars(),
            so only one batch of text is ever in memory at a time.

        Progress step transitions (chunk -> embed -> index) happen once, on the
        first batch, so overall_progress is monotonically increasing and never
        drops backwards between batches.

        Per-batch granularity is pushed via _emit_progress() at each sub-step
        and stored in metadata["batch_progress"] for frontend display.

        Args:
            config:      RagConfig dict with incremental_mode=True
            progress_id: Pipeline progress ID
            source_type: 'file', 'upload', or 'source_db'
            source_data: File path string (file/upload) or source_db dict

        Returns:
            Execution result dict
        """
        if progress_id not in self.active_pipelines:
            return {"success": False, "error": "Invalid progress_id"}

        progress = self.active_pipelines[progress_id]
        progress.config_name = config.get('config_name')
        progress.config_payload = config # type: ignore

        ingestion = DataIngestion()
        chunking  = TextChunking()
        embedding = EmbeddingGeneration()
        indexing  = IndexBuilding()

        shared_index_obj = None  # Keep index in memory during the loop

        try:
            db_config       = config['database']
            pipeline_config = config['pipeline']
            storage_config  = config['storage']
            batch_size      = config.get('batch_size', 1000)

            # Step 0: Connect DB
            progress.start_step("connect_db")
            await asyncio.sleep(0.05)
            progress.complete_step("connect_db")

            # Step 1: Plan batches WITHOUT loading any document content
            # source_db: one COUNT query only.
            # file/upload: one stat() syscall only.
            progress.start_step("ingest")
            logger.info("-" * 35)
            logger.info("INCREMENTAL PIPELINE: Planning batches")
            logger.info("-" * 35)

            if source_type == 'source_db' and source_data:
                # Option 1: source_db
                source_db_reader = DatabaseSource(
                    db_config=source_data['db_config'],
                    chunk_column=source_data['chunk_column'],
                    id_column=source_data['id_column']
                )
                total_units   = int(source_db_reader.count_rows())
                total_batches = int((total_units + batch_size - 1) // batch_size)
                use_char_batching = False
                logger.info(
                    f"source_db: {total_units} rows -> "
                    f"{total_batches} batches of {batch_size} rows"
                )

            elif source_type in ('file', 'upload'):
                # Option 2: file / upload
                # For 'file':   source_data is the file path string.
                # For 'upload': the temp file path is in source_data (string) or
                #               config['source']['temp_path'] as a fallback.
                if source_type == 'file':
                    file_path = source_data
                else:
                    file_path = (
                        source_data if isinstance(source_data, str)
                        else config['source']['temp_path']
                    )

                file_source = ingestion.get_file_source(file_path) #type: ignore
                total_units = file_source.get_char_count()

                # Scale batch_size (row units) -> characters using chunk_size as
                # the proxy; chars/batch is approximately batch_size rows x chunk_size chars/row.
                chunk_size      = pipeline_config['chunking'].get('chunk_size', 500)
                batch_size_chars = batch_size * chunk_size
                total_batches   = int((total_units + batch_size_chars - 1) // batch_size_chars)
                use_char_batching = True
                logger.info(
                    f"file: {total_units} chars -> "
                    f"{total_batches} batches of about {batch_size_chars} characters each"
                )

            else:
                raise ValueError(f"Unsupported source type for incremental mode: {source_type}")

            config['total_batches'] = total_batches
            progress.total_batches = total_batches #type:ignore
            progress.batches_completed = 0
            progress.complete_step("ingest", {"total_units": total_units, "total_batches": total_batches})

            config["stage"]            = "Batch Planning Complete"
            await self._emit_progress(progress_id, config)
            logger.info(f"Will process in {total_batches} batches")

            # Steps 2-4: Chunk -> Embed -> Index - one batch at a time
            # chunk step is started here; it is completed (and embed/index started
            # in turn) after the FIRST batch only, so overall_progress only moves
            # forward and never resets between batches.
            progress.start_step("chunk")
            
            total_chunks_processed = 0
            total_embeddings_generated = 0

            for batch_num in range(1, total_batches + 1):
                # Check for cancellation request at each batch boundary
                if progress.is_cancelled:
                    return await self.handle_cancellation(progress_id, config, f"Batch {batch_num}")


                logger.info(f"BATCH {batch_num}/{total_batches}")
                logger.info("-" * 35)

                # Update progress object with current batch context
                progress.batches_completed = batch_num - 1 # or wait until it's actually DONE?
                # User wants to see "Processing batch X/Y" or "X-1/Y completed".
                # Usually "0/Y" means none are fully done yet.
                
                # Emit "Starting Batch X/Y" so UI reflects progress immediately
                config["stage"] = f"Processing Batch {batch_num}/{total_batches}: Reading..."
                
                # Update progress object with current batch context BEFORE emission
                progress.metadata["batch_progress"] = {
                    "current_batch": batch_num,
                    "total_batches": total_batches,
                    "percentage":    round(((batch_num - 1) / total_batches) * 100, 1)
                }
                
                await self._emit_progress(progress_id, config)

                # -- Read this batch from source (no full-file/full-table load) -
                if source_type == 'source_db' and source_data:
                    offset     = (batch_num - 1) * batch_size
                    batch_data = await source_db_reader.read_batch(offset, batch_size)  # type: ignore
                    logger.info(f"Read {len(batch_data)} rows from source DB (offset={offset})")
                else:
                    char_offset = (batch_num - 1) * batch_size_chars  # type: ignore
                    batch_data  = await file_source.read_batch_chars(  # type: ignore
                        char_offset, batch_size_chars  # type: ignore
                    )
                    logger.info(
                        f"Read file batch: {sum(len(d) for d in batch_data)} chars "
                        f"(char_offset={char_offset})"
                    )

                if not batch_data:
                    logger.warning(f"Batch {batch_num} returned no data - skipping")
                    continue

                # Chunk — DB source rows are already standalone text units; skip re-chunking
                if source_type == 'source_db':
                    batch_chunks = [str(d) if not isinstance(d, str) else d for d in batch_data]
                    batch_chunks = [c for c in batch_chunks if c and len(c) >= 10]
                    logger.info(f"DB source: using {len(batch_chunks)} rows directly as chunks")
                else:
                    chunking_config = pipeline_config['chunking']
                    batch_chunks = await chunking.chunk_documents(batch_data, chunking_config)
                    batch_chunks = chunking.validate_chunks(batch_chunks)
                    logger.info(f"Generated {len(batch_chunks)} chunks")
                total_chunks_processed += len(batch_chunks)
                progress.update_stats(chunks=total_chunks_processed)

                # Transition chunk -> embed on the first batch only
                if batch_num == 1:
                    progress.complete_step("chunk")
                    config["stage"] = f"Batch {batch_num}/{total_batches}: Chunking Complete"
                    await self._emit_progress(progress_id, config)
                    progress.start_step("embed")

                # Embed
                embedding_config = pipeline_config['embedding']
                embed_result = await embedding.generate_embeddings(
                    chunks=batch_chunks,
                    embedding_config=embedding_config,
                    db_config=db_config,
                    progress_id=progress_id,
                    batch_number=batch_num
                )
                total_embeddings_generated += embed_result['num_embeddings']
                progress.update_stats(embeddings=total_embeddings_generated, model_loading=False)
                logger.info(f"Generated {embed_result['num_embeddings']} embeddings")

                # Transition embed -> index on the first batch only
                if batch_num == 1:
                    progress.complete_step("embed")
                    config["stage"] = f"Batch {batch_num}/{total_batches}: Embedding Complete"
                    await self._emit_progress(progress_id, config)
                    progress.start_step("index")

                # Build / merge FAISS index
                indexing_config = pipeline_config['indexing']
                is_final        = (batch_num == total_batches)

                # Keep index in memory to avoid O(N^2) disk I/O
                # We pass the shared_index object to the builder
                index_files = await indexing.build_index_incremental(
                    db_config=db_config,
                    embedding_config=embedding_config,
                    indexing_config=indexing_config,
                    storage_config=storage_config,
                    batch_number=batch_num,
                    is_final_batch=is_final,
                    existing_index_obj=shared_index_obj  # new param
                )
                
                # Update the shared reference if the builder returns the index object
                if isinstance(index_files, tuple):
                    index_files, shared_index_obj = index_files
                    
                # -- If index_files is still a tuple here something is wrong, 
                # but we handle it by unpacking.
                
                # If shared_index_obj was updated (or created) in-place,
                # we don't need to faiss.read_index() next time.
                logger.info(f"Index updated with batch {batch_num}")

                # -- Force FAISS manager to reload the NEW index files from disk ------
                # This ensures _generate_realtime_insights can find the table
                if index_files and len(index_files) == 2:
                    try:
                        idx_dir = Path(storage_config['index_path'])
                        full_idx_path = str(idx_dir / index_files[0])
                        full_ids_path = str(idx_dir / index_files[1])
                        
                        # table_name calculation (consistent with indexing.py)
                        model_short = MODEL_SHORT_NAMES.get(embedding_config['model'], embedding_config['model'].split('/')[-1].lower())
                        table = db_config.get('table', 'documents')
                        dbname = db_config.get('dbname', 'unknown_db')
                        table_name = f"{dbname}__{table}__{model_short}_{indexing_config['type']}"
                        
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(
                            None,
                            lambda: faiss_manager.load_index_from_path(full_idx_path, full_ids_path, table_name)
                        )
                        logger.info(f"FAISS manager refreshed for {table_name}")
                    except Exception as e:
                        logger.error(f"Failed to refresh FAISS manager: {e}")

                # Update config state
                config['batches_completed']       = batch_num
                progress.batches_completed        = batch_num
                config['storage']['index_files']  = index_files

                # Store per-batch granularity for frontend
                progress.metadata["batch_progress"] = {
                    "current_batch": batch_num,
                    "total_batches": total_batches,
                    "percentage":    round((batch_num / total_batches) * 100, 1)
                }

                # -- Generate Real-time Insights (after first batch or any subsequent) ----------
                # This fulfills the user request: "updates the value while running itself in real time"
                if batch_num == 1 or batch_num % 5 == 0 or batch_num == total_batches:
                    try:
                        insights = await self._generate_realtime_insights(config)
                        config.update(insights)
                        logger.info(f"Real-time insights updated for batch {batch_num}")
                    except Exception as e:
                        logger.error(f"Failed to generate real-time insights: {e}")

                if batch_num == 1:
                    config['mode'] = 'partially_ready'
                    logger.info("First batch complete - search now available (partial results)")
                elif is_final:
                    config['mode']                  = 'ready'
                    config['pipeline_completed']    = True
                    config['pipeline_completed_at'] = datetime.now().isoformat()
                    logger.info("All batches complete - search now uses full dataset")
                else:
                    logger.info(f"Batch {batch_num}/{total_batches} complete, continuing...")

                # Emit batch-completion SSE event
                config["batches_completed"] = batch_num
                config["total_batches"]     = total_batches
                config["stage"]             = f"Completed Batch {batch_num}/{total_batches}"
                await self._emit_progress(progress_id, config)

                # Persist config after each batch so progress survives restarts.
                # Use CONFIGS_DIR (absolute Path) - never a relative string.
                config_name = config.get('config_name')
                if config_name:
                    safe_name   = re.sub(r'[^\w\-_]', '_', config_name)
                    config_path = CONFIGS_DIR / f"{safe_name}.json"

                    # Only persist stable config state - not runtime counters
                    persistent_config = {k: v for k, v in config.items()
                                        if k not in ('batches_completed', 'total_batches', 'stage', 'progress')}
                    
                    with open(config_path, 'w') as f:
                        json.dump(persistent_config, f, indent=2)

                # Reload FAISS index after batch 1 (enables search) and after
                # the final batch (refreshes to the complete dataset).
                if batch_num == 1 or is_final:
                    await self._reload_index_for_search(config)

            # Step 5: Retrieval
            progress.complete_step("index")
            progress.start_step("retrieval")
            logger.info("Retrieval system ready")
            progress.complete_step("retrieval")

            # Final persistence BEFORE emission
            config["progress"]          = 100
            config["pipeline_completed"] = True
            config['mode']              = 'ready'
            config["stage"]             = "Incremental Pipeline Complete"
            
            config_name = config.get('config_name')
            if config_name:
                safe_name   = re.sub(r'[^\w\-_]', '_', config_name)
                config_path = CONFIGS_DIR / f"{safe_name}.json"
                persistent_config = {k: v for k, v in config.items()
                                    if k not in ('batches_completed', 'total_batches', 'stage', 'progress')}
                with open(config_path, 'w') as f:
                    json.dump(persistent_config, f, indent=2)

            # Final SSE emission
            await self._emit_progress(progress_id, config)

            logger.info("-" * 35)
            logger.info("INCREMENTAL PIPELINE COMPLETE")
            logger.info("-" * 35)

            return {
                "success":              True,
                "message":              "Incremental pipeline completed successfully",
                "config":               config,
                "progress_id":          progress_id,
                "batches_completed":    total_batches,
                "total_batches":        total_batches,
                "search_available_from": "batch_1"
            }

        except asyncio.CancelledError:
            logger.info(f"Incremental pipeline {progress_id} task was cancelled. Handling cleanup...")
            return await self.handle_cancellation(progress_id, config, "Asyncio Cancellation")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Incremental pipeline failed: {error_msg}", exc_info=True)

            if progress.current_step:
                progress.fail_step(progress.current_step, error_msg)

            # CRITICAL: set pipeline_completed so SSE stream closes on error
            # (same pattern as standard execute() - without this the frontend hangs)
            config["pipeline_completed"] = True
            config["status"] = "failed"
            config["error"]  = error_msg
            await self._emit_progress(progress_id, config)

            return {
                "success":           False,
                "error":             error_msg,
                "progress_id":       progress_id,
                "batches_completed": config.get('batches_completed', 0)
            }
        finally:
            if 'file_source' in locals():
                try:
                    file_source.cleanup() # type: ignore
                except:
                    pass

    def request_cancellation(self, progress_id: str) -> bool:
        """Signal a running pipeline to cancel immediately via asyncio.Task.cancel()"""
        if progress_id not in self.active_pipelines:
            return False
            
        self.active_pipelines[progress_id].cancellation_requested = True
        logger.info(f"Cancellation requested for pipeline {progress_id}")
        
        # NEW: Hard cancellation of the asyncio task
        if progress_id in self.running_tasks:
            logger.info(f"Hard-cancelling asyncio task for pipeline {progress_id}")
            self.running_tasks[progress_id].cancel()
            return True
            
        return True
    
    async def _reload_index_for_search(self, config: Dict[str, Any]):
        """
        Reload index into FAISS manager after a batch completes.
        Enables immediate search with the latest merged index.
        """
        embedding_model = config['pipeline']['embedding']['model']
        index_type      = config['pipeline']['indexing']['type']
        db_config       = config['database']
        storage_config  = config['storage']

        model_short = MODEL_SHORT_NAMES.get(embedding_model, embedding_model.split('/')[-1])
        table       = db_config.get('table', 'documents')
        dbname      = db_config.get('dbname', 'unknown_db')
        table_name  = f"{dbname}__{table}__{model_short}_{index_type}"

        stored_dir = storage_config['index_path']
        resolved_dir = Path(stored_dir) if Path(stored_dir).exists() else BASE_DIR / "faiss_indexes" / f"{config.get('config_name', 'unknown')}.index"
        index_path = str(resolved_dir / f"{table_name}.index")
        ids_path   = str(resolved_dir / f"{table_name}_ids.pkl")

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, 
                lambda: faiss_manager.load_index_from_path(index_path, ids_path, table_name)
            )
            logger.info(f"Index reloaded for search: {table_name}")
        except Exception as e:
            logger.error(f"Failed to reload index: {e}")

    async def _setup_retrieval(
        self,
        index_files: List[str],
        storage_config: Dict[str, Any],
        embedding_model: str,
        index_type: str,
        db_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Setup retrieval system with newly built indexes."""
        logger.info("Setting up retrieval with index files:")
        for file in index_files:
            logger.info(f"  - {file}")

        model_short = MODEL_SHORT_NAMES.get(
            embedding_model, embedding_model.split('/')[-1].lower()
        )
        table      = db_config.get('table', 'documents')
        dbname     = db_config.get('dbname', 'unknown_db')
        table_name = f"{dbname}__{table}__{model_short}_{index_type}"

        stored_dir = storage_config['index_path']
        index_path = Path(stored_dir) if Path(stored_dir).exists() else BASE_DIR / "faiss_indexes"
        index_file = index_path / f"{table_name}.index"
        ids_file   = index_path / f"{table_name}_ids.pkl"

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: faiss_manager.load_index_from_path(str(index_file), str(ids_file), table_name)
            )
            stats = faiss_manager.get_index_stats(table_name)
            logger.info("Retrieval system configured successfully")
            return stats or {"status": "ready"}
        except Exception as e:
            logger.error(f"Failed to load index into manager: {e}")
            # Non-fatal - indexes are saved to disk, can be loaded on next request
            return {"status": "indexes_saved", "note": "Load manually if needed"}


    async def _generate_realtime_insights(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a partial LLM summary and sample search results using the 
        currently indexed data.
        """
        config_name = config.get('config_name')
        if not config_name:
            return {}

        preview_query = config.get('preview_query', "Provide a comprehensive summary of the ingested data")
        
        embedding_model = config['pipeline']['embedding']['model']
        index_type = config['pipeline']['indexing']['type']
        db_config = config['database']
        
        # table_name calculation (consistent with indexing.py)
        model_short = MODEL_SHORT_NAMES.get(embedding_model, embedding_model.split('/')[-1].lower())
        table = db_config.get('table', 'documents')
        dbname = db_config.get('dbname', 'unknown_db')
        table_name = f"{dbname}__{table}__{model_short}_{index_type}"
        
        logger.info(f"Generating realtime insights for table: {table_name}")

        results = []
        summary = None
        llm_model = config['pipeline'].get('llm', {}).get('model') or 'qwen3-0.6b'

        try:
            loop = asyncio.get_running_loop()
            
            # 1. Encode query (offload)
            query_vec = await loop.run_in_executor(
                None,
                lambda: embedding_service.encode(preview_query, model_name=embedding_model)
            )
            
            # System-wide padding to max dimension (consistency with embedding.py)
            max_dim = max(MODEL_DIMENSIONS.values())
            if len(query_vec) < max_dim:
                padded = np.zeros(max_dim, dtype=np.float32)
                padded[:len(query_vec)] = query_vec
                query_vec = padded

            # 2. Search (offload)
            top_k = config.get('pipeline', {}).get('top_k', 3)
            distances, indices = await loop.run_in_executor(
                None,
                lambda: faiss_manager.search(table_name, query_vec, top_k=top_k)
            )
            
            # Build id->distance map before DB fetch so scores align with rows
            # regardless of the order PostgreSQL returns them.
            valid_positions = [j for j, idx in enumerate(indices[0]) if idx != -1]
            db_ids = faiss_manager.get_database_ids(table_name, indices[0])
            id_to_dist = {db_ids[k]: float(distances[0][valid_positions[k]]) for k in range(len(db_ids))}

            if db_ids:
                # 3. DB fetch (offload)
                db_conn = DynamicDatabaseConnection(db_config)
                rows = await loop.run_in_executor(
                    None,
                    lambda: db_conn.fetch_documents(db_ids)
                )

                results = [
                    {
                        "id": row[0],
                        "id_val": row[0],
                        "text": row[1],
                        "content": row[1],
                        "score": round(max(0.0, 1.0 - id_to_dist.get(row[0], 0.0) / 2.0), 4),
                        "similarity_score": round(max(0.0, 1.0 - id_to_dist.get(row[0], 0.0) / 2.0), 4),
                        "embedding_model": row[2],
                        "metadata": {"source": row[2]}
                    }
                    for row in rows
                ]
                
                # 4. LLM Summary — disabled, skip silently
                chunks = [r['text'] for r in results]
                summary = None

        except Exception as e:
            logger.error(f"Insight generation error: {e}", exc_info=True)
            
        return {
            "summary": summary,
            "llm_summary": summary,
            "partial_results": results,
            "model_used": llm_model,
            "query": preview_query
        }
    
    async def handle_cancellation(self, progress_id: str, config: Dict[str, Any], stage_name: str) -> Dict[str, Any]:
        """Centralized handler for confirmed cancellation requests."""
        if progress_id in self.active_pipelines and self.active_pipelines[progress_id].status == "cancelled":
             logger.info(f"Pipeline {progress_id} already being cancelled. Ignoring duplicate request.")
             return {"status": "cancelled", "config": config}

        logger.info(f"Pipeline {progress_id} cancellation confirmed at stage: {stage_name}")
        
        # 1. Update progress state
        if progress_id in self.active_pipelines:
            self.active_pipelines[progress_id].status = "cancelled"
            
        # 2. Update config for terminal emission
        config['mode'] = 'cancelled'
        config['pipeline_completed'] = True  # CRITICAL: Mark as terminal so SSE loop closes
        
        # 3. Emit final terminal signal with high-visibility log
        logger.info("################################################################")
        logger.info(f"TERMINATING PIPELINE {progress_id} - CLEANUP STARTING")
        logger.info("################################################################")
        await self._emit_progress(progress_id, {
            **config,
            "stage": "Cancelled and Cleaned",
            "status": "cancelled",
            "message": f"Pipeline cancelled at {stage_name}"
        })
        
        # 4. Perform cleanup
        await self._cleanup_pipeline(config, progress_id)
        
        return {"status": "cancelled", "config": config}

    async def _cleanup_pipeline(self, config: Dict[str, Any], progress_id: str) -> None:
        """Delete partial DB data and index files created by a cancelled pipeline."""
        import shutil

        # --- DB cleanup ---
        db_config = config.get('database', {})
        db_name = db_config.get('dbname', '')
        auto_created = db_name.startswith('rag_db_')

        def _do_db_cleanup():
            try:
                if auto_created:
                    # Drop the entire auto-created database
                    conn = psycopg2.connect(
                        host=db_config['host'], port=db_config['port'],
                        user=db_config['user'], password=db_config['password'],
                        dbname='postgres'
                    )
                    conn.autocommit = True
                    cursor = conn.cursor()
                    
                    # Force disconnect other users before dropping
                    try:
                        cursor.execute(f"""
                            SELECT pg_terminate_backend(pid) 
                            FROM pg_stat_activity 
                            WHERE datname = '{db_name}' 
                            AND pid <> pg_backend_pid();
                        """)
                    except Exception as e:
                        logger.warning(f"Failed to kill active sessions for {db_name}: {e}")

                    cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
                    cursor.close()
                    conn.close()
                    logger.info(f"Dropped auto-created DB: {db_name}")
                else:
                    # User-provided DB — only delete rows inserted by this pipeline
                    table = db_config.get('table', 'documents')
                    conn = psycopg2.connect(
                        host=db_config['host'], port=db_config['port'],
                        user=db_config['user'], password=db_config['password'],
                        dbname=db_name
                    )
                    cursor = conn.cursor()
                    cursor.execute(f"DELETE FROM {table} WHERE source = 'pipeline'")
                    conn.commit()
                    cursor.close()
                    conn.close()
                    logger.info(f"Deleted pipeline rows from user-provided table: {table}")
            except Exception as e:
                logger.error(f"DB cleanup failed: {e}")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _do_db_cleanup)

        # --- Index directory cleanup ---
        try:
            index_path = Path(config.get('storage', {}).get('index_path', ''))
            if index_path.exists():
                # Unload from memory first to release file handles (Windows requirement)
                try:
                    embedding_model = config.get('pipeline', {}).get('embedding', {}).get('model', '')
                    index_type = config.get('pipeline', {}).get('indexing', {}).get('type', '')
                    if embedding_model and index_type:
                        model_short = MODEL_SHORT_NAMES.get(
                            embedding_model, embedding_model.split('/')[-1].lower()
                        )
                        table = config.get('database', {}).get('table', 'documents')
                        dbname = config.get('database', {}).get('dbname', 'unknown_db')
                        table_name = f"{dbname}__{table}__{model_short}_{index_type}"
                        faiss_manager.unload_index(table_name)
                except Exception as e:
                    logger.warning(f"Failed to unload index before cleanup (non-fatal): {e}")

                shutil.rmtree(str(index_path))
                logger.info(f"Deleted index directory: {index_path}")
        except Exception as e:
            logger.error(f"Index cleanup failed: {e}")

    async def execute_preembedded(
        self,
        config: Dict[str, Any],
        progress_id: str,
        config_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the pre-embedded pipeline.

        Source table already contains text + vectors — no chunking or re-embedding
        is performed. Steps 1-3 (ingest/chunk/embed) are marked as skipped and the
        pipeline jumps straight to index building and retrieval.

        Config must include:
            source_db.db_config     – connection to the source table
            source_db.text_column   – column with display text
            source_db.vector_column – column with pre-computed vectors
            source_db.id_column     – primary key column
            pipeline.embedding.model – model name (for query encoding during search)
            pipeline.indexing.type  – hnsw | flat | ivf
        """
        if progress_id not in self.active_pipelines:
            return {"success": False, "error": "Invalid progress_id"}

        progress = self.active_pipelines[progress_id]
        progress.config_name = config_name or config.get('config_name')  # type: ignore
        progress.config_payload = config  # type: ignore
        progress.total_batches = 1
        progress.batches_completed = 0

        indexing = IndexBuilding()

        try:
            source_db_cfg  = config['source_db']['db_config']
            vector_column  = config['source_db']['vector_column']
            id_column      = config['source_db']['id_column']
            text_column    = config['source_db'].get('text_column', 'chunks')
            pipeline_cfg   = config['pipeline']
            storage_cfg    = config['storage']
            embedding_cfg  = pipeline_cfg['embedding']
            indexing_cfg   = pipeline_cfg['indexing']

            # Ensure the db_config used for retrieval knows the text column
            source_db_cfg['text_column'] = text_column

            # Step 0: Connect DB
            progress.start_step("connect_db")
            await self._emit_progress(progress_id, {"stage": "Connecting to source database..."})
            await asyncio.sleep(0.05)
            progress.complete_step("connect_db")
            await self._emit_progress(progress_id, {"stage": "Connected"})

            # Steps 1–3: Skipped (data already embedded)
            for step in ("ingest", "chunk", "embed"):
                progress.steps[step]["status"] = "skipped"
                progress.steps[step]["started_at"] = datetime.now().isoformat()
                progress.steps[step]["completed_at"] = datetime.now().isoformat()

            config["stage"] = "Skipping ingest/chunk/embed — vectors already present"
            await self._emit_progress(progress_id, config)

            # Step 4: Build FAISS index from pre-existing vectors
            progress.start_step("index")
            config["stage"] = "Building FAISS index from existing vectors..."
            await self._emit_progress(progress_id, config)

            index_files = await indexing.build_indexes_preembedded(
                source_db_config=source_db_cfg,
                vector_column=vector_column,
                id_column=id_column,
                embedding_config=embedding_cfg,
                indexing_config=indexing_cfg,
                storage_config=storage_cfg,
            )

            progress.complete_step("index", {"index_files": index_files})
            config["stage"] = "FAISS Index Built"
            await self._emit_progress(progress_id, config)

            # Step 5: Retrieval setup
            progress.start_step("retrieval")
            config["stage"] = "Setting up retrieval..."
            await self._emit_progress(progress_id, config)

            # The database for search is the SOURCE table (it already has text+vectors).
            # id_column must be carried so fetch_documents uses the right PK column.
            config['database'] = {**source_db_cfg, 'text_column': text_column, 'id_column': id_column}

            retrieval_stats = await self._setup_retrieval(
                index_files=index_files,
                storage_config=storage_cfg,
                embedding_model=embedding_cfg['model'],
                index_type=indexing_cfg['type'],
                db_config=source_db_cfg,
            )
            progress.complete_step("retrieval", retrieval_stats)
            progress.batches_completed = 1

            # Persist config
            config['pipeline_completed']    = True
            config['pipeline_completed_at'] = datetime.now().isoformat()
            config['mode']                  = 'ready'
            config['storage']['index_files'] = index_files
            config["stage"] = "Pre-embedded Pipeline Complete"
            config["progress"] = 100

            cfg_name = config.get('config_name')
            if cfg_name:
                safe_name   = re.sub(r'[^\w\-_]', '_', cfg_name)
                config_path = CONFIGS_DIR / f"{safe_name}.json"
                persistent  = {k: v for k, v in config.items()
                               if k not in ('batches_completed', 'total_batches', 'stage', 'progress')}
                with open(config_path, 'w') as f:
                    json.dump(persistent, f, indent=2)

            await self._emit_progress(progress_id, config)

            return {
                "success":     True,
                "message":     "Pre-embedded pipeline complete — index ready for search",
                "config":      config,
                "progress_id": progress_id,
                "index_files": index_files,
            }

        except asyncio.CancelledError:
            return await self.handle_cancellation(progress_id, config, "Pre-embedded Cancellation")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Pre-embedded pipeline failed: {error_msg}", exc_info=True)
            if progress.current_step:
                progress.fail_step(progress.current_step, error_msg)
            config["pipeline_completed"] = True
            config["status"] = "failed"
            config["error"]  = error_msg
            await self._emit_progress(progress_id, config)
            return {"success": False, "error": error_msg, "progress_id": progress_id}


# Global orchestrator instance
orchestrator = PipelineOrchestrator()