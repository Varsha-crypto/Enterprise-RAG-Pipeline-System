import asyncio
import logging
import re
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    async def chunk(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Input text
            
        Returns:
            List of text chunks
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return strategy configuration."""
        pass


class FixedSizeChunking(ChunkingStrategy):    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if chunk_overlap >= chunk_size:
            raise ValueError("Overlap must be less than chunk size")
        
        logger.info(f"FixedSizeChunking: size={chunk_size}, overlap={chunk_overlap}")
    
    async def chunk(self, text: str) -> List[str]:
        if not text or len(text) == 0:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # Get chunk
            chunk = text[start:end]
            
            # Only add non-empty chunks
            if chunk.strip():
                chunks.append(chunk)
            
            # Move start position (accounting for overlap)
            start += (self.chunk_size - self.chunk_overlap)
            
            # Yield control every few iterations if it's a huge doc
            if len(chunks) % 50 == 0:
                await asyncio.sleep(0)

            # Prevent infinite loop on very small texts
            if start >= len(text):
                break
        
        logger.debug(f"Created {len(chunks)} chunks from {len(text)} characters")
        return chunks
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration."""
        return {
            "strategy": "fixed_size",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }


class SentenceBasedChunking(ChunkingStrategy):
    def __init__(self, target_size: int = 500, max_size: int = 750):
        self.target_size = target_size
        self.max_size = max_size
        logger.info(f"SentenceBasedChunking: target={target_size}, max={max_size}")
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (can be enhanced later)
        sentence_endings = r'[.!?]+[\s\n]+'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]
    
    async def chunk(self, text: str) -> List[str]:
        """Split text into chunks on sentence boundaries."""
        if not text or len(text) == 0:
            return []
        
        sentences = self._split_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed max_size, start new chunk
            if current_length + sentence_length > self.max_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            
            # If near target size, start new chunk
            elif current_length + sentence_length > self.target_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            
            # Otherwise, add to current chunk
            else:
                current_chunk.append(sentence)
                current_length += sentence_length + 1  # +1 for space
            
            # Yield periodically
            if len(chunks) % 20 == 0:
                await asyncio.sleep(0)
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        logger.debug(f"Created {len(chunks)} sentence-based chunks")
        return chunks
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration."""
        return {
            "strategy": "sentence_based",
            "target_size": self.target_size,
            "max_size": self.max_size
        }



class TextChunking:
    """
    Main text chunking coordinator.
    
    Supports multiple chunking strategies through extensible architecture.
    """
    
    def __init__(self):
        self.strategies = {
            'fixed_size': FixedSizeChunking,
            'sentence_based': SentenceBasedChunking,
        }
        logger.info("TextChunking initialized")
    
    def _get_strategy(self, chunking_config: Dict[str, Any]) -> ChunkingStrategy:
        strategy_name = chunking_config['strategy']
        
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown chunking strategy: {strategy_name}")
        
        strategy_class = self.strategies[strategy_name]
        
        # Create instance with appropriate parameters
        if strategy_name == 'fixed_size':
            return strategy_class(
                chunk_size=chunking_config.get('chunk_size', 500),
                chunk_overlap=chunking_config.get('chunk_overlap', 50)
            )
        elif strategy_name == 'sentence_based':
            return strategy_class(
                target_size=chunking_config.get('target_size', 500),
                max_size=chunking_config.get('max_size', 750)
            )
        else:
            return strategy_class()
    
    async def chunk_documents(
        self, 
        documents: List[str], 
        chunking_config: Dict[str, Any],
        progress_id: Optional[str] = None
    ) -> List[str]:
        logger.info("-" * 35)
        logger.info("CHUNKING: Splitting documents into chunks")
        logger.info("-" * 35)
        
        strategy = self._get_strategy(chunking_config)
        logger.info(f"Using strategy: {strategy.get_config()}")
        
        all_chunks = []
        
        for i, document in enumerate(documents, 1):
            doc_chunks = await strategy.chunk(document)
            all_chunks.extend(doc_chunks)
            
            # Frequent progress updates (after every document or large doc part)
            if i % 1 == 0: 
                logger.info(f"Processed {i}/{len(documents)} documents. Chunks so far: {len(all_chunks)}")
                
                # Update progress if possible
                if progress_id:
                    try:
                        from .orchestrator import orchestrator
                        if progress_id in orchestrator.active_pipelines:
                            orchestrator.active_pipelines[progress_id].update_stats(
                                chunks=len(all_chunks)
                            )
                            # Trigger immediate SSE emission so the UI updates
                            await orchestrator._emit_progress(progress_id, {
                                "stage": f"Chunking documents ({i}/{len(documents)})..."
                            })
                    except Exception as e:
                        logger.debug(f"Could not update progress stats: {e}")
                
                # Yield control
                await asyncio.sleep(0)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        
        # Log statistics
        stats = self.get_statistics(all_chunks)
        logger.info(f"Chunk statistics: {stats}")
        
        return all_chunks
    
    def get_statistics(self, chunks: List[str]) -> Dict[str, Any]:
        if not chunks:
            return {
                "num_chunks": 0,
                "avg_length": 0,
                "min_length": 0,
                "max_length": 0
            }
        
        lengths = [len(chunk) for chunk in chunks]
        
        return {
            "num_chunks": len(chunks),
            "avg_length": round(sum(lengths) / len(lengths), 2),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "total_characters": sum(lengths)
        }
    
    def validate_chunks(self, chunks: List[str], min_length: int = 10) -> List[str]:
        valid_chunks = []
        
        for chunk in chunks:
            # Remove whitespace-only chunks
            if not chunk.strip():
                continue
            
            # Remove very short chunks
            if len(chunk.strip()) < min_length:
                logger.debug(f"Skipping short chunk: {len(chunk)} chars")
                continue
            
            valid_chunks.append(chunk.strip())
        
        logger.info(f"Validated {len(valid_chunks)}/{len(chunks)} chunks")
        return valid_chunks