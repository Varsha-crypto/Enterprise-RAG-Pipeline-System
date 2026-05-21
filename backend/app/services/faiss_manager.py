"""
FAISS Manager Class, to handle loading, reloading, and searching FAISS indexes.
"""
import os
import faiss
faiss.omp_set_num_threads(1)

import numpy as np
import pickle
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class FAISSManager:
    
    def __init__(self, index_dir: str = "indexes"):
        self.index_dir = Path(index_dir)
        self.indexes: Dict[str, faiss.Index] = {}
        self.ids_dict: Dict[str, List[int]] = {}
        
        # If index directory doesn't exist, create it
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"FAISSManager initialized with index directory: {self.index_dir}")
    
    def set_index_directory(self, index_dir: str):
        """Change index directory dynamically."""
        self.index_dir = Path(index_dir)
        logger.info(f"Index directory set to: {index_dir}")
    
    def load_index_from_path(self, index_path: str, ids_path: str, table_name: str):
        """Load a specific index from given paths."""
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found: {index_path}")
        
        index = faiss.read_index(index_path)
        
        with open(ids_path, 'rb') as f:
            ids = pickle.load(f)
        
        self.indexes[table_name] = index
        self.ids_dict[table_name] = ids
        
        logger.info(f"Loaded index: {table_name} from {index_path}")

    def _load_single_index(self, table_name: str) -> bool:
        index_path = self.index_dir / f"{table_name}.index"
        ids_path = self.index_dir / f"{table_name}_ids.pkl"
        
        if not index_path.exists():
            logger.warning(f"Index file not found: {index_path}")
            return False
        
        if not ids_path.exists():
            logger.warning(f"IDs file not found: {ids_path}")
            return False
        
        try:
            index = faiss.read_index(str(index_path))
            
            with open(ids_path, 'rb') as f:
                ids = pickle.load(f)
            
            self.indexes[table_name] = index
            self.ids_dict[table_name] = ids
            
            logger.info(f"Loaded index for table '{table_name}': {index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index for table '{table_name}': {e}")
            return False
    
    def load_all_indexes(self) -> int:
        loaded_count = 0
        
        if not self.index_dir.exists():
            logger.warning(f"Index directory does not exist: {self.index_dir}")
            return loaded_count
        
        for index_file in self.index_dir.glob("*.index"):
            table_name = index_file.stem 
            
            if self._load_single_index(table_name):
                loaded_count += 1
        
        logger.info(f"Loaded {loaded_count} FAISS indexes")
        return loaded_count
    
    def reload_index(self, table_name: str) -> bool:

        logger.info(f"Reloading index for table: {table_name}")
        
        self.indexes.pop(table_name, None)
        self.ids_dict.pop(table_name, None)
        
        return self._load_single_index(table_name)
    
    def reload_all_indexes(self) -> int:
    
        logger.info("Reloading all indexes...")
        
        self.indexes.clear()
        self.ids_dict.clear()
        
        return self.load_all_indexes()
    
    def unload_index(self, table_name: str):
        """Unload an index from memory and clear its references."""
        if table_name in self.indexes:
            del self.indexes[table_name]
        if table_name in self.ids_dict:
            del self.ids_dict[table_name]
        logger.info(f"Unloaded index for table: {table_name}")
    
    def search(
        self, 
        table_name: str, 
        query_vector: np.ndarray, 
        top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        if table_name not in self.indexes:
            raise ValueError(f"Table '{table_name}' not indexed. Available tables: {list(self.indexes.keys())}")
        
        # Ensure query_vector is 2-dimensional and has dtype as float32 (FAISS requirement)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)
        
        # Perform search
        index = self.indexes[table_name]
        distances, indices = index.search(query_vector, top_k) # type: ignore
        
        logger.debug(f"FAISS search on '{table_name}': found {len(indices[0])} results")
        
        return distances, indices
    
    def get_database_ids(self, table_name: str, faiss_indices: np.ndarray) -> List[int]:
        if table_name not in self.ids_dict:
            raise ValueError(f"Table '{table_name}' not found in ID mappings")
        
        # Filter out -1 indices (FAISS returns -1 for missing results)
        valid_indices = [i for i in faiss_indices if i != -1]
        
        db_ids = [self.ids_dict[table_name][i] for i in valid_indices]
        
        return db_ids
    
    def is_table_indexed(self, table_name: str) -> bool:
        return table_name in self.indexes
    
    def get_indexed_tables(self) -> List[str]:
        return list(self.indexes.keys())
    
    def get_index_stats(self, table_name: str) -> Optional[Dict]:
        if table_name not in self.indexes:
            return None
        
        index = self.indexes[table_name]
        
        return {
            "table_name": table_name,
            "num_vectors": index.ntotal,
            "dimension": index.d,
            "index_type": type(index).__name__,
            "is_trained": index.is_trained,
        }
    
    def get_all_stats(self) -> Dict[str, Dict]: 
        return {                                # type: ignore
            table: self.get_index_stats(table)
            for table in self.get_indexed_tables()
        }
    
# Global instance
faiss_manager = FAISSManager()