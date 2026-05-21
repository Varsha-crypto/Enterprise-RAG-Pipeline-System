"""
FAISS Indexer - Demo Version

Builds multiple FAISS index types (HNSW, Flat, IVF) for each embedding model.
Creates separate index files named: documents_{model}_{type}.index
Saves FAISS index files and ID mappings to disk.
"""

import os
import sys
import psycopg2
import faiss
import numpy as np
import pickle
import ast
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.config import (
    DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME,
    INDEX_DIR, HNSW_M,
    AVAILABLE_EMBEDDING_MODELS, AVAILABLE_INDEX_TYPES,
    MODEL_DIMENSIONS, MODEL_SHORT_NAMES
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_vector(vector_value, target_dim):
    """
    Parse a vector value from PostgreSQL and extract first target_dim dimensions.
    """
    if isinstance(vector_value, str):
        vec = np.array(ast.literal_eval(vector_value))
    elif isinstance(vector_value, list):
        vec = np.array(vector_value)
    else:
        vec = np.array(vector_value)
    
    # Extract only the dimensions this model actually uses
    return vec[:target_dim].astype(np.float32)


def fetch_vectors_by_model(cursor, embedding_model):
    """
    Fetch all vectors for a specific embedding model.
    """
    logger.info(f"Fetching vectors for model: {embedding_model}")
    
    cursor.execute("""
        SELECT id, vector
        FROM documents
        WHERE embedding_model = %s
        ORDER BY id
    """, (embedding_model,))
    
    rows = cursor.fetchall()
    
    if not rows:
        logger.warning(f"No data found for model: {embedding_model}")
        return None, None
    
    logger.info(f"Fetched {len(rows)} vectors")
    
    # Get the actual dimension for this model
    target_dim = MODEL_DIMENSIONS[embedding_model]
    
    # Extract IDs and vectors
    ids = [row[0] for row in rows]
    vectors = [parse_vector(row[1], target_dim) for row in rows]
    vectors_array = np.array(vectors, dtype=np.float32)
    
    logger.info(f"Vector array shape: {vectors_array.shape}")
    
    return ids, vectors_array


def create_index(vectors_array, dimension, index_type):
    """
    Create a FAISS index of specified type.
    """
    logger.info(f"Building {index_type.upper()} index (dimension={dimension})")
    
    if index_type == "hnsw":
        # HNSW index
        index = faiss.IndexHNSWFlat(dimension, HNSW_M)
        logger.info(f"  HNSW parameters: M={HNSW_M}")
    
    elif index_type == "flat":
        # Flat index
        index = faiss.IndexFlatL2(dimension)
        logger.info("  Using exact L2 search")
    
    elif index_type == "ivf":
        # Inverted file (IVF) index 
        n_vectors = len(vectors_array)
        nlist = min(100, max(10, n_vectors // 10))  # Number of clusters
        
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        
        logger.info(f"  IVF parameters: nlist={nlist}")
        logger.info("  Training IVF index...")
        index.train(vectors_array) # type: ignore
    
    else:
        raise ValueError(f"Unknown index type: {index_type}")
    
    # Add vectors
    logger.info("  Adding vectors to index...")
    index.add(vectors_array) # type: ignore
    logger.info(f" Index built: {index.ntotal} vectors indexed")
    
    return index


def save_index(index, ids, model_name, index_type, index_dir):
    """
    Save FAISS index and ID mapping to disk.
    """
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # Use short model name for filename
    model_short = MODEL_SHORT_NAMES.get(model_name, model_name.split('/')[-1])
    
    # Filenames: documents_{model}_{type}.index
    index_filename = f"documents_{model_short}_{index_type}.index"
    ids_filename = f"documents_{model_short}_{index_type}_ids.pkl"
    
    index_path = index_dir / index_filename
    ids_path = index_dir / ids_filename
    
    # Save FAISS index
    faiss.write_index(index, str(index_path))
    logger.info(f"  Saved index: {index_path}")
    
    # Save ID mapping
    with open(ids_path, 'wb') as f:
        pickle.dump(ids, f)
    logger.info(f"  Saved IDs: {ids_path}")


def main():
    """Main indexing function."""
    try:
        logger.info("Starting FAISS indexing (DEMO MODE)")
        logger.info(f"Models: {len(AVAILABLE_EMBEDDING_MODELS)}")
        logger.info(f"Index types: {len(AVAILABLE_INDEX_TYPES)}")
        logger.info(f"Total indexes to build: {len(AVAILABLE_EMBEDDING_MODELS) * len(AVAILABLE_INDEX_TYPES)}")
        
        # Connect to database
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            dbname=DB_NAME
        )
        cursor = conn.cursor()
        logger.info("Connected to PostgreSQL database")
        
        total_built = 0
        
        # Build indexes for each model × each type
        for model_name in AVAILABLE_EMBEDDING_MODELS:
            logger.info("")
            logger.info(f"Processing model: {model_name}")
            
            # Fetch vectors for this model
            ids, vectors_array = fetch_vectors_by_model(cursor, model_name)
            
            if vectors_array is None:
                logger.warning(f"Skipping {model_name} - no data found")
                continue
            
            dimension = MODEL_DIMENSIONS[model_name]
            
            # Build each index type
            for index_type in AVAILABLE_INDEX_TYPES:
                logger.info("")
                logger.info(f"Building {index_type.upper()} index...")
                
                try:
                    # Create index
                    index = create_index(vectors_array, dimension, index_type)
                    
                    # Save to disk
                    save_index(index, ids, model_name, index_type, INDEX_DIR)
                    
                    total_built += 1
                    logger.info(f"✓ Successfully built {model_name.split('/')[-1]}_{index_type}")
                    
                except Exception as e:
                    logger.error(f"✗ Failed to build {model_name}_{index_type}: {e}")
        
        cursor.close()
        conn.close()
        
        logger.info("")
        logger.info(f"FAISS indexing completed")
        logger.info(f"  Total indexes built: {total_built}")
        logger.info(f"  Index directory: {Path(INDEX_DIR).absolute()}")
        logger.info("")
        logger.info("Next step: Start backend with 'python backend/app/main.py'")
        
    except Exception as e:
        logger.error(f"Error in FAISS indexing: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()