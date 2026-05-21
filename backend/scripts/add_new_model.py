"""
Add a new embedding model to existing data without re-ingesting everything.
"""

import os
import sys
import psycopg2
import logging
from sentence_transformers import SentenceTransformer
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.config import (
    DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME,
    MODEL_DIMENSIONS
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def add_new_model_embeddings(new_model_name):
    try:
        logger.info(f"Adding embeddings for new model: {new_model_name}")
        
        # Connect to database
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, user=DB_USER,
            password=DB_PASSWORD, dbname=DB_NAME
        )
        cursor = conn.cursor()
        
        # Get unique chunks (only need text, not existing vectors)
        cursor.execute("""
            SELECT DISTINCT chunks 
            FROM documents 
            ORDER BY chunks
        """)
        
        chunks = [row[0] for row in cursor.fetchall()]
        logger.info(f"Found {len(chunks)} unique chunks to embed")
        
        # Load new model
        logger.info(f"Loading model: {new_model_name}")
        model = SentenceTransformer(new_model_name)
        
        # Get target dimension
        target_dim = MODEL_DIMENSIONS[new_model_name]
        max_dim = max(MODEL_DIMENSIONS.values())
        
        # Embed and insert
        logger.info("Generating embeddings...")
        for i, chunk in enumerate(chunks, 1):
            # Generate embedding
            embedding = model.encode([chunk], normalize_embeddings=True)[0]
            
            # Pad to max dimension
            if len(embedding) < max_dim:
                padded = np.zeros(max_dim, dtype=np.float32)
                padded[:len(embedding)] = embedding
                embedding = padded
            
            # Convert to string
            vector_str = '[' + ','.join(map(str, embedding)) + ']'
            
            # Insert
            cursor.execute("""
                INSERT INTO documents (chunks, vector, embedding_model, source)
                VALUES (%s, %s, %s, %s)
            """, (chunk, vector_str, new_model_name, 'wikipedia'))
            
            if i % 50 == 0:
                logger.info(f"Processed {i}/{len(chunks)} chunks")
                conn.commit()
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Successfully added {len(chunks)} embeddings for {new_model_name}")
        logger.info("Next step: Run backend/scripts/faiss_indexer.py to build indexes for new model")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Currently want E5-Large-v2 embeddings
    add_new_model_embeddings("intfloat/e5-large-v2")