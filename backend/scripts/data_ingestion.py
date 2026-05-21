"""
Data Ingestion - Demo Version

Generates dummy data and embeds it with ALL available embedding models.
Each chunk is inserted once per model, creating multiple rows with different embeddings.
"""

import os
import sys
import psycopg2
import logging
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import numpy as np
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.config import (
    DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME,
    AVAILABLE_EMBEDDING_MODELS, MODEL_DIMENSIONS
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dummy text generation
sentence_starts = [
    "The quick brown fox", "In a galaxy far away", "Once upon a time",
    "Scientists have discovered", "The weather today is", "Technology continues to evolve",
    "Humans are capable of", "The future holds", "Learning new skills",
    "Exploring the unknown", "Artificial intelligence enables", "Machine learning algorithms",
    "Data analysis reveals", "Research indicates", "Studies have shown",
    "Experts believe", "Innovation drives", "Systems integrate", "Networks connect"
]

sentence_middles = [
    "jumps over the lazy dog", "a new species exists", "there was a magical kingdom",
    "that the Earth is round", "sunny and warm", "at an unprecedented rate",
    "amazing feats of creativity", "many exciting possibilities", "can be incredibly rewarding",
    "leads to great discoveries", "better decision making", "improved accuracy",
    "hidden patterns", "significant correlations", "important trends",
    "future developments", "breakthrough solutions", "seamless operations", "global collaboration"
]

sentence_ends = [
    "which is quite remarkable.", "changing our perspective forever.",
    "filled with wonder and adventure.", "proving ancient theories wrong.",
    "perfect for outdoor activities.", "shaping the world we live in.",
    "inspiring innovation worldwide.", "waiting to be explored.",
    "opening new doors of opportunity.", "expanding our horizons.",
    "transforming industries.", "creating new possibilities.", "driving progress forward.",
    "enabling breakthrough research.", "fostering global connections."
]


def generate_random_chunk():
    start = random.choice(sentence_starts)
    middle = random.choice(sentence_middles)
    end = random.choice(sentence_ends)
    return f"{start} {middle} {end}"


def create_table(cursor):
    """Create documents table with embedding_model column."""
    try:
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                chunks TEXT NOT NULL,
                vector VECTOR({max(MODEL_DIMENSIONS.values())}),
                embedding_model TEXT NOT NULL
            );
        """)
        logger.info("Table 'documents' created or already exists")
        
        # Create index on embedding_model for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_embedding_model 
            ON documents(embedding_model);
        """)
        logger.info("Index on 'embedding_model' created")
        
    except Exception as e:
        logger.error(f"Error creating table: {e}")
        raise


def insert_data_with_models(cursor, num_chunks=100):
    logger.info(f"Generating {num_chunks} chunks with {len(AVAILABLE_EMBEDDING_MODELS)} models")
    logger.info(f"Total rows to insert: {num_chunks * len(AVAILABLE_EMBEDDING_MODELS)}")

    # Load all models
    models = {}
    for model_name in AVAILABLE_EMBEDDING_MODELS:
        logger.info(f"Loading model: {model_name}")
        models[model_name] = SentenceTransformer(model_name)
    
    # Generate unique chunks
    chunks = [generate_random_chunk() for _ in range(num_chunks)]
    
    # Insert each chunk with each model
    total_inserted = 0
    for i, chunk in enumerate(chunks, 1):
        for model_name, model in models.items():
            # Generate embedding
            embedding = model.encode([chunk], normalize_embeddings=True)[0]
            
            # Pad or truncate to max dimension (for consistent VECTOR type)
            max_dim = max(MODEL_DIMENSIONS.values())
            current_dim = len(embedding)
            
            if current_dim < max_dim:
                # Pad with zeros
                padded = np.zeros(max_dim, dtype=np.float32)
                padded[:current_dim] = embedding
                embedding = padded
            elif current_dim > max_dim:
                # Truncate
                embedding = embedding[:max_dim]
            
            # Convert to string for PostgreSQL
            vector_str = '[' + ','.join(map(str, embedding)) + ']'
            
            # Insert
            cursor.execute("""
                INSERT INTO documents (chunks, vector, embedding_model)
                VALUES (%s, %s, %s)
            """, (chunk, vector_str, model_name))
            
            total_inserted += 1
        
        if i % 10 == 0:
            logger.info(f"Processed {i}/{num_chunks} chunks ({total_inserted} total rows)")
    
    logger.info(f"Inserted {total_inserted} rows total")


def main():
    """Main ingestion function."""
    try:
        logger.info("Starting data ingestion for demo")
        
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
        
        # Create table
        create_table(cursor)
        conn.commit()
        
        # Insert data
        insert_data_with_models(cursor, num_chunks=100)
        conn.commit()
        
        # Verify
        cursor.execute("SELECT COUNT(*), embedding_model FROM documents GROUP BY embedding_model")
        results = cursor.fetchall()
        
        logger.info("Data distribution by model:")
        for count, model in results:
            logger.info(f"  {model}: {count} rows")
        
        cursor.close()
        conn.close()
        
        logger.info("Data ingestion completed successfully")
        logger.info("Next step: Run scripts/faiss_indexer.py to build indexes")
        
    except Exception as e:
        logger.error(f"Error in data ingestion: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()