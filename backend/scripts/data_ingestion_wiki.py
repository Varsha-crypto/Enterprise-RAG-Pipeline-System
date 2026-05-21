"""
Data Ingestion - Demo Version with Wikipedia Data

Uses real Wikipedia articles from HuggingFace datasets for meaningful semantic search.
Each article is chunked and embedded with all models.
"""

import os
import sys
import psycopg2
import logging
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import numpy as np

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


def load_wikipedia_data(num_chunks=200):
    """
    Load diverse real content from HuggingFace datasets, using WikiText-103.
    """
    try:
        from datasets import load_dataset
        
        logger.info(f"Loading WikiText-103 dataset (cleaned Wikipedia)...")
        
        # WikiText-103: Pre-processed Wikipedia text
        dataset = load_dataset(
            "Salesforce/wikitext",
            "wikitext-103-v1",
            split="train"
        )
        
        chunks = []
        seen_chunks = set()  # Avoid duplicates
        
        # Keywords to filter for diverse topics
        topic_keywords = {
            'climate', 'weather', 'environment', 'temperature', 'pollution',
            'india', 'delhi', 'mumbai', 'bengaluru', 'chennai', 'kolkata', 'hyderabad', 'london', 'paris', 'new york', 'china', 'japan',
            'science', 'physics', 'biology', 'chemistry', 'astronomy', 'mathematics',
            'technology', 'computer', 'internet', 'software', 'artificial', 'digital',
            'history', 'war', 'century', 'revolution', 'empire', 'ancient',
            'health', 'medical', 'disease', 'hospital', 'medicine', 'treatment',
            'economy', 'business', 'trade', 'finance', 'market', 'industry',
            'sport', 'football', 'cricket', 'olympic', 'basketball', 'tennis',
            'music', 'art', 'literature', 'culture', 'philosophy', 'religion',
            'education', 'university', 'research', 'study', 'learning'
        }
        
        for article in dataset:
            text = str(article['text']) #type: ignore
            text_lower = text.lower()
            
            # Skip empty or very short texts
            if len(text) < 200:
                continue
            
            # Check if text contains any topic keywords
            has_topic = any(keyword in text_lower for keyword in topic_keywords)
            
            if has_topic:
                # Split into chunks of ~500 characters with 100 char overlap
                words = text.split()
                current_chunk_words = []
                current_length = 0
                
                for word in words:
                    current_chunk_words.append(word)
                    current_length += len(word) + 1  # +1 for space
                    
                    # Create chunk when we reach ~500 chars
                    if current_length >= 500:
                        chunk_text = ' '.join(current_chunk_words)
                        
                        # Only add if unique and substantial
                        if chunk_text not in seen_chunks and len(chunk_text) > 150:
                            chunks.append(chunk_text)
                            seen_chunks.add(chunk_text)
                        
                        # Keep last 20 words for overlap
                        current_chunk_words = current_chunk_words[-20:]
                        current_length = sum(len(w) + 1 for w in current_chunk_words)
                        
                        if len(chunks) >= num_chunks:
                            break
            
            if len(chunks) >= num_chunks:
                break
        
        logger.info(f"Generated {len(chunks)} unique chunks from WikiText")
        logger.info(f"Sample chunk: {chunks[0][:100]}...")  # Log first chunk to verify
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error loading WikiText: {e}")
        raise Exception(f"Failed to load dataset: {e}")  


def generate_diverse_content(num_chunks=200):
    """Generate diverse sample content across multiple domains."""
    
    topics = {
        'climate': [
            "Climate change refers to long-term shifts in global temperatures and weather patterns. Scientists agree that human activities, particularly burning fossil fuels, are the main driver of recent climate change.",
            "Global warming has led to rising sea levels, melting ice caps, and more frequent extreme weather events. Countries worldwide are working to reduce greenhouse gas emissions.",
            "The Paris Agreement aims to limit global temperature rise to well below 2 degrees Celsius. Renewable energy sources like solar and wind are becoming increasingly important."
        ],
        'cities': [
            "New Delhi is the capital of India and part of the larger Delhi metropolitan area. It serves as the center of Indian government and is home to numerous historical monuments.",
            "London is the capital of the United Kingdom and a major global financial center. The city has a rich history spanning over two millennia.",
            "Paris, known as the City of Light, is famous for its art, fashion, and culture. The Eiffel Tower is one of the most recognizable landmarks in the world."
        ],
        'technology': [
            "Artificial intelligence is transforming industries from healthcare to finance. Machine learning algorithms can now recognize patterns and make predictions with remarkable accuracy.",
            "The internet has revolutionized communication and information access. Billions of people worldwide are connected through digital networks.",
            "Quantum computing promises to solve complex problems that are intractable for classical computers. Major tech companies are investing heavily in this emerging field."
        ],
        'science': [
            "DNA carries genetic information in living organisms. The discovery of its double helix structure was a breakthrough in molecular biology.",
            "The theory of evolution explains how species change over time through natural selection. Charles Darwin developed this foundational concept in biology.",
            "Black holes are regions of spacetime with gravity so strong that nothing can escape. They form when massive stars collapse at the end of their lives."
        ],
        'health': [
            "Vaccines have saved millions of lives by preventing infectious diseases. They work by training the immune system to recognize and fight pathogens.",
            "Regular exercise and a balanced diet are essential for maintaining good health. Physical activity reduces the risk of many chronic diseases.",
            "Mental health is as important as physical health. Stress management and adequate sleep are crucial for overall wellbeing."
        ],
        'history': [
            "World War II was a global conflict that lasted from 1939 to 1945. It involved most of the world's nations and resulted in significant geopolitical changes.",
            "The Industrial Revolution began in Britain in the late 18th century. It marked a shift from agrarian economies to industrial manufacturing.",
            "Ancient civilizations like Egypt, Rome, and China made lasting contributions to human culture. Their achievements in architecture, governance, and science influence us today."
        ],
        'sports': [
            "The Olympic Games bring together athletes from around the world to compete in various sports. The modern Olympics were revived in 1896.",
            "Football, known as soccer in some countries, is the world's most popular sport. The FIFA World Cup is watched by billions of people.",
            "Cricket is especially popular in countries like India, England, and Australia. The sport has a rich history dating back centuries."
        ]
    }
    
    chunks = []
    import random
    
    # Ensure even distribution across topics
    chunks_per_topic = num_chunks // len(topics)
    
    for topic, sentences in topics.items():
        for _ in range(chunks_per_topic):
            chunk = random.choice(sentences)
            chunks.append(chunk)
    
    # Shuffle for variety
    random.shuffle(chunks)
    
    return chunks[:num_chunks]


def create_table(cursor):
    """Create documents table with embedding_model column."""
    try:
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                chunks TEXT NOT NULL,
                vector VECTOR({max(MODEL_DIMENSIONS.values())}),
                embedding_model TEXT NOT NULL,
                source TEXT DEFAULT 'wikipedia'
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


def insert_data_with_models(cursor, chunks):
    """
    Insert Wikipedia chunks with ALL embedding models.
    """
    logger.info(f"Embedding {len(chunks)} chunks with {len(AVAILABLE_EMBEDDING_MODELS)} models")
    logger.info(f"Total rows to insert: {len(chunks) * len(AVAILABLE_EMBEDDING_MODELS)}")
    
    # Load all models
    models = {}
    for model_name in AVAILABLE_EMBEDDING_MODELS:
        logger.info(f"Loading model: {model_name}")
        if "nomic" in model_name.lower():
            models[model_name] = SentenceTransformer(model_name, trust_remote_code=True)
        elif "bge-m3" in model_name.lower():
            try:
                # Try loading with safetensors
                models[model_name] = SentenceTransformer(
                    model_name,
                    revision="refs/pr/130"  # This PR has safetensors
                )
            except:
                # Fallback: skip BGE-M3
                logger.warning(f"Skipping {model_name} - couldn't load safely")
                continue
        else:
            models[model_name] = SentenceTransformer(model_name)
    
    # Insert each chunk with each model
    total_inserted = 0
    for i, chunk in enumerate(chunks, 1):
        for model_name, model in models.items():
            # Generate embedding
            embedding = model.encode([chunk], normalize_embeddings=True)[0]
            
            # Pad or truncate to max dimension
            max_dim = max(MODEL_DIMENSIONS.values())
            current_dim = len(embedding)
            
            if current_dim < max_dim:
                padded = np.zeros(max_dim, dtype=np.float32)
                padded[:current_dim] = embedding
                embedding = padded
            elif current_dim > max_dim:
                embedding = embedding[:max_dim]
            
            # Convert to string for PostgreSQL
            vector_str = '[' + ','.join(map(str, embedding)) + ']'
            
            # Insert
            cursor.execute("""
                INSERT INTO documents (chunks, vector, embedding_model, source)
                VALUES (%s, %s, %s, %s)
            """, (chunk, vector_str, model_name, 'wikipedia'))
            
            total_inserted += 1
        
        if i % 50 == 0:
            logger.info(f"Processed {i}/{len(chunks)} chunks ({total_inserted} total rows)")
    
    logger.info(f"Inserted {total_inserted} rows total")


def main():
    """Main ingestion function."""
    try:
        logger.info("Starting data ingestion with Wikipedia articles for demo")
        
        # Load Wikipedia data
        chunks = load_wikipedia_data(num_chunks=200)  
        
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
        insert_data_with_models(cursor, chunks)
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
        logger.info("Next step: Run backend/scripts/faiss_indexer.py to build indexes")
        
    except Exception as e:
        logger.error(f"Error in data ingestion: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()