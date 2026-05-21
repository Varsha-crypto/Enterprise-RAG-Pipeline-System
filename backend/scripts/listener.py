"""
FAISS Listener - Monitors PostgreSQL for vector tables and triggers FAISS indexing. Runs continuously as a background process.
"""
import time
import subprocess
import psycopg2
import os
import sys
import logging
from pathlib import Path
from typing import Set, List, Tuple
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Database configuration
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5433')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'postgres')
DB_NAME = os.getenv('DB_NAME', 'appdb')

# Listener configuration
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', '60'))  
INDEX_DIR = Path(os.getenv('INDEX_DIR', 'indexes'))


def get_vector_tables(cursor) -> List[Tuple[str, str]]:
    cursor.execute("""
        SELECT table_schema, table_name
        FROM information_schema.columns
        WHERE udt_name = 'vector'
        GROUP BY table_schema, table_name
        ORDER BY table_schema, table_name
    """)
    return cursor.fetchall()


def is_table_indexed(table_name: str) -> bool:
    index_file = INDEX_DIR / f"{table_name}.index"
    ids_file = INDEX_DIR / f"{table_name}_ids.pkl"
    
    return index_file.exists() and ids_file.exists()


def run_indexer() -> bool:
    logger.info("Triggering FAISS indexer...")
    
    try:
        script_dir = Path(__file__).parent
        indexer_path = script_dir / "faiss_indexer.py"
        
        result = subprocess.run(
            [sys.executable, str(indexer_path)],
            capture_output=True,
            text=True,
            timeout=300 
        )
        
        # Output logs
        if result.stdout:
            logger.info(f"Indexer output:\n{result.stdout}")
        
        if result.returncode == 0:
            logger.info("Indexing completed successfully")
            return True
        else:
            logger.error(f"Indexing failed with return code {result.returncode}")
            if result.stderr:
                logger.error(f"Indexer errors:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Indexer timed out after 5 minutes")
        return False
    except Exception as e:
        logger.error(f"Failed to run indexer: {e}")
        return False


def check_and_index(cursor) -> Tuple[int, int]:
    vector_tables = get_vector_tables(cursor)
    
    if not vector_tables:
        logger.debug("No vector tables found in database")
        return 0, 0
    
    unindexed_tables = []
    indexed_tables = []
    
    for schema, table in vector_tables:
        full_name = f"{schema}.{table}"
        
        if is_table_indexed(table):
            indexed_tables.append(full_name)
            logger.debug(f"Table '{full_name}' is already indexed")
        else:
            unindexed_tables.append(full_name)
            logger.info(f"Table '{full_name}' needs indexing")
    
    logger.info(f"Vector tables status: {len(indexed_tables)} indexed, {len(unindexed_tables)} need indexing")
    
    if unindexed_tables:
        logger.info(f"Indexing {len(unindexed_tables)} table(s): {', '.join(unindexed_tables)}")
        
        if run_indexer():
            return len(vector_tables), len(unindexed_tables)
        else:
            logger.error("Indexing failed, will retry on next check")
            return len(vector_tables), 0
    
    return len(vector_tables), 0


def connect_to_db():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            dbname=DB_NAME
        )
        logger.debug("Connected to PostgreSQL database")
        return conn
    except psycopg2.Error as e:
        logger.error(f"Failed to connect to database: {e}")
        return None


def main():
    
    logger.info("-" * 35)
    logger.info("FAISS Listener started")
    logger.info(f"Database: {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    logger.info(f"Check interval: {CHECK_INTERVAL} seconds")
    logger.info(f"Index directory: {INDEX_DIR}")
    logger.info("-" * 35)
    
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    consecutive_failures = 0
    max_consecutive_failures = 5
    
    while True:
        try:
            conn = connect_to_db()
            
            if conn is None:
                consecutive_failures += 1
                logger.error(f"Database connection failed (attempt {consecutive_failures}/{max_consecutive_failures})")
                
                if consecutive_failures >= max_consecutive_failures:
                    logger.critical("Too many consecutive failures, exiting")
                    sys.exit(1)
                
                time.sleep(CHECK_INTERVAL)
                continue
            
            # Reset failure counter on successful connection
            consecutive_failures = 0
            
            # Check for tables and index if needed
            cursor = conn.cursor()
            tables_found, tables_indexed = check_and_index(cursor)
            cursor.close()
            conn.close()
            
            if tables_indexed > 0:
                logger.info(f"Successfully indexed {tables_indexed} table(s)")
            
            logger.debug(f"Sleeping for {CHECK_INTERVAL} seconds...")
            time.sleep(CHECK_INTERVAL)
                
        except psycopg2.Error as e:
            consecutive_failures += 1
            logger.error(f"Database error: {e}")
            
            if consecutive_failures >= max_consecutive_failures:
                logger.critical("Too many consecutive failures, exiting")
                sys.exit(1)
            
            time.sleep(CHECK_INTERVAL)
            
        except Exception as e:
            logger.error(f"Unexpected error in listener: {e}", exc_info=True)
            time.sleep(CHECK_INTERVAL)
    
    logger.info("Listener shutdown complete")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Listener interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)