"""
Database Schema Update

Add batch_number column to documents table for incremental processing.
Run this SQL migration on your database.
"""

-- Add batch_number column (NULL for existing data)
ALTER TABLE documents 
ADD COLUMN IF NOT EXISTS batch_number INTEGER DEFAULT NULL;

-- Create index for faster batch filtering
CREATE INDEX IF NOT EXISTS idx_batch_number ON documents(batch_number);

-- Optional: Add index for combined filtering
CREATE INDEX IF NOT EXISTS idx_model_batch ON documents(embedding_model, batch_number);

-- Verify schema
\d documents

/*
Expected schema after migration:

Table "public.documents"
Column          | Type    | Nullable | Default
----------------+---------+----------+---------
id              | serial  | not null | nextval('documents_id_seq')
chunks          | text    | not null | 
vector          | vector  | null     |
embedding_model | text    | not null |
source          | text    | null     | 'pipeline'
batch_number    | integer | null     |

Indexes:
- documents_pkey PRIMARY KEY (id)
- idx_embedding_model (embedding_model)
- idx_batch_number (batch_number)
- idx_model_batch (embedding_model, batch_number)
*/