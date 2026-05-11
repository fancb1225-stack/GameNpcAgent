-- PostgreSQL pgvector migration for CoffeeNpcAgent long-term memories
-- Run this inside the coffee_npc_agent database.

CREATE EXTENSION IF NOT EXISTS vector;

-- If long_term_memories already exists, add vector column.
-- Default dimension here is 384. Keep it consistent with EmbeddingService.EMBEDDING_DIM.
ALTER TABLE long_term_memories
ADD COLUMN IF NOT EXISTS embedding vector(384);

-- Optional metadata column if your old table does not have it.
ALTER TABLE long_term_memories
ADD COLUMN IF NOT EXISTS embedding_model VARCHAR(128);

-- HNSW index for cosine search. Requires pgvector extension.
CREATE INDEX IF NOT EXISTS idx_long_term_memories_embedding_hnsw
ON long_term_memories
USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_long_term_memories_player_npc_type
ON long_term_memories (player_id, npc_id, memory_type);
