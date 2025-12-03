-- Initialize cognigraph database with pgvector extension and tables

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Memory vectors table
CREATE TABLE IF NOT EXISTS memories (
    id BIGSERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    embedding VECTOR(1536),  -- OpenAI text-embedding-3-small dimension
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create HNSW index for fast similarity search (Postgres 16+)
CREATE INDEX IF NOT EXISTS memories_embedding_idx 
ON memories 
USING hnsw (embedding vector_cosine_ops);

-- Alternative: IVFFlat index (if HNSW not available)
-- CREATE INDEX memories_embedding_idx ON memories USING ivfflat (embedding vector_cosine_ops);

-- Graph edges table
CREATE TABLE IF NOT EXISTS graph_edges (
    id BIGSERIAL PRIMARY KEY,
    source_id BIGINT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_id BIGINT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    weight FLOAT NOT NULL DEFAULT 0.0,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(source_id, target_id)
);

-- Index for efficient edge queries
CREATE INDEX IF NOT EXISTS graph_edges_source_idx ON graph_edges(source_id);
CREATE INDEX IF NOT EXISTS graph_edges_target_idx ON graph_edges(target_id);

-- Function to search for similar memories
CREATE OR REPLACE FUNCTION match_memories(
    query_embedding VECTOR(1536),
    match_threshold FLOAT DEFAULT 0.7,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    id BIGINT,
    text TEXT,
    similarity FLOAT,
    metadata JSONB
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        memories.id,
        memories.text,
        1 - (memories.embedding <=> query_embedding) AS similarity,
        memories.metadata
    FROM memories
    WHERE 1 - (memories.embedding <=> query_embedding) > match_threshold
    ORDER BY memories.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers to auto-update updated_at
CREATE TRIGGER update_memories_updated_at
    BEFORE UPDATE ON memories
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_graph_edges_updated_at
    BEFORE UPDATE ON graph_edges
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions (for local dev)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO cognigraph;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO cognigraph;

-- Print success message
DO $$
BEGIN
    RAISE NOTICE 'cognigraph database initialized successfully!';
    RAISE NOTICE 'Tables created: memories, graph_edges';
    RAISE NOTICE 'pgvector extension enabled with HNSW indexing';
END $$;
