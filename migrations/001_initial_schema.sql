-- RAG Brain Initial Schema
-- Enable pgvector extension for vector similarity search

CREATE EXTENSION IF NOT EXISTS vector;

-- Enum types
CREATE TYPE memory_category AS ENUM (
    'decision',
    'bug_fix',
    'pattern',
    'outcome',
    'insight',
    'code_snippet',
    'documentation',
    'other'
);

CREATE TYPE memory_tier AS ENUM (
    'core',
    'active',
    'archive',
    'quarantine'
);

CREATE TYPE relationship_type AS ENUM (
    'supports',
    'contradicts',
    'extends',
    'caused_by',
    'related'
);

CREATE TYPE event_type AS ENUM (
    'retrieval',
    'helpful',
    'not_helpful',
    'stale'
);

-- Main memories table
CREATE TABLE memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    embedding vector(768),
    category memory_category NOT NULL DEFAULT 'other',
    tags TEXT[] DEFAULT '{}',
    source TEXT,
    project TEXT,
    extra_data JSONB DEFAULT '{}',
    predicted_quality FLOAT DEFAULT 0.5,
    usefulness_score FLOAT DEFAULT 0.5,
    access_count INTEGER DEFAULT 0,
    tier memory_tier NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Memory relationships
CREATE TABLE memory_links (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    relationship relationship_type NOT NULL,
    strength FLOAT DEFAULT 0.5,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(source_id, target_id, relationship)
);

-- Event log for training signals
CREATE TABLE memory_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    event event_type NOT NULL,
    context JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Training state tracker
CREATE TABLE training_state (
    id INTEGER PRIMARY KEY DEFAULT 1,
    memories_since_last_train INTEGER DEFAULT 0,
    current_model_version INTEGER DEFAULT 0,
    last_train_at TIMESTAMPTZ,
    CONSTRAINT single_row CHECK (id = 1)
);

-- Initialize training state
INSERT INTO training_state (id, memories_since_last_train, current_model_version)
VALUES (1, 0, 0)
ON CONFLICT (id) DO NOTHING;

-- Model storage
CREATE TABLE models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version INTEGER NOT NULL UNIQUE,
    metrics JSONB NOT NULL DEFAULT '{}',
    model_blob BYTEA,
    feature_importance JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Concepts (emerged clusters)
CREATE TABLE concepts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    centroid vector(768),
    memory_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Concept memberships
CREATE TABLE concept_members (
    concept_id UUID NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    similarity FLOAT NOT NULL,
    PRIMARY KEY (concept_id, memory_id)
);

-- Indexes for vector similarity search
CREATE INDEX idx_memories_embedding ON memories
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Indexes for filtering
CREATE INDEX idx_memories_tier ON memories(tier);
CREATE INDEX idx_memories_category ON memories(category);
CREATE INDEX idx_memories_project ON memories(project);
CREATE INDEX idx_memories_quality ON memories(predicted_quality);
CREATE INDEX idx_memories_usefulness ON memories(usefulness_score);
CREATE INDEX idx_memories_created ON memories(created_at);
CREATE INDEX idx_memories_tags ON memories USING GIN(tags);

-- Indexes for memory links
CREATE INDEX idx_memory_links_source ON memory_links(source_id);
CREATE INDEX idx_memory_links_target ON memory_links(target_id);

-- Indexes for events
CREATE INDEX idx_memory_events_memory ON memory_events(memory_id);
CREATE INDEX idx_memory_events_type ON memory_events(event);
CREATE INDEX idx_memory_events_created ON memory_events(created_at);

-- Indexes for concepts
CREATE INDEX idx_concepts_centroid ON concepts
USING ivfflat (centroid vector_cosine_ops)
WITH (lists = 50);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at
CREATE TRIGGER memories_updated_at
    BEFORE UPDATE ON memories
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER concepts_updated_at
    BEFORE UPDATE ON concepts
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();
