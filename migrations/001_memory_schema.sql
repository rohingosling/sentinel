-------------------------------------------------------------------------------------------------------------------------
-- Migration: 001_memory_schema
-- Project:   Sentinel
-- Phase:     3 -- Memory System
--
-- Description:
--
--   Tier 2 (episodic) and Tier 3 (semantic) storage, per architecture 3.2.4.
--
--   Tier 1 is deliberately absent: working memory is a diskcache store on its own SQLite file, not a table here, so it
--   can be evicted and thrown away without touching durable state.
--
--   The identity_snapshots and event_log tables from the same architecture section are NOT created here. They belong to
--   Phase 8 and Phase 5 and arrive in their own numbered migrations -- a migration that creates tables no code yet reads
--   cannot be verified by the phase that ships it.
--
-- Note:
--
--   The FLOAT[384] width is Decision 2 and is fixed in the schema on purpose. It is checked at startup against both the
--   configured database.vector_dimensions and the embedding model's actual output width; a disagreement aborts rather
--   than storing vectors that would search silently badly.
-------------------------------------------------------------------------------------------------------------------------

-- Tier 2: episodic memory. Conversation summaries, interaction events, and heartbeat outcomes.
--
-- timestamp is the event's own ISO 8601 instant and is what retrieval orders by; created_at is when the row was
-- written. They differ whenever an episode is backfilled, and conflating them would silently re-date imported history.

CREATE TABLE episodes
(
    id          TEXT PRIMARY KEY,                   -- UUID.
    timestamp   TEXT NOT NULL,                      -- ISO 8601, timezone-aware.
    category    TEXT NOT NULL,                      -- 'conversation' | 'heartbeat' | 'tool' | ...
    summary     TEXT NOT NULL,
    tags        TEXT,                               -- JSON array, queried through json_each().
    session_id  TEXT,
    created_at  TEXT DEFAULT ( datetime ( 'now' ) )
);

CREATE INDEX idx_episodes_ts  ON episodes ( timestamp DESC );
CREATE INDEX idx_episodes_cat ON episodes ( category, timestamp DESC );

-- Tier 3: semantic memory. One sqlite-vec virtual table holding the vector and every field needed to render a hit,
-- so a search answers from one statement instead of a KNN followed by a join.

CREATE VIRTUAL TABLE semantic_vectors USING vec0
(
    id               TEXT PRIMARY KEY,
    embedding        FLOAT[384] distance_metric=cosine,   -- all-MiniLM-L6-v2 width (Decision 2).
                                                          -- MUST equal config vector_dimensions.
    +content         TEXT,                                -- Auxiliary: the original text.
    +category        TEXT,                                -- Auxiliary: memory type.
    +timestamp       TEXT,                                -- Auxiliary: creation time, ISO 8601.
    +embedding_model TEXT                                 -- Auxiliary: model identity, e.g. 'all-MiniLM-L6-v2'.
                                                          -- Lets startup detect a model change and demand a
                                                          -- re-embed instead of silently degrading search.
);
