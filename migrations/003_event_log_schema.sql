-------------------------------------------------------------------------------------------------------------------------
-- Migration: 003_event_log_schema
-- Project:   Sentinel
-- Phase:     5 -- Event Logging
--
-- Description:
--
--   The generic append-only event log of architecture 3.2.10: one row per event, whatever produced it, carrying a
--   category, an event name, a correlation identifier, and two JSON payloads.
--
--   This is the audit trail. Where heartbeat_ticks (migration 002) records the eight specific fields of one tick, this
--   table records everything -- user messages, agent responses, tool calls, security decisions, startup and shutdown --
--   in one shape that can be queried across subsystems. The two coexist: Phase 5 mirrors tick boundaries in here as
--   'heartbeat.start' and 'heartbeat.end'; it does not replace heartbeat_ticks.
--
-- Note:
--
--   The append-only rule is enforced by the storage engine, not by convention. The two triggers below make UPDATE and
--   DELETE fail on this table for every caller -- Sentinel's own code, a future migration that forgets, and a user
--   poking at the file with sqlite3 alike. An audit log that the audited process can rewrite is not an audit log, and
--   an application-layer rule is exactly the kind that survives until the first module that does not know about it.
--
--   The consequence is deliberate and worth stating: rows here are never pruned. logging.retention governs the exported
--   JSON files, which are the rotating copy; the table grows for the life of the installation. At the ~150 bytes a
--   typical row occupies that is a few megabytes a year for an agent in constant use, which is a price worth paying for
--   a trail that cannot be quietly edited.
-------------------------------------------------------------------------------------------------------------------------

-- One row per event, matching the log entry schema of architecture 3.2.10 field for field.
--
-- data and metadata are JSON text rather than columns because their shape is per-event: a tool invocation carries
-- arguments, a security denial carries a rule, and no fixed column set covers both without being mostly NULL.
--
-- correlation_id is nullable rather than defaulted. An event genuinely outside any turn -- system.startup is the
-- clearest case -- has nothing to correlate with, and inventing an identifier for it would put a value in the index
-- that no second row will ever match.

CREATE TABLE event_log
(
    id             TEXT PRIMARY KEY,        -- UUID.
    timestamp      TEXT NOT NULL,           -- ISO 8601, timezone-aware, as everything persisted is.
    category       TEXT NOT NULL,           -- One of the eight categories of architecture 3.2.10.
    event          TEXT NOT NULL,           -- Dotted event name, always prefixed by its own category.
    source         TEXT,                    -- Component that emitted it.
    correlation_id TEXT,                    -- Ties every event of one turn together. NULL outside a turn.
    data           TEXT,                    -- JSON object. Event-specific payload.
    metadata       TEXT                     -- JSON object: agent_version, identity_version, session_id.
);

-- Reads are "the recent events", "this turn's events", and "this category's events", in that order of frequency.
--
-- The timestamp index is DESC because every listing is newest-first, and ISO 8601 sorts lexicographically in the same
-- order it sorts chronologically -- which is the whole reason instants are stored as text (see timestamps.py).

CREATE INDEX idx_events_ts       ON event_log ( timestamp DESC );
CREATE INDEX idx_events_corr     ON event_log ( correlation_id );
CREATE INDEX idx_events_category ON event_log ( category, timestamp DESC );

-- Append-only, enforced here rather than in the application layer.
--
-- RAISE(ABORT) rolls back the statement and surfaces the message, so a caller that tries either sees why rather than
-- silently succeeding against a table it believes it changed.

CREATE TRIGGER event_log_forbid_update
BEFORE UPDATE ON event_log
BEGIN
    SELECT RAISE ( ABORT, 'event_log is append-only: UPDATE is not permitted.' );
END;

CREATE TRIGGER event_log_forbid_delete
BEFORE DELETE ON event_log
BEGIN
    SELECT RAISE ( ABORT, 'event_log is append-only: DELETE is not permitted.' );
END;
