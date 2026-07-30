-------------------------------------------------------------------------------------------------------------------------
-- Migration: 002_heartbeat_schema
-- Project:   Sentinel
-- Phase:     4 -- Heartbeat Loop
--
-- Description:
--
--   The three tables the autonomous tick reads and writes, per architecture 3.2.2: the pending-task queue, the cron
--   schedule, and the tick record itself.
--
--   All three are durable rather than in-memory because the heartbeat must survive a restart. A queue held in process
--   memory loses every pending task when the application closes, and a cron schedule held in APScheduler's own store
--   would have to be rebuilt from somewhere on each launch -- so this is that somewhere.
--
-- Note:
--
--   heartbeat_ticks is NOT the event log. The generic append-only event_log of architecture 3.2.10 belongs to Phase 5
--   and arrives as migration 003. This table carries exactly the eight fields of the heartbeat event schema in 3.2.2,
--   which are too specific to live as JSON in a generic envelope and are what the tick's own criteria are checked
--   against. Phase 5 mirrors tick boundaries into event_log as 'heartbeat.start' / 'heartbeat.end'; it does not
--   replace this table.
-------------------------------------------------------------------------------------------------------------------------

-- The pending-task queue. Ordered by priority first and arrival second, so an urgent task added late is still picked
-- up before a routine one that has been waiting.
--
-- scheduled_for defers a task without involving cron: a task with a future instant is simply not yet pending. NULL
-- means "as soon as possible", which is the common case and therefore the default.

CREATE TABLE tasks
(
    id            TEXT PRIMARY KEY,                   -- UUID.
    description   TEXT    NOT NULL,                   -- What the agent is being asked to do.
    status        TEXT    NOT NULL DEFAULT 'pending', -- 'pending' | 'running' | 'done' | 'failed' | 'cancelled'.
    priority      INTEGER NOT NULL DEFAULT 0,         -- Higher runs first.
    source        TEXT    NOT NULL DEFAULT 'user',    -- 'user' | 'schedule' | 'signal' | 'agent'.
    scheduled_for TEXT,                               -- ISO 8601. NULL means eligible immediately.
    created_at    TEXT    NOT NULL DEFAULT ( datetime ( 'now' ) ),
    started_at    TEXT,
    completed_at  TEXT,
    attempts      INTEGER NOT NULL DEFAULT 0,         -- Incremented on each claim, so a poison task is visible.
    result        TEXT,
    error         TEXT
);

CREATE INDEX idx_tasks_queue     ON tasks ( status, priority DESC, created_at );
CREATE INDEX idx_tasks_scheduled ON tasks ( scheduled_for );

-- The cron schedule. Jobs are due-checked inside the tick rather than registered as independent APScheduler jobs, so
-- that every autonomous action -- queued, scheduled, or signalled -- passes through the one max_autonomous_actions
-- budget. A job firing on its own timer would bypass that ceiling entirely.
--
-- next_run_at is materialised rather than recomputed on every read: it is what the due query indexes on, and computing
-- it needs the cron parser, which SQL does not have.

CREATE TABLE scheduled_jobs
(
    id          TEXT PRIMARY KEY,                   -- UUID.
    name        TEXT    NOT NULL UNIQUE,            -- Stable handle, so re-registering updates rather than duplicates.
    cron        TEXT    NOT NULL,                   -- Five-field crontab expression.
    description TEXT    NOT NULL DEFAULT '',        -- What the agent should do when it fires.
    enabled     INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT    NOT NULL DEFAULT ( datetime ( 'now' ) ),
    last_run_at TEXT,                               -- ISO 8601, timezone-aware.
    next_run_at TEXT                                -- ISO 8601. NULL means the expression yields no further fire time.
);

CREATE INDEX idx_scheduled_jobs_due ON scheduled_jobs ( enabled, next_run_at );

-- One row per tick, carrying the heartbeat event schema of architecture 3.2.2 verbatim.
--
-- The row is written at tick start with ended_at NULL and updated at tick end. That ordering is the point: a tick that
-- crashes mid-flight leaves a row with a start and no end, which is exactly the evidence needed to find it. A row
-- written only on success would make a crashed tick indistinguishable from one that never ran.

CREATE TABLE heartbeat_ticks
(
    tick_id         TEXT PRIMARY KEY,               -- UUID.
    started_at      TEXT    NOT NULL,               -- ISO 8601, timezone-aware.
    ended_at        TEXT,                           -- NULL while the tick is in flight, or if it crashed.
    tasks_evaluated INTEGER NOT NULL DEFAULT 0,     -- Candidates inspected, before the action budget applied.
    actions_taken   INTEGER NOT NULL DEFAULT 0,     -- Candidates actually processed. Never exceeds the budget.
    actions         TEXT    NOT NULL DEFAULT '[]',  -- JSON array of action summaries.
    errors          TEXT    NOT NULL DEFAULT '[]',  -- JSON array of error descriptions.
    next_tick_at    TEXT                            -- ISO 8601. NULL when the loop is stopping or paused.
);

CREATE INDEX idx_heartbeat_ticks_started ON heartbeat_ticks ( started_at DESC );
