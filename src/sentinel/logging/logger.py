#-----------------------------------------------------------------------------------------------------------------------
# Module:  logger.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   The structured event logger (architecture 3.2.10).
#
#   One call -- EventLogger.log ( category, event, data, correlation_id ) -- writes an append-only row and fans the same
#   event out to every configured exporter. Reading is the mirror image: query() applies the filters the log API
#   exposes, newest first.
#
#   Two failure modes are handled deliberately differently, and the distinction is the whole design:
#
#     * A BAD EVENT raises. An unknown category or a malformed name is an authoring mistake that a test will catch on
#       the first run, and swallowing it would leave a hole in the audit trail that nothing ever reports.
#     * A FAILED WRITE does not. The database being locked, the disk being full, the connection having been closed
#       during shutdown -- none of these are the caller's fault, and none justify aborting the operation being audited.
#       They are logged through the ordinary logger and log() returns the event it could not store.
#
#   Correlation identifiers propagate through a contextvar rather than through every function signature between the
#   gateway and the tool dispatcher. Threading an identifier through fifteen call sites means fourteen places to forget
#   it, and the fifteenth is the one that logs the security denial. A contextvar follows an asyncio task and its
#   children automatically, which is exactly the shape of a turn. An explicit argument still wins where one is passed,
#   so nothing is trapped by the ambient value.
#
#   The logger holds a connection it does not own. Its lifetime belongs to whoever opened it -- the process manager in
#   the running agent, a fixture in the suite -- for the same reason the task queue and the cron schedule work that way.
#-----------------------------------------------------------------------------------------------------------------------

import contextvars
import logging
import uuid

from collections.abc import Iterator, Sequence
from contextlib      import contextmanager
from datetime        import datetime
from typing          import Any

import aiosqlite

from sentinel.config     import SentinelConfig
from sentinel.database   import DATABASE_FAULTS
from sentinel.errors     import DatabaseError
from sentinel.timestamps import to_iso_timestamp

from sentinel.logging.exporters import EventExporter
from sentinel.logging.schemas   import LogEvent, build_event, row_to_event

logger = logging.getLogger ( __name__ )

# Every column a reader sees, in one place, so the row-to-event mapping cannot drift from the SELECT list. Held to
# being static identifier text by tests/phase5/test_sql_safety.py, which is what keeps the S608 suppression honest.

EVENT_COLUMNS = "id, timestamp, category, event, source, correlation_id, data, metadata"

# Default ceiling on a single query. A caller asking for everything gets the newest page of it rather than the whole
# table, because the table is append-only and grows for the life of the installation.

DEFAULT_QUERY_LIMIT = 100

# The ambient turn identifier. None outside a turn, which is the correct answer for system.startup and friends -- an
# invented identifier there would put a value in the correlation index that no second row will ever match.

CORRELATION_ID: contextvars.ContextVar [ str | None ] = contextvars.ContextVar (
    "sentinel_correlation_id", default = None
)


#-----------------------------------------------------------------------------------------------------------------------
# Correlation
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Function: new_correlation_id
#
# Description:
#
#   Mint a fresh turn identifier.
#
# Arguments:
#
#   None.
#
# Returns:
#
#   A new UUID string.
#
#-----------------------------------------------------------------------------------------------------------------------

def new_correlation_id () -> str:

    # Return data to caller.

    return str ( uuid.uuid4 () )


#-----------------------------------------------------------------------------------------------------------------------
# Function: current_correlation_id
#
# Description:
#
#   Read the turn identifier in force.
#
# Arguments:
#
#   None.
#
# Returns:
#
#   The ambient identifier, or None when the caller is not inside a turn.
#
#-----------------------------------------------------------------------------------------------------------------------

def current_correlation_id () -> str | None:

    # Return data to caller.

    return CORRELATION_ID.get ()


#-----------------------------------------------------------------------------------------------------------------------
# Function: correlation_scope
#
# Description:
#
#   Establish a turn identifier for the duration of a block.
#
#   Everything logged inside the block -- including from any coroutine the block awaits -- carries this identifier
#   unless it passes one of its own. That is what makes "every event of one turn shares a correlation_id" a property of
#   the runtime rather than a discipline each author has to remember.
#
# Arguments:
#
#   correlation_id : Identifier to establish. A fresh one is minted when omitted.
#
# Returns:
#
#   The identifier in force inside the block. The previous value is restored on exit, so nested scopes -- a tool
#   dispatch inside a turn -- unwind correctly rather than leaving the outer turn's events uncorrelated.
#
#-----------------------------------------------------------------------------------------------------------------------

@contextmanager
def correlation_scope ( correlation_id: str | None = None ) -> Iterator [ str ]:

    identifier = correlation_id if correlation_id is not None else new_correlation_id ()
    token      = CORRELATION_ID.set ( identifier )

    try:
        yield identifier
    finally:

        # A token may only be reset in the context that created it. That normally holds -- the scope opens and closes
        # in one coroutine -- but an async generator abandoned mid-iteration is finalised by the event loop's
        # asyncgen hook, which runs its close in a different task and therefore a different context. Clearing the
        # value is the correct fallback there, and turning that into an unhandled exception during garbage collection
        # would be a strange way for an audit facility to make itself known.

        try:
            CORRELATION_ID.reset ( token )
        except ValueError:
            CORRELATION_ID.set ( None )


#-----------------------------------------------------------------------------------------------------------------------
# Logger
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Class: EventLogger
#
# Description:
#
#   Writes and reads the append-only event log.
#
# Attributes:
#
#   connection       : Open aiosqlite connection whose schema is at migration 003 or later, or None to run without a
#                      database and export only.
#   exporters        : Additional destinations each event is fanned out to.
#   source           : Component name recorded on events that do not name their own.
#   identity_version : Identity generation stamped into every event's metadata.
#   written          : Events successfully persisted, for tests and diagnostics.
#   failures         : Events that could not be persisted.
#-----------------------------------------------------------------------------------------------------------------------

class EventLogger:

    #-------------------------------------------------------------------------------------------------------------------
    # Function: __init__
    #
    # Description:
    #
    #   Construct the logger.
    #
    # Arguments:
    #
    #   connection       : An open aiosqlite connection, or None to export without persisting.
    #   exporters        : Destinations each event is also sent to.
    #   source           : Default component name for events that do not name one.
    #   identity_version : Identity generation to stamp. Zero until Phase 8 supplies one.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def __init__ ( self,
                   connection: aiosqlite.Connection | None      = None,
                   exporters: Sequence [ EventExporter ] | None = None,
                   source: str                                  = "sentinel",
                   identity_version: int = 0 ) -> None:

        # Record the collaborators. The connection's lifetime belongs to whoever opened it.

        self.connection       = connection
        self.exporters        = list ( exporters or [] )
        self.source           = source
        self.identity_version = identity_version

        self.written  = 0
        self.failures = 0

    #-------------------------------------------------------------------------------------------------------------------
    # Writing
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: log
    #
    # Description:
    #
    #   Record one event.
    #
    # Arguments:
    #
    #   category       : One of the eight categories of architecture 3.2.10.
    #   event          : Dotted event name, prefixed by its category.
    #   data           : Event-specific payload. Redacted before it is written anywhere.
    #   correlation_id : Turn identifier. The ambient one from correlation_scope when omitted.
    #   source         : Component emitting the event. The logger's own default when omitted.
    #   session_id     : Conversation this belongs to, or None.
    #   timestamp      : Instant to record. Now when omitted.
    #
    # Returns:
    #
    #   The assembled event, whether or not it reached the database.
    #
    #   Raises EventLogError when the category or event name is unacceptable -- an authoring mistake, distinct from a
    #   storage failure, which is swallowed. See the module header.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def log ( self,
                    category: str,
                    event: str,
                    data: dict [ str, Any ] | None = None,
                    correlation_id: str | None     = None,
                    source: str | None             = None,
                    session_id: str | None         = None,
                    timestamp: str | None = None ) -> LogEvent:

        # An explicit identifier wins over the ambient one, so a caller correlating across turns is never overridden by
        # the scope it happens to be running inside.

        identifier = correlation_id if correlation_id is not None else current_correlation_id ()

        record = build_event (
            category         = category,
            event            = event,
            data             = data,
            correlation_id   = identifier,
            source           = source if source is not None else self.source,
            session_id       = session_id,
            identity_version = self.identity_version,
            timestamp        = timestamp,
        )

        await self._persist ( [ record ] )

        self._fan_out ( record )

        # Return data to caller.

        return record

    #-------------------------------------------------------------------------------------------------------------------
    # Function: log_many
    #
    # Description:
    #
    #   Record several events in one transaction.
    #
    #   The bulk path. One commit per event is the right default for a live agent, where events arrive seconds apart and
    #   durability per event matters; it is the wrong shape for a caller with a batch already in hand, where the commits
    #   dominate the cost entirely.
    #
    # Arguments:
    #
    #   events : The events to write. Already assembled, so validation and redaction have happened.
    #
    # Returns:
    #
    #   The events, unchanged.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def log_many ( self, events: Sequence [ LogEvent ] ) -> Sequence [ LogEvent ]:

        if not events:
            return events

        await self._persist ( events )

        for record in events:
            self._fan_out ( record )

        # Return data to caller.

        return events

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _persist
    #
    # Description:
    #
    #   Write events to the event_log table.
    #
    # Arguments:
    #
    #   events : The events to write.
    #
    # Returns:
    #
    #   None. A storage failure is counted and logged, never raised -- losing an audit row must not abort the operation
    #   being audited, and during shutdown the connection may already be closed, which aiosqlite reports as a
    #   ValueError rather than an sqlite3.Error (hence DATABASE_FAULTS).
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def _persist ( self, events: Sequence [ LogEvent ] ) -> None:

        if self.connection is None:
            return

        try:
            await self.connection.executemany (
                "INSERT INTO event_log "
                "( id, timestamp, category, event, source, correlation_id, data, metadata ) "
                "VALUES ( ?, ?, ?, ?, ?, ?, ?, ? )",
                [ record.as_row () for record in events ],
            )

            await self.connection.commit ()

            self.written += len ( events )

        except DATABASE_FAULTS as error:
            self.failures += len ( events )

            logger.warning (
                "Could not write %d event(s) to the event log: %s. The events still reached "
                "every configured exporter.",
                len ( events ), error,
            )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _fan_out
    #
    # Description:
    #
    #   Send one event to every exporter.
    #
    # Arguments:
    #
    #   record : The event to send.
    #
    # Returns:
    #
    #   None. An exporter that raises is reported and removed from the list: a destination that has failed once will
    #   almost certainly fail on every later event, and the alternative is the same warning on every event for the rest
    #   of the run.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def _fan_out ( self, record: LogEvent ) -> None:

        broken: list [ EventExporter ] = []

        for exporter in self.exporters:

            try:
                exporter.export ( record )
            except Exception as error:                                                 # noqa: BLE001
                logger.warning (
                    "Event exporter %s failed and has been disconnected: %s",
                    type ( exporter ).__name__, error,
                )

                broken.append ( exporter )

        for exporter in broken:
            self.exporters.remove ( exporter )

    #-------------------------------------------------------------------------------------------------------------------
    # Reading
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: query
    #
    # Description:
    #
    #   Read events, filtered and newest first.
    #
    #   Every filter is optional and they combine with AND, which is what the log API exposes. Instants are compared as
    #   ISO 8601 text, which sorts chronologically because that is the whole reason timestamps are stored as text.
    #
    # Arguments:
    #
    #   category       : Restrict to one category.
    #   event          : Restrict to one event name.
    #   correlation_id : Restrict to one turn.
    #   after          : Only events strictly after this instant.
    #   before         : Only events strictly before this instant.
    #   limit          : Maximum rows to return.
    #   offset         : Rows to skip, for paging.
    #
    # Returns:
    #
    #   The matching events, newest first.
    #
    #   Raises DatabaseError if the query failed, or if the logger has no connection -- a caller asking to read from a
    #   logger that only exports has made a mistake worth hearing about, unlike a caller writing to one.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def query ( self,
                      category: str | None          = None,
                      event: str | None             = None,
                      correlation_id: str | None    = None,
                      after: datetime | str | None  = None,
                      before: datetime | str | None = None,
                      limit: int                    = DEFAULT_QUERY_LIMIT,
                      offset: int = 0 ) -> list [ LogEvent ]:

        if self.connection is None:
            raise DatabaseError ( "This event logger has no database connection, so it cannot be queried." )

        clauses, parameters = self._filters ( category, event, correlation_id, after, before )

        where = f" WHERE {' AND '.join ( clauses )}" if clauses else ""

        # Ties are broken on rowid, which SQLite assigns in insertion order, and NOT on the identifier -- which is a
        # random UUID and would order two same-instant events arbitrarily. Two events of one turn genuinely can share a
        # timestamp: the clock's resolution is finite and a turn that fails fast emits its opening and closing rows
        # within the same tick of it. Ordering them at random makes the trail read as though the agent answered before
        # it was asked, which is exactly the question a trail is consulted to settle.

        try:
            async with self.connection.execute (
                f"SELECT {EVENT_COLUMNS} FROM event_log{where} "                        # noqa: S608
                f"ORDER BY timestamp DESC, rowid DESC LIMIT ? OFFSET ?",
                ( *parameters, max ( 0, limit ), max ( 0, offset ) ),
            ) as cursor:
                rows = await cursor.fetchall ()

        except DATABASE_FAULTS as error:
            raise DatabaseError ( f"Cannot read the event log: {error}" ) from error

        # Return data to caller.

        return [ row_to_event ( tuple ( row ) ) for row in rows ]

    #-------------------------------------------------------------------------------------------------------------------
    # Function: count
    #
    # Description:
    #
    #   Count events matching the same filters query() takes.
    #
    # Arguments:
    #
    #   category       : Restrict to one category.
    #   event          : Restrict to one event name.
    #   correlation_id : Restrict to one turn.
    #   after          : Only events strictly after this instant.
    #   before         : Only events strictly before this instant.
    #
    # Returns:
    #
    #   The number of matching rows.
    #
    #   Raises DatabaseError if the query failed, or if the logger has no connection.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def count ( self,
                      category: str | None         = None,
                      event: str | None            = None,
                      correlation_id: str | None   = None,
                      after: datetime | str | None = None,
                      before: datetime | str | None = None ) -> int:

        if self.connection is None:
            raise DatabaseError ( "This event logger has no database connection, so it cannot be queried." )

        clauses, parameters = self._filters ( category, event, correlation_id, after, before )

        where = f" WHERE {' AND '.join ( clauses )}" if clauses else ""

        try:
            async with self.connection.execute (
                f"SELECT COUNT(*) FROM event_log{where}", parameters                    # noqa: S608
            ) as cursor:
                row = await cursor.fetchone ()

        except DATABASE_FAULTS as error:
            raise DatabaseError ( f"Cannot count event log rows: {error}" ) from error

        # Return data to caller.

        return int ( row [ 0 ] ) if row else 0

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _filters
    #
    # Description:
    #
    #   Build the WHERE clauses and bound parameters shared by query() and count().
    #
    #   One implementation rather than two, so a filter that means one thing in a listing cannot come to mean something
    #   else in its own count -- which is the classic way a paged API reports "417 results" above a page of three.
    #
    # Arguments:
    #
    #   category       : Restrict to one category.
    #   event          : Restrict to one event name.
    #   correlation_id : Restrict to one turn.
    #   after          : Only events strictly after this instant.
    #   before         : Only events strictly before this instant.
    #
    # Returns:
    #
    #   The clause fragments and their bound values, in matching order. Every value is bound; the only text interpolated
    #   into a query anywhere in this module is the column-list constant.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def _filters ( self,
                   category: str | None,
                   event: str | None,
                   correlation_id: str | None,
                   after: datetime | str | None,
                   before: datetime | str | None ) -> tuple [ list [ str ], tuple [ object, ... ] ]:

        clauses: list [ str ]    = []
        values:  list [ object ] = []

        if category:
            clauses.append ( "category = ?" )
            values.append ( category )

        if event:
            clauses.append ( "event = ?" )
            values.append ( event )

        if correlation_id:
            clauses.append ( "correlation_id = ?" )
            values.append ( correlation_id )

        if after is not None:
            clauses.append ( "timestamp > ?" )
            values.append ( _as_timestamp ( after ) )

        if before is not None:
            clauses.append ( "timestamp < ?" )
            values.append ( _as_timestamp ( before ) )

        # Return data to caller.

        return clauses, tuple ( values )

    #-------------------------------------------------------------------------------------------------------------------
    # Lifecycle
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: close
    #
    # Description:
    #
    #   Close every exporter.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   None. The database connection is deliberately NOT closed: it belongs to whoever opened it, and the heartbeat
    #   and the memory system share the same arrangement.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def close ( self ) -> None:

        for exporter in self.exporters:

            try:
                exporter.close ()
            except Exception as error:                                                 # noqa: BLE001
                logger.warning ( "Event exporter %s did not close cleanly: %s", type ( exporter ).__name__, error )

        self.exporters.clear ()


#-----------------------------------------------------------------------------------------------------------------------
# Helpers
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Function: _as_timestamp
#
# Description:
#
#   Normalise a filter bound to the stored ISO 8601 form.
#
# Arguments:
#
#   moment : A datetime, or a string already in the stored form.
#
# Returns:
#
#   The comparable ISO 8601 text. A string is passed through untouched, so a caller can hand the API's own query
#   parameter straight in without a parse-and-reformat round trip that could only lose precision.
#
#-----------------------------------------------------------------------------------------------------------------------

def _as_timestamp ( moment: datetime | str ) -> str:

    # Return data to caller.

    return moment if isinstance ( moment, str ) else to_iso_timestamp ( moment )


#-----------------------------------------------------------------------------------------------------------------------
# Assembly
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Function: build_exporters
#
# Description:
#
#   Construct the exporters the configuration asks for.
#
# Arguments:
#
#   configuration : The loaded configuration.
#
# Returns:
#
#   The exporters, in the order events reach them. Empty when every exporter is switched off, which is legitimate: the
#   database alone is a complete audit trail, and the file copy exists for reading without SQLite.
#
#-----------------------------------------------------------------------------------------------------------------------

def build_exporters ( configuration: SentinelConfig ) -> list [ EventExporter ]:

    from sentinel.logging.exporters import PrometheusExporter, RotatingFileExporter, StdoutExporter

    exporters: list [ EventExporter ] = []

    if configuration.logging.file_export:
        exporters.append (
            RotatingFileExporter (
                directory = configuration.logs_directory,
                max_bytes = configuration.logging.max_file_size,
                retention = configuration.logging.retention,
            )
        )

    if configuration.logging.stdout_export:
        exporters.append ( StdoutExporter ( json_output = configuration.logging.json_output ) )

    if configuration.logging.prometheus_enabled:
        exporters.append ( PrometheusExporter () )

    # Return data to caller.

    return exporters


#-----------------------------------------------------------------------------------------------------------------------
# Function: open_event_logger
#
# Description:
#
#   Open the database and assemble an event logger over it.
#
#   Async, and therefore called from inside the runtime's event loop rather than from the command that launches it. An
#   aiosqlite handle binds to the loop that opened it, so a logger assembled on one loop and written from another
#   deadlocks rather than failing -- the same reason open_heartbeat is a factory and not a constructor.
#
#   A connection of its own rather than the heartbeat's. Event logging happens on every interactive turn as well as
#   every tick, and SQLite serialises statements per connection: sharing one would make a user's message wait behind
#   whatever autonomous work happened to be in flight. WAL is what makes the second connection cheap.
#
# Arguments:
#
#   configuration : The loaded configuration.
#   connection    : An already-open connection to use. One is opened when omitted.
#
# Returns:
#
#   The assembled logger. The caller owns closing it, and owns closing the connection it opened.
#
#   Raises DatabaseError if the database could not be opened.
#
#-----------------------------------------------------------------------------------------------------------------------

async def open_event_logger ( configuration: SentinelConfig,
                              connection: aiosqlite.Connection | None = None ) -> EventLogger:

    from sentinel.database import connect

    handle = connection

    # event_log_enabled false means "export only". The exporters still run; nothing is persisted, and query() then
    # refuses rather than silently returning nothing.

    if handle is None and configuration.logging.event_log_enabled:
        handle = await connect (
            database_path      = configuration.database_path,
            wal_mode           = configuration.database.wal_mode,
            busy_timeout       = configuration.database.busy_timeout,
            journal_size_limit = configuration.database.journal_size_limit,
            load_vectors       = False,
        )

    # Return data to caller.

    return EventLogger (
        connection = handle,
        exporters  = build_exporters ( configuration ),
        source     = configuration.agent.name.lower (),
    )
