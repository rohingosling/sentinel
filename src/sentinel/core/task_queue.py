#-----------------------------------------------------------------------------------------------------------------------
# Module:  task_queue.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   The SQLite-backed pending-task queue the heartbeat drains (architecture 3.2.2, step 3).
#
#   Durable rather than in-memory, for one reason: the agent is a desktop application the user closes. A queue held in
#   process memory would silently discard everything outstanding on every exit, and "remind me tomorrow" would survive
#   only until bedtime.
#
#   Claiming is a conditional UPDATE, not a SELECT followed by an UPDATE. The difference matters even though Sentinel
#   is single-process: the heartbeat, an API caller, and a future skill can all reach this table from different tasks
#   on the same event loop, and a read-then-write pair leaves a window in which two of them claim the same row. The
#   conditional form makes the claim itself the test -- exactly one caller sees rowcount 1.
#
#   Ordering is priority first, arrival second. An urgent task added late is picked up before a routine one that has
#   been waiting, which is the behaviour anyone would expect of a priority queue and is not what a pure FIFO gives.
#-----------------------------------------------------------------------------------------------------------------------

import logging
import uuid

from dataclasses import dataclass
from datetime    import datetime

import aiosqlite

from sentinel.database   import DATABASE_FAULTS
from sentinel.errors     import DatabaseError
from sentinel.timestamps import from_iso_timestamp, to_iso_timestamp

logger = logging.getLogger ( __name__ )

# Every column a caller sees, in one place, so the row-to-Task mapping cannot drift from the SELECT list.

TASK_COLUMNS = (
    "id, description, status, priority, source, scheduled_for, "
    "created_at, started_at, completed_at, attempts, result, error"
)

# Lifecycle states. A task is pending until claimed, running until it finishes, and then terminal.

STATUS_PENDING   = "pending"
STATUS_RUNNING   = "running"
STATUS_DONE      = "done"
STATUS_FAILED    = "failed"
STATUS_CANCELLED = "cancelled"

TERMINAL_STATUSES = frozenset ( { STATUS_DONE, STATUS_FAILED, STATUS_CANCELLED } )

# Where a task came from. Recorded because the answer changes how a failure should be read: a user's task that fails
# deserves a message back, where a signal-derived one that fails is usually just noise that has stopped being true.

SOURCE_USER     = "user"
SOURCE_SCHEDULE = "schedule"
SOURCE_SIGNAL   = "signal"
SOURCE_AGENT    = "agent"


#-----------------------------------------------------------------------------------------------------------------------
# Types
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Class: Task
#
# Description:
#
#   One queued unit of work.
#
# Attributes:
#
#   id            : UUID.
#   description   : What the agent is being asked to do.
#   status        : One of the STATUS_* constants.
#   priority      : Higher runs first.
#   source        : One of the SOURCE_* constants.
#   scheduled_for : ISO 8601 instant before which the task is not eligible, or None for immediately.
#   created_at    : ISO 8601 instant the row was written.
#   started_at    : ISO 8601 instant the task was claimed, or None.
#   completed_at  : ISO 8601 instant the task reached a terminal state, or None.
#   attempts      : Claims so far. A value above one means the task was requeued after a crash.
#   result        : Outcome text on success, or None.
#   error         : Failure description, or None.
#-----------------------------------------------------------------------------------------------------------------------

@dataclass ( frozen = True )
class Task:

    id:            str
    description:   str
    status:        str
    priority:      int
    source:        str
    scheduled_for: str | None
    created_at:    str
    started_at:    str | None
    completed_at:  str | None
    attempts:      int
    result:        str | None
    error:         str | None

    #-------------------------------------------------------------------------------------------------------------------
    # Function: moment
    #
    # Description:
    #
    #   The task's creation instant.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   A timezone-aware datetime, or None when the stored value cannot be parsed.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def moment ( self ) -> datetime | None:

        # Return data to caller.

        return from_iso_timestamp ( self.created_at )


#-----------------------------------------------------------------------------------------------------------------------
# Function: row_to_task
#
# Description:
#
#   Map one database row onto a Task.
#
# Arguments:
#
#   row : A row selected with TASK_COLUMNS, in that order.
#
# Returns:
#
#   The reconstructed task.
#
#-----------------------------------------------------------------------------------------------------------------------

def row_to_task ( row: aiosqlite.Row | tuple [ object, ... ] ) -> Task:

    # Return data to caller.

    return Task (
        id            = str ( row [ 0 ] ),
        description   = str ( row [ 1 ] ),
        status        = str ( row [ 2 ] ),
        priority      = int ( row [ 3 ] ),            # type: ignore [ arg-type ]
        source        = str ( row [ 4 ] ),
        scheduled_for = str ( row [ 5 ] ) if row [ 5 ] is not None else None,
        created_at    = str ( row [ 6 ] ),
        started_at    = str ( row [ 7 ] ) if row [ 7 ] is not None else None,
        completed_at  = str ( row [ 8 ] ) if row [ 8 ] is not None else None,
        attempts      = int ( row [ 9 ] ),            # type: ignore [ arg-type ]
        result        = str ( row [ 10 ] ) if row [ 10 ] is not None else None,
        error         = str ( row [ 11 ] ) if row [ 11 ] is not None else None,
    )


#-----------------------------------------------------------------------------------------------------------------------
# Queue
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Class: TaskQueue
#
# Description:
#
#   The pending-task queue.
#
#   Holds an open connection rather than opening one per call: the heartbeat touches this table several times per tick,
#   and SQLite connection setup includes the pragma set.
#
# Attributes:
#
#   connection : An open aiosqlite connection whose schema is at migration 002 or later.
#-----------------------------------------------------------------------------------------------------------------------

class TaskQueue:

    #-------------------------------------------------------------------------------------------------------------------
    # Function: __init__
    #
    # Description:
    #
    #   Construct the queue over an open connection.
    #
    # Arguments:
    #
    #   connection : An open aiosqlite connection.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def __init__ ( self, connection: aiosqlite.Connection ) -> None:

        # Record the connection. The caller owns its lifetime.

        self.connection = connection

    #-------------------------------------------------------------------------------------------------------------------
    # Writing
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: enqueue
    #
    # Description:
    #
    #   Add a task to the queue.
    #
    # Arguments:
    #
    #   description   : What the agent is being asked to do.
    #   priority      : Higher runs first.
    #   source        : One of the SOURCE_* constants.
    #   scheduled_for : Instant before which the task is not eligible. Immediately eligible when omitted.
    #   task_id       : Identifier to use. Generated when omitted.
    #
    # Returns:
    #
    #   The stored task.
    #
    #   Raises DatabaseError if the row could not be written.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def enqueue ( self,
                        description: str,
                        priority: int                  = 0,
                        source: str                    = SOURCE_USER,
                        scheduled_for: datetime | None = None,
                        task_id: str | None = None ) -> Task:

        # A task with no description is not a task; it would reach the model as an empty instruction.

        if not description.strip ():
            raise DatabaseError ( "A task needs a description. Refusing to queue an empty one." )

        identifier = task_id if task_id is not None else str ( uuid.uuid4 () )
        created    = to_iso_timestamp ()
        eligible   = to_iso_timestamp ( scheduled_for ) if scheduled_for is not None else None

        try:
            await self.connection.execute (
                "INSERT INTO tasks "
                "( id, description, status, priority, source, scheduled_for, created_at, attempts ) "
                "VALUES ( ?, ?, ?, ?, ?, ?, ?, 0 )",
                ( identifier, description, STATUS_PENDING, priority, source, eligible, created ),
            )

            await self.connection.commit ()

        except DATABASE_FAULTS as error:
            raise DatabaseError ( f"Cannot queue the task: {error}" ) from error

        logger.debug ( "Queued task %s (priority=%d, source=%s).", identifier, priority, source )

        # Return data to caller.

        return Task (
            id            = identifier,
            description   = description,
            status        = STATUS_PENDING,
            priority      = priority,
            source        = source,
            scheduled_for = eligible,
            created_at    = created,
            started_at    = None,
            completed_at  = None,
            attempts      = 0,
            result        = None,
            error         = None,
        )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: claim
    #
    # Description:
    #
    #   Take ownership of a pending task, moving it to running.
    #
    #   The status test lives inside the UPDATE rather than in a preceding SELECT, so the claim is atomic: two callers
    #   racing for the same row produce one success and one failure rather than two successes.
    #
    # Arguments:
    #
    #   task_id : The task to claim.
    #   moment  : Instant to record as the start. Now when omitted.
    #
    # Returns:
    #
    #   True when this caller claimed the task, False when it was already claimed, terminal, or absent.
    #
    #   Raises DatabaseError if the update could not be applied.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def claim ( self, task_id: str, moment: datetime | None = None ) -> bool:

        started = to_iso_timestamp ( moment )

        try:
            cursor = await self.connection.execute (
                "UPDATE tasks "
                "SET status = ?, started_at = ?, attempts = attempts + 1 "
                "WHERE id = ? AND status = ?",
                ( STATUS_RUNNING, started, task_id, STATUS_PENDING ),
            )

            await self.connection.commit ()

        except DATABASE_FAULTS as error:
            raise DatabaseError ( f"Cannot claim task {task_id}: {error}" ) from error

        # Return data to caller.

        return cursor.rowcount == 1

    #-------------------------------------------------------------------------------------------------------------------
    # Function: complete
    #
    # Description:
    #
    #   Mark a task finished successfully.
    #
    # Arguments:
    #
    #   task_id : The task to complete.
    #   result  : Outcome text to record.
    #   moment  : Instant to record as the completion. Now when omitted.
    #
    # Returns:
    #
    #   True when a row was updated.
    #
    #   Raises DatabaseError if the update could not be applied.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def complete ( self, task_id: str, result: str = "", moment: datetime | None = None ) -> bool:

        # Return data to caller.

        return await self._finish ( task_id, STATUS_DONE, result = result, error = None, moment = moment )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: fail
    #
    # Description:
    #
    #   Mark a task finished unsuccessfully.
    #
    # Arguments:
    #
    #   task_id : The task to fail.
    #   error   : Failure description to record.
    #   moment  : Instant to record as the completion. Now when omitted.
    #
    # Returns:
    #
    #   True when a row was updated.
    #
    #   Raises DatabaseError if the update could not be applied.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def fail ( self, task_id: str, error: str, moment: datetime | None = None ) -> bool:

        # Return data to caller.

        return await self._finish ( task_id, STATUS_FAILED, result = None, error = error, moment = moment )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: cancel
    #
    # Description:
    #
    #   Withdraw a task that has not yet been claimed.
    #
    # Arguments:
    #
    #   task_id : The task to cancel.
    #   moment  : Instant to record as the completion. Now when omitted.
    #
    # Returns:
    #
    #   True when a pending task was cancelled, False when it was already running or terminal. A running task is left
    #   alone on purpose: the work is in flight and cancelling the row would not stop it.
    #
    #   Raises DatabaseError if the update could not be applied.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def cancel ( self, task_id: str, moment: datetime | None = None ) -> bool:

        completed = to_iso_timestamp ( moment )

        try:
            cursor = await self.connection.execute (
                "UPDATE tasks SET status = ?, completed_at = ? WHERE id = ? AND status = ?",
                ( STATUS_CANCELLED, completed, task_id, STATUS_PENDING ),
            )

            await self.connection.commit ()

        except DATABASE_FAULTS as error:
            raise DatabaseError ( f"Cannot cancel task {task_id}: {error}" ) from error

        # Return data to caller.

        return cursor.rowcount == 1

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _finish
    #
    # Description:
    #
    #   Move a running task to a terminal state.
    #
    # Arguments:
    #
    #   task_id : The task to finish.
    #   status  : The terminal status to record.
    #   result  : Outcome text, or None.
    #   error   : Failure description, or None.
    #   moment  : Instant to record as the completion. Now when omitted.
    #
    # Returns:
    #
    #   True when a row was updated.
    #
    #   Raises DatabaseError if the update could not be applied.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def _finish ( self,
                        task_id: str,
                        status: str,
                        result: str | None,
                        error: str | None,
                        moment: datetime | None ) -> bool:

        completed = to_iso_timestamp ( moment )

        try:
            cursor = await self.connection.execute (
                "UPDATE tasks SET status = ?, completed_at = ?, result = ?, error = ? WHERE id = ?",
                ( status, completed, result, error, task_id ),
            )

            await self.connection.commit ()

        except DATABASE_FAULTS as database_error:
            raise DatabaseError ( f"Cannot finish task {task_id}: {database_error}" ) from database_error

        # Return data to caller.

        return cursor.rowcount == 1

    #-------------------------------------------------------------------------------------------------------------------
    # Function: requeue_stale
    #
    # Description:
    #
    #   Return tasks abandoned mid-flight to the pending state.
    #
    #   A task left running belongs to a tick that never finished -- the application was closed or crashed while the
    #   agent was working on it. Nothing will ever complete it, so on the next start it goes back into the queue. The
    #   attempts counter is what keeps this from becoming an infinite retry: a task that has crashed the agent
    #   repeatedly is visible as a high count rather than hidden behind a fresh-looking row.
    #
    # Arguments:
    #
    #   max_attempts : Give up on a task that has already been claimed this many times, failing it instead.
    #
    # Returns:
    #
    #   The number of tasks returned to pending.
    #
    #   Raises DatabaseError if the update could not be applied.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def requeue_stale ( self, max_attempts: int = 3 ) -> int:

        abandoned = to_iso_timestamp ()

        try:

            # Anything already at the attempt ceiling is terminal, not retryable. Done first, so the requeue below
            # cannot pick it up again in the same call.

            await self.connection.execute (
                "UPDATE tasks SET status = ?, completed_at = ?, error = ? "
                "WHERE status = ? AND attempts >= ?",
                (
                    STATUS_FAILED, abandoned,
                    f"Abandoned mid-flight {max_attempts} times; not retried again.",
                    STATUS_RUNNING, max_attempts,
                ),
            )

            cursor = await self.connection.execute (
                "UPDATE tasks SET status = ?, started_at = NULL WHERE status = ?",
                ( STATUS_PENDING, STATUS_RUNNING ),
            )

            await self.connection.commit ()

        except DATABASE_FAULTS as error:
            raise DatabaseError ( f"Cannot requeue abandoned tasks: {error}" ) from error

        requeued = cursor.rowcount if cursor.rowcount > 0 else 0

        if requeued:
            logger.info ( "Requeued %d task(s) abandoned by an earlier run.", requeued )

        # Return data to caller.

        return requeued

    #-------------------------------------------------------------------------------------------------------------------
    # Reading
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: pending
    #
    # Description:
    #
    #   Read the tasks eligible to run now (architecture 3.2.2, step 3).
    #
    # Arguments:
    #
    #   limit  : Maximum tasks to return.
    #   moment : Instant to judge eligibility against. Now when omitted.
    #
    # Returns:
    #
    #   Eligible tasks, highest priority first and oldest first within a priority. A task deferred to a future instant
    #   is excluded until that instant passes.
    #
    #   Raises DatabaseError if the query failed.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def pending ( self, limit: int = 50, moment: datetime | None = None ) -> list [ Task ]:

        now = to_iso_timestamp ( moment )

        try:
            async with self.connection.execute (
                f"SELECT {TASK_COLUMNS} FROM tasks "                                   # noqa: S608
                f"WHERE status = ? AND ( scheduled_for IS NULL OR scheduled_for <= ? ) "
                f"ORDER BY priority DESC, created_at ASC LIMIT ?",
                ( STATUS_PENDING, now, limit ),
            ) as cursor:
                rows = await cursor.fetchall ()

        except DATABASE_FAULTS as error:
            raise DatabaseError ( f"Cannot read pending tasks: {error}" ) from error

        # Return data to caller.

        return [ row_to_task ( row ) for row in rows ]

    #-------------------------------------------------------------------------------------------------------------------
    # Function: get
    #
    # Description:
    #
    #   Read one task by identifier.
    #
    # Arguments:
    #
    #   task_id : The task to read.
    #
    # Returns:
    #
    #   The task, or None when no such row exists.
    #
    #   Raises DatabaseError if the query failed.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def get ( self, task_id: str ) -> Task | None:

        try:
            async with self.connection.execute (
                f"SELECT {TASK_COLUMNS} FROM tasks WHERE id = ?",                      # noqa: S608
                ( task_id, ),
            ) as cursor:
                row = await cursor.fetchone ()

        except DATABASE_FAULTS as error:
            raise DatabaseError ( f"Cannot read task {task_id}: {error}" ) from error

        # Return data to caller.

        return row_to_task ( row ) if row else None

    #-------------------------------------------------------------------------------------------------------------------
    # Function: count
    #
    # Description:
    #
    #   Count tasks, optionally in one status.
    #
    # Arguments:
    #
    #   status : Status to count. Every task when omitted.
    #
    # Returns:
    #
    #   The number of matching rows.
    #
    #   Raises DatabaseError if the query failed.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def count ( self, status: str | None = None ) -> int:

        query = "SELECT COUNT(*) FROM tasks"
        parameters: tuple [ object, ... ] = ()

        if status is not None:
            query      = f"{query} WHERE status = ?"
            parameters = ( status, )

        try:
            async with self.connection.execute ( query, parameters ) as cursor:
                row = await cursor.fetchone ()

        except DATABASE_FAULTS as error:
            raise DatabaseError ( f"Cannot count tasks: {error}" ) from error

        # Return data to caller.

        return int ( row [ 0 ] ) if row else 0

    #-------------------------------------------------------------------------------------------------------------------
    # Function: purge
    #
    # Description:
    #
    #   Delete terminal tasks completed before a cut-off.
    #
    # Arguments:
    #
    #   before : Instant before which completed tasks are removed.
    #
    # Returns:
    #
    #   The number of rows deleted. Pending and running tasks are never touched, whatever their age.
    #
    #   Raises DatabaseError if the delete failed.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def purge ( self, before: datetime ) -> int:

        cut_off  = to_iso_timestamp ( before )
        terminal = ", ".join ( "?" for _ in TERMINAL_STATUSES )

        try:
            cursor = await self.connection.execute (
                f"DELETE FROM tasks WHERE status IN ( {terminal} ) "                   # noqa: S608
                f"AND completed_at IS NOT NULL AND completed_at < ?",
                ( *sorted ( TERMINAL_STATUSES ), cut_off ),
            )

            await self.connection.commit ()

        except DATABASE_FAULTS as error:
            raise DatabaseError ( f"Cannot purge completed tasks: {error}" ) from error

        # Return data to caller.

        return cursor.rowcount if cursor.rowcount > 0 else 0
