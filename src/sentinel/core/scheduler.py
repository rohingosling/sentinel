#-----------------------------------------------------------------------------------------------------------------------
# Module:  scheduler.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   The cron schedule the heartbeat consults each tick (architecture 3.2.2, step 4).
#
#   APScheduler is used here only as a cron *parser* and next-fire-time calculator, not as the thing that runs the
#   jobs. That is the important design point in this module, and it is deliberate:
#
#     * A job registered with APScheduler fires on its own timer, in its own task, outside the tick. It would
#       therefore bypass max_autonomous_actions entirely -- the one ceiling that bounds how much the agent may do
#       unsupervised. Ten cron jobs coming due together would launch ten concurrent agent turns.
#     * Due-checking inside the tick puts scheduled jobs, queued tasks, and environmental signals through the same
#       budget, in one ordering, with one tick record describing what happened.
#
#   The cost is granularity: a job cannot fire more precisely than the heartbeat interval, so at the 240 s default a
#   job due at 12:00:00 runs somewhere in 12:00:00-12:04:00. For an autonomous assistant that is the right trade --
#   nothing here is a real-time control loop, and the alternative is an unbounded action rate.
#
#   next_run_at is materialised in the table rather than recomputed on read. It is what the due query indexes on, and
#   computing it needs the cron parser, which SQL does not have.
#
#   Expressions are interpreted against the MACHINE'S LOCAL CLOCK, and the resulting instants are stored in UTC. Those
#   are two separate decisions and both are deliberate:
#
#     * Interpretation is local because a person writing "0 9 * * *" means nine in the morning where they are.
#       Sentinel is a single-user desktop application (Decision 4), not a server farm, and a digest arriving at 13:00
#       because the author's offset is +04 is simply wrong. This was found by running the heartbeat and reading the
#       timezone in its own log; every test until then used explicit UTC instants and so could not have caught it.
#     * Storage is UTC because a stored local time is ambiguous for one hour every autumn, and because every other
#       instant Sentinel persists is UTC (see sentinel.timestamps). The two do not conflict: local is how the
#       expression is read, UTC is how the answer is written down.
#
#   Every function here takes an explicit `timezone` so a test can pin one. A test that inherits the developer's zone
#   asserts something different on every machine.
#-----------------------------------------------------------------------------------------------------------------------

import logging
import uuid

from dataclasses import dataclass
from datetime    import UTC, datetime, timedelta, tzinfo

import aiosqlite

from apscheduler.triggers.cron import CronTrigger

from sentinel.database   import DATABASE_FAULTS
from sentinel.errors     import DatabaseError, ScheduleError
from sentinel.timestamps import from_iso_timestamp, to_iso_timestamp

logger = logging.getLogger ( __name__ )

# Every column a caller sees, in one place, so the row-to-ScheduledJob mapping cannot drift from the SELECT list.

JOB_COLUMNS = "id, name, cron, description, enabled, created_at, last_run_at, next_run_at"

# A crontab expression has five fields: minute, hour, day, month, day-of-week. Checked before handing the string to
# APScheduler, whose own error for a wrong field count is less specific than the one below.

CRON_FIELD_COUNT = 5

# Day-of-week names, indexed by the STANDARD crontab numbering: 0 and 7 are both Sunday, 1 is Monday.
#
# This table exists because APScheduler numbers the same field differently -- 0 is MONDAY, not Sunday -- while
# from_crontab passes the field through untranslated. The expression "0 9 * * 1-5" therefore means Monday to Friday to
# every user who has ever written a crontab, and Tuesday to Saturday to APScheduler. Every schedule a user wrote would
# run a day late, and the failure is invisible: the job does fire, just never on the day that was asked for.
#
# Numbers are rewritten to names before parsing, because the names mean the same thing in both systems.

CRON_DAY_NAMES = ( "sun", "mon", "tue", "wed", "thu", "fri", "sat" )

# The smallest step forward that makes the next-fire-time search exclusive of the instant it starts from.
#
# APScheduler's search is inclusive: asked what comes next at exactly 12:00:00 for "0 * * * *", it answers 12:00:00.
# Left alone, that makes a job whose firing instant lands exactly on its schedule advance to itself and stay due for
# ever, re-firing on every tick.

FIRE_TIME_EPSILON = timedelta ( microseconds = 1 )


#-----------------------------------------------------------------------------------------------------------------------
# Function: local_timezone
#
# Description:
#
#   The machine's own timezone.
#
# Arguments:
#
#   None.
#
# Returns:
#
#   The local zone as a tzinfo. Falls back to UTC if the platform will not report one, which is the safe direction: a
#   schedule that runs on the wrong clock is better than one that cannot be parsed at all.
#
#-----------------------------------------------------------------------------------------------------------------------

def local_timezone () -> tzinfo:

    zone = datetime.now ().astimezone ().tzinfo

    # Return data to caller.

    return zone if zone is not None else UTC


#-----------------------------------------------------------------------------------------------------------------------
# Cron parsing
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Function: parse_cron
#
# Description:
#
#   Build a trigger from a crontab expression.
#
# Arguments:
#
#   expression : A five-field crontab expression, e.g. "*/15 * * * *".
#   timezone   : Clock the expression is written against. The machine's own zone when omitted.
#
# Returns:
#
#   The parsed trigger.
#
#   Raises ScheduleError if the expression has the wrong field count or cannot be parsed. The message quotes the
#   expression, because a cron string is almost always wrong in a way its author cannot see by reading it.
#
#-----------------------------------------------------------------------------------------------------------------------

def parse_cron ( expression: str, timezone: tzinfo | None = None ) -> CronTrigger:

    fields = expression.strip ().split ()

    if len ( fields ) != CRON_FIELD_COUNT:
        raise ScheduleError (
            f"The cron expression {expression!r} has {len ( fields )} field(s); "
            f"{CRON_FIELD_COUNT} are required -- minute, hour, day, month, day-of-week. "
            f"Example: '0 9 * * 1-5' for 09:00 on weekdays."
        )

    # Translate the day-of-week field out of standard crontab numbering and into names, which both systems read the
    # same way. See CRON_DAY_NAMES for why this is not optional.

    fields [ -1 ] = normalise_day_of_week ( fields [ -1 ] )

    zone = timezone if timezone is not None else local_timezone ()

    try:
        trigger = CronTrigger.from_crontab ( " ".join ( fields ), timezone = zone )
    except ValueError as error:
        raise ScheduleError ( f"Cannot parse the cron expression {expression!r}: {error}" ) from error

    # Return data to caller.

    return trigger


#-----------------------------------------------------------------------------------------------------------------------
# Function: normalise_day_of_week
#
# Description:
#
#   Rewrite a crontab day-of-week field from standard numbering into day names.
#
#   Standard crontab numbers Sunday as 0 (and accepts 7 for it too); APScheduler numbers Monday as 0. Names are
#   unambiguous in both, so every number in the field becomes the name it denotes in the standard scheme, and the
#   result means to APScheduler exactly what the user wrote.
#
# Arguments:
#
#   field : The fifth field of a crontab expression, e.g. "1-5", "0,6", "*/2", "*".
#
# Returns:
#
#   The field with its numbers replaced by names. Ranges, lists, steps, and anything already named pass through with
#   their structure intact; a value that is not a plain number is left exactly as written, so an expression this
#   function does not understand is still rejected by the parser rather than silently altered here.
#
#-----------------------------------------------------------------------------------------------------------------------

def normalise_day_of_week ( field: str ) -> str:

    #-------------------------------------------------------------------------------------------------------------------
    # Function: to_name
    #
    # Description:
    #
    #   Convert one day-of-week value to its name.
    #
    # Arguments:
    #
    #   value : A single value from the field.
    #
    # Returns:
    #
    #   The day name for a number in 0 to 7, or the value unchanged for anything else.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def to_name ( value: str ) -> str:

        # Return data to caller.

        if value.isdigit () and int ( value ) <= 7:
            return CRON_DAY_NAMES [ int ( value ) % 7 ]

        return value

    terms: list [ str ] = []

    for item in field.split ( "," ):

        # Split a trailing step off first, so "1-5/2" keeps its step and only the range is rewritten.

        body, separator, step = item.partition ( "/" )
        suffix                = separator + step

        low, dash, high = body.partition ( "-" )

        if dash:
            terms.append ( f"{to_name ( low )}-{to_name ( high )}{suffix}" )
        else:
            terms.append ( f"{to_name ( body )}{suffix}" )

    # Return data to caller.

    return ",".join ( terms )


#-----------------------------------------------------------------------------------------------------------------------
# Function: next_fire_time
#
# Description:
#
#   Compute when a cron expression next comes due.
#
# Arguments:
#
#   expression : A five-field crontab expression.
#   after      : Instant to search forward from. Now when omitted.
#   timezone   : Clock the expression is written against. The machine's own zone when omitted.
#
# Returns:
#
#   The next fire time STRICTLY AFTER the given instant, as a timezone-aware datetime CONVERTED TO UTC, or None when
#   the expression has no further fire time -- which a date-bounded expression such as "0 0 30 2 *" genuinely does not.
#   The expression is read against `timezone`; the answer is always returned in UTC, which is what gets stored.
#
#   Strictly after, not at or after. APScheduler's own search is inclusive, and both callers here need exclusion:
#   registration must not make a job instantly due because it happened to be registered on the minute, and mark_fired
#   must advance past the instant it just fired at rather than back onto it -- which would leave the job due for ever.
#
#   Raises ScheduleError if the expression cannot be parsed.
#
#-----------------------------------------------------------------------------------------------------------------------

def next_fire_time ( expression: str,
                     after: datetime | None = None,
                     timezone: tzinfo | None = None ) -> datetime | None:

    trigger = parse_cron ( expression, timezone )
    moment  = after if after is not None else datetime.now ( UTC )

    if moment.tzinfo is None:
        moment = moment.replace ( tzinfo = UTC )

    # A cron trigger derives everything from the wall clock, so the search is driven by `now` with no previous fire
    # time. Nudging `now` forward by the smallest representable amount is what makes the result exclusive.

    fire_time = trigger.get_next_fire_time ( None, moment + FIRE_TIME_EPSILON )

    # Return data to caller.

    return fire_time.astimezone ( UTC ) if fire_time is not None else None


#-----------------------------------------------------------------------------------------------------------------------
# Types
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Class: ScheduledJob
#
# Description:
#
#   One recurring job.
#
# Attributes:
#
#   id          : UUID.
#   name        : Stable handle. Re-registering the same name updates rather than duplicating.
#   cron        : The five-field crontab expression.
#   description : What the agent should do when the job fires.
#   enabled     : Whether the job is considered when checking what is due.
#   created_at  : ISO 8601 instant the job was registered.
#   last_run_at : ISO 8601 instant the job last fired, or None.
#   next_run_at : ISO 8601 instant the job next comes due, or None when the expression is exhausted.
#-----------------------------------------------------------------------------------------------------------------------

@dataclass ( frozen = True )
class ScheduledJob:

    id:          str
    name:        str
    cron:        str
    description: str
    enabled:     bool
    created_at:  str
    last_run_at: str | None
    next_run_at: str | None

    #-------------------------------------------------------------------------------------------------------------------
    # Function: due_at
    #
    # Description:
    #
    #   The job's next due instant.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   A timezone-aware datetime, or None when the job has no further fire time or the stored value is unparseable.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def due_at ( self ) -> datetime | None:

        # Return data to caller.

        return from_iso_timestamp ( self.next_run_at )


#-----------------------------------------------------------------------------------------------------------------------
# Function: row_to_job
#
# Description:
#
#   Map one database row onto a ScheduledJob.
#
# Arguments:
#
#   row : A row selected with JOB_COLUMNS, in that order.
#
# Returns:
#
#   The reconstructed job.
#
#-----------------------------------------------------------------------------------------------------------------------

def row_to_job ( row: aiosqlite.Row | tuple [ object, ... ] ) -> ScheduledJob:

    # Return data to caller.

    return ScheduledJob (
        id          = str ( row [ 0 ] ),
        name        = str ( row [ 1 ] ),
        cron        = str ( row [ 2 ] ),
        description = str ( row [ 3 ] ),
        enabled     = bool ( row [ 4 ] ),
        created_at  = str ( row [ 5 ] ),
        last_run_at = str ( row [ 6 ] ) if row [ 6 ] is not None else None,
        next_run_at = str ( row [ 7 ] ) if row [ 7 ] is not None else None,
    )


#-----------------------------------------------------------------------------------------------------------------------
# Scheduler
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Class: CronScheduler
#
# Description:
#
#   The persistent cron schedule.
#
#   Durable because the schedule is the user's, not the process's: "back up my notes every Sunday" must survive the
#   application closing on Saturday.
#
# Attributes:
#
#   connection : An open aiosqlite connection whose schema is at migration 002 or later.
#-----------------------------------------------------------------------------------------------------------------------

class CronScheduler:

    #-------------------------------------------------------------------------------------------------------------------
    # Function: __init__
    #
    # Description:
    #
    #   Construct the scheduler over an open connection.
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
    # Function: register
    #
    # Description:
    #
    #   Add a job, or update the one already holding this name.
    #
    #   Upsert rather than insert, keyed on name: registering the same job twice is what happens when a skill declares
    #   its schedule at load time and the application is restarted. Two rows would fire the job twice.
    #
    # Arguments:
    #
    #   name        : Stable handle for the job.
    #   cron        : Five-field crontab expression.
    #   description : What the agent should do when the job fires.
    #   enabled     : Whether the job should be considered when checking what is due.
    #   moment      : Instant to compute the first fire time from. Now when omitted.
    #
    # Returns:
    #
    #   The stored job.
    #
    #   Raises ScheduleError if the expression cannot be parsed, or DatabaseError if the row could not be written.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def register ( self,
                         name: str,
                         cron: str,
                         description: str = "",
                         enabled: bool    = True,
                         moment: datetime | None = None ) -> ScheduledJob:

        if not name.strip ():
            raise ScheduleError ( "A scheduled job needs a name. Refusing to register an unnamed one." )

        # Parse before writing. A row carrying an unparseable expression would fail on every due check for ever,
        # somewhere far away from the call that created it.

        due        = next_fire_time ( cron, moment )
        identifier = str ( uuid.uuid4 () )
        created    = to_iso_timestamp ( moment )
        next_run   = to_iso_timestamp ( due ) if due is not None else None

        try:
            await self.connection.execute (
                "INSERT INTO scheduled_jobs "
                "( id, name, cron, description, enabled, created_at, next_run_at ) "
                "VALUES ( ?, ?, ?, ?, ?, ?, ? ) "
                "ON CONFLICT ( name ) DO UPDATE SET "
                "cron = excluded.cron, description = excluded.description, "
                "enabled = excluded.enabled, next_run_at = excluded.next_run_at",
                ( identifier, name, cron, description, int ( enabled ), created, next_run ),
            )

            await self.connection.commit ()

        except DATABASE_FAULTS as error:
            raise DatabaseError ( f"Cannot register the scheduled job {name!r}: {error}" ) from error

        logger.debug ( "Registered scheduled job %r (%s), next due %s.", name, cron, next_run )

        stored = await self.get ( name )

        if stored is None:
            raise DatabaseError ( f"The scheduled job {name!r} was written but could not be read back." )

        # Return data to caller.

        return stored

    #-------------------------------------------------------------------------------------------------------------------
    # Function: remove
    #
    # Description:
    #
    #   Delete a job.
    #
    # Arguments:
    #
    #   name : The job to remove.
    #
    # Returns:
    #
    #   True when a row was deleted.
    #
    #   Raises DatabaseError if the delete failed.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def remove ( self, name: str ) -> bool:

        try:
            cursor = await self.connection.execute ( "DELETE FROM scheduled_jobs WHERE name = ?", ( name, ) )

            await self.connection.commit ()

        except DATABASE_FAULTS as error:
            raise DatabaseError ( f"Cannot remove the scheduled job {name!r}: {error}" ) from error

        # Return data to caller.

        return cursor.rowcount == 1

    #-------------------------------------------------------------------------------------------------------------------
    # Function: set_enabled
    #
    # Description:
    #
    #   Enable or disable a job without deleting it.
    #
    # Arguments:
    #
    #   name    : The job to change.
    #   enabled : Whether the job should be considered when checking what is due.
    #
    # Returns:
    #
    #   True when a row was updated.
    #
    #   Raises DatabaseError if the update failed.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def set_enabled ( self, name: str, enabled: bool ) -> bool:

        try:
            cursor = await self.connection.execute (
                "UPDATE scheduled_jobs SET enabled = ? WHERE name = ?",
                ( int ( enabled ), name ),
            )

            await self.connection.commit ()

        except DATABASE_FAULTS as error:
            raise DatabaseError ( f"Cannot change the scheduled job {name!r}: {error}" ) from error

        # Return data to caller.

        return cursor.rowcount == 1

    #-------------------------------------------------------------------------------------------------------------------
    # Function: mark_fired
    #
    # Description:
    #
    #   Record that a job has fired, and advance it to its next due instant.
    #
    #   Advancing is computed from the firing instant rather than from the stored next_run_at, so a job whose due time
    #   passed while the application was closed fires once on the next start and then resumes its normal cadence --
    #   instead of firing repeatedly to "catch up" on every occurrence it slept through.
    #
    # Arguments:
    #
    #   name   : The job that fired.
    #   moment : Instant the job fired. Now when omitted.
    #
    # Returns:
    #
    #   The updated job, or None when no such row exists.
    #
    #   Raises ScheduleError if the stored expression cannot be parsed, or DatabaseError if the update failed.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def mark_fired ( self, name: str, moment: datetime | None = None ) -> ScheduledJob | None:

        job = await self.get ( name )

        if job is None:
            return None

        fired    = moment if moment is not None else datetime.now ( UTC )
        due      = next_fire_time ( job.cron, fired )
        next_run = to_iso_timestamp ( due ) if due is not None else None

        try:
            await self.connection.execute (
                "UPDATE scheduled_jobs SET last_run_at = ?, next_run_at = ? WHERE name = ?",
                ( to_iso_timestamp ( fired ), next_run, name ),
            )

            await self.connection.commit ()

        except DATABASE_FAULTS as error:
            raise DatabaseError ( f"Cannot advance the scheduled job {name!r}: {error}" ) from error

        # Return data to caller.

        return await self.get ( name )

    #-------------------------------------------------------------------------------------------------------------------
    # Reading
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: due
    #
    # Description:
    #
    #   Read the enabled jobs whose next fire time has arrived (architecture 3.2.2, step 4).
    #
    # Arguments:
    #
    #   moment : Instant to judge against. Now when omitted.
    #   limit  : Maximum jobs to return.
    #
    # Returns:
    #
    #   Due jobs, oldest due first, so a job that has been waiting longest is processed before one that came due this
    #   instant when the action budget cannot cover both.
    #
    #   Raises DatabaseError if the query failed.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def due ( self, moment: datetime | None = None, limit: int = 50 ) -> list [ ScheduledJob ]:

        now = to_iso_timestamp ( moment )

        try:
            async with self.connection.execute (
                f"SELECT {JOB_COLUMNS} FROM scheduled_jobs "                           # noqa: S608
                f"WHERE enabled = 1 AND next_run_at IS NOT NULL AND next_run_at <= ? "
                f"ORDER BY next_run_at ASC LIMIT ?",
                ( now, limit ),
            ) as cursor:
                rows = await cursor.fetchall ()

        except DATABASE_FAULTS as error:
            raise DatabaseError ( f"Cannot read due scheduled jobs: {error}" ) from error

        # Return data to caller.

        return [ row_to_job ( row ) for row in rows ]

    #-------------------------------------------------------------------------------------------------------------------
    # Function: get
    #
    # Description:
    #
    #   Read one job by name.
    #
    # Arguments:
    #
    #   name : The job to read.
    #
    # Returns:
    #
    #   The job, or None when no such row exists.
    #
    #   Raises DatabaseError if the query failed.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def get ( self, name: str ) -> ScheduledJob | None:

        try:
            async with self.connection.execute (
                f"SELECT {JOB_COLUMNS} FROM scheduled_jobs WHERE name = ?",            # noqa: S608
                ( name, ),
            ) as cursor:
                row = await cursor.fetchone ()

        except DATABASE_FAULTS as error:
            raise DatabaseError ( f"Cannot read the scheduled job {name!r}: {error}" ) from error

        # Return data to caller.

        return row_to_job ( row ) if row else None

    #-------------------------------------------------------------------------------------------------------------------
    # Function: all_jobs
    #
    # Description:
    #
    #   Read every registered job.
    #
    # Arguments:
    #
    #   include_disabled : Include jobs that are currently disabled.
    #
    # Returns:
    #
    #   The jobs, ordered by name so the listing is stable between calls.
    #
    #   Raises DatabaseError if the query failed.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def all_jobs ( self, include_disabled: bool = True ) -> list [ ScheduledJob ]:

        condition = "" if include_disabled else "WHERE enabled = 1 "

        try:
            async with self.connection.execute (
                f"SELECT {JOB_COLUMNS} FROM scheduled_jobs {condition}ORDER BY name ASC"   # noqa: S608
            ) as cursor:
                rows = await cursor.fetchall ()

        except DATABASE_FAULTS as error:
            raise DatabaseError ( f"Cannot list scheduled jobs: {error}" ) from error

        # Return data to caller.

        return [ row_to_job ( row ) for row in rows ]
