#-----------------------------------------------------------------------------------------------------------------------
# Module:  heartbeat.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   The autonomous execution tick (architecture 3.2.2).
#
#       wake -> collect candidates -> spend the action budget -> record the tick -> sleep
#
#   Candidates come from three sources, in this order: the pending-task queue, the cron schedule, and the registered
#   environmental signal gatherers. Order is not arbitrary -- it is the priority the action budget spends itself in.
#   Explicitly queued work outranks a recurring job, which outranks something the agent noticed by itself.
#
#   Three properties are worth stating plainly, because each is a decision rather than a detail:
#
#     * The tick never raises. A task that fails, a job whose expression has rotted, a signal gatherer that throws --
#       each is recorded in the tick's errors column and the loop moves to the next candidate. An agent that stops
#       waking up after one bad task is worse than one that occasionally achieves nothing, and the failure mode of a
#       heartbeat that silently died is the hardest kind to notice.
#     * max_autonomous_actions bounds the whole tick, not each source. Five queued tasks and five due jobs still
#       produce five actions, not ten. This is the only ceiling on how much the agent may do unsupervised, so nothing
#       is allowed to route around it -- which is why cron jobs are due-checked here rather than registered as
#       independent APScheduler jobs (see scheduler.py).
#     * Ticks are serialised. APScheduler is configured with max_instances=1, so a tick that outruns the interval
#       delays the next one instead of overlapping it. Two concurrent ticks would double the action budget and race
#       for the same pending tasks.
#
#   Pause is a flag checked at the top of the scheduled callback rather than a call into APScheduler. It has to hold
#   whether or not the scheduler is running -- the API can pause a heartbeat that has not started yet -- and a flag is
#   the only form that means the same thing in both cases.
#-----------------------------------------------------------------------------------------------------------------------

import json
import logging
import uuid

from collections.abc import Sequence
from dataclasses     import dataclass, field
from datetime        import UTC, datetime, timedelta
from typing          import Any, Protocol, runtime_checkable

import aiosqlite

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval  import IntervalTrigger

from sentinel.config            import SentinelConfig
from sentinel.core.loop         import TRIGGER_HEARTBEAT_TICK, LoopResult, Trigger
from sentinel.core.orchestrator import HEARTBEAT_SESSION, AgentOrchestrator
from sentinel.core.scheduler    import CronScheduler, ScheduledJob
from sentinel.core.task_queue   import Task, TaskQueue
from sentinel.database          import DATABASE_FAULTS
from sentinel.errors            import DatabaseError, HeartbeatError, SentinelError
from sentinel.logging.logger    import EventLogger
from sentinel.logging.schemas import (
    CATEGORY_HEARTBEAT,
    EVENT_HEARTBEAT_END,
    EVENT_HEARTBEAT_ERROR,
    EVENT_HEARTBEAT_START,
)
from sentinel.timestamps import to_iso_timestamp

logger = logging.getLogger ( __name__ )

# The APScheduler job identifier. Fixed rather than generated, so re-adding it replaces the previous registration
# instead of scheduling a second tick.

TICK_JOB_ID = "sentinel-heartbeat"

# How late a tick may fire and still be worth running. A tick delayed beyond this is skipped rather than run against
# stale candidates -- the next one is at most one interval away and will see the same queue, only fresher.

MISFIRE_GRACE_SECONDS = 60

# What kind of candidate produced an action. Recorded in the tick's actions array so the record says what the agent
# spent its budget on, not merely how much of it.

CANDIDATE_TASK   = "task"
CANDIDATE_JOB    = "job"
CANDIDATE_SIGNAL = "signal"

# How many candidates to read from each source. Larger than any plausible action budget on purpose: the tick record's
# tasks_evaluated is meant to show the backlog, which a limit equal to the budget would hide.

CANDIDATE_READ_LIMIT = 50

# What one tick has to choose between: queued tasks, due jobs, and gathered signals, in the order the action budget
# spends itself.

Candidates = tuple [ list [ Task ], list [ ScheduledJob ], list [ "Signal" ] ]


#-----------------------------------------------------------------------------------------------------------------------
# Signals
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Class: Signal
#
# Description:
#
#   One thing the agent noticed by itself.
#
# Attributes:
#
#   source  : Name of the gatherer that produced it.
#   summary : One-line description, used as the trigger content.
#   data    : Structured detail for the gatherer's own use.
#-----------------------------------------------------------------------------------------------------------------------

@dataclass ( frozen = True )
class Signal:

    source:  str
    summary: str
    data:    dict [ str, Any ] = field ( default_factory = dict )


#-----------------------------------------------------------------------------------------------------------------------
# Class: SignalSource
#
# Description:
#
#   Something that can report environmental signals (architecture 3.2.2, step 5).
#
#   A Protocol rather than a base class, so a channel, a skill, or a test double can satisfy it without importing
#   anything from here. Phase 12's channels are the first real implementations; until then the registry is empty and
#   the tick simply finds nothing, which is the correct behaviour for an agent with nothing plugged in.
#
# Attributes:
#
#   name : Identifies the gatherer in logs and in the tick record.
#-----------------------------------------------------------------------------------------------------------------------

@runtime_checkable
class SignalSource ( Protocol ):

    name: str

    #-------------------------------------------------------------------------------------------------------------------
    # Function: gather
    #
    # Description:
    #
    #   Collect whatever this source currently has to report.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   The signals found, or an empty sequence. Returning nothing is the common case and is not an error.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def gather ( self ) -> Sequence [ Signal ]: ...


#-----------------------------------------------------------------------------------------------------------------------
# Records
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Class: HeartbeatTick
#
# Description:
#
#   One tick's record, carrying the heartbeat event schema of architecture 3.2.2.
#
# Attributes:
#
#   tick_id         : UUID.
#   started_at      : ISO 8601 instant the tick began.
#   ended_at        : ISO 8601 instant the tick finished, or None while in flight.
#   tasks_evaluated : Candidates inspected, before the action budget was applied.
#   actions_taken   : Candidates actually processed. Never exceeds max_autonomous_actions.
#   actions         : Summaries of what was done.
#   errors          : Descriptions of what went wrong.
#   next_tick_at    : ISO 8601 instant of the next scheduled tick, or None when stopping or paused.
#-----------------------------------------------------------------------------------------------------------------------

@dataclass
class HeartbeatTick:

    tick_id:         str
    started_at:      str
    ended_at:        str | None                 = None
    tasks_evaluated: int                        = 0
    actions_taken:   int                        = 0
    actions:         list [ dict [ str, Any ] ] = field ( default_factory = list )
    errors:          list [ str ]               = field ( default_factory = list )
    next_tick_at:    str | None                 = None


#-----------------------------------------------------------------------------------------------------------------------
# Class: HeartbeatStatus
#
# Description:
#
#   What the heartbeat is currently doing. The body of the status endpoint.
#
# Attributes:
#
#   enabled      : Whether the configuration asks for a heartbeat at all.
#   running      : Whether the scheduler is started.
#   paused       : Whether ticks are being skipped.
#   interval     : Seconds between ticks.
#   next_tick_at : ISO 8601 instant of the next scheduled tick, or None.
#   last_tick_id : The most recent tick's identifier, or None when none has run.
#-----------------------------------------------------------------------------------------------------------------------

@dataclass ( frozen = True )
class HeartbeatStatus:

    enabled:      bool
    running:      bool
    paused:       bool
    interval:     int
    next_tick_at: str | None
    last_tick_id: str | None

    #-------------------------------------------------------------------------------------------------------------------
    # Function: as_dict
    #
    # Description:
    #
    #   Render the status as a JSON-serialisable mapping.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   The status as plain data, for an API response body.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def as_dict ( self ) -> dict [ str, Any ]:

        # Return data to caller.

        return {
            "enabled": self.enabled,
            "running": self.running,
            "paused": self.paused,
            "interval": self.interval,
            "next_tick_at": self.next_tick_at,
            "last_tick_id": self.last_tick_id,
        }


#-----------------------------------------------------------------------------------------------------------------------
# Loop
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Class: HeartbeatLoop
#
# Description:
#
#   Drives the autonomous tick.
#
# Attributes:
#
#   configuration  : The loaded configuration.
#   orchestrator   : Turns a candidate into a completed agent turn.
#   queue          : The pending-task queue.
#   cron           : The cron schedule.
#   connection     : Connection used to write tick records.
#   signal_sources : Registered environmental signal gatherers.
#   events         : Event logger the tick boundaries are mirrored into, or None to run without one.
#-----------------------------------------------------------------------------------------------------------------------

class HeartbeatLoop:

    #-------------------------------------------------------------------------------------------------------------------
    # Function: __init__
    #
    # Description:
    #
    #   Construct the loop.
    #
    # Arguments:
    #
    #   configuration  : The loaded configuration.
    #   orchestrator   : Turns a candidate into a completed agent turn.
    #   queue          : The pending-task queue.
    #   cron           : The cron schedule.
    #   connection     : Connection used to write tick records.
    #   signal_sources : Environmental signal gatherers. None registers none, which is the Phase 4 default.
    #   events         : Event logger the tick boundaries are mirrored into. Omitting it makes every emission a no-op.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def __init__ ( self,
                   configuration: SentinelConfig,
                   orchestrator: AgentOrchestrator,
                   queue: TaskQueue,
                   cron: CronScheduler,
                   connection: aiosqlite.Connection,
                   signal_sources: Sequence [ SignalSource ] | None = None,
                   events: EventLogger | None = None ) -> None:

        # Record the collaborators.

        self.configuration  = configuration
        self.orchestrator   = orchestrator
        self.queue          = queue
        self.cron           = cron
        self.connection     = connection
        self.signal_sources = list ( signal_sources or [] )
        self.events         = events

        # Runtime state. The scheduler is built on start() rather than here, because AsyncIOScheduler binds to the
        # running event loop and construction may happen before there is one.

        # `started` is Sentinel's own record of intent, and it is what the status reports -- NOT the scheduler's own
        # `running` property. AsyncIOScheduler.shutdown() dispatches its teardown onto the event loop rather than
        # performing it inline, so `running` stays True until the loop next turns. A status endpoint answering
        # "running" immediately after a successful stop would be reporting that lag as fact.

        self.scheduler: AsyncIOScheduler | None = None
        self.started = False
        self.paused  = False
        self.last_tick_id: str | None = None

    #-------------------------------------------------------------------------------------------------------------------
    # Lifecycle
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: start
    #
    # Description:
    #
    #   Begin ticking.
    #
    #   Any task left running by an earlier process is returned to the queue first. Such a task belongs to a tick that
    #   never finished -- the application was closed or crashed mid-turn -- and nothing else will ever complete it.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   None.
    #
    #   Raises HeartbeatError if the scheduler could not be started.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def start ( self ) -> None:

        if not self.configuration.heartbeat.enabled:
            logger.info ( "Heartbeat is disabled in the configuration; not starting it." )

            return

        if self.started:
            logger.debug ( "Heartbeat is already running." )

            return

        # Recover work abandoned by an earlier run before the first tick reads the queue.

        try:
            await self.queue.requeue_stale ()
        except DatabaseError as error:
            logger.warning ( "Could not requeue abandoned tasks: %s. Starting anyway.", error )

        interval = self.configuration.heartbeat.interval

        try:
            self.scheduler = AsyncIOScheduler ( timezone = UTC )

            self.scheduler.add_job (
                self._scheduled_tick,
                trigger            = IntervalTrigger ( seconds = interval, timezone = UTC ),
                id                 = TICK_JOB_ID,
                replace_existing   = True,
                max_instances      = 1,
                coalesce           = True,
                misfire_grace_time = MISFIRE_GRACE_SECONDS,
                next_run_time      = datetime.now ( UTC ) + timedelta ( seconds = interval ),
            )

            self.scheduler.start ()

            self.started = True

        except Exception as error:
            raise HeartbeatError ( f"Cannot start the heartbeat scheduler: {error}" ) from error

        logger.info ( "Heartbeat started; ticking every %d s.", interval )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: stop
    #
    # Description:
    #
    #   Stop ticking.
    #
    # Arguments:
    #
    #   wait : Block until a tick in flight has finished.
    #
    # Returns:
    #
    #   None. Stopping an already-stopped heartbeat is a no-op rather than an error, because shutdown paths run it
    #   without knowing whether start() ever succeeded.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def stop ( self, wait: bool = False ) -> None:

        if not self.started or self.scheduler is None:
            return

        self.started = False

        self.scheduler.shutdown ( wait = wait )

        logger.info ( "Heartbeat stopped." )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: pause
    #
    # Description:
    #
    #   Suspend ticking without tearing the scheduler down.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   None. A tick already in flight is allowed to finish; pausing means "start no more", not "abandon this one".
    #
    #-------------------------------------------------------------------------------------------------------------------

    def pause ( self ) -> None:

        self.paused = True

        logger.info ( "Heartbeat paused." )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: resume
    #
    # Description:
    #
    #   Resume ticking after a pause.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   None. The next tick arrives on the existing schedule rather than immediately -- resuming is not a request to
    #   catch up on everything missed.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def resume ( self ) -> None:

        self.paused = False

        logger.info ( "Heartbeat resumed." )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: status
    #
    # Description:
    #
    #   Report what the heartbeat is currently doing.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   The current status.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def status ( self ) -> HeartbeatStatus:

        # Return data to caller.

        return HeartbeatStatus (
            enabled      = self.configuration.heartbeat.enabled,
            running      = self.started,
            paused       = self.paused,
            interval     = self.configuration.heartbeat.interval,
            next_tick_at = self.next_tick_at (),
            last_tick_id = self.last_tick_id,
        )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: next_tick_at
    #
    # Description:
    #
    #   Read the scheduler's next fire time.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   The next tick instant as an ISO 8601 string, or None when the heartbeat is stopped or paused. A paused
    #   heartbeat reports None even though APScheduler still holds a fire time, because that fire time will not
    #   produce a tick and reporting it would be a promise the loop does not intend to keep.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def next_tick_at ( self ) -> str | None:

        if self.scheduler is None or not self.started or self.paused:
            return None

        job = self.scheduler.get_job ( TICK_JOB_ID )

        if job is None or job.next_run_time is None:
            return None

        # Return data to caller.

        return to_iso_timestamp ( job.next_run_time )

    #-------------------------------------------------------------------------------------------------------------------
    # Ticking
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _scheduled_tick
    #
    # Description:
    #
    #   The callback APScheduler invokes on each interval.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   None. Skips the tick while paused, and swallows any failure the tick itself did not already record: an
    #   exception escaping into APScheduler would be logged by it and the schedule would survive, but the tick record
    #   would not exist and the failure would be invisible to everything that reads this table.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def _scheduled_tick ( self ) -> None:

        if self.paused:
            logger.debug ( "Heartbeat is paused; skipping this tick." )

            return

        try:
            await self.tick ()
        except Exception as error:                                                     # noqa: BLE001
            logger.exception ( "Heartbeat tick failed: %s", error )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: tick
    #
    # Description:
    #
    #   Run one tick to completion (architecture 3.2.2).
    #
    #   Public so tests and an operator can drive a tick without waiting for the interval. It runs whether or not the
    #   heartbeat is paused -- pause governs the schedule, not an explicit request.
    #
    # Arguments:
    #
    #   moment : Instant to treat as the tick's start. Now when omitted.
    #
    # Returns:
    #
    #   The completed tick record.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def tick ( self, moment: datetime | None = None ) -> HeartbeatTick:

        started = moment if moment is not None else datetime.now ( UTC )
        record  = HeartbeatTick ( tick_id = str ( uuid.uuid4 () ), started_at = to_iso_timestamp ( started ) )

        self.last_tick_id = record.tick_id

        # Write the opening row before doing any work. A tick that crashes then leaves a row with a start and no end,
        # which is exactly the evidence needed to find it; a row written only on success would make a crashed tick
        # indistinguishable from one that never ran.

        await self.record_tick_start ( record )

        # Mirror the boundary into the generic event log (architecture 3.2.10). The tick's own identifier is the
        # correlation identifier here, so every boundary event of one tick groups together -- while each turn the tick
        # drives keeps the identifier the loop mints for it, which is the granularity "what happened when I asked X"
        # actually needs.

        await self.record_event (
            EVENT_HEARTBEAT_START, record.tick_id,
            { "interval": self.configuration.heartbeat.interval },
        )

        logger.debug ( "Heartbeat tick %s starting.", record.tick_id )

        tasks, jobs, signals = await self.collect_candidates ( record, started )

        record.tasks_evaluated = len ( tasks ) + len ( jobs ) + len ( signals )

        budget = self.configuration.heartbeat.max_autonomous_actions

        # Spend the budget across all three sources in one pass. The ceiling is the tick's, not each source's.

        for task in tasks:
            if record.actions_taken >= budget:
                break

            await self.process_task ( task, record )

        for job in jobs:
            if record.actions_taken >= budget:
                break

            await self.process_job ( job, record, started )

        for signal in signals:
            if record.actions_taken >= budget:
                break

            await self.process_signal ( signal, record )

        if record.tasks_evaluated > record.actions_taken:
            logger.info (
                "Heartbeat tick %s reached its action budget of %d with %d candidate(s) left over.",
                record.tick_id, budget, record.tasks_evaluated - record.actions_taken,
            )

        record.ended_at     = to_iso_timestamp ()
        record.next_tick_at = self.next_tick_at ()

        await self.record_tick_end ( record )

        # One event per error rather than one carrying them all. An error is the thing anyone querying this log is
        # looking for, and a row per occurrence is what makes "how often does the cron table fail to read" countable.

        for description in record.errors:
            await self.record_event ( EVENT_HEARTBEAT_ERROR, record.tick_id, { "error": description } )

        await self.record_event (
            EVENT_HEARTBEAT_END, record.tick_id,
            {
                "tasks_evaluated": record.tasks_evaluated,
                "actions_taken": record.actions_taken,
                "actions": record.actions,
                "errors": len ( record.errors ),
                "next_tick_at": record.next_tick_at,
            },
        )

        # A tick that did nothing is not news, and at the 240 s default it happens hundreds of times a day. A tick
        # that acted, or that hit an error, is the only record the user has of what the agent did unsupervised, so it
        # is reported at INFO. This is also why APScheduler's own per-execution chatter is silenced (see
        # main.configure_logging): without that, the noise would be third-party and the signal would be hidden.

        eventful = record.actions_taken > 0 or bool ( record.errors )

        logger.log (
            logging.INFO if eventful else logging.DEBUG,
            "Heartbeat tick %s finished: %d evaluated, %d acted on, %d error(s).",
            record.tick_id, record.tasks_evaluated, record.actions_taken, len ( record.errors ),
        )

        # Return data to caller.

        return record

    #-------------------------------------------------------------------------------------------------------------------
    # Function: collect_candidates
    #
    # Description:
    #
    #   Read everything this tick could act on (architecture 3.2.2, steps 3 to 5).
    #
    # Arguments:
    #
    #   record : The tick record, for logging a source that failed.
    #   moment : Instant to judge eligibility and due-ness against.
    #
    # Returns:
    #
    #   Pending tasks, due jobs, and gathered signals. A source that fails contributes an empty list and an entry in
    #   the record's errors, rather than aborting the tick: a broken cron table must not stop queued work.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def collect_candidates ( self, record: HeartbeatTick, moment: datetime ) -> Candidates:

        tasks: list [ Task ]         = []
        jobs:  list [ ScheduledJob ] = []

        try:
            tasks = await self.queue.pending ( limit = CANDIDATE_READ_LIMIT, moment = moment )
        except SentinelError as error:
            record.errors.append ( f"Could not read pending tasks: {error}" )

        try:
            jobs = await self.cron.due ( moment = moment, limit = CANDIDATE_READ_LIMIT )
        except SentinelError as error:
            record.errors.append ( f"Could not read due scheduled jobs: {error}" )

        signals = await self.gather_signals ( record )

        # Return data to caller.

        return tasks, jobs, signals

    #-------------------------------------------------------------------------------------------------------------------
    # Function: gather_signals
    #
    # Description:
    #
    #   Poll every registered signal source (architecture 3.2.2, step 5).
    #
    # Arguments:
    #
    #   record : The tick record, for logging a gatherer that failed.
    #
    # Returns:
    #
    #   Every signal reported. A gatherer that raises is recorded and skipped -- one broken integration must not
    #   silence the others, and third-party code is where a raise is most likely to come from.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def gather_signals ( self, record: HeartbeatTick ) -> list [ Signal ]:

        signals: list [ Signal ] = []

        for source in self.signal_sources:

            name = getattr ( source, "name", source.__class__.__name__ )

            try:
                found = await source.gather ()
            except Exception as error:                                                 # noqa: BLE001
                logger.warning ( "Signal source %r failed: %s", name, error )

                record.errors.append ( f"Signal source {name!r} failed: {error}" )

                continue

            signals.extend ( found )

        # Return data to caller.

        return signals

    #-------------------------------------------------------------------------------------------------------------------
    # Processing
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: process_task
    #
    # Description:
    #
    #   Claim and run one queued task.
    #
    # Arguments:
    #
    #   task   : The task to run.
    #   record : The tick record to update.
    #
    # Returns:
    #
    #   None. A task that cannot be claimed was taken by someone else and is skipped silently -- that is the claim
    #   working, not a failure.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def process_task ( self, task: Task, record: HeartbeatTick ) -> None:

        try:
            claimed = await self.queue.claim ( task.id )
        except SentinelError as error:
            record.errors.append ( f"Could not claim task {task.id}: {error}" )

            return

        if not claimed:
            logger.debug ( "Task %s was already claimed; skipping it.", task.id )

            return

        result = await self.run_turn ( task.description, record, CANDIDATE_TASK, task.id )

        try:
            if result is None:
                await self.queue.fail ( task.id, "The agent turn did not complete." )
            elif result.error:
                await self.queue.fail ( task.id, result.error )
            else:
                await self.queue.complete ( task.id, result.text )

        except SentinelError as error:
            record.errors.append ( f"Could not record the outcome of task {task.id}: {error}" )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: process_job
    #
    # Description:
    #
    #   Run one due scheduled job and advance its schedule.
    #
    # Arguments:
    #
    #   job    : The job to run.
    #   record : The tick record to update.
    #   moment : Instant to treat as the firing time.
    #
    # Returns:
    #
    #   None. The job is advanced whether or not the turn succeeded, because that is what cron means: a missed or
    #   failed occurrence is not retried, it simply comes round again.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def process_job ( self, job: ScheduledJob, record: HeartbeatTick, moment: datetime ) -> None:

        description = job.description or f"Run the scheduled job {job.name!r}."

        await self.run_turn ( description, record, CANDIDATE_JOB, job.name )

        try:
            await self.cron.mark_fired ( job.name, moment )
        except SentinelError as error:
            logger.warning ( "Could not advance the scheduled job %r: %s", job.name, error )

            record.errors.append ( f"Could not advance the scheduled job {job.name!r}: {error}" )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: process_signal
    #
    # Description:
    #
    #   Run one environmental signal as an agent turn.
    #
    # Arguments:
    #
    #   signal : The signal to act on.
    #   record : The tick record to update.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def process_signal ( self, signal: Signal, record: HeartbeatTick ) -> None:

        await self.run_turn ( signal.summary, record, CANDIDATE_SIGNAL, signal.source )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: run_turn
    #
    # Description:
    #
    #   Hand one candidate to the orchestrator and record what came back.
    #
    #   The turn runs in its own session, so autonomous work never threads itself into whatever the user last said.
    #
    # Arguments:
    #
    #   description : What the agent is being asked to do. Becomes the trigger content.
    #   record      : The tick record to update.
    #   kind        : One of the CANDIDATE_* constants.
    #   reference   : Identifier of the candidate, for the action summary.
    #
    # Returns:
    #
    #   The turn outcome, or None when the turn itself raised. Either way the action is counted: the budget bounds
    #   what the agent attempts, not what it manages to finish, and a failing task that did not count would let a
    #   crash loop run unbounded.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def run_turn ( self,
                         description: str,
                         record: HeartbeatTick,
                         kind: str,
                         reference: str ) -> LoopResult | None:

        record.actions_taken += 1

        trigger = Trigger (
            kind      = TRIGGER_HEARTBEAT_TICK,
            content   = description,
            heartbeat = { "pending_tasks": [ description ], "scheduled_jobs": [] },
        )

        try:
            result = await self.orchestrator.process ( trigger, session_id = HEARTBEAT_SESSION )

        except Exception as error:                                                     # noqa: BLE001
            logger.exception ( "Autonomous turn for %s %r failed: %s", kind, reference, error )

            record.errors.append ( f"{kind} {reference!r}: {error}" )
            record.actions.append ( { "kind": kind, "reference": reference, "status": "error" } )

            return None

        status = "error" if result.error else "ok"

        record.actions.append (
            {
                "kind": kind,
                "reference": reference,
                "status": status,
                "iterations": result.iterations,
                "stop_reason": result.stop_reason,
            }
        )

        if result.error:
            record.errors.append ( f"{kind} {reference!r}: {result.error}" )

        # Return data to caller.

        return result

    #-------------------------------------------------------------------------------------------------------------------
    # Tick records
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: record_event
    #
    # Description:
    #
    #   Mirror one tick boundary into the generic event log.
    #
    # Arguments:
    #
    #   event          : One of the heartbeat event names.
    #   correlation_id : The tick's own identifier.
    #   data           : Event payload.
    #
    # Returns:
    #
    #   None. An absent logger makes this a no-op, and a failing one is reported and moved past -- the tick's contract
    #   is that it never raises, and an audit row is not worth breaking it for.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def record_event ( self,
                             event: str,
                             correlation_id: str,
                             data: dict [ str, Any ] | None = None ) -> None:

        if self.events is None:
            return

        try:
            await self.events.log (
                category       = CATEGORY_HEARTBEAT,
                event          = event,
                data           = data,
                correlation_id = correlation_id,
                source         = "heartbeat",
            )

        except SentinelError as error:
            logger.warning ( "Could not record the %s event: %s", event, error )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: record_tick_start
    #
    # Description:
    #
    #   Write the opening row for a tick.
    #
    # Arguments:
    #
    #   record : The tick record.
    #
    # Returns:
    #
    #   None. A write failure is logged rather than raised -- losing the audit row is bad, and refusing to do the
    #   work because the audit row could not be written is worse.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def record_tick_start ( self, record: HeartbeatTick ) -> None:

        try:
            await self.connection.execute (
                "INSERT INTO heartbeat_ticks ( tick_id, started_at ) VALUES ( ?, ? )",
                ( record.tick_id, record.started_at ),
            )

            await self.connection.commit ()

        except DATABASE_FAULTS as error:
            logger.warning ( "Could not record the start of heartbeat tick %s: %s", record.tick_id, error )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: record_tick_end
    #
    # Description:
    #
    #   Complete the row for a finished tick.
    #
    # Arguments:
    #
    #   record : The tick record.
    #
    # Returns:
    #
    #   None. A write failure is logged rather than raised, as for the opening row.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def record_tick_end ( self, record: HeartbeatTick ) -> None:

        try:
            await self.connection.execute (
                "UPDATE heartbeat_ticks SET ended_at = ?, tasks_evaluated = ?, actions_taken = ?, "
                "actions = ?, errors = ?, next_tick_at = ? WHERE tick_id = ?",
                (
                    record.ended_at, record.tasks_evaluated, record.actions_taken,
                    json.dumps ( record.actions ), json.dumps ( record.errors ),
                    record.next_tick_at, record.tick_id,
                ),
            )

            await self.connection.commit ()

        except DATABASE_FAULTS as error:
            logger.warning ( "Could not record the end of heartbeat tick %s: %s", record.tick_id, error )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: recent_ticks
    #
    # Description:
    #
    #   Read the most recent tick records.
    #
    # Arguments:
    #
    #   limit : Maximum records to return.
    #
    # Returns:
    #
    #   The records, newest first.
    #
    #   Raises DatabaseError if the query failed.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def recent_ticks ( self, limit: int = 10 ) -> list [ HeartbeatTick ]:

        try:
            async with self.connection.execute (
                "SELECT tick_id, started_at, ended_at, tasks_evaluated, actions_taken, actions, errors, next_tick_at "
                "FROM heartbeat_ticks ORDER BY started_at DESC LIMIT ?",
                ( limit, ),
            ) as cursor:
                rows = await cursor.fetchall ()

        except DATABASE_FAULTS as error:
            raise DatabaseError ( f"Cannot read heartbeat tick records: {error}" ) from error

        # Return data to caller.

        return [ row_to_tick ( row ) for row in rows ]


#-----------------------------------------------------------------------------------------------------------------------
# Assembly
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Function: open_heartbeat
#
# Description:
#
#   Open the database and assemble a heartbeat over it.
#
#   Async, and therefore called from inside the runtime's event loop rather than from the command that launches it.
#   The connection an aiosqlite handle binds to is the loop that opened it, so a heartbeat assembled on one loop and
#   ticked on another deadlocks rather than failing -- which is why this is a factory and not a constructor.
#
# Arguments:
#
#   configuration  : The loaded configuration.
#   adapter        : The LLM adapter autonomous turns should call.
#   signal_sources : Environmental signal gatherers. None registers none, which is the Phase 4 default.
#   events         : Event logger for the tick boundaries and the turns they drive. Supplied by the process manager,
#                    which owns its lifetime; omitting it runs the tick without an audit trail.
#
# Returns:
#
#   The assembled loop, not yet started. The caller owns starting it and closing its connection.
#
#   Raises DatabaseError if the database could not be opened.
#
#-----------------------------------------------------------------------------------------------------------------------

async def open_heartbeat ( configuration: SentinelConfig,
                           adapter: Any,
                           signal_sources: Sequence [ SignalSource ] | None = None,
                           events: EventLogger | None = None ) -> HeartbeatLoop:

    from sentinel.cache          import open_cache
    from sentinel.database       import connect
    from sentinel.memory.working import WorkingMemory

    connection = await connect (
        database_path      = configuration.database_path,
        wal_mode           = configuration.database.wal_mode,
        busy_timeout       = configuration.database.busy_timeout,
        journal_size_limit = configuration.database.journal_size_limit,
        load_vectors       = False,
    )

    # A connection of its own rather than one shared with the memory system. The tick runs concurrently with whatever
    # the user is doing, and SQLite serialises statements on a connection -- sharing one would make an autonomous turn
    # and an interactive one wait for each other. WAL is what makes two connections to the same file cheap.

    cache   = open_cache ( configuration.cache_directory, configuration.memory.working_size_limit )
    working = WorkingMemory ( cache, session_id = HEARTBEAT_SESSION )

    # Return data to caller.

    return HeartbeatLoop (
        configuration  = configuration,
        orchestrator   = AgentOrchestrator ( configuration, adapter, working = working, events = events ),
        queue          = TaskQueue ( connection ),
        cron           = CronScheduler ( connection ),
        connection     = connection,
        signal_sources = signal_sources,
        events         = events,
    )


#-----------------------------------------------------------------------------------------------------------------------
# Function: row_to_tick
#
# Description:
#
#   Map one database row onto a HeartbeatTick.
#
# Arguments:
#
#   row : A row selected as tick_id, started_at, ended_at, tasks_evaluated, actions_taken, actions, errors,
#         next_tick_at.
#
# Returns:
#
#   The reconstructed record. A JSON column that will not decode yields an empty list rather than raising, so one
#   corrupt row does not take down a listing.
#
#-----------------------------------------------------------------------------------------------------------------------

def row_to_tick ( row: aiosqlite.Row | tuple [ object, ... ] ) -> HeartbeatTick:

    # Return data to caller.

    return HeartbeatTick (
        tick_id         = str ( row [ 0 ] ),
        started_at      = str ( row [ 1 ] ),
        ended_at        = str ( row [ 2 ] ) if row [ 2 ] is not None else None,
        tasks_evaluated = int ( row [ 3 ] ),                                           # type: ignore [ arg-type ]
        actions_taken   = int ( row [ 4 ] ),                                           # type: ignore [ arg-type ]
        actions         = decode_json_list ( row [ 5 ] ),
        errors          = decode_json_list ( row [ 6 ] ),
        next_tick_at    = str ( row [ 7 ] ) if row [ 7 ] is not None else None,
    )


#-----------------------------------------------------------------------------------------------------------------------
# Function: decode_json_list
#
# Description:
#
#   Decode a stored JSON array column.
#
# Arguments:
#
#   value : The stored column value.
#
# Returns:
#
#   The decoded list, or an empty list when the value is absent, unparseable, or not an array.
#
#-----------------------------------------------------------------------------------------------------------------------

def decode_json_list ( value: object ) -> list [ Any ]:

    if not isinstance ( value, str ) or not value:
        return []

    try:
        decoded = json.loads ( value )
    except json.JSONDecodeError:
        logger.warning ( "Ignoring an unparseable heartbeat JSON column: %r.", value [ : 80 ] )

        return []

    # Return data to caller.

    return decoded if isinstance ( decoded, list ) else []
