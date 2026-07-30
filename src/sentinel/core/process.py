#-----------------------------------------------------------------------------------------------------------------------
# Module:  process.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   Process lifecycle manager (architecture 4.3).
#
#   Coordinates four components with independent lifecycles -- the agent API, the Open WebUI subprocess, the system
#   tray, and the native window -- through the explicit state machine in states.py. Every state change goes through
#   dispatch(), so the sequence a component saw is always reconstructable from the log.
#
#   Threading model, which is the one genuinely awkward part:
#
#     * Everything here runs on an asyncio loop, on a background thread.
#     * The native window must own the main thread (see window.py), so it is NOT started from here. ProcessManager
#       exposes wait_until_ready() and notify_window_closed() for the main thread to drive, and every cross-thread
#       entry point marshals onto the loop rather than touching state directly.
#
#   Crash detection does not wait for a health poll. A subprocess that exits wakes its watcher task immediately, so the
#   restart begins in milliseconds rather than after the 10 s health interval -- which is what makes the "restarts
#   within 5 seconds" criterion achievable without polling aggressively.
#-----------------------------------------------------------------------------------------------------------------------

import asyncio
import contextlib
import logging
import sqlite3
import threading
import time

from collections.abc import Awaitable, Callable
from typing          import Any

import httpx

from fastapi import FastAPI

from sentinel           import __version__
from sentinel.config    import SentinelConfig
from sentinel.core      import webui as webui_support
from sentinel.core.tray import TrayIcon
from sentinel.errors    import SentinelError
from sentinel.logging.schemas import (
    CATEGORY_SYSTEM,
    EVENT_SYSTEM_HEALTH,
    EVENT_SYSTEM_SHUTDOWN,
    EVENT_SYSTEM_STARTUP,
)
from sentinel.core.states import (
    ComponentName,
    EventName,
    GuardContext,
    ProcessState,
    is_required,
    resolve_transition,
)

logger = logging.getLogger ( __name__ )

# Seconds between health probes while waiting for a component to come up. Distinct from the steady-state health
# interval: during startup the answer is expected to change soon, so polling is tighter.

STARTUP_POLL_INTERVAL = 0.25

# How long the main thread blocks in one slice while waiting for shutdown.
#
# It waits in slices rather than once, because on Windows `threading.Event.wait()` with no timeout is NOT interruptible
# by Ctrl+C: the thread blocks in a native call, the interpreter never regains control, and the signal handler
# therefore never runs. Slicing hands control back regularly, which is what lets KeyboardInterrupt actually be
# delivered. Short enough to feel immediate, long enough not to spin.

SHUTDOWN_POLL_INTERVAL = 0.25


#-----------------------------------------------------------------------------------------------------------------------
# Class: ComponentStatus
#
# Description:
#
#   Live status of one managed component.
#
# Attributes:
#
#   name          : Which component this is.
#   started       : Whether a start has ever been attempted.
#   healthy       : Whether it is currently up.
#   failures      : Recovery attempts since the last sustained period of health.
#   last_healthy  : Monotonic time it was last observed healthy, used to age out the failure counter.
#-----------------------------------------------------------------------------------------------------------------------

class ComponentStatus:

    #-------------------------------------------------------------------------------------------------------------------
    # Function: __init__
    #
    # Description:
    #
    #   Construct a status record for one component.
    #
    # Arguments:
    #
    #   name : Which component this is.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def __init__ ( self, name: ComponentName ) -> None:

        self.name         = name
        self.started      = False
        self.healthy      = False
        self.failures     = 0
        self.last_healthy = 0.0


#-----------------------------------------------------------------------------------------------------------------------
# Class: ProcessManager
#
# Description:
#
#   The finite state machine that owns component lifecycle (architecture 4.3).
#
#   Constructed with the loaded configuration and the already-built FastAPI application. Collaborators that touch the
#   outside world -- the health probe and the Open WebUI launcher -- are injectable, so the whole machine can be
#   exercised without binding a port or installing Open WebUI.
#
# Attributes:
#
#   configuration : The loaded configuration.
#   application   : The FastAPI application to serve.
#   state         : Current FSM state.
#   components    : Per-component status records.
#-----------------------------------------------------------------------------------------------------------------------

class ProcessManager:

    #-------------------------------------------------------------------------------------------------------------------
    # Function: __init__
    #
    # Description:
    #
    #   Construct the process manager.
    #
    # Arguments:
    #
    #   configuration : The loaded configuration.
    #   application   : The FastAPI application the agent API serves. None runs without an API, which only a test does.
    #   api_token     : Bearer token handed to Open WebUI. Read from the keyring when omitted.
    #   health_probe  : Async callable ( url ) -> bool used for readiness checks. A real HTTP GET when omitted.
    #   webui_launcher: Async callable () -> process used to start Open WebUI. A real subprocess when omitted.
    #   tray          : Tray icon to show. A Phase 2 stub when omitted.
    #   adapter       : LLM adapter the heartbeat's autonomous turns call. Omitting it runs the agent with no
    #                   heartbeat, which is what every test that does not exercise autonomy wants.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def __init__ ( self,
                   configuration: SentinelConfig,
                   application: FastAPI | None                                   = None,
                   api_token: str | None                                         = None,
                   health_probe: Callable [ [ str ], Awaitable [ bool ] ] | None = None,
                   webui_launcher: Callable [ [], Awaitable [ Any ] ] | None     = None,
                   tray: TrayIcon | None                                         = None,
                   adapter: Any = None ) -> None:

        self.configuration = configuration
        self.application   = application
        self.api_token     = api_token
        self.adapter       = adapter
        self.state         = ProcessState.STOPPED

        self.components = { name: ComponentStatus ( name ) for name in ComponentName }

        self._health_probe   = health_probe if health_probe is not None else probe_url
        self._webui_launcher = webui_launcher if webui_launcher is not None else self._spawn_webui
        self._tray           = tray if tray is not None else TrayIcon ( title = configuration.agent.name )

        # Runtime handles, populated as components start.

        # The autonomous tick. Assembled in start() rather than here, because its database connection binds to the
        # event loop that opens it and there is no loop yet at construction. The gateway reaches it through a callable
        # for exactly the same reason -- see create_application's `heartbeat` argument.

        self.heartbeat: Any = None

        # The event logger, on the same reasoning: its connection binds to the loop that opens it. Opened before the
        # heartbeat so the tick can be handed the same instance, and closed after everything else in shutdown so the
        # system.shutdown event has somewhere to go.

        self.events: Any = None

        # What interactive turns run through, built alongside the event logger it holds. The gateway resolves it per
        # request through a callable, so the application can be constructed before the runtime exists.

        self.orchestrator: Any = None

        self._api_server:    Any                         = None
        self._api_task:      asyncio.Task [ Any ] | None = None
        self._webui_process: Any                         = None
        self._webui_watcher: asyncio.Task [ Any ] | None = None
        self._monitor_task:  asyncio.Task [ Any ] | None = None
        self._recovery_lock = asyncio.Lock ()

        # Cross-thread coordination. The main thread waits on these while the loop runs on a background thread.

        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready_event   = threading.Event ()
        self._stopped_event = threading.Event ()
        self._shutdown_requested: asyncio.Event | None = None

        # The one invariant the FSM exists to protect: nothing restarts once shutdown has begun.

        self.auto_restart_enabled = True

    #-------------------------------------------------------------------------------------------------------------------
    # State machine
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: dispatch
    #
    # Description:
    #
    #   Apply one event to the state machine.
    #
    #   The single place state changes. An event with no matching row in the transition table is dropped and logged --
    #   the table decides what is legal, not the caller.
    #
    # Arguments:
    #
    #   event     : The event to apply.
    #   component : The component it concerns, where the event has one.
    #
    # Returns:
    #
    #   The resulting state, which is the previous state when the event was not legal here.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def dispatch ( self, event: EventName, component: ComponentName | None = None ) -> ProcessState:

        context = GuardContext (
            component            = component,
            required             = component is not None and is_required ( component ),
            all_required_healthy = self.all_required_healthy (),
            retries_remaining    = self.retries_remaining ( component ),
            all_stopped          = self.all_stopped (),
            window_open          = self.components [ ComponentName.WINDOW ].healthy,
        )

        target = resolve_transition ( self.state, event, context )

        if target is None:
            logger.debug ( "Event %s is not legal in state %s; ignored.", event, self.state )

            return self.state

        if target != self.state:
            logger.info ( "State %s -> %s on %s%s.",
                          self.state, target, event,
                          f" ({component})" if component is not None else "" )

            self.state = target

            # Entering shutdown disables restarts before anything else happens. Doing this in dispatch rather than in
            # shutdown() closes the window in which a crash event arriving between the two could still trigger a
            # restart -- the exact race the FSM was introduced to remove.

            if target is ProcessState.SHUTTING_DOWN:
                self.auto_restart_enabled = False

            if target in ( ProcessState.RUNNING, ProcessState.STOPPED, ProcessState.FATAL ):
                self._signal_main_thread ()

        # Return data to caller.

        return self.state

    #-------------------------------------------------------------------------------------------------------------------
    # Function: all_required_healthy
    #
    # Description:
    #
    #   Report whether every required component is up.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   True when the agent API and Open WebUI are both healthy.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def all_required_healthy ( self ) -> bool:

        # Return data to caller.

        return all (
            status.healthy
            for name, status in self.components.items ()
            if is_required ( name ) and self._is_enabled ( name )
        )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: retries_remaining
    #
    # Description:
    #
    #   Report whether a component has recovery attempts left.
    #
    # Arguments:
    #
    #   component : The component to check, or None.
    #
    # Returns:
    #
    #   True when the component's failure count is still below process.max_retries.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def retries_remaining ( self, component: ComponentName | None ) -> bool:

        if component is None:
            return False

        # Return data to caller.

        return self.components [ component ].failures < self.configuration.process.max_retries

    #-------------------------------------------------------------------------------------------------------------------
    # Function: all_stopped
    #
    # Description:
    #
    #   Report whether every started component has stopped.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   True when nothing that was started is still healthy.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def all_stopped ( self ) -> bool:

        # Return data to caller.

        return not any ( status.healthy for status in self.components.values () )

    #-------------------------------------------------------------------------------------------------------------------
    # Startup
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: start
    #
    # Description:
    #
    #   Run the startup sequence (architecture 4.3.6).
    #
    #   Sequential by design: Open WebUI is started only after the gateway answers, so the interface never comes up
    #   pointed at an API that is not yet listening.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   True when every required component came up and the machine reached RUNNING. False leaves it in FATAL, with the
    #   partial startup already torn down by the caller's shutdown.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def start ( self ) -> bool:

        self._loop               = asyncio.get_running_loop ()
        self._shutdown_requested = asyncio.Event ()

        self.dispatch ( EventName.LAUNCH )

        # 0. The event log, before anything it might want to record. Not a tracked component either: it has no health
        #    endpoint, and an agent that runs without an audit trail is degraded rather than broken.

        await self._start_events ()

        # 1. Agent API.

        if not await self._start_agent_api ():
            self.dispatch ( EventName.COMPONENT_FAILED, ComponentName.AGENT_API )

            return False

        self.dispatch ( EventName.COMPONENT_READY, ComponentName.AGENT_API )

        # 1a. The heartbeat. Not a tracked component: it has no health endpoint and nothing to restart, and a failure
        #     to start it degrades autonomy rather than the application. The agent still answers when spoken to.

        await self._start_heartbeat ()

        # 2. Open WebUI.

        if self._is_enabled ( ComponentName.WEBUI ):

            if not await self._start_webui ():
                self.dispatch ( EventName.COMPONENT_FAILED, ComponentName.WEBUI )

                return False

            self.dispatch ( EventName.COMPONENT_READY, ComponentName.WEBUI )

        # 3. Tray. Optional -- a failure here is a warning, never a degraded system.

        self.components [ ComponentName.TRAY ].started = True
        self.components [ ComponentName.TRAY ].healthy = self._tray.start ()

        self.dispatch (
            EventName.COMPONENT_READY if self.components [ ComponentName.TRAY ].healthy
            else EventName.COMPONENT_FAILED,
            ComponentName.TRAY,
        )

        # The window is step 4 of the architecture's sequence but is launched by the main thread, not from here, so
        # RUNNING is reached without it. It is optional, so that is the correct reading of the transition table.

        if self.state is ProcessState.STARTING:
            self.dispatch ( EventName.COMPONENT_READY )

        # The first row of this run's audit trail, written once the outcome of startup is actually known. Emitted even
        # when the machine did not reach RUNNING, because "it tried to start and did not" is the more interesting of
        # the two records.

        await self.record_event (
            EVENT_SYSTEM_STARTUP,
            {
                "state": str ( self.state ),
                "version": __version__,
                "api": self.health_url (),
                "heartbeat": self.heartbeat is not None,
                "components": {
                    str ( name ): status.healthy for name, status in self.components.items ()
                },
            },
        )

        self._monitor_task = asyncio.create_task ( self._monitor () )

        # Return data to caller.

        return self.state is ProcessState.RUNNING

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _start_events
    #
    # Description:
    #
    #   Open the event logger this run records itself against.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   None. A failure is logged and swallowed, on the same reasoning as the heartbeat's: an agent with no audit trail
    #   is diminished, and one that refuses to start because it could not open a log file is unusable. Every emission
    #   site treats an absent logger as a no-op, so nothing downstream has to check.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def _start_events ( self ) -> None:

        from sentinel.core.orchestrator import AgentOrchestrator
        from sentinel.logging.logger    import open_event_logger

        try:
            self.events = await open_event_logger ( self.configuration )

        except SentinelError as error:
            logger.warning ( "The event log could not be opened: %s. The agent runs without an audit trail.", error )

            self.events = None

        # The orchestrator interactive turns go through, holding whichever logger came back -- including none, which is
        # a turn that runs and is not recorded rather than a turn that does not run.
        #
        # No session history store: an OpenAI-compatible client sends its whole conversation on every request, so the
        # gateway hands the history in and the store would never be read. The heartbeat's orchestrator is the one that
        # has no client and therefore does have one (see open_heartbeat).

        if self.adapter is not None:
            self.orchestrator = AgentOrchestrator (
                configuration = self.configuration,
                adapter       = self.adapter,
                events        = self.events,
            )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _stop_events
    #
    # Description:
    #
    #   Close the event logger and the connection it was given.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def _stop_events ( self ) -> None:

        if self.events is None:
            return

        events            = self.events
        self.events       = None
        self.orchestrator = None

        # Exporters first: an exporter holds a file handle, and flushing it is the part that matters most on a machine
        # that is about to lose power or be closed.

        events.close ()

        if events.connection is not None:

            try:
                await events.connection.close ()
            except sqlite3.Error as error:
                logger.warning ( "The event log's database connection did not close cleanly: %s", error )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: record_event
    #
    # Description:
    #
    #   Write one system event, if there is anywhere to write it.
    #
    # Arguments:
    #
    #   event : One of the system event names.
    #   data  : Event payload.
    #
    # Returns:
    #
    #   None. Absent logger, no-op; failing logger, a warning. Lifecycle transitions must not be blocked by the log
    #   that describes them.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def record_event ( self, event: str, data: dict [ str, Any ] | None = None ) -> None:

        if self.events is None:
            return

        try:
            await self.events.log (
                category = CATEGORY_SYSTEM,
                event    = event,
                data     = data,
                source   = "process",
            )

        except SentinelError as error:
            logger.warning ( "Could not record the %s event: %s", event, error )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _start_heartbeat
    #
    # Description:
    #
    #   Assemble and start the autonomous tick.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   None. A failure here is logged and swallowed rather than failing startup: it costs the agent its autonomy, not
    #   its ability to answer, and refusing to launch at all over a heartbeat that would not start would be a worse
    #   outcome than launching without one.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def _start_heartbeat ( self ) -> None:

        if not self.configuration.heartbeat.enabled or self.adapter is None:
            logger.debug ( "No heartbeat: it is disabled, or no LLM adapter was supplied." )

            return

        from sentinel.core.heartbeat import open_heartbeat

        try:
            self.heartbeat = await open_heartbeat ( self.configuration, self.adapter, events = self.events )

            await self.heartbeat.start ()

        except SentinelError as error:
            logger.warning ( "The heartbeat could not be started: %s. The agent runs without autonomy.", error )

            self.heartbeat = None

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _stop_heartbeat
    #
    # Description:
    #
    #   Stop the autonomous tick and close the connection it owns.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def _stop_heartbeat ( self ) -> None:

        if self.heartbeat is None:
            return

        self.heartbeat.stop ()

        try:
            await self.heartbeat.connection.close ()
        except sqlite3.Error as error:
            logger.warning ( "The heartbeat's database connection did not close cleanly: %s", error )

        self.heartbeat = None

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _start_agent_api
    #
    # Description:
    #
    #   Serve the FastAPI application and wait for it to answer.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   True once GET /health returns success within process.startup_timeout.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def _start_agent_api ( self ) -> bool:

        status         = self.components [ ComponentName.AGENT_API ]
        status.started = True

        if self.application is None:

            # No application to serve. Only a test reaches this, and it wants the rest of the machine.

            status.healthy      = True
            status.last_healthy = time.monotonic ()

            return True

        import uvicorn

        server_config = uvicorn.Config (
            self.application,
            host      = self.configuration.api.host,
            port      = self.configuration.api.port,
            log_level = self.configuration.logging.level.lower (),
        )

        self._api_server = uvicorn.Server ( server_config )

        # uvicorn installs SIGINT and SIGTERM handlers, which only the main thread may do. The manager runs on a
        # background thread, so the handlers are suppressed and shutdown is driven through the FSM instead.

        self._api_server.install_signal_handlers = lambda: None

        self._api_task = asyncio.create_task ( self._api_server.serve () )

        healthy = await self._wait_for_health ( self.health_url (), self.configuration.process.startup_timeout )

        status.healthy = healthy

        if healthy:
            status.last_healthy = time.monotonic ()
        else:
            logger.error ( "The agent API did not answer on %s within %d s.",
                           self.health_url (), self.configuration.process.startup_timeout )

        # Return data to caller.

        return healthy

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _start_webui
    #
    # Description:
    #
    #   Start the Open WebUI subprocess and wait for it to answer.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   True once the interface answers within process.startup_timeout.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def _start_webui ( self ) -> bool:

        status         = self.components [ ComponentName.WEBUI ]
        status.started = True

        try:
            self._webui_process = await self._webui_launcher ()
        except FileNotFoundError as error:

            # The commonest real failure: an install without the `ui` extra. Say which command fixes it rather than
            # reporting a bare OSError from deep inside asyncio.

            logger.error (
                "Open WebUI could not be started (%s). Install it with `pip install \"sentinel[ui]\"`, "
                "or set ui.webui_enabled to false in agent.yaml to run the API alone.", error
            )

            return False

        except OSError as error:
            logger.error ( "Open WebUI could not be started: %s", error )

            return False

        logger.info ( "Open WebUI starting on %s (pid %s).",
                      webui_support.webui_url ( self.configuration ), getattr ( self._webui_process, "pid", "?" ) )

        # Watch for an unexpected exit. This is what makes a killed subprocess detectable immediately rather than at
        # the next health poll.

        self._webui_watcher = asyncio.create_task ( self._watch_webui ( self._webui_process ) )

        healthy = await self._wait_for_health (
            webui_support.webui_url ( self.configuration ),
            self.configuration.process.startup_timeout,
            process = self._webui_process,
        )

        status.healthy = healthy

        if healthy:
            status.last_healthy = time.monotonic ()
        else:
            logger.error (
                "Open WebUI is not serving on %s. Its own output above is the authoritative reason; a non-zero "
                "exit code there means the interface failed, not that Sentinel timed out.",
                webui_support.webui_url ( self.configuration ),
            )

        # Return data to caller.

        return healthy

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _spawn_webui
    #
    # Description:
    #
    #   Launch the real Open WebUI subprocess.
    #
    #   The default launcher. Separated from _start_webui so a test can substitute a fake process without also
    #   replacing the readiness and restart logic that is the thing worth testing.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   The launched asyncio subprocess.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def _spawn_webui ( self ) -> asyncio.subprocess.Process:

        command     = webui_support.build_command ( self.configuration )
        environment = webui_support.build_environment ( self.configuration, api_token = self.api_token )

        self.configuration.webui_directory.mkdir ( parents = True, exist_ok = True )

        # Started IN the data directory, not merely pointed at it.
        #
        # Open WebUI writes .webui_secret_key to Path.cwd() -- not to DATA_DIR -- so the working directory decides
        # where a credential lands. Inheriting Sentinel's own cwd puts it wherever the user happened to launch from:
        # a developer's repository (where it is one `git add -A` from being committed), or, for the installed
        # product, a directory such as Program Files that may not be writable at all.

        # Return data to caller.

        return await asyncio.create_subprocess_exec (
            *command,
            env = environment,
            cwd = str ( self.configuration.webui_directory ),
        )

    #-------------------------------------------------------------------------------------------------------------------
    # Recovery
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _watch_webui
    #
    # Description:
    #
    #   Wait for the Open WebUI subprocess to exit and react to it.
    #
    # Arguments:
    #
    #   process : The process to watch.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def _watch_webui ( self, process: Any ) -> None:

        with contextlib.suppress ( asyncio.CancelledError ):

            code = await process.wait ()

            # An exit during shutdown is the shutdown working. Anywhere else it is a crash.

            if not self.auto_restart_enabled or self.state is ProcessState.SHUTTING_DOWN:
                logger.debug ( "Open WebUI exited with code %s during shutdown.", code )

                return

            logger.warning ( "Open WebUI exited unexpectedly with code %s.", code )

            self.components [ ComponentName.WEBUI ].healthy = False

            self.dispatch ( EventName.COMPONENT_CRASHED, ComponentName.WEBUI )

            await self.recover ( ComponentName.WEBUI )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: recover
    #
    # Description:
    #
    #   Attempt to bring one crashed component back (architecture 4.3.5).
    #
    #   Serialised behind a lock: two components crashing seconds apart would otherwise run two recoveries at once and
    #   could restart the interface against an API that is still down.
    #
    # Arguments:
    #
    #   component : The component to recover.
    #
    # Returns:
    #
    #   True when the component is healthy again.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def recover ( self, component: ComponentName ) -> bool:

        # Guard 1: recovery happens only in DEGRADED. In SHUTTING_DOWN this is the restart-during-shutdown race, and
        # returning here is what prevents the zombie.

        if self.state is not ProcessState.DEGRADED or not self.auto_restart_enabled:
            logger.debug ( "Not recovering %s in state %s.", component, self.state )

            return False

        async with self._recovery_lock:

            status = self.components [ component ]

            # Guard 2: the retry budget.

            if status.failures >= self.configuration.process.max_retries:
                logger.error ( "%s failed %d times; giving up.", component, status.failures )

                self.dispatch ( EventName.RECOVERY_EXHAUSTED, component )

                return False

            status.failures += 1

            logger.warning ( "Recovering %s (attempt %d of %d).",
                             component, status.failures, self.configuration.process.max_retries )

            recovered = False

            if component is ComponentName.WEBUI:
                recovered = await self._start_webui ()

            if recovered:
                logger.info ( "%s recovered.", component )

                self.dispatch ( EventName.COMPONENT_RECOVERED, component )
            else:
                logger.error ( "%s did not come back.", component )

                self.dispatch ( EventName.COMPONENT_CRASHED, component )

            # Return data to caller.

            return recovered

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _monitor
    #
    # Description:
    #
    #   Poll component health for as long as the system runs.
    #
    #   The subprocess watcher catches an outright exit; this catches the other case -- a process that is still alive
    #   but has stopped answering. It also ages out the failure counter, so a component that has been healthy for
    #   process.retry_window gets its full retry budget back.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def _monitor ( self ) -> None:

        interval = float ( self.configuration.process.health_check_interval )

        with contextlib.suppress ( asyncio.CancelledError ):

            while self._is_serving ():

                await asyncio.sleep ( interval )

                if not self._is_serving ():
                    break

                await self._check_component ( ComponentName.AGENT_API, self.health_url () )

                if self._is_enabled ( ComponentName.WEBUI ):
                    await self._check_component (
                        ComponentName.WEBUI, webui_support.webui_url ( self.configuration )
                    )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _is_serving
    #
    # Description:
    #
    #   Report whether the system is in a state that should still be health-monitored.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   True in RUNNING and DEGRADED. Read through a method rather than inline so the monitor's loop condition is
    #   re-evaluated after every await -- the state can and does change while the loop is sleeping.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def _is_serving ( self ) -> bool:

        # Return data to caller.

        return self.state in ( ProcessState.RUNNING, ProcessState.DEGRADED )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _check_component
    #
    # Description:
    #
    #   Probe one component and feed the result into the machine.
    #
    # Arguments:
    #
    #   component : The component to probe.
    #   url       : Its health URL.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def _check_component ( self, component: ComponentName, url: str ) -> None:

        status  = self.components [ component ]
        healthy = await self._health_probe ( url )

        if healthy:

            # Sustained health earns the retry budget back, so an install that crashes once a day does not eventually
            # exhaust a counter that was never reset.

            if status.failures and time.monotonic () - status.last_healthy >= self.configuration.process.retry_window:
                logger.info ( "%s has been healthy for %d s; resetting its failure count.",
                              component, self.configuration.process.retry_window )

                status.failures = 0

            was_unhealthy  = not status.healthy
            status.healthy = True

            if not status.failures or not was_unhealthy:
                status.last_healthy = time.monotonic ()

            if was_unhealthy:
                self.dispatch ( EventName.COMPONENT_RECOVERED, component )

                await self.record_event (
                    EVENT_SYSTEM_HEALTH,
                    { "component": str ( component ), "healthy": True, "url": url },
                )

            return

        if not status.healthy:
            return

        logger.warning ( "%s stopped answering on %s.", component, url )

        status.healthy = False

        # Only the transition is recorded, never the poll. At a ten-second interval a row per probe would be 8,640
        # rows a day saying nothing changed, which would bury the handful of rows that say something did.

        await self.record_event (
            EVENT_SYSTEM_HEALTH,
            { "component": str ( component ), "healthy": False, "url": url, "failures": status.failures },
        )

        self.dispatch ( EventName.COMPONENT_CRASHED, component )

        await self.recover ( component )

    #-------------------------------------------------------------------------------------------------------------------
    # Shutdown
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: shutdown
    #
    # Description:
    #
    #   Run the shutdown sequence (architecture 4.3.7).
    #
    #   Terminate, then wait, then kill. The wait is what makes it graceful: Open WebUI flushes its own SQLite database
    #   on a clean exit, and killing it outright risks the user's chat history.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def shutdown ( self ) -> None:

        if self.state is ProcessState.STOPPED:
            return

        self.dispatch ( EventName.SHUTDOWN )

        # 1. Stop restarting anything. dispatch has already done this; repeated here because the invariant matters more
        #    than the duplication, and a future caller may reach shutdown by another path.

        self.auto_restart_enabled = False

        # Record the intent to stop before anything is torn down, rather than the fact of having stopped afterwards. A
        # teardown that hangs on a subprocess that will not die leaves this row behind either way, and a shutdown with
        # no closing record is exactly the case worth being able to see.

        await self.record_event (
            EVENT_SYSTEM_SHUTDOWN,
            {
                "components": {
                    str ( name ): status.healthy for name, status in self.components.items ()
                },
            },
        )

        # 2. Stop the health monitor before tearing components down, so its probes cannot race the teardown.

        await cancel_task ( self._monitor_task )

        self._monitor_task = None

        # 3. Open WebUI: terminate, wait, kill.

        await self._stop_webui ()

        # 4. The heartbeat, before the API it shares a process with. A tick in flight is left to finish; what stops
        #    here is the schedule that would start another one.

        await self._stop_heartbeat ()

        # 5. The agent API.

        await self._stop_agent_api ()

        # 5a. The event log, after everything that could still want to write to it and before the tray, which cannot.

        await self._stop_events ()

        # 6. The tray.

        self._tray.stop ()

        self.components [ ComponentName.TRAY ].healthy   = False
        self.components [ ComponentName.WINDOW ].healthy = False

        for name in ComponentName:
            self.dispatch ( EventName.COMPONENT_STOPPED, name )

        logger.info ( "Shutdown complete." )

        self._stopped_event.set ()

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _stop_webui
    #
    # Description:
    #
    #   Terminate the Open WebUI subprocess, killing it if it will not go.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def _stop_webui ( self ) -> None:

        await cancel_task ( self._webui_watcher )

        self._webui_watcher = None

        process = self._webui_process

        if process is None:
            return

        self.components [ ComponentName.WEBUI ].healthy = False

        if process.returncode is not None:
            self._webui_process = None

            return

        logger.info ( "Stopping Open WebUI (pid %s).", getattr ( process, "pid", "?" ) )

        with contextlib.suppress ( ProcessLookupError, OSError ):
            process.terminate ()

        try:
            await asyncio.wait_for ( process.wait (), timeout = self.configuration.process.shutdown_timeout )

        except TimeoutError:
            logger.warning ( "Open WebUI ignored the terminate request after %d s; killing it.",
                             self.configuration.process.shutdown_timeout )

            with contextlib.suppress ( ProcessLookupError, OSError ):
                process.kill ()

            with contextlib.suppress ( TimeoutError ):
                await asyncio.wait_for ( process.wait (), timeout = self.configuration.process.shutdown_timeout )

        self._webui_process = None

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _stop_agent_api
    #
    # Description:
    #
    #   Ask uvicorn to stop serving and wait for the task to finish.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def _stop_agent_api ( self ) -> None:

        self.components [ ComponentName.AGENT_API ].healthy = False

        if self._api_server is not None:
            self._api_server.should_exit = True

        if self._api_task is not None:

            try:
                await asyncio.wait_for (
                    asyncio.shield ( self._api_task ),
                    timeout = self.configuration.process.shutdown_timeout,
                )

            except TimeoutError:
                logger.warning ( "The agent API did not stop in time; cancelling it." )

                await cancel_task ( self._api_task )

        self._api_task   = None
        self._api_server = None

    #-------------------------------------------------------------------------------------------------------------------
    # Function: run_until_shutdown
    #
    # Description:
    #
    #   Start every component, then serve until shutdown is requested.
    #
    #   The whole lifetime of the agent, as one awaitable. Intended to be handed to asyncio.run on a background thread
    #   while the main thread shows the window.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   True when startup succeeded and shutdown was clean. False when startup failed, in which case whatever did start
    #   has still been torn down.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def run_until_shutdown ( self ) -> bool:

        started = await self.start ()

        if started:

            # start() guarantees the event exists, but mypy cannot see that through the branch.

            if self._shutdown_requested is not None:
                await self._shutdown_requested.wait ()

        else:
            logger.error ( "Startup failed in state %s. Tearing down what did start.", self.state )

        await self.shutdown ()

        # Release any main thread still waiting for a readiness that will never arrive.

        self._ready_event.set ()
        self._stopped_event.set ()

        # Return data to caller.

        return started

    #-------------------------------------------------------------------------------------------------------------------
    # Cross-thread entry points
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: request_shutdown
    #
    # Description:
    #
    #   Ask the manager to shut down. Safe to call from any thread.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def request_shutdown ( self ) -> None:

        event = self._shutdown_requested

        if event is None or self._loop is None:

            # Nothing has started yet, so there is nothing to unwind.

            self._stopped_event.set ()

            return

        self._loop.call_soon_threadsafe ( event.set )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: notify_window_closed
    #
    # Description:
    #
    #   Record that the user closed the native window.
    #
    #   PHASE 2 BEHAVIOUR, DELIBERATELY REVERSED IN PHASE 9. With no tray icon there is nowhere to minimise to, so a
    #   closed window with a still-running agent would be an agent the user cannot reach or stop. Once the tray exists,
    #   this stops requesting shutdown and the FSM's RUNNING/window_open=false path -- which is already implemented and
    #   already tested -- becomes the live one.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def notify_window_closed ( self ) -> None:

        self.components [ ComponentName.WINDOW ].healthy = False

        if self._loop is not None:
            self._loop.call_soon_threadsafe ( self.dispatch, EventName.WINDOW_CLOSED, ComponentName.WINDOW )
        else:
            self.dispatch ( EventName.WINDOW_CLOSED, ComponentName.WINDOW )

        logger.info ( "Window closed. Shutting down (Phase 2 behaviour; Phase 9 minimises to the tray instead)." )

        self.request_shutdown ()

    #-------------------------------------------------------------------------------------------------------------------
    # Function: notify_window_opened
    #
    # Description:
    #
    #   Record that the native window is on screen.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def notify_window_opened ( self ) -> None:

        self.components [ ComponentName.WINDOW ].started = True
        self.components [ ComponentName.WINDOW ].healthy = True

    #-------------------------------------------------------------------------------------------------------------------
    # Function: wait_until_ready
    #
    # Description:
    #
    #   Block the calling thread until the system is RUNNING or has given up.
    #
    # Arguments:
    #
    #   timeout : Seconds to wait. None waits indefinitely.
    #
    # Returns:
    #
    #   True when the machine reached RUNNING.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def wait_until_ready ( self, timeout: float | None = None ) -> bool:

        self._ready_event.wait ( timeout )

        # Return data to caller.

        return self.state is ProcessState.RUNNING

    #-------------------------------------------------------------------------------------------------------------------
    # Function: wait_until_stopped
    #
    # Description:
    #
    #   Block the calling thread until shutdown has completed.
    #
    #   Waits in short slices rather than in one call. On Windows, `threading.Event.wait()` with no timeout cannot be
    #   interrupted by Ctrl+C -- the thread sits in a native wait, the interpreter never regains control, and the
    #   signal handler never runs. This is the main thread in a headless run (no window to block in), so an
    #   uninterruptible wait here means Ctrl+C does nothing at all and the only way to stop the agent is
    #   `sentinel stop` from another terminal, or killing the process.
    #
    # Arguments:
    #
    #   timeout : Seconds to wait. None waits indefinitely.
    #
    # Returns:
    #
    #   True when the system is stopped.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def wait_until_stopped ( self, timeout: float | None = None ) -> bool:

        deadline = None if timeout is None else time.monotonic () + timeout

        while not self._stopped_event.is_set ():

            if deadline is None:
                self._stopped_event.wait ( SHUTDOWN_POLL_INTERVAL )

                continue

            remaining = deadline - time.monotonic ()

            if remaining <= 0.0:
                break

            self._stopped_event.wait ( min ( SHUTDOWN_POLL_INTERVAL, remaining ) )

        # Return data to caller.

        return self._stopped_event.is_set ()

    #-------------------------------------------------------------------------------------------------------------------
    # Helpers
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: health_url
    #
    # Description:
    #
    #   The gateway's health endpoint.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   The URL the readiness probe polls. Unauthenticated by design, so the probe cannot be defeated by a missing
    #   credential.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def health_url ( self ) -> str:

        # Return data to caller.

        return f"http://{self.configuration.api.host}:{self.configuration.api.port}/health"

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _is_enabled
    #
    # Description:
    #
    #   Report whether a component is configured to run at all.
    #
    # Arguments:
    #
    #   component : The component to check.
    #
    # Returns:
    #
    #   False for Open WebUI when ui.webui_enabled is off -- a headless run of the API alone, which is what a developer
    #   or a clean-room verification wants. True otherwise.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def _is_enabled ( self, component: ComponentName ) -> bool:

        if component is ComponentName.WEBUI:
            return self.configuration.ui.webui_enabled

        # Return data to caller.

        return True

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _wait_for_health
    #
    # Description:
    #
    #   Poll a URL until it answers or the timeout expires.
    #
    # Arguments:
    #
    #   url      : The URL to poll.
    #   timeout  : Seconds to keep trying.
    #   process  : Subprocess whose death ends the wait early, or None when there is no process behind the URL.
    #
    # Returns:
    #
    #   True when the URL answered in time.
    #
    #   A process that has already exited ends the wait at once. Without that, a component that dies on its first line
    #   -- a missing module, a bad argument, a port clash -- is reported sixty seconds later as "did not answer in
    #   time", which points at the timeout rather than at the exit code that actually explains it.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def _wait_for_health ( self, url: str, timeout: float, process: Any = None ) -> bool:

        deadline = time.monotonic () + timeout

        while time.monotonic () < deadline:

            if await self._health_probe ( url ):
                return True

            if process is not None and process.returncode is not None:
                logger.error ( "The process behind %s exited with code %s before it answered.",
                               url, process.returncode )

                return False

            await asyncio.sleep ( STARTUP_POLL_INTERVAL )

        # Return data to caller.

        return False

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _signal_main_thread
    #
    # Description:
    #
    #   Release a main thread waiting on readiness.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def _signal_main_thread ( self ) -> None:

        self._ready_event.set ()

        if self.state is ProcessState.STOPPED:
            self._stopped_event.set ()


#-----------------------------------------------------------------------------------------------------------------------
# Module helpers
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Function: probe_url
#
# Description:
#
#   Ask a URL whether it is alive.
#
# Arguments:
#
#   url     : The URL to probe.
#   timeout : Seconds to wait for an answer.
#
# Returns:
#
#   True for any response below 500. Open WebUI redirects its root, and a 3xx from a server that is up should not read
#   as a server that is down; a 5xx is a genuine failure and does.
#
#-----------------------------------------------------------------------------------------------------------------------

async def probe_url ( url: str, timeout: float = 2.0 ) -> bool:

    try:
        async with httpx.AsyncClient ( timeout = timeout ) as client:

            response = await client.get ( url )

            return response.status_code < 500

    except ( httpx.HTTPError, OSError ):

        # Return data to caller.

        return False


#-----------------------------------------------------------------------------------------------------------------------
# Function: cancel_task
#
# Description:
#
#   Cancel a task and wait for it to finish.
#
# Arguments:
#
#   task : The task to cancel, or None.
#
# Returns:
#
#   None.
#
#-----------------------------------------------------------------------------------------------------------------------

async def cancel_task ( task: asyncio.Task [ Any ] | None ) -> None:

    if task is None or task.done ():
        return

    task.cancel ()

    with contextlib.suppress ( asyncio.CancelledError, Exception ):
        await task
