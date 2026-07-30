#-----------------------------------------------------------------------------------------------------------------------
# Module:  states.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   Process lifecycle states, events, and the transition table (architecture 4.3).
#
#   The table is data rather than a nest of conditionals for one reason: it is the specification. A transition the
#   architecture document lists and the code omits is then a missing row, visible by reading, instead of a branch nobody
#   wrote. The three bug classes the FSM exists to prevent -- restart-during-shutdown, cascading recovery, and window
#   close mistaken for failure -- are each a single row here.
#
#   Guards are named rather than inlined so the table stays declarative. ProcessManager supplies the context they read;
#   nothing in this module knows what a subprocess is.
#-----------------------------------------------------------------------------------------------------------------------

from collections.abc import Callable
from dataclasses     import dataclass
from enum            import StrEnum


#-----------------------------------------------------------------------------------------------------------------------
# Class: ProcessState
#
# Description:
#
#   The six lifecycle states (architecture 4.3.2).
#
#   SHUTTING_DOWN is the one state with a hard behavioural rule attached: no component may be restarted while in it.
#-----------------------------------------------------------------------------------------------------------------------

class ProcessState ( StrEnum ):

    STOPPED       = "stopped"
    STARTING      = "starting"
    RUNNING       = "running"
    DEGRADED      = "degraded"
    SHUTTING_DOWN = "shutting_down"
    FATAL         = "fatal"


#-----------------------------------------------------------------------------------------------------------------------
# Class: ComponentName
#
# Description:
#
#   The four managed components (architecture 4.3.3).
#
#   Criticality is not a property of the component itself but of the deployment, so it lives in REQUIRED_COMPONENTS
#   rather than on the enum.
#-----------------------------------------------------------------------------------------------------------------------

class ComponentName ( StrEnum ):

    AGENT_API = "agent_api"
    WEBUI     = "webui"
    TRAY      = "tray"
    WINDOW    = "window"


#-----------------------------------------------------------------------------------------------------------------------
# Class: EventName
#
# Description:
#
#   Every event the FSM accepts (architecture 4.3.4).
#-----------------------------------------------------------------------------------------------------------------------

class EventName ( StrEnum ):

    LAUNCH              = "Launch"
    COMPONENT_READY     = "ComponentReady"
    COMPONENT_FAILED    = "ComponentFailed"
    COMPONENT_CRASHED   = "ComponentCrashed"
    COMPONENT_RECOVERED = "ComponentRecovered"
    COMPONENT_STOPPED   = "ComponentStopped"
    RECOVERY_EXHAUSTED  = "RecoveryExhausted"
    WINDOW_CLOSED       = "WindowClosed"
    OPEN_WINDOW         = "OpenWindow"
    SHUTDOWN            = "Shutdown"

# Failure of a required component degrades the system; failure of an optional one is logged and tolerated. The window
# is deliberately optional: closing it is a user action, and Phase 9 makes that explicit by minimising to the tray.

REQUIRED_COMPONENTS = frozenset ( { ComponentName.AGENT_API, ComponentName.WEBUI } )


#-----------------------------------------------------------------------------------------------------------------------
# Class: GuardContext
#
# Description:
#
#   Everything a transition guard is allowed to read.
#
#   Passed as a value object rather than letting guards reach into ProcessManager: a guard that can see the manager can
#   also change it, and a transition table with side effects is no longer a table.
#
# Attributes:
#
#   component            : The component the event concerns, or None for whole-system events such as Shutdown.
#   required             : Whether that component is required.
#   all_required_healthy : Whether every required component is currently healthy.
#   retries_remaining    : Whether the component has recovery attempts left.
#   all_stopped          : Whether every started component has now stopped.
#   window_open          : Whether the native window is currently open.
#-----------------------------------------------------------------------------------------------------------------------

@dataclass ( frozen = True )
class GuardContext:

    component:            ComponentName | None = None
    required:             bool                 = False
    all_required_healthy: bool                 = False
    retries_remaining:    bool                 = False
    all_stopped:          bool                 = False
    window_open:          bool                 = False

# Guards, by name. Each answers one question about the context and nothing else.

GUARDS: dict [ str, Callable [ [ GuardContext ], bool ] ] = {
    "required": lambda context: context.required,
    "optional": lambda context: not context.required,
    "all_required_healthy": lambda context: context.all_required_healthy,
    "retries_remaining": lambda context: context.retries_remaining,
    "all_stopped": lambda context: context.all_stopped,
    "window_closed": lambda context: not context.window_open,
}


#-----------------------------------------------------------------------------------------------------------------------
# Class: Transition
#
# Description:
#
#   One row of the transition table.
#
# Attributes:
#
#   source : State the transition leaves.
#   event  : Event that triggers it.
#   guard  : Name of the guard that must pass, or None for an unconditional row.
#   target : State the transition enters.
#-----------------------------------------------------------------------------------------------------------------------

@dataclass ( frozen = True )
class Transition:

    source: ProcessState
    event:  EventName
    guard:  str | None
    target: ProcessState

# The transition table (architecture 4.3.4).
#
# Order is significant: the first matching row wins, so a guarded row must precede the unguarded fallback for the same
# (state, event) pair. Every pair that can occur has a fallback, because an event with no matching row is dropped, and a
# silently dropped Shutdown is exactly the failure this table exists to prevent.

# Aliases used only by the table below. The rows are far easier to compare when each fits on one line, and
# ProcessState/EventName spelled out in every cell costs more width than the clarity is worth here.

State = ProcessState
Event = EventName


TRANSITIONS: tuple [ Transition, ... ] = (

    # STOPPED.

    Transition ( State.STOPPED, Event.LAUNCH, None, State.STARTING ),

    # STARTING. Components come up in sequence; the last required one to report ready promotes the system to RUNNING.

    Transition ( State.STARTING, Event.COMPONENT_READY, "all_required_healthy", State.RUNNING ),
    Transition ( State.STARTING, Event.COMPONENT_READY, None, State.STARTING ),
    Transition ( State.STARTING, Event.COMPONENT_FAILED, "required", State.FATAL ),
    Transition ( State.STARTING, Event.COMPONENT_FAILED, None, State.STARTING ),
    Transition ( State.STARTING, Event.WINDOW_CLOSED, None, State.STARTING ),
    Transition ( State.STARTING, Event.SHUTDOWN, None, State.SHUTTING_DOWN ),

    # RUNNING. WindowClosed is not a failure -- it stays here, and Phase 9 keeps the agent alive behind the tray.

    Transition ( State.RUNNING, Event.COMPONENT_CRASHED, "required", State.DEGRADED ),
    Transition ( State.RUNNING, Event.COMPONENT_CRASHED, None, State.RUNNING ),
    Transition ( State.RUNNING, Event.WINDOW_CLOSED, None, State.RUNNING ),
    Transition ( State.RUNNING, Event.OPEN_WINDOW, "window_closed", State.RUNNING ),
    Transition ( State.RUNNING, Event.COMPONENT_READY, None, State.RUNNING ),
    Transition ( State.RUNNING, Event.SHUTDOWN, None, State.SHUTTING_DOWN ),

    # DEGRADED. Recovery is serialised here, one component at a time, until either everything is healthy again or the
    # retry budget runs out.

    Transition ( State.DEGRADED, Event.COMPONENT_RECOVERED, "all_required_healthy", State.RUNNING ),
    Transition ( State.DEGRADED, Event.COMPONENT_RECOVERED, None, State.DEGRADED ),
    Transition ( State.DEGRADED, Event.COMPONENT_CRASHED, "retries_remaining", State.DEGRADED ),
    Transition ( State.DEGRADED, Event.COMPONENT_CRASHED, None, State.FATAL ),
    Transition ( State.DEGRADED, Event.RECOVERY_EXHAUSTED, None, State.FATAL ),
    Transition ( State.DEGRADED, Event.WINDOW_CLOSED, None, State.DEGRADED ),
    Transition ( State.DEGRADED, Event.SHUTDOWN, None, State.SHUTTING_DOWN ),

    # SHUTTING_DOWN. The critical guard: a crash here is ignored rather than recovered, because restarting a component
    # while tearing the system down is how zombie processes are made.

    Transition ( State.SHUTTING_DOWN, Event.COMPONENT_STOPPED, "all_stopped", State.STOPPED ),
    Transition ( State.SHUTTING_DOWN, Event.COMPONENT_STOPPED, None, State.SHUTTING_DOWN ),
    Transition ( State.SHUTTING_DOWN, Event.COMPONENT_CRASHED, None, State.SHUTTING_DOWN ),
    Transition ( State.SHUTTING_DOWN, Event.WINDOW_CLOSED, None, State.SHUTTING_DOWN ),
    Transition ( State.SHUTTING_DOWN, Event.SHUTDOWN, None, State.SHUTTING_DOWN ),

    # FATAL. Terminal until the user intervenes, either by quitting or by launching again.

    Transition ( State.FATAL, Event.SHUTDOWN, None, State.SHUTTING_DOWN ),
    Transition ( State.FATAL, Event.LAUNCH, None, State.STARTING ),
)


#-----------------------------------------------------------------------------------------------------------------------
# Function: resolve_transition
#
# Description:
#
#   Find the state an event moves the machine to.
#
# Arguments:
#
#   state   : Current state.
#   event   : The event being applied.
#   context : Values the guards may read.
#
# Returns:
#
#   The target state, or None when no row matches -- which means the event is not legal in this state and must be
#   dropped rather than acted on.
#
#-----------------------------------------------------------------------------------------------------------------------

def resolve_transition ( state: ProcessState,
                         event: EventName,
                         context: GuardContext ) -> ProcessState | None:

    # First matching row wins, so guarded rows are listed before their fallbacks.

    for transition in TRANSITIONS:

        if transition.source != state or transition.event != event:
            continue

        if transition.guard is None or GUARDS [ transition.guard ] ( context ):
            return transition.target

    # Return data to caller.

    return None


#-----------------------------------------------------------------------------------------------------------------------
# Function: is_required
#
# Description:
#
#   Report whether a component's failure should degrade the system.
#
# Arguments:
#
#   component : The component to classify.
#
# Returns:
#
#   True for the agent API and Open WebUI, False for the tray and the native window.
#
#-----------------------------------------------------------------------------------------------------------------------

def is_required ( component: ComponentName ) -> bool:

    # Return data to caller.

    return component in REQUIRED_COMPONENTS
