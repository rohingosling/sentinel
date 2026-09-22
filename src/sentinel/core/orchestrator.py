#-----------------------------------------------------------------------------------------------------------------------
# Module:  orchestrator.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   The agent orchestrator (architecture 3.2.1).
#
#   One entry point for every event that can start an agent turn -- a user message, a channel message, a heartbeat
#   tick, a system event -- responsible for the session history around the turn, while the agentic loop (3.3) owns
#   everything inside it.
#
#   Session history is the working-memory tier from Phase 3 rather than a store of its own. That is the whole reason
#   Tier 1 is session-scoped and TTL-evicted: an abandoned conversation expires on its own instead of accumulating for
#   ever in a table nobody prunes.
#
#   Deliberately thin, and it stays thin. Steps 3 and 5 of the algorithm in 3.2.1 name an EventLogger, which Phase 5
#   supplies: the orchestrator hands it to the loop, and the loop -- which is where the events actually happen -- emits
#   them under one correlation identifier per turn.
#
#   There is one entry point per shape of caller and no more. process() runs a turn to completion, which is what a
#   heartbeat tick wants; process_stream() runs the same turn and forwards its progress, which is what the API gateway
#   wants. Both read and write the same session history around the same loop. Until Phase 5 the gateway went straight
#   to AgenticLoop because only process() existed and it could not stream, which meant interactive turns quietly had
#   no session handling at all -- two paths into the loop, one of them missing a step.
#-----------------------------------------------------------------------------------------------------------------------

import logging

from collections.abc import AsyncIterator
from typing          import Any

from sentinel.config         import SentinelConfig
from sentinel.core.loop      import AgenticLoop, LoopEvent, LoopResult, Trigger
from sentinel.llm.adapter    import LlmAdapter
from sentinel.logging.logger import EventLogger
from sentinel.memory.working import DEFAULT_SESSION, WorkingMemory

logger = logging.getLogger ( __name__ )

# Heartbeat turns are kept out of the interactive session on purpose. Autonomous work threading itself into whatever
# the user last said would let a tick's tool chatter reappear as context in their next message.

HEARTBEAT_SESSION = "heartbeat"


#-----------------------------------------------------------------------------------------------------------------------
# Class: AgentOrchestrator
#
# Description:
#
#   Turns an event into a completed agent turn.
#
# Attributes:
#
#   configuration : The loaded configuration.
#   adapter       : The LLM adapter the loop should call.
#   working       : Session history store, or None to run every turn without history.
#   events        : Event logger handed to the loop, or None to run without an audit trail.
#-----------------------------------------------------------------------------------------------------------------------

class AgentOrchestrator:

    #-------------------------------------------------------------------------------------------------------------------
    # Function: __init__
    #
    # Description:
    #
    #   Construct the orchestrator.
    #
    # Arguments:
    #
    #   configuration : The loaded configuration.
    #   adapter       : The LLM adapter the loop should call.
    #   working       : Session history store. Omitting it makes every turn stateless, which is what a test wants and
    #                   what a first run before the cache directory exists gets.
    #   events        : Event logger the turn records itself against. Passed straight through to the loop, which is
    #                   where the events happen; the orchestrator emits none of its own.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def __init__ ( self,
                   configuration: SentinelConfig,
                   adapter: LlmAdapter,
                   working: WorkingMemory | None = None,
                   events: EventLogger | None = None ) -> None:

        # Record the collaborators and build the loop once -- it is stateless between turns.

        self.configuration = configuration
        self.adapter       = adapter
        self.working       = working
        self.events        = events
        self.loop          = AgenticLoop ( configuration, adapter, events = events )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: process
    #
    # Description:
    #
    #   Run one event to a completed turn (architecture 3.2.1).
    #
    # Arguments:
    #
    #   trigger    : What started the turn.
    #   session_id : Session whose history the turn continues.
    #   history    : Conversation to continue instead of the session store's. See resolve_history.
    #
    # Returns:
    #
    #   The turn outcome. A guard firing produces a result carrying an error description rather than an exception, as
    #   in the loop itself -- the caller is a heartbeat tick that must go on to the next task either way.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def process ( self,
                        trigger: Trigger,
                        session_id: str = DEFAULT_SESSION,
                        history: list [ dict [ str, Any ] ] | None = None ) -> LoopResult:

        history = self.resolve_history ( session_id, history )

        logger.debug (
            "Processing a %s trigger in session %r with %d prior message(s).",
            trigger.kind, session_id, len ( history ),
        )

        result = await self.loop.run ( trigger, history )

        self.write_history ( session_id, result.history )

        # Return data to caller.

        return result

    #-------------------------------------------------------------------------------------------------------------------
    # Function: process_stream
    #
    # Description:
    #
    #   Run one event to a completed turn, forwarding progress as it goes (architecture 3.2.1 over 3.3.8).
    #
    #   The same turn process() runs, in the shape a streaming client needs. Session history is read before the first
    #   event and written after the terminal one, so a streamed turn is remembered exactly as a completed one is --
    #   which is the whole reason the gateway now comes through here instead of reaching past to AgenticLoop.
    #
    # Arguments:
    #
    #   trigger    : What started the turn.
    #   session_id : Session whose history the turn continues.
    #   history    : Conversation to continue instead of the session store's. See resolve_history.
    #
    # Returns:
    #
    #   The loop's own events, unaltered, ending in exactly one terminal "result". History is written from that event;
    #   a stream abandoned before it arrives -- a client that disconnected -- writes nothing, which is correct: there
    #   is no completed turn to remember.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def process_stream ( self,
                               trigger: Trigger,
                               session_id: str = DEFAULT_SESSION,
                               history: list [ dict [ str, Any ] ] | None = None ) -> AsyncIterator [ LoopEvent ]:

        history = self.resolve_history ( session_id, history )

        logger.debug (
            "Streaming a %s trigger in session %r with %d prior message(s).",
            trigger.kind, session_id, len ( history ),
        )

        async for event in self.loop.run_stream ( trigger, history ):

            if event.type == "result" and event.result is not None:
                self.write_history ( session_id, event.result.history )

            yield event

    #-------------------------------------------------------------------------------------------------------------------
    # Session history
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: resolve_history
    #
    # Description:
    #
    #   Decide which conversation a turn continues.
    #
    #   Two kinds of caller need opposite answers, and the difference is who holds the transcript. An OpenAI-compatible
    #   client sends its whole conversation on every request -- that is what makes a stateless-looking API work against
    #   a stateful agent -- so its history must be used as given; reading the session store as well would replay every
    #   message twice. A heartbeat tick has no client and no transcript, so the session store is the only record there
    #   is.
    #
    # Arguments:
    #
    #   session_id : Session to fall back to.
    #   supplied   : History the caller provided, or None to use the session store. An explicitly empty list means "a
    #                new conversation" and is honoured as such, which is why the test is against None and not falsiness.
    #
    # Returns:
    #
    #   The conversation the turn should continue.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def resolve_history ( self,
                          session_id: str,
                          supplied: list [ dict [ str, Any ] ] | None ) -> list [ dict [ str, Any ] ]:

        if supplied is not None:
            return list ( supplied )

        # Return data to caller.

        return self.read_history ( session_id )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: read_history
    #
    # Description:
    #
    #   Read a session's conversation history.
    #
    # Arguments:
    #
    #   session_id : The session to read.
    #
    # Returns:
    #
    #   The stored messages, or an empty list when there is no history store or the session is new. A cache read that
    #   fails is treated as an empty history rather than an error: losing context degrades a turn, where refusing to
    #   run it loses the turn entirely.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def read_history ( self, session_id: str ) -> list [ dict [ str, Any ] ]:

        if self.working is None:
            return []

        # Return data to caller.

        return self.working.for_session ( session_id ).get_conversation ()

    #-------------------------------------------------------------------------------------------------------------------
    # Function: write_history
    #
    # Description:
    #
    #   Store a session's conversation history.
    #
    # Arguments:
    #
    #   session_id : The session to write.
    #   history    : The messages as they stood at the end of the turn.
    #
    # Returns:
    #
    #   None. A failed write is logged and swallowed -- the turn already happened and its result is on its way to the
    #   caller, so raising here would discard a completed answer over a cache miss.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def write_history ( self, session_id: str, history: list [ dict [ str, Any ] ] ) -> None:

        if self.working is None or not history:
            return

        if not self.working.for_session ( session_id ).set_conversation ( history ):
            logger.warning ( "Could not store the conversation history for session %r.", session_id )
