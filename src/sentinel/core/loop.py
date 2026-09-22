#-----------------------------------------------------------------------------------------------------------------------
# Module:  loop.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   Agentic execution loop (architecture 3.3.2).
#
#       Trigger -> prompt assembly -> LLM call -> dispatch
#                                                  |- text     -> emit response
#                                                  |- tool_use -> execute, inject result, loop back
#                                                  '- end_turn  -> finalise
#
#   Three guards terminate a runaway turn, and all three produce a *response* rather than an exception -- the user gets
#   told what happened instead of seeing a traceback:
#
#     * max_iterations   ceiling on tool-use cycles. On reaching it the loop calls once more with tools=[] to force a
#                        final text answer, rather than truncating mid-thought. A HeartbeatTick uses the lower
#                        heartbeat.max_iterations instead, and may call only heartbeat.allowed_tools (3.3.9).
#     * loop_timeout     wall-clock ceiling on the whole turn.
#     * max_total_tokens context budget. Over budget triggers compression, and only a still-over-budget history after
#                        compression ends the turn.
#
#   Tool dispatch is a stub in Phase 1. It returns a plain error string rather than raising, which is the point:
#   the model sees the failure as a tool result and can self-correct on the next iteration (architecture 3.3.7). Phase 7
#   replaces the stub body without changing this contract.
#
#   Token accounting prefers the exact usage the provider already reported over any local estimate. After the first
#   iteration the previous response's usage gives the true prompt size for free, so the loop only asks the provider to
#   count when it has no prior figure to work from.
#
#   Every turn is also an audit trail. The loop mints one correlation identifier, establishes it as the ambient one for
#   everything the turn calls into, and records the turn's opening, each tool dispatch, each permission denial, and the
#   closing response against it -- so "what did the agent do when I asked X" is one indexed query rather than a search
#   through prose. Event logging is optional: an absent logger makes every emission a no-op, which is what a unit test
#   of the loop's control flow wants.
#-----------------------------------------------------------------------------------------------------------------------

import logging
import time
import uuid

from collections.abc import AsyncIterator, Sequence
from typing          import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from sentinel.config import SentinelConfig
from sentinel.core   import prompt as prompt_module
from sentinel.errors import LlmError, SentinelError
from sentinel.llm.adapter import (
    STOP_END_TURN,
    STOP_TOOL_USE,
    LlmAdapter,
    LlmResponse,
    PromptBlock,
    TokenUsage,
    ToolCall,
)
from sentinel.logging.logger import EventLogger, correlation_scope
from sentinel.logging.schemas import (
    CATEGORY_AGENT,
    CATEGORY_SECURITY,
    CATEGORY_TOOL,
    CATEGORY_USER,
    EVENT_AGENT_DECISION,
    EVENT_AGENT_RESPONSE,
    EVENT_SECURITY_DENY,
    EVENT_TOOL_ERROR,
    EVENT_TOOL_INVOKE,
    EVENT_TOOL_RESULT,
    EVENT_USER_MESSAGE,
)

logger = logging.getLogger ( __name__ )

# Trigger kinds (architecture 3.3.2).

TRIGGER_USER_MESSAGE    = "UserMessage"
TRIGGER_CHANNEL_MESSAGE = "ChannelMessage"
TRIGGER_HEARTBEAT_TICK  = "HeartbeatTick"
TRIGGER_SYSTEM_EVENT    = "SystemEvent"

# How each trigger kind opens its own audit trail. A message from a person is a user.message however it arrived, which
# is why a channel message maps there too -- the channel is recorded in the payload, not in the event name. An
# autonomous turn has no user behind it, so it opens as an agent.decision instead: what the agent chose to do next.

TRIGGER_EVENTS: dict [ str, tuple [ str, str ] ] = {
    TRIGGER_USER_MESSAGE: ( CATEGORY_USER, EVENT_USER_MESSAGE ),
    TRIGGER_CHANNEL_MESSAGE: ( CATEGORY_USER, EVENT_USER_MESSAGE ),
    TRIGGER_HEARTBEAT_TICK: ( CATEGORY_AGENT, EVENT_AGENT_DECISION ),
    TRIGGER_SYSTEM_EVENT: ( CATEGORY_AGENT, EVENT_AGENT_DECISION ),
}

# How much of a message or a tool payload is copied into an event. The event log is an audit trail, so the content
# matters -- but a 16,000-token tool result stored verbatim on every dispatch would make the log larger than the
# conversation it describes. The length is always recorded in full alongside the preview, so a truncated entry says so.

EVENT_CONTENT_PREVIEW = 500

# What the stub returns for any tool call, until Phase 7 supplies real dispatch.

TOOL_STUB_MESSAGE = "Tool system not yet available"

# What an autonomous turn is told when it reaches for a tool outside the heartbeat permission set (architecture 3.3.9).
# Phrased as a tool result rather than raised, for the same reason as every other dispatch failure: the model can pick
# a different approach on the next iteration, where an exception would end a turn it could have completed.

TOOL_DENIED_MESSAGE = (
    "Permission denied: {name} is not in the heartbeat permission set. Autonomous turns "
    "may use only the tools listed in heartbeat.allowed_tools."
)

# Instruction used to force a final answer once max_iterations is reached.

FINAL_ANSWER_INSTRUCTION = (
    "[System] You have reached the maximum number of tool calls. "
    "Provide your final response now, using what you already know."
)

# Compression geometry (architecture 3.3.6): keep the edges, summarise the middle.

PROTECTED_HEAD_MESSAGES = 2
PROTECTED_TAIL_MESSAGES = 6
SUMMARY_MAX_TOKENS      = 2000

COMPRESSION_SYSTEM_PROMPT = (
    "Summarise this conversation segment concisely. Preserve all factual content, tool "
    "results, and decisions made. Omit pleasantries and repetition."
)

COMPRESSION_ACKNOWLEDGEMENT = "Understood. I have the context from our earlier discussion."


#-----------------------------------------------------------------------------------------------------------------------
# Types
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Class: Trigger
#
# Description:
#
#   What started a turn.
#
#   The loop stays generic across trigger kinds on purpose: channel-specific behaviour belongs in a channel skill, not
#   here. All this type does is carry the content and the metadata the prompt assembler needs.
#
# Attributes:
#   kind      : One of the TRIGGER_* constants.
#   content   : The message or event text.
#   channel   : Channel metadata, for a ChannelMessage.
#   heartbeat : Heartbeat state, for a HeartbeatTick.
#-----------------------------------------------------------------------------------------------------------------------

class Trigger ( BaseModel ):

    model_config = ConfigDict ( extra = "forbid" )

    kind:      str                      = TRIGGER_USER_MESSAGE
    content:   str                      = ""
    channel:   dict [ str, Any ] | None = None
    heartbeat: dict [ str, Any ] | None = None


#-----------------------------------------------------------------------------------------------------------------------
# Class: ToolResult
#
# Description:
#
#   The outcome of one tool dispatch.
#
# Attributes:
#
#   success : Whether the tool ran and produced output.
#   output  : Serialised tool output.
#   error   : Failure description, or None on success.
#-----------------------------------------------------------------------------------------------------------------------

class ToolResult ( BaseModel ):

    model_config = ConfigDict ( extra = "forbid" )

    success: bool
    output:  str        = ""
    error:   str | None = None


#-----------------------------------------------------------------------------------------------------------------------
# Class: LoopResult
#
# Description:
#
#   The outcome of one turn.
#
# Attributes:
#
#   text           : Final response text.
#   iterations     : Iterations actually run.
#   stop_reason    : Why the loop ended -- a provider stop reason, or one of the guard names.
#   usage          : Token usage summed across every call in the turn.
#   correlation_id : Identifier tying every event of this turn together in the event log.
#   error          : Failure description when the turn ended on a guard or a provider failure.
#   history        : The conversation history as it stood at the end of the turn.
#-----------------------------------------------------------------------------------------------------------------------

class LoopResult ( BaseModel ):

    model_config = ConfigDict ( extra = "forbid" )

    text:           str                        = ""
    iterations:     int                        = 0
    stop_reason:    str                        = STOP_END_TURN
    usage:          TokenUsage                 = Field ( default_factory = TokenUsage )
    correlation_id: str                        = ""
    error:          str | None                 = None
    history:        list [ dict [ str, Any ] ] = Field ( default_factory = list )


#-----------------------------------------------------------------------------------------------------------------------
# Class: LoopEvent
#
# Description:
#
#   One event emitted while streaming a turn (architecture 3.3.8).
#
#   Tool-use iterations are reported as "status" events, not as final text: the user sees "Searching the web..." rather
#   than the model's internal tool chatter. Only the closing turn's text is streamed as "text".
#
# Attributes:
#
#   type   : Event kind.
#   text   : Incremental response text, for "text" events.
#   status : Human-readable progress description, for "status" events.
#   done   : Whether the described activity has finished, for "status" events.
#   result : The completed turn, for the terminal "result" event.
#-----------------------------------------------------------------------------------------------------------------------

class LoopEvent ( BaseModel ):

    model_config = ConfigDict ( extra = "forbid" )

    type:   Literal [ "status", "text", "result" ]
    text:   str                                    = ""
    status: str                                    = ""
    done:   bool                                   = False
    result: LoopResult | None                      = None


#-----------------------------------------------------------------------------------------------------------------------
# Loop
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Class: AgenticLoop
#
# Description:
#
#   Drives one turn from trigger to final response.
#
# Attributes:
#
#   configuration : The loaded configuration.
#   adapter       : The LLM adapter to call.
#   events        : Event logger for the turn's audit trail, or None to run without one.
#-----------------------------------------------------------------------------------------------------------------------

class AgenticLoop:

    #-------------------------------------------------------------------------------------------------------------------
    # Function: __init__
    #
    # Description:
    #
    #   Construct the loop.
    #
    # Arguments:
    #
    #   configuration : The loaded configuration.
    #   adapter       : The LLM adapter to call.
    #   events        : Event logger the turn records itself against. Omitting it makes every emission a no-op, which
    #                   is what a unit test of the loop's control flow wants and what a first run before the database
    #                   exists gets.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def __init__ ( self,
                   configuration: SentinelConfig,
                   adapter: LlmAdapter,
                   events: EventLogger | None = None ) -> None:

        # Record the configuration, the adapter, and the audit trail.

        self.configuration = configuration
        self.adapter       = adapter
        self.events        = events

    #-------------------------------------------------------------------------------------------------------------------
    # Event logging
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: record
    #
    # Description:
    #
    #   Write one event against this turn.
    #
    # Arguments:
    #
    #   category       : One of the event categories.
    #   event          : The dotted event name.
    #   correlation_id : This turn's identifier.
    #   data           : Event payload.
    #
    # Returns:
    #
    #   None. A logger that is absent makes this a no-op, and a logger that fails is reported and moved past: the loop
    #   owes the user a response, and abandoning a completed turn over an audit row would be the wrong trade.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def record ( self,
                       category: str,
                       event: str,
                       correlation_id: str,
                       data: dict [ str, Any ] | None = None ) -> None:

        if self.events is None:
            return

        try:
            await self.events.log (
                category       = category,
                event          = event,
                data           = data,
                correlation_id = correlation_id,
                source         = "loop",
            )

        except SentinelError as error:
            logger.warning ( "Could not record the %s event: %s", event, error )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: preview
    #
    # Description:
    #
    #   Cut a payload to the length an event records, keeping its true size alongside.
    #
    # Arguments:
    #
    #   text : The payload to describe.
    #
    # Returns:
    #
    #   A mapping carrying the truncated text, its full character length, and whether it was cut. The length is always
    #   the real one, so a reader can tell a short message from a long one that was trimmed.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def preview ( self, text: str ) -> dict [ str, Any ]:

        # Return data to caller.

        return {
            "preview": text [ : EVENT_CONTENT_PREVIEW ],
            "length": len ( text ),
            "truncated": len ( text ) > EVENT_CONTENT_PREVIEW,
        }

    #-------------------------------------------------------------------------------------------------------------------
    # Tool dispatch
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: get_tool_schemas
    #
    # Description:
    #
    #   Return the tool schemas advertised to the model.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   An empty list in Phase 1. Phase 7 replaces this with SkillRegistry.get_tool_schemas (architecture 3.3.3).
    #
    #-------------------------------------------------------------------------------------------------------------------

    def get_tool_schemas ( self ) -> list [ dict [ str, Any ] ]:

        # Return data to caller.

        return []

    #-------------------------------------------------------------------------------------------------------------------
    # Function: permitted_tools
    #
    # Description:
    #
    #   Determine which tools this trigger's turn may call (architecture 3.3.9).
    #
    # Arguments:
    #
    #   trigger : What started the turn.
    #
    # Returns:
    #
    #   The permitted tool names for an autonomous turn, or None when the turn is interactive and carries the user's
    #   full permissions. None and the empty set are deliberately different answers: None is "unrestricted", the empty
    #   set is "nothing at all", and the default for an unconfigured heartbeat is the latter.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def permitted_tools ( self, trigger: Trigger ) -> frozenset [ str ] | None:

        # An interactive turn is bounded by the user watching it, not by this list.

        if trigger.kind != TRIGGER_HEARTBEAT_TICK:
            return None

        # Return data to caller.

        return frozenset ( self.configuration.heartbeat.allowed_tools )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: iteration_ceiling
    #
    # Description:
    #
    #   Determine the tool-use cycle ceiling for this trigger's turn (architecture 3.3.9).
    #
    # Arguments:
    #
    #   trigger : What started the turn.
    #
    # Returns:
    #
    #   heartbeat.max_iterations for an autonomous turn, loop.max_iterations otherwise. The autonomous ceiling is lower
    #   because no one is present to interrupt a turn that has started going in circles.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def iteration_ceiling ( self, trigger: Trigger ) -> int:

        # Return data to caller.

        if trigger.kind == TRIGGER_HEARTBEAT_TICK:
            return self.configuration.heartbeat.max_iterations

        return self.configuration.loop.max_iterations

    #-------------------------------------------------------------------------------------------------------------------
    # Function: dispatch_tool
    #
    # Description:
    #
    #   Execute one tool call.
    #
    #   Phase 1 stub. It returns a failed ToolResult rather than raising, because the model should see the failure as
    #   a tool result and adapt -- an exception here would abort a turn the model could have recovered from.
    #
    # Arguments:
    #
    #   tool_call      : The tool the model asked to run.
    #   correlation_id : Identifier tying this dispatch to the rest of the turn.
    #
    # Returns:
    #
    #   The dispatch outcome. Always a failure in Phase 1.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def dispatch_tool ( self, tool_call: ToolCall, correlation_id: str ) -> ToolResult:

        # Phase 1: no skill runtime exists, so every dispatch reports the same failure.

        logger.info (
            "Tool dispatch requested for %r but the skill system is not available yet "
            "(correlation_id=%s).",
            tool_call.name, correlation_id,
        )

        # Return data to caller.

        return ToolResult ( success = False, error = TOOL_STUB_MESSAGE )

    #-------------------------------------------------------------------------------------------------------------------
    # Context management
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: count_prompt_tokens
    #
    # Description:
    #
    #   Determine the prompt token count for the next call.
    #
    # Arguments:
    #
    #   system       : Assembled system prompt blocks.
    #   history      : Conversation history.
    #   tools        : Tool schemas.
    #   last_usage   : Usage reported by the previous call in this turn, if any.
    #
    # Returns:
    #
    #   The prompt token count. Prefers the exact figure the provider already reported, because it is authoritative,
    #   and already in hand; falls back to asking the provider to count only when there is no prior figure.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def count_prompt_tokens ( self,
                                    system: Sequence [ PromptBlock ],
                                    history: Sequence [ dict [ str, Any ] ],
                                    tools: Sequence [ dict [ str, Any ] ] | None,
                                    last_usage: TokenUsage | None ) -> int:

        # An earlier call in this turn already measured the prompt exactly.

        if last_usage is not None and last_usage.total_prompt_tokens > 0:
            return last_usage.total_prompt_tokens

        # Return data to caller.

        return await self.adapter.count_tokens ( system, history, tools )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: compress_history
    #
    # Description:
    #
    #   Compress a conversation history by summarising its middle (architecture 3.3.6).
    #
    #   Strategy: keep the first exchange and the last three turns verbatim, replace everything between them with an
    #   LLM-written summary. Truncation would lose the early context that carries the task definition; this keeps
    #   it and discards the repetition instead.
    #
    # Arguments:
    #
    #   history : The full conversation history.
    #
    # Returns:
    #
    #   The compressed history: protected head, summary exchange, protected tail. Returned unchanged when there is no
    #   middle to compress, or when the summarising call itself fails -- a failed summary must not lose the history.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def compress_history ( self, history: list [ dict [ str, Any ] ] ) -> list [ dict [ str, Any ] ]:

        # Nothing to compress when head and tail already cover everything.

        protected_head = history [ : PROTECTED_HEAD_MESSAGES ]
        protected_tail = history [ -PROTECTED_TAIL_MESSAGES : ]
        middle         = history [ PROTECTED_HEAD_MESSAGES : -PROTECTED_TAIL_MESSAGES ]

        if not middle:
            return history

        # Summarise the middle. A failure here is survivable: return the history intact
        # rather than dropping the segment we could not summarise.

        serialised = "\n\n".join (
            f"{message.get ( 'role', 'user' )}: {self._content_to_text ( message.get ( 'content', '' ) )}"
            for message in middle
        )

        try:
            summary_response = await self.adapter.complete (
                system     = [ PromptBlock ( text = COMPRESSION_SYSTEM_PROMPT, cached = False ) ],
                messages   = [ { "role": "user", "content": serialised } ],
                tools      = None,
                max_tokens = SUMMARY_MAX_TOKENS,
            )

            summary_text = summary_response.text

        except LlmError as error:
            logger.warning (
                "History compression failed (%s). Keeping the full history; the context "
                "budget guard will decide what happens next.",
                error,
            )

            return history

        if not summary_text.strip ():
            return history

        # Return data to caller.

        return [
            *protected_head,
            { "role": "user", "content": f"[Conversation summary]\n{summary_text}" },
            { "role": "assistant", "content": COMPRESSION_ACKNOWLEDGEMENT },
            *protected_tail,
        ]

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _content_to_text
    #
    # Description:
    #
    #   Flatten message content to plain text for summarisation.
    #
    # Arguments:
    #
    #   content : Message content, a string or a list of content blocks.
    #
    # Returns:
    #
    #   The text of the content. Non-text blocks are described by their type rather than serialised in full, so a tool
    #   result does not drown the summary in JSON.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def _content_to_text ( self, content: Any ) -> str:

        # A plain string needs nothing.

        if isinstance ( content, str ):
            return content

        if isinstance ( content, list ):
            parts: list [ str ] = []

            for block in content:

                if isinstance ( block, dict ):
                    text = block.get ( "text" )

                    if isinstance ( text, str ):
                        parts.append ( text )
                    elif block.get ( "type" ) == "tool_result":
                        parts.append ( f"[tool result: {block.get ( 'content', '' )}]" )
                    else:
                        parts.append ( f"[{block.get ( 'type', 'block' )}]" )
                else:
                    parts.append ( str ( block ) )

            return "\n".join ( parts )

        # Return data to caller.

        return str ( content )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: truncate_tool_output
    #
    # Description:
    #
    #   Cut a tool result to the configured token budget.
    #
    # Arguments:
    #
    #   output : The tool's serialised output.
    #
    # Returns:
    #
    #   The output, truncated to loop.max_tool_result with a visible marker when it was too long.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def truncate_tool_output ( self, output: str ) -> str:

        # Return data to caller.

        return prompt_module.truncate_to_tokens ( output, self.configuration.loop.max_tool_result )

    #-------------------------------------------------------------------------------------------------------------------
    # Prompt assembly
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: build_system_prompt
    #
    # Description:
    #
    #   Assemble the system prompt for a trigger.
    #
    # Arguments:
    #
    #   trigger : What started the turn.
    #
    # Returns:
    #
    #   Prompt blocks, stable before volatile, with the cache breakpoint on the last stable block. Identity,
    #   and memory are empty in Phase 1; the sections exist so the order is settled early.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def build_system_prompt ( self, trigger: Trigger ) -> list [ PromptBlock ]:

        # Channel and heartbeat context appear only for their own trigger kinds.

        channel   = trigger.channel if trigger.kind == TRIGGER_CHANNEL_MESSAGE else None
        heartbeat = trigger.heartbeat if trigger.kind == TRIGGER_HEARTBEAT_TICK else None

        tool_names = [ str ( schema.get ( "name", "" ) ) for schema in self.get_tool_schemas () ]

        # Return data to caller.

        return prompt_module.assemble (
            configuration = self.configuration,
            allowed_tools = [ name for name in tool_names if name ],
            channel       = channel,
            heartbeat     = heartbeat,
        )

    #-------------------------------------------------------------------------------------------------------------------
    # Execution
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: run
    #
    # Description:
    #
    #   Run one turn to completion.
    #
    # Arguments:
    #
    #   trigger : What started the turn.
    #   history : Prior conversation history. A new conversation when omitted.
    #
    # Returns:
    #
    #   The turn outcome. A guard firing produces a result carrying an error description, not an exception. The user is
    #   owed an explanation rather than a stack trace.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def run ( self,
                    trigger: Trigger,
                    history: list [ dict [ str, Any ] ] | None = None ) -> LoopResult:

        # Consume the streaming form and keep only its terminal result.

        result = LoopResult ()

        async for event in self.run_stream ( trigger, history ):
            if event.type == "result" and event.result is not None:
                result = event.result

        # Return data to caller.

        return result

    #-------------------------------------------------------------------------------------------------------------------
    # Function: run_stream
    #
    # Description:
    #
    #   Run one turn, emitting progress as it goes (architecture 3.3.8).
    #
    #   Iterations that use tools emit "status" events; only the closing turn emits "text". The terminal "result" event
    #   always arrives, on every path including every guard, so a consumer can rely on it to end its own loop.
    #
    # Arguments:
    #
    #   trigger : What started the turn.
    #   history : Prior conversation history. A new conversation when omitted.
    #
    # Returns:
    #
    #   Status and text events as the turn proceeds, then exactly one terminal "result" event.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def run_stream ( self,
                           trigger: Trigger,
                           history: list [ dict [ str, Any ] ] | None = None ) -> AsyncIterator [ LoopEvent ]:

        # The turn's identifier is minted here and established as the ambient one for everything the turn calls into,
        # so a component several frames down -- the tool dispatcher in Phase 7, the permission gate in Phase 6 -- can
        # record itself against this turn without the identifier being threaded through every signature between here
        # and there. The body is a separate generator purely so this stays a five-line wrapper; the contextvar
        # bookkeeping is the one part of the loop worth being able to read at a glance.

        correlation_id = str ( uuid.uuid4 () )

        with correlation_scope ( correlation_id ):

            async for event in self._run_stream ( trigger, history, correlation_id ):
                yield event

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _run_stream
    #
    # Description:
    #
    #   The body of one turn, under an established correlation identifier.
    #
    # Arguments:
    #
    #   trigger        : What started the turn.
    #   history        : Prior conversation history.
    #   correlation_id : This turn's identifier, already ambient.
    #
    # Returns:
    #
    #   Status and text events as the turn proceeds, then exactly one terminal "result" event.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def _run_stream ( self,
                            trigger: Trigger,
                            history: list [ dict [ str, Any ] ] | None,
                            correlation_id: str ) -> AsyncIterator [ LoopEvent ]:

        limits     = self.configuration.loop
        loop_start = time.monotonic ()

        conversation: list [ dict [ str, Any ] ] = list ( history or [] )

        system = self.build_system_prompt ( trigger )
        tools  = self.get_tool_schemas ()

        # Autonomous turns run under a lower iteration ceiling and a restricted tool set (architecture 3.3.9).

        max_iterations = self.iteration_ceiling ( trigger )
        permitted      = self.permitted_tools ( trigger )

        # Open the turn's audit trail before anything can fail, so even a turn that dies on its first LLM call leaves
        # a record of what was asked of it.

        category, opening_event = TRIGGER_EVENTS.get (
            trigger.kind, ( CATEGORY_AGENT, EVENT_AGENT_DECISION )
        )

        await self.record (
            category, opening_event, correlation_id,
            {
                "trigger": trigger.kind,
                "channel": trigger.channel.get ( "name" ) if trigger.channel else None,
                "history_messages": len ( conversation ),
                **self.preview ( trigger.content ),
            },
        )

        # Step 2: append the trigger content as a user message.

        if trigger.content:
            conversation.append ( { "role": "user", "content": trigger.content } )

        totals = TokenUsage ()
        last_usage: TokenUsage | None = None
        iteration = 0
        response: LlmResponse | None = None

        # Step 3: the loop proper.

        while iteration < max_iterations:
            iteration += 1

            # Guard: wall-clock timeout.

            elapsed = time.monotonic () - loop_start

            if elapsed > limits.loop_timeout:
                logger.warning (
                    "Loop timeout after %.1f s on iteration %d (correlation_id=%s).",
                    elapsed, iteration, correlation_id,
                )

                yield await self.fail_turn (
                    f"Agent timed out after {limits.loop_timeout}s.",
                    "loop_timeout", iteration - 1, totals, correlation_id, conversation,
                )

                return

            # Guard: context window budget. Compress first, fail only if still over.

            tokens_used = await self.count_prompt_tokens ( system, conversation, tools, last_usage )

            if tokens_used > limits.max_total_tokens:
                logger.info (
                    "Context budget exceeded (%d > %d). Compressing history (correlation_id=%s).",
                    tokens_used, limits.max_total_tokens, correlation_id,
                )

                conversation = await self.compress_history ( conversation )
                last_usage   = None
                tokens_used  = await self.count_prompt_tokens ( system, conversation, tools, None )

                if tokens_used > limits.max_total_tokens:
                    yield await self.fail_turn (
                        "Context window exhausted.",
                        "context_exhausted", iteration - 1, totals, correlation_id, conversation,
                    )

                    return

            # The LLM call, streamed.
            #
            # Deltas are forwarded to the caller as they arrive rather than assembled and
            # emitted at the end. Calling complete() here instead was the whole reason the
            # gateway's SSE stream delivered one giant frame: the wire format was correct
            # and nothing streamed, which no mocked test caught because they all asserted
            # on the envelope rather than on arrival times.
            #
            # Text is streamed on every iteration, including one that goes on to call a
            # tool. Prose the model writes before reaching for a tool is a genuine part of
            # its answer; what architecture 3.3.8 keeps off the transcript is the tool
            # traffic itself, and that is already rendered as status events below.

            try:
                response = None

                async for stream_event in self.adapter.stream (
                    system   = system,
                    messages = conversation,
                    tools    = tools if tools else None,
                ):
                    if stream_event.type == "text" and stream_event.text:
                        yield LoopEvent ( type = "text", text = stream_event.text )

                    elif stream_event.type == "final":
                        response = stream_event.response

                # A stream that ends without its terminal event has told us nothing about
                # why it stopped, so it cannot be treated as a completed turn.

                if response is None:
                    raise LlmError ( "The model stream ended without a final response." )

            except LlmError as error:
                logger.error ( "LLM call failed (correlation_id=%s): %s", correlation_id, error )

                yield await self.fail_turn (
                    f"The language model could not be reached: {error}",
                    "llm_error", iteration, totals, correlation_id, conversation,
                )

                return

            last_usage = response.usage

            self._accumulate ( totals, response.usage )

            logger.debug (
                "Iteration %d: stop_reason=%s prompt_tokens=%d cache_read=%d (correlation_id=%s).",
                iteration, response.stop_reason, response.usage.total_prompt_tokens,
                response.usage.cache_read_input_tokens, correlation_id,
            )

            # Dispatch on the stop reason.

            if response.stop_reason == STOP_TOOL_USE and response.tool_calls:

                conversation.append ( self._assistant_message ( response ) )

                tool_result_blocks: list [ dict [ str, Any ] ] = []

                for tool_call in response.tool_calls:

                    yield LoopEvent (
                        type   = "status",
                        status = f"Running {tool_call.name}...",
                        done   = False,
                    )

                    # The permission gate sits here rather than inside dispatch_tool, so Phase 7 can replace the
                    # dispatcher wholesale without inheriting responsibility for enforcing the heartbeat permission
                    # set -- and so a denial is recorded even while dispatch is still a stub.

                    if permitted is not None and tool_call.name not in permitted:
                        logger.warning (
                            "Denied %r to an autonomous turn: not in heartbeat.allowed_tools "
                            "(correlation_id=%s).",
                            tool_call.name, correlation_id,
                        )

                        # A refusal is a security event, not a tool event. It is logged whether or not dispatch is
                        # real, which is the point of gating here: Phase 6's audit questions are answerable from this
                        # row already.

                        await self.record (
                            CATEGORY_SECURITY, EVENT_SECURITY_DENY, correlation_id,
                            {
                                "tool": tool_call.name,
                                "reason": "not in heartbeat.allowed_tools",
                                "trigger": trigger.kind,
                            },
                        )

                        outcome = ToolResult (
                            success = False,
                            error   = TOOL_DENIED_MESSAGE.format ( name = tool_call.name ),
                        )
                    else:
                        tool_started = time.monotonic ()

                        await self.record (
                            CATEGORY_TOOL, EVENT_TOOL_INVOKE, correlation_id,
                            {
                                "tool": tool_call.name,
                                "tool_use_id": tool_call.id,
                                "arguments": tool_call.arguments,
                                "iteration": iteration,
                            },
                        )

                        outcome = await self.dispatch_tool ( tool_call, correlation_id )

                        await self.record (
                            CATEGORY_TOOL,
                            EVENT_TOOL_RESULT if outcome.success else EVENT_TOOL_ERROR,
                            correlation_id,
                            {
                                "tool": tool_call.name,
                                "tool_use_id": tool_call.id,
                                "duration_ms": round ( ( time.monotonic () - tool_started ) * 1000.0, 1 ),
                                "error": outcome.error,
                                **self.preview ( outcome.output if outcome.success else ( outcome.error or "" ) ),
                            },
                        )

                    payload = outcome.output if outcome.success else ( outcome.error or "Tool failed." )

                    tool_result_blocks.append (
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_call.id,
                            "content": self.truncate_tool_output ( payload ),
                            "is_error": not outcome.success,
                        }
                    )

                    yield LoopEvent (
                        type   = "status",
                        status = f"{tool_call.name} finished.",
                        done   = True,
                    )

                conversation.append ( { "role": "user", "content": tool_result_blocks } )

                continue

            # Any non-tool stop reason ends the loop. An unexpected one is logged, but the
            # text produced so far is still returned -- partial output beats none.

            if response.stop_reason != STOP_END_TURN:
                logger.warning (
                    "Unexpected stop reason %r on iteration %d (correlation_id=%s).",
                    response.stop_reason, iteration, correlation_id,
                )

            conversation.append ( self._assistant_message ( response ) )

            completed = LoopResult (
                text           = response.text,
                iterations     = iteration,
                stop_reason    = response.stop_reason,
                usage          = totals,
                correlation_id = correlation_id,
                history        = conversation,
            )

            await self.record_response ( completed, correlation_id )

            # No text event here: the deltas were forwarded as they arrived. Emitting the
            # assembled text again would duplicate the whole response for any client that
            # renders both.

            yield LoopEvent ( type = "result", result = completed )

            return

        # Guard: max iterations. Force a final answer by calling once more without tools,
        # so the turn ends with something useful rather than an apology.

        logger.warning (
            "Max iterations (%d) reached (correlation_id=%s). Forcing a final response.",
            max_iterations, correlation_id,
        )

        forced = list ( conversation )
        forced.append ( { "role": "user", "content": FINAL_ANSWER_INSTRUCTION } )

        # Streamed like any other turn. This is the answer the user actually reads, so it
        # would be a strange one to withhold until complete.

        try:
            final = None

            async for stream_event in self.adapter.stream (
                system   = system,
                messages = forced,
                tools    = [],
            ):
                if stream_event.type == "text" and stream_event.text:
                    yield LoopEvent ( type = "text", text = stream_event.text )

                elif stream_event.type == "final":
                    final = stream_event.response

            if final is None:
                raise LlmError ( "The model stream ended without a final response." )

        except LlmError as error:
            yield await self.fail_turn (
                f"Agent reached the tool-call limit and could not produce a final response: {error}",
                "max_iterations", iteration, totals, correlation_id, conversation,
            )

            return

        self._accumulate ( totals, final.usage )

        conversation.append ( self._assistant_message ( final ) )

        forced_result = LoopResult (
            text           = final.text,
            iterations     = iteration,
            stop_reason    = "max_iterations",
            usage          = totals,
            correlation_id = correlation_id,
            history        = conversation,
        )

        await self.record_response ( forced_result, correlation_id )

        yield LoopEvent ( type = "result", result = forced_result )

    #-------------------------------------------------------------------------------------------------------------------
    # Helpers
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _assistant_message
    #
    # Description:
    #
    #   Build the history entry for one assistant turn.
    #
    # Arguments:
    #
    #   response : The completed turn.
    #
    # Returns:
    #
    #   A history message carrying the provider's own content blocks when it supplied them, so thinking signatures and
    #   tool_use blocks replay intact; otherwise a plain text message.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def _assistant_message ( self, response: LlmResponse ) -> dict [ str, Any ]:

        # Return data to caller.

        if response.raw_content:
            return { "role": "assistant", "content": response.raw_content }

        return { "role": "assistant", "content": response.text }

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _accumulate
    #
    # Description:
    #
    #   Add one call's usage into the running total for the turn.
    #
    # Arguments:
    #
    #   totals : The running total, mutated in place.
    #   usage  : Usage from one call.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def _accumulate ( self, totals: TokenUsage, usage: TokenUsage ) -> None:

        # Sum each field. Pydantic validate_assignment is off for this model, so plain
        # attribute arithmetic is safe and cheap.

        totals.input_tokens += usage.input_tokens
        totals.output_tokens += usage.output_tokens
        totals.cache_creation_input_tokens += usage.cache_creation_input_tokens
        totals.cache_read_input_tokens += usage.cache_read_input_tokens

    #-------------------------------------------------------------------------------------------------------------------
    # Function: record_response
    #
    # Description:
    #
    #   Close the turn's audit trail with what the agent produced.
    #
    #   One place rather than one per exit path, so the closing record of a turn that succeeded and one that hit a
    #   guard carry the same fields -- which is what makes "how often does the loop time out" a query rather than a
    #   comparison of two differently-shaped rows.
    #
    # Arguments:
    #
    #   result         : The turn's outcome.
    #   correlation_id : Identifier for this turn.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def record_response ( self, result: LoopResult, correlation_id: str ) -> None:

        await self.record (
            CATEGORY_AGENT, EVENT_AGENT_RESPONSE, correlation_id,
            {
                "stop_reason": result.stop_reason,
                "iterations": result.iterations,
                "error": result.error,
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "cache_read_tokens": result.usage.cache_read_input_tokens,
                **self.preview ( result.text ),
            },
        )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: fail_turn
    #
    # Description:
    #
    #   Record and build the terminal event for a turn that ended on a guard or a provider failure.
    #
    #   The audit trail closes on every path, not only the happy one. A turn that timed out and one that was never
    #   started look identical in the log otherwise, and the first is the one worth finding.
    #
    # Arguments:
    #
    #   message        : User-facing explanation.
    #   stop_reason    : Which guard or failure ended the turn.
    #   iterations     : Iterations completed.
    #   totals         : Accumulated usage.
    #   correlation_id : Identifier for this turn.
    #   history        : Conversation history as it stands.
    #
    # Returns:
    #
    #   The terminal "result" event.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def fail_turn ( self,
                          message: str,
                          stop_reason: str,
                          iterations: int,
                          totals: TokenUsage,
                          correlation_id: str,
                          history: list [ dict [ str, Any ] ] ) -> LoopEvent:

        event = self._error_result ( message, stop_reason, iterations, totals, correlation_id, history )

        if event.result is not None:
            await self.record_response ( event.result, correlation_id )

        # Return data to caller.

        return event

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _error_result
    #
    # Description:
    #
    #   Build the terminal event for a turn that ended on a guard or a provider failure.
    #
    # Arguments:
    #
    #   message        : User-facing explanation.
    #   stop_reason    : Which guard or failure ended the turn.
    #   iterations     : Iterations completed.
    #   totals         : Accumulated usage.
    #   correlation_id : Identifier for this turn.
    #   history        : Conversation history as it stands.
    #
    # Returns:
    #
    #   A terminal "result" event. The explanation is placed in both text and error, so a client that renders only text
    #   still tells the user what happened.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def _error_result ( self,
                        message: str,
                        stop_reason: str,
                        iterations: int,
                        totals: TokenUsage,
                        correlation_id: str,
                        history: list [ dict [ str, Any ] ] ) -> LoopEvent:

        # Return data to caller.

        return LoopEvent (
            type = "result",
            result = LoopResult (
                text           = message,
                iterations     = iterations,
                stop_reason    = stop_reason,
                usage          = totals,
                correlation_id = correlation_id,
                error          = message,
                history        = history,
            ),
        )
