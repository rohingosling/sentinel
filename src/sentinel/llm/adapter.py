#-----------------------------------------------------------------------------------------------------------------------
# Module:  adapter.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   Provider-neutral LLM interface.
#
#   Defines the wire types the agentic loop works in and the LlmAdapter contract every provider implements. Two design
#   choices are load-bearing:
#
#     * The system prompt crosses this boundary as a list of blocks, not a string. A cache breakpoint is a property of a
#       block, so flattening to a string here would make prompt caching impossible to express (architecture Decision 3).
#     * Retry lives in the base class, not in each provider. The backoff schedule is a Sentinel policy, and duplicating
#       it per provider guarantees the two eventually disagree.
#-----------------------------------------------------------------------------------------------------------------------

import asyncio
import logging

from abc             import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing          import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from sentinel.errors import LlmAuthenticationError, LlmError, LlmRequestError, LlmTransportError

logger = logging.getLogger ( __name__ )

# Stop reasons the loop branches on. "end_turn" and "tool_use" are the two expected
# outcomes; everything else is handled as an unexpected stop (architecture 3.3.2).

STOP_END_TURN   = "end_turn"
STOP_TOOL_USE   = "tool_use"
STOP_MAX_TOKENS = "max_tokens"


#-----------------------------------------------------------------------------------------------------------------------
# Wire types
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Class: PromptBlock
#
# Description:
#
#   One block of the system prompt.
#
#   `cached` marks the block as the end of the cacheable prefix. Exactly one block in an assembled prompt carries it --
#   the last stable one (architecture Decision 3).
#
# Attributes:
#
#   text   : The block's rendered text.
#   cached : Emit a cache_control breakpoint on this block.
#-----------------------------------------------------------------------------------------------------------------------

class PromptBlock ( BaseModel ):

    model_config = ConfigDict ( extra = "forbid" )

    text:   str
    cached: bool = False


#-----------------------------------------------------------------------------------------------------------------------
# Class: ToolCall
#
# Description:
#
#   A tool the model asked to run.
#
# Attributes:
#
#   id        : Provider-assigned identifier. The tool result must quote it back.
#   name      : Tool name as advertised to the model.
#   arguments : Validated input the model supplied.
#-----------------------------------------------------------------------------------------------------------------------

class ToolCall ( BaseModel ):

    model_config = ConfigDict ( extra = "forbid" )

    id:        str
    name:      str
    arguments: dict [ str, Any ] = Field ( default_factory = dict )


#-----------------------------------------------------------------------------------------------------------------------
# Class: TokenUsage
#
# Description:
#
#   Token accounting for one call.
#
#   The two cache fields are the only direct evidence that prompt caching works. T1.30 asserts cache_read is non-zero on
#   the second loop iteration, which is why they are carried through the abstraction rather than left in the provider.
#
# Attributes:
#
#   input_tokens                : Uncached prompt tokens, billed in full.
#   output_tokens               : Generated tokens, including thinking.
#   cache_creation_input_tokens : Tokens written to the cache this call.
#   cache_read_input_tokens     : Tokens served from the cache this call.
#-----------------------------------------------------------------------------------------------------------------------

class TokenUsage ( BaseModel ):

    model_config = ConfigDict ( extra = "forbid" )

    input_tokens:                int = 0
    output_tokens:               int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens:     int = 0

    #-------------------------------------------------------------------------------------------------------------------
    # Function: total_prompt_tokens
    #
    # Description:
    #
    #   Every prompt token processed, cached or not.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   The sum of the three input fields. `input_tokens` alone is the uncached remainder, so reading it as the prompt
    #   size under-reports badly once caching is working.
    #
    #-------------------------------------------------------------------------------------------------------------------

    @property
    def total_prompt_tokens ( self ) -> int:

        # Return data to caller.

        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )


#-----------------------------------------------------------------------------------------------------------------------
# Class: LlmResponse
#
# Description:
#
#   One completed model turn.
#
# Attributes:
#
#   text        : Concatenated text content. Empty when the model only asked for tools.
#   tool_calls  : Tools the model asked to run. Non-empty exactly when stop_reason is "tool_use".
#   stop_reason : Why generation ended.
#   usage       : Token accounting.
#   model       : Model that produced the turn, as reported by the provider.
#   raw_content : Provider-native content blocks, replayed verbatim into the next request so thinking-block signatures
#                 survive the round trip.
#-----------------------------------------------------------------------------------------------------------------------

class LlmResponse ( BaseModel ):

    model_config = ConfigDict ( extra = "forbid" )

    text:        str                        = ""
    tool_calls:  list [ ToolCall ]          = Field ( default_factory = list )
    stop_reason: str                        = STOP_END_TURN
    usage:       TokenUsage                 = Field ( default_factory = TokenUsage )
    model:       str                        = ""
    raw_content: list [ dict [ str, Any ] ] = Field ( default_factory = list )


#-----------------------------------------------------------------------------------------------------------------------
# Class: StreamEvent
#
# Description:
#
#   One event from a streaming call.
#
#   "text" events carry an incremental delta; the single terminal "final" event carries the assembled response. A
#   consumer that only wants text can ignore everything else and still be correct.
#
# Attributes:
#
#   type     : Event kind.
#   text     : Incremental text, for "text" events.
#   response : The assembled turn, for the terminal "final" event.
#-----------------------------------------------------------------------------------------------------------------------

class StreamEvent ( BaseModel ):

    model_config = ConfigDict ( extra = "forbid" )

    type:     Literal [ "text", "final" ]
    text:     str                         = ""
    response: LlmResponse | None          = None


#-----------------------------------------------------------------------------------------------------------------------
# Adapter contract
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Class: LlmAdapter
#
# Description:
#
#   The interface every provider implements.
#
#   Subclasses implement _complete and _stream. Callers use complete and stream, which wrap those in the shared retry
#   policy, so no provider can accidentally opt out of it.
#
# Attributes:
#
#   model          : Model identifier passed to the provider.
#   max_tokens     : Ceiling on generated tokens per call.
#   timeout        : Seconds before a call is abandoned.
#   retry_attempts : Attempts before giving up.
#   retry_backoff  : Seconds before the first retry; doubles per attempt.
#-----------------------------------------------------------------------------------------------------------------------

class LlmAdapter ( ABC ):

    #-------------------------------------------------------------------------------------------------------------------
    # Function: __init__
    #
    # Description:
    #
    #   Record the call parameters shared by every provider.
    #
    # Arguments:
    #
    #   model          : Model identifier.
    #   max_tokens     : Ceiling on generated tokens per call.
    #   timeout        : Seconds before a call is abandoned.
    #   retry_attempts : Attempts before giving up.
    #   retry_backoff  : Seconds before the first retry.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def __init__ ( self,
                   model: str,
                   max_tokens: int     = 4096,
                   timeout: float      = 120.0,
                   retry_attempts: int = 3,
                   retry_backoff: float = 1.0 ) -> None:

        # Record the shared call parameters.

        self.model          = model
        self.max_tokens     = max_tokens
        self.timeout        = timeout
        self.retry_attempts = retry_attempts
        self.retry_backoff  = retry_backoff

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _complete
    #
    # Description:
    #
    #   Provider-specific single completion. Called by complete, which owns the retry policy.
    #
    # Arguments:
    #
    #   system     : System prompt blocks, in assembly order.
    #   messages   : Conversation history in provider-neutral form.
    #   tools      : Tool schemas advertised to the model.
    #   max_tokens : Override for this call only.
    #
    # Returns:
    #
    #   The completed turn.
    #
    #-------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    async def _complete ( self,
                          system: Sequence [ PromptBlock ],
                          messages: Sequence [ dict [ str, Any ] ],
                          tools: Sequence [ dict [ str, Any ] ] | None = None,
                          max_tokens: int | None = None ) -> LlmResponse:

        raise NotImplementedError

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _stream
    #
    # Description:
    #
    #   Provider-specific streaming completion. Called by stream, which owns the retry policy.
    #
    # Arguments:
    #
    #   system     : System prompt blocks, in assembly order.
    #   messages   : Conversation history in provider-neutral form.
    #   tools      : Tool schemas advertised to the model.
    #   max_tokens : Override for this call only.
    #
    # Returns:
    #
    #   Text deltas as they arrive, then exactly one terminal "final" event.
    #
    #-------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def _stream ( self,
                  system: Sequence [ PromptBlock ],
                  messages: Sequence [ dict [ str, Any ] ],
                  tools: Sequence [ dict [ str, Any ] ] | None = None,
                  max_tokens: int | None = None ) -> AsyncIterator [ StreamEvent ]:

        raise NotImplementedError

    #-------------------------------------------------------------------------------------------------------------------
    # Function: count_tokens
    #
    # Description:
    #
    #   Count the tokens a request would consume.
    #
    # Arguments:
    #
    #   system   : System prompt blocks.
    #   messages : Conversation history.
    #   tools    : Tool schemas.
    #
    # Returns:
    #
    #   The prompt token count. The default implementation estimates locally; a provider that exposes an authoritative
    #   counter should override this.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def count_tokens ( self,
                             system: Sequence [ PromptBlock ],
                             messages: Sequence [ dict [ str, Any ] ],
                             tools: Sequence [ dict [ str, Any ] ] | None = None ) -> int:

        # Estimate locally. Imported here to keep the module import graph acyclic.

        from sentinel.llm.tokens import estimate_request_tokens

        # Return data to caller.

        return estimate_request_tokens ( system, messages, tools )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: complete
    #
    # Description:
    #
    #   Run a completion, retrying transport failures with exponential backoff.
    #
    #   Backoff is 1 s, 2 s, 4 s by default (architecture 3.3.7). An authentication failure is never retried: a rejected
    #   credential will be rejected again, and retrying only delays a clear error.
    #
    # Arguments:
    #
    #   system     : System prompt blocks, in assembly order.
    #   messages   : Conversation history in provider-neutral form.
    #   tools      : Tool schemas advertised to the model.
    #   max_tokens : Override for this call only.
    #
    # Returns:
    #
    #   The completed turn.
    #
    #   Raises LlmAuthenticationError immediately on a rejected credential, or LlmTransportError once every attempt has
    #   failed.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def complete ( self,
                         system: Sequence [ PromptBlock ],
                         messages: Sequence [ dict [ str, Any ] ],
                         tools: Sequence [ dict [ str, Any ] ] | None = None,
                         max_tokens: int | None = None ) -> LlmResponse:

        # Retry transport failures; surface everything else at once.

        last_error: LlmError | None = None

        for attempt in range ( self.retry_attempts ):

            try:
                return await self._complete ( system, messages, tools, max_tokens )

            except ( LlmAuthenticationError, LlmRequestError ):

                # Neither a rejected credential nor a rejected request changes on a retry.

                raise

            except LlmTransportError as error:
                last_error = error

                await self._sleep_before_retry ( attempt, error )

        # Every attempt failed. Report the last cause rather than inventing a summary.

        raise LlmTransportError (
            f"LLM call failed after {self.retry_attempts} attempts: {last_error}"
        ) from last_error

    #-------------------------------------------------------------------------------------------------------------------
    # Function: stream
    #
    # Description:
    #
    #   Run a streaming completion, retrying transport failures before the first event.
    #
    #   Retry stops being safe the moment a delta has been handed to the caller -- replaying the call would duplicate
    #   text the user already saw. So a mid-stream failure propagates rather than retrying.
    #
    # Arguments:
    #
    #   system     : System prompt blocks, in assembly order.
    #   messages   : Conversation history in provider-neutral form.
    #   tools      : Tool schemas advertised to the model.
    #   max_tokens : Override for this call only.
    #
    # Returns:
    #
    #   Text deltas as they arrive, then exactly one terminal "final" event.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def stream ( self,
                       system: Sequence [ PromptBlock ],
                       messages: Sequence [ dict [ str, Any ] ],
                       tools: Sequence [ dict [ str, Any ] ] | None = None,
                       max_tokens: int | None = None ) -> AsyncIterator [ StreamEvent ]:

        # Retry only while nothing has been emitted.

        last_error: LlmError | None = None

        for attempt in range ( self.retry_attempts ):

            emitted = False

            try:
                async for event in self._stream ( system, messages, tools, max_tokens ):
                    emitted = True

                    yield event

                return

            except ( LlmAuthenticationError, LlmRequestError ):

                # Neither a rejected credential nor a rejected request changes on a retry.

                raise

            except LlmTransportError as error:
                if emitted:
                    raise

                last_error = error

                await self._sleep_before_retry ( attempt, error )

        raise LlmTransportError (
            f"LLM stream failed after {self.retry_attempts} attempts: {last_error}"
        ) from last_error

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _sleep_before_retry
    #
    # Description:
    #
    #   Wait out the backoff interval for one failed attempt.
    #
    #   Skips the wait entirely after the final attempt, so a fully failed call does not pay a pointless delay before
    #   reporting.
    #
    # Arguments:
    #
    #   attempt : Zero-based index of the attempt that just failed.
    #   error   : The failure being retried, logged for diagnosis.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def _sleep_before_retry ( self, attempt: int, error: LlmError ) -> None:

        # The last attempt has nothing to wait for.

        if attempt >= self.retry_attempts - 1:
            return

        delay = self.retry_backoff * ( 2 ** attempt )

        logger.warning (
            "LLM call failed (attempt %d/%d): %s. Retrying in %.1f s.",
            attempt + 1, self.retry_attempts, error, delay,
        )

        await asyncio.sleep ( delay )
