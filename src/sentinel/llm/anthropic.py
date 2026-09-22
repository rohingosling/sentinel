#-----------------------------------------------------------------------------------------------------------------------
# Module:  anthropic.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   Anthropic Claude adapter.
#
#   Wraps AsyncAnthropic and translates between Sentinel's provider-neutral types and the Messages API. Four details are
#   deliberate and worth not undoing:
#
#     * The cache breakpoint is emitted as cache_control on the last stable system block (Decision 3). It is
#       the whole reason the system prompt crosses the adapter boundary as blocks rather than a string.
#     * Sampling parameters are not sent. temperature, top_p, and top_k are removed on Claude Opus 5 and return a 400.
#       agent.yaml still carries llm.temperature for the Ollama fallback, which does accept it.
#     * Thinking is left at its default. On Claude Opus 5 thinking is on by default, and max_tokens caps thinking plus
#       response text together -- which is why the loop budgets max_tokens rather than assuming it is all answer.
#     * SDK retries are disabled (max_retries=0). LlmAdapter owns the retry policy; two layers of backoff would multiply
#       into a wait far longer than loop_timeout.
#-----------------------------------------------------------------------------------------------------------------------

import logging

from collections.abc import AsyncIterator, Sequence
from typing          import Any

import anthropic

from sentinel.errors import LlmAuthenticationError, LlmRequestError, LlmTransportError
from sentinel.llm.adapter import (
    LlmAdapter,
    LlmResponse,
    PromptBlock,
    StreamEvent,
    TokenUsage,
    ToolCall,
)
from sentinel.security.secrets import SECRET_ANTHROPIC_API_KEY, require_secret

logger = logging.getLogger ( __name__ )

# Marker the Messages API expects for a five-minute cache breakpoint.

CACHE_CONTROL_EPHEMERAL = { "type": "ephemeral" }

# Client-error statuses that a later attempt might survive: request timeout, conflict, and
# rate limit. Every other 4xx describes the request itself and will fail identically.

RETRYABLE_CLIENT_STATUSES = frozenset ( { 408, 409, 429 } )


#-----------------------------------------------------------------------------------------------------------------------
# Class: AnthropicAdapter
#
# Description:
#
#   LlmAdapter backed by the Anthropic Messages API.
#
# Attributes:
#
#   client : The underlying AsyncAnthropic client.
#-----------------------------------------------------------------------------------------------------------------------

class AnthropicAdapter ( LlmAdapter ):

    #-------------------------------------------------------------------------------------------------------------------
    # Function: __init__
    #
    # Description:
    #
    #   Construct the adapter and its client.
    #
    #   The credential is read through the secrets module, so the keyring is consulted before the environment and an
    #   absent key produces an actionable ConfigurationError rather than a 401 at first use.
    #
    # Arguments:
    #
    #   model          : Model identifier.
    #   max_tokens     : Ceiling on generated tokens per call.
    #   timeout        : Seconds before a call is abandoned.
    #   retry_attempts : Attempts before giving up.
    #   retry_backoff  : Seconds before the first retry.
    #   api_key        : Explicit credential. Read from the keyring or environment when omitted.
    #   client         : Pre-built client, used by tests to inject a fake transport.
    #
    # Returns:
    #
    #   None.
    #
    #   Raises ConfigurationError when no credential can be found.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def __init__ ( self,
                   model: str           = "claude-opus-5",
                   max_tokens: int      = 4096,
                   timeout: float       = 120.0,
                   retry_attempts: int  = 3,
                   retry_backoff: float = 1.0,
                   api_key: str | None  = None,
                   client: Any = None ) -> None:

        # Record the shared call parameters.

        super ().__init__ (
            model          = model,
            max_tokens     = max_tokens,
            timeout        = timeout,
            retry_attempts = retry_attempts,
            retry_backoff  = retry_backoff,
        )

        # An injected client wins outright, so a test never needs a credential at all.

        if client is not None:
            self.client = client

            return

        key = api_key if api_key is not None else require_secret ( SECRET_ANTHROPIC_API_KEY )

        # max_retries=0: LlmAdapter owns retry. See the module header.

        self.client = anthropic.AsyncAnthropic (
            api_key     = key,
            timeout     = timeout,
            max_retries = 0,
        )

    #-------------------------------------------------------------------------------------------------------------------
    # Request construction
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: build_system_blocks
    #
    # Description:
    #
    #   Render prompt blocks into Messages API system blocks.
    #
    # Arguments:
    #
    #   system : System prompt blocks, in assembly order.
    #
    # Returns:
    #
    #   A list of text blocks. Any block marked cached carries a cache_control breakpoint, so everything up to and
    #   including it is cacheable and everything after it is not.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def build_system_blocks ( self, system: Sequence [ PromptBlock ] ) -> list [ dict [ str, Any ] ]:

        # Render each block, attaching the breakpoint where the assembler asked for one.

        blocks: list [ dict [ str, Any ] ] = []

        for block in system:

            rendered: dict [ str, Any ] = { "type": "text", "text": block.text }

            if block.cached:
                rendered [ "cache_control" ] = dict ( CACHE_CONTROL_EPHEMERAL )

            blocks.append ( rendered )

        # Return data to caller.

        return blocks

    #-------------------------------------------------------------------------------------------------------------------
    # Function: build_request
    #
    # Description:
    #
    #   Assemble the keyword arguments for a Messages API call.
    #
    # Arguments:
    #
    #   system     : System prompt blocks, in assembly order.
    #   messages   : Conversation history.
    #   tools      : Tool schemas advertised to the model.
    #   max_tokens : Override for this call only.
    #
    # Returns:
    #
    #   Keyword arguments ready to pass to messages.create or messages.stream. Sampling parameters are deliberately
    #   absent; see the module header.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def build_request ( self,
                        system: Sequence [ PromptBlock ],
                        messages: Sequence [ dict [ str, Any ] ],
                        tools: Sequence [ dict [ str, Any ] ] | None = None,
                        max_tokens: int | None = None ) -> dict [ str, Any ]:

        # Assemble the request.

        request: dict [ str, Any ] = {
            "model": self.model,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "system": self.build_system_blocks ( system ),
            "messages": list ( messages ),
        }

        # An empty tool list is not the same as no tools: it is how the max-iterations
        # guard forces a text response. Send it only when tools were actually supplied.

        if tools is not None:
            request [ "tools" ] = list ( tools )

        # Return data to caller.

        return request

    #-------------------------------------------------------------------------------------------------------------------
    # Response translation
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: translate_message
    #
    # Description:
    #
    #   Convert a Messages API response into an LlmResponse.
    #
    # Arguments:
    #
    #   message : The provider's response object.
    #
    # Returns:
    #
    #   The provider-neutral turn. Content blocks are preserved verbatim in raw_content so thinking-block signatures
    #   survive being replayed into the next request.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def translate_message ( self, message: Any ) -> LlmResponse:

        # Split content into text and tool calls, keeping the raw blocks for replay.

        text_parts: list [ str ]               = []
        tool_calls: list [ ToolCall ]          = []
        raw_blocks: list [ dict [ str, Any ] ] = []

        for block in message.content or []:

            block_type = getattr ( block, "type", None )

            if block_type == "text":
                text_parts.append ( block.text )

            elif block_type == "tool_use":
                tool_calls.append (
                    ToolCall (
                        id        = block.id,
                        name      = block.name,
                        arguments = dict ( block.input or {} ),
                    )
                )

            raw_blocks.append ( self._block_to_dict ( block ) )

        # Token accounting, including the two cache fields that prove caching works.

        usage_object = getattr ( message, "usage", None )

        usage = TokenUsage (
            input_tokens                = getattr ( usage_object, "input_tokens", 0 ) or 0,
            output_tokens               = getattr ( usage_object, "output_tokens", 0 ) or 0,
            cache_creation_input_tokens = getattr ( usage_object, "cache_creation_input_tokens", 0 ) or 0,
            cache_read_input_tokens     = getattr ( usage_object, "cache_read_input_tokens", 0 ) or 0,
        )

        # Return data to caller.

        return LlmResponse (
            text        = "".join ( text_parts ),
            tool_calls  = tool_calls,
            stop_reason = getattr ( message, "stop_reason", None ) or "end_turn",
            usage       = usage,
            model       = getattr ( message, "model", self.model ) or self.model,
            raw_content = raw_blocks,
        )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _block_to_dict
    #
    # Description:
    #
    #   Convert one provider content block to a plain dictionary.
    #
    # Arguments:
    #
    #   block : A provider content block, or an already-plain dictionary from a test fake.
    #
    # Returns:
    #
    #   The block as a dictionary suitable for replay in a later request.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def _block_to_dict ( self, block: Any ) -> dict [ str, Any ]:

        # Real SDK models expose model_dump; test fakes may already be dictionaries.

        if isinstance ( block, dict ):
            return dict ( block )

        dumper = getattr ( block, "model_dump", None )

        if callable ( dumper ):
            dumped: dict [ str, Any ] = dumper ( exclude_none = True )

            return dumped

        # Return data to caller.

        return { "type": getattr ( block, "type", "text" ) }

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _translate_error
    #
    # Description:
    #
    #   Map a provider exception onto Sentinel's retryable / non-retryable split.
    #
    # Arguments:
    #
    #   error : The exception the SDK raised.
    #
    # Returns:
    #
    #   LlmAuthenticationError for a rejected credential, which must never be retried. LlmTransportError for anything
    #   that a later attempt might survive.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def _translate_error ( self, error: Exception ) -> Exception:

        # A rejected or forbidden credential will be rejected again. Do not retry it.

        if isinstance ( error, anthropic.AuthenticationError | anthropic.PermissionDeniedError ):
            return LlmAuthenticationError ( f"Anthropic rejected the credential: {error}" )

        # Neither will a request the provider considers malformed. A misspelled model name
        # is a 404 and a rejected parameter is a 400; retrying either three times with
        # backoff delays an unchanging answer by several seconds and buries the actual
        # cause under a retry summary. Only 408, 409, and 429 are worth another attempt.

        status = getattr ( error, "status_code", None )

        if isinstance ( status, int ) and 400 <= status < 500 and status not in RETRYABLE_CLIENT_STATUSES:
            return LlmRequestError ( f"Anthropic rejected the request ({status}): {error}" )

        # Everything else -- 5xx, timeouts, dropped connections -- may survive a retry.

        return LlmTransportError ( f"Anthropic call failed: {error}" )

    #-------------------------------------------------------------------------------------------------------------------
    # Provider calls
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _complete
    #
    # Description:
    #
    #   Run one non-streaming completion.
    #
    # Arguments:
    #
    #   system     : System prompt blocks, in assembly order.
    #   messages   : Conversation history.
    #   tools      : Tool schemas advertised to the model.
    #   max_tokens : Override for this call only.
    #
    # Returns:
    #
    #   The completed turn.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def _complete ( self,
                          system: Sequence [ PromptBlock ],
                          messages: Sequence [ dict [ str, Any ] ],
                          tools: Sequence [ dict [ str, Any ] ] | None = None,
                          max_tokens: int | None = None ) -> LlmResponse:

        # Call the provider, translating both success and failure into Sentinel types.

        request = self.build_request ( system, messages, tools, max_tokens )

        try:
            message = await self.client.messages.create ( **request )
        except anthropic.AnthropicError as error:
            raise self._translate_error ( error ) from error

        # Return data to caller.

        return self.translate_message ( message )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _stream
    #
    # Description:
    #
    #   Run one streaming completion.
    #
    # Arguments:
    #
    #   system     : System prompt blocks, in assembly order.
    #   messages   : Conversation history.
    #   tools      : Tool schemas advertised to the model.
    #   max_tokens : Override for this call only.
    #
    # Returns:
    #
    #   Text deltas as they arrive, then one terminal "final" event carrying the assembled turn.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def _stream ( self,
                        system: Sequence [ PromptBlock ],
                        messages: Sequence [ dict [ str, Any ] ],
                        tools: Sequence [ dict [ str, Any ] ] | None = None,
                        max_tokens: int | None = None ) -> AsyncIterator [ StreamEvent ]:

        # Stream text as it arrives, then emit the assembled turn once.

        request = self.build_request ( system, messages, tools, max_tokens )

        try:
            async with self.client.messages.stream ( **request ) as stream:

                async for delta in stream.text_stream:
                    yield StreamEvent ( type = "text", text = delta )

                final = await stream.get_final_message ()

        except anthropic.AnthropicError as error:
            raise self._translate_error ( error ) from error

        yield StreamEvent ( type = "final", response = self.translate_message ( final ) )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: count_tokens
    #
    # Description:
    #
    #   Count the prompt tokens of a request using the provider's own counter.
    #
    #   The /count_tokens endpoint is free and authoritative, which makes it strictly better than approximating the
    #   tokeniser locally. It has its own rate limit, separate from message creation, so it cannot starve the agent of
    #   inference capacity. A failure degrades to the local estimate rather than aborting the turn -- an unavailable
    #   counter is not a reason to refuse to think.
    #
    # Arguments:
    #
    #   system   : System prompt blocks.
    #   messages : Conversation history.
    #   tools    : Tool schemas advertised to the model.
    #
    # Returns:
    #
    #   The prompt token count.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def count_tokens ( self,
                             system: Sequence [ PromptBlock ],
                             messages: Sequence [ dict [ str, Any ] ],
                             tools: Sequence [ dict [ str, Any ] ] | None = None ) -> int:

        # An empty history has nothing to count, and the endpoint rejects it.

        if not messages:
            return sum (
                len ( block.text ) // 3 for block in system
            )

        request: dict [ str, Any ] = {
            "model": self.model,
            "system": self.build_system_blocks ( system ),
            "messages": list ( messages ),
        }

        if tools:
            request [ "tools" ] = list ( tools )

        try:
            counted = await self.client.messages.count_tokens ( **request )

            return int ( counted.input_tokens )

        except Exception as error:

            # Fall back to the local estimate rather than failing the turn.

            logger.warning (
                "Token counting failed (%s). Falling back to the local estimate, which is "
                "accurate to roughly +/-30%%.",
                error,
            )

            return await super ().count_tokens ( system, messages, tools )
