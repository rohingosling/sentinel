#-----------------------------------------------------------------------------------------------------------------------
# Module:  ollama.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   Ollama adapter -- local model fallback.
#
#   Phase 1 scope is availability detection plus a working non-streaming and streaming text path. Tool use is explicitly
#   not implemented: Ollama's tool support varies by model, and the loop has no real tools to dispatch until Phase 7. A
#   request carrying tools raises rather than silently dropping them, because a fallback that quietly ignores the tools
#   the model was told it had is worse than one that refuses.
#
#   Availability is checked against /api/tags, the cheapest endpoint that proves a daemon is actually answering. It is
#   never assumed: fallback_enabled defaults to false precisely because most installs have no Ollama.
#-----------------------------------------------------------------------------------------------------------------------

import json
import logging

from collections.abc import AsyncIterator, Sequence
from typing          import Any

import httpx

from sentinel.errors import LlmError, LlmRequestError, LlmTransportError
from sentinel.llm.adapter import (
    STOP_END_TURN,
    LlmAdapter,
    LlmResponse,
    PromptBlock,
    StreamEvent,
    TokenUsage,
)

logger = logging.getLogger ( __name__ )

# Default endpoint of a local Ollama daemon.

DEFAULT_BASE_URL = "http://127.0.0.1:11434"

# Seconds allowed for the availability probe. Short by design -- this runs on the startup
# path, and a hung probe would stall the agent to check an optional feature.

AVAILABILITY_TIMEOUT = 2.0

# Client-error statuses that a later attempt might survive. Every other 4xx describes the
# request itself -- most often a model that has not been pulled, which Ollama answers 404.

RETRYABLE_CLIENT_STATUSES = frozenset ( { 408, 409, 429 } )


#-----------------------------------------------------------------------------------------------------------------------
# Class: OllamaAdapter
#
# Description:
#
#   LlmAdapter backed by a local Ollama daemon.
#
# Attributes:
#
#   base_url : Root URL of the Ollama daemon.
#-----------------------------------------------------------------------------------------------------------------------

class OllamaAdapter ( LlmAdapter ):

    #-------------------------------------------------------------------------------------------------------------------
    # Function: __init__
    #
    # Description:
    #
    #   Construct the adapter.
    #
    # Arguments:
    #
    #   model          : Model identifier, e.g. "llama3.1:8b".
    #   max_tokens     : Ceiling on generated tokens per call.
    #   timeout        : Seconds before a call is abandoned.
    #   retry_attempts : Attempts before giving up.
    #   retry_backoff  : Seconds before the first retry.
    #   base_url       : Root URL of the daemon.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def __init__ ( self,
                   model: str           = "llama3.1:8b",
                   max_tokens: int      = 4096,
                   timeout: float       = 120.0,
                   retry_attempts: int  = 3,
                   retry_backoff: float = 1.0,
                   base_url: str = DEFAULT_BASE_URL ) -> None:

        # Record the shared call parameters, then the endpoint.

        super ().__init__ (
            model          = model,
            max_tokens     = max_tokens,
            timeout        = timeout,
            retry_attempts = retry_attempts,
            retry_backoff  = retry_backoff,
        )

        self.base_url = base_url.rstrip ( "/" )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: is_available
    #
    # Description:
    #
    #   Report whether a local Ollama daemon is answering.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   True when /api/tags responds 200. False on any failure -- a connection refused, a timeout, a non-200 status. An
    #   absent optional dependency is never an exception.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def is_available ( self ) -> bool:

        # Probe the cheapest endpoint that proves a daemon is really there.

        try:
            async with httpx.AsyncClient ( timeout = AVAILABILITY_TIMEOUT ) as client:
                response = await client.get ( f"{self.base_url}/api/tags" )

                return response.status_code == 200

        except Exception as error:
            logger.debug ( "Ollama is not available at %s: %s", self.base_url, error )

            return False

    #-------------------------------------------------------------------------------------------------------------------
    # Function: build_request
    #
    # Description:
    #
    #   Assemble an /api/chat request body.
    #
    # Arguments:
    #
    #   system     : System prompt blocks, flattened into one system message.
    #   messages   : Conversation history.
    #   max_tokens : Override for this call only.
    #   stream     : Ask the daemon to stream.
    #
    # Returns:
    #
    #   The request body. Prompt blocks flatten to a single string here -- Ollama has no prompt cache, so the block
    #   structure that exists for Anthropic's cache breakpoint carries no meaning.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def build_request ( self,
                        system: Sequence [ PromptBlock ],
                        messages: Sequence [ dict [ str, Any ] ],
                        max_tokens: int | None = None,
                        stream: bool = False ) -> dict [ str, Any ]:

        # Flatten the system blocks, then prepend them as a single system message.

        system_text = "\n\n".join ( block.text for block in system if block.text )

        chat: list [ dict [ str, Any ] ] = []

        if system_text:
            chat.append ( { "role": "system", "content": system_text } )

        for message in messages:
            content = message.get ( "content", "" )

            chat.append (
                {
                    "role": message.get ( "role", "user" ),
                    "content": content if isinstance ( content, str ) else json.dumps ( content ),
                }
            )

        # Return data to caller.

        return {
            "model": self.model,
            "messages": chat,
            "stream": stream,
            "options": {
                "num_predict": max_tokens if max_tokens is not None else self.max_tokens,
            },
        }

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _reject_tools
    #
    # Description:
    #
    #   Refuse a request that advertises tools.
    #
    # Arguments:
    #
    #   tools : Tool schemas supplied by the caller.
    #
    # Returns:
    #
    #   None.
    #
    #   Raises LlmError when tools were supplied, rather than dropping them silently.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def _reject_tools ( self, tools: Sequence [ dict [ str, Any ] ] | None ) -> None:

        # An empty list is fine -- that is the max-iterations guard asking for text only.

        if tools:
            raise LlmError (
                "The Ollama fallback does not support tool use. Tool dispatch arrives in Phase 7; "
                "until then a tool-bearing request must go to the primary provider."
            )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _translate_error
    #
    # Description:
    #
    #   Map a daemon failure onto Sentinel's retryable / non-retryable split.
    #
    #   The common case by a wide margin is a model that has not been pulled: Ollama answers 404, and retrying three
    #   times with backoff turns an instant, actionable "model not found" into several seconds of silence followed by a
    #   retry summary that hides it.
    #
    # Arguments:
    #
    #   error     : The exception raised while calling the daemon.
    #   streaming : Whether the failure came from the streaming path, for the message only.
    #
    # Returns:
    #
    #   LlmRequestError for a 4xx the daemon will reject identically next time, LlmTransportError for anything a retry
    #   might survive -- a refused connection, a timeout, a 5xx.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def _translate_error ( self, error: Exception, streaming: bool = False ) -> Exception:

        # A 4xx describes the request. Only 408, 409, and 429 are worth another attempt.

        action = "stream" if streaming else "call"

        if isinstance ( error, httpx.HTTPStatusError ):
            status = error.response.status_code

            if 400 <= status < 500 and status not in RETRYABLE_CLIENT_STATUSES:

                detail = ""

                try:
                    detail = str ( error.response.json ().get ( "error", "" ) )
                except Exception:
                    detail = error.response.text [ : 200 ]

                hint = (
                    f" Run `ollama pull {self.model}` if the model has not been downloaded."
                    if status == 404
                    else ""
                )

                return LlmRequestError (
                    f"Ollama rejected the {action} ({status}): {detail or error}.{hint}"
                )

        # Return data to caller.

        return LlmTransportError ( f"Ollama {action} failed: {error}" )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _complete
    #
    # Description:
    #
    #   Run one non-streaming completion.
    #
    # Arguments:
    #
    #   system     : System prompt blocks.
    #   messages   : Conversation history.
    #   tools      : Tool schemas. Must be empty or absent.
    #   max_tokens : Override for this call only.
    #
    # Returns:
    #
    #   The completed turn, always with stop_reason "end_turn" -- this path cannot produce a tool call.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def _complete ( self,
                          system: Sequence [ PromptBlock ],
                          messages: Sequence [ dict [ str, Any ] ],
                          tools: Sequence [ dict [ str, Any ] ] | None = None,
                          max_tokens: int | None = None ) -> LlmResponse:

        # Refuse tools, then call the daemon.

        self._reject_tools ( tools )

        request = self.build_request ( system, messages, max_tokens, stream = False )

        try:
            async with httpx.AsyncClient ( timeout = self.timeout ) as client:
                response = await client.post ( f"{self.base_url}/api/chat", json = request )

                response.raise_for_status ()

                document = response.json ()

        except Exception as error:
            raise self._translate_error ( error ) from error

        # Return data to caller.

        return LlmResponse (
            text        = str ( document.get ( "message", {} ).get ( "content", "" ) ),
            stop_reason = STOP_END_TURN,
            usage = TokenUsage (
                input_tokens  = int ( document.get ( "prompt_eval_count", 0 ) or 0 ),
                output_tokens = int ( document.get ( "eval_count", 0 ) or 0 ),
            ),
            model = str ( document.get ( "model", self.model ) ),
        )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _stream
    #
    # Description:
    #
    #   Run one streaming completion.
    #
    # Arguments:
    #
    #   system     : System prompt blocks.
    #   messages   : Conversation history.
    #   tools      : Tool schemas. Must be empty or absent.
    #   max_tokens : Override for this call only.
    #
    # Returns:
    #
    #   Text deltas as they arrive, then one terminal "final" event.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def _stream ( self,
                        system: Sequence [ PromptBlock ],
                        messages: Sequence [ dict [ str, Any ] ],
                        tools: Sequence [ dict [ str, Any ] ] | None = None,
                        max_tokens: int | None = None ) -> AsyncIterator [ StreamEvent ]:

        # Refuse tools, then stream newline-delimited JSON from the daemon.

        self._reject_tools ( tools )

        request = self.build_request ( system, messages, max_tokens, stream = True )

        collected: list [ str ] = []
        usage = TokenUsage ()

        try:
            async with httpx.AsyncClient ( timeout = self.timeout ) as client:
                async with client.stream ( "POST", f"{self.base_url}/api/chat", json = request ) as response:

                    response.raise_for_status ()

                    async for line in response.aiter_lines ():

                        if not line.strip ():
                            continue

                        document = json.loads ( line )
                        delta    = str ( document.get ( "message", {} ).get ( "content", "" ) )

                        if delta:
                            collected.append ( delta )

                            yield StreamEvent ( type = "text", text = delta )

                        # The terminal record carries the token counts.

                        if document.get ( "done" ):
                            usage = TokenUsage (
                                input_tokens  = int ( document.get ( "prompt_eval_count", 0 ) or 0 ),
                                output_tokens = int ( document.get ( "eval_count", 0 ) or 0 ),
                            )

        except Exception as error:
            raise self._translate_error ( error, streaming = True ) from error

        yield StreamEvent (
            type = "final",
            response = LlmResponse (
                text        = "".join ( collected ),
                stop_reason = STOP_END_TURN,
                usage       = usage,
                model       = self.model,
            ),
        )
