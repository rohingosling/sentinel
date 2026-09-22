#-----------------------------------------------------------------------------------------------------------------------
# Module:  openai_compat.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   OpenAI-compatible request and response schemas.
#
#   Open WebUI speaks the OpenAI chat-completions wire format, so Sentinel presents that shape and translates it.
#   Two consequences to keep in mind:
#
#     * The `model` field in a request is accepted and ignored. Sentinel exposes exactly one model -- itself -- and the
#       model actually used is whatever agent.yaml configures. Silently honouring a client-supplied model name would let
#       the UI override a deliberate configuration choice.
#     * Only the trailing user message drives the turn; earlier messages become conversation history. That is what makes
#       an OpenAI-shaped stateless request work against a stateful agent.
#-----------------------------------------------------------------------------------------------------------------------

import time

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Identifiers the API reports for itself.

MODEL_IDENTIFIER = "sentinel"
MODEL_OWNER      = "sentinel"

# Object type strings the OpenAI schema requires.

OBJECT_MODEL           = "model"
OBJECT_MODEL_LIST      = "list"
OBJECT_COMPLETION      = "chat.completion"
OBJECT_COMPLETION_PART = "chat.completion.chunk"

# Finish reasons, mapped from Sentinel's loop stop reasons.

FINISH_STOP   = "stop"
FINISH_LENGTH = "length"

# Loop stop reasons that mean "ran out of room" rather than "finished speaking".

LENGTH_STOP_REASONS = frozenset ( { "max_tokens", "context_exhausted", "max_iterations" } )


#-----------------------------------------------------------------------------------------------------------------------
# Requests
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Class: ChatMessage
#
# Description:
#
#   One message in an incoming chat request.
#
# Attributes:
#
#   role    : Who produced the message.
#   content : The message text. A null content is permitted by the OpenAI schema and treated as empty.
#   name    : Optional speaker name, accepted and ignored.
#-----------------------------------------------------------------------------------------------------------------------

class ChatMessage ( BaseModel ):

    model_config = ConfigDict ( extra = "ignore" )

    role:    Literal [ "system", "user", "assistant", "tool" ]
    content: str | None                                        = None
    name:    str | None                                        = None


#-----------------------------------------------------------------------------------------------------------------------
# Class: ChatCompletionRequest
#
# Description:
#
#   An incoming chat-completions request.
#
#   extra = "ignore" is deliberate. Open WebUI sends fields Sentinel has no use for, and rejecting an unrecognised field
#   would break the UI on its next release for no benefit. This is the opposite of the config loader, and for
#   the opposite reason: a config typo is the author's mistake to see, whereas an unknown wire field is someone else's
#   client evolving.
#
# Attributes:
#
#   model       : Requested model. Accepted and ignored; see the module header.
#   messages    : Conversation, oldest first. The trailing user message drives the turn.
#   stream      : Stream the response as server-sent events.
#   max_tokens  : Ceiling on generated tokens. Falls back to the configured value.
#   temperature : Accepted and ignored -- sampling parameters are rejected by Claude Opus 5.
#-----------------------------------------------------------------------------------------------------------------------

class ChatCompletionRequest ( BaseModel ):

    model_config = ConfigDict ( extra = "ignore" )

    model:       str | None           = None
    messages:    list [ ChatMessage ] = Field ( min_length = 1 )
    stream:      bool                 = False
    max_tokens:  int | None           = Field ( default = None, gt = 0 )
    temperature: float | None         = None

    #-------------------------------------------------------------------------------------------------------------------
    # Function: split_history
    #
    # Description:
    #
    #   Split the request into prior history and the message that drives this turn.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   A pair: the history in Sentinel's internal form, and the trailing user message text. System messages are dropped
    #   from the history -- Sentinel assembles its own system prompt, and letting a client inject one would bypass the
    #   persona, permissions, and cache ordering that prompt assembly exists to guarantee.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def split_history ( self ) -> tuple [ list [ dict [ str, Any ] ], str ]:

        # Keep only user and assistant turns; Sentinel owns the system prompt.

        usable = [
            message for message in self.messages
            if message.role in ( "user", "assistant" )
        ]

        # The trailing user message is the trigger. Anything after the last user message
        # is an assistant turn with nothing to answer, so it stays in the history.

        trigger_index = None

        for index in range ( len ( usable ) - 1, -1, -1 ):
            if usable [ index ].role == "user":
                trigger_index = index

                break

        if trigger_index is None:
            return (
                [
                    { "role": message.role, "content": message.content or "" }
                    for message in usable
                ],
                "",
            )

        history = [
            { "role": message.role, "content": message.content or "" }
            for message in usable [ : trigger_index ]
        ]

        # Return data to caller.

        return history, usable [ trigger_index ].content or ""


#-----------------------------------------------------------------------------------------------------------------------
# Responses
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Function: model_entry
#
# Description:
#
#   Build the single model entry Sentinel advertises.
#
# Arguments:
#
#   created : Unix timestamp to report. The current time when omitted.
#
# Returns:
#
#   One OpenAI-shaped model object.
#
#-----------------------------------------------------------------------------------------------------------------------

def model_entry ( created: int | None = None ) -> dict [ str, Any ]:

    # Return data to caller.

    return {
        "id": MODEL_IDENTIFIER,
        "object": OBJECT_MODEL,
        "created": created if created is not None else int ( time.time () ),
        "owned_by": MODEL_OWNER,
    }


#-----------------------------------------------------------------------------------------------------------------------
# Function: model_list
#
# Description:
#
#   Build the /v1/models response body.
#
# Arguments:
#
#   None.
#
# Returns:
#
#   An OpenAI-shaped model list carrying exactly one entry.
#
#-----------------------------------------------------------------------------------------------------------------------

def model_list () -> dict [ str, Any ]:

    # Return data to caller.

    return { "object": OBJECT_MODEL_LIST, "data": [ model_entry () ] }


#-----------------------------------------------------------------------------------------------------------------------
# Function: finish_reason_for
#
# Description:
#
#   Map a Sentinel loop stop reason onto an OpenAI finish reason.
#
# Arguments:
#
#   stop_reason : The loop's stop reason.
#
# Returns:
#
#   "length" when the turn ended because it ran out of room, "stop" otherwise. The OpenAI schema has no vocabulary for
#   "a guard fired", so a guard is reported as the closest honest equivalent.
#
#-----------------------------------------------------------------------------------------------------------------------

def finish_reason_for ( stop_reason: str ) -> str:

    # Return data to caller.

    return FINISH_LENGTH if stop_reason in LENGTH_STOP_REASONS else FINISH_STOP


#-----------------------------------------------------------------------------------------------------------------------
# Function: completion_response
#
# Description:
#
#   Build a non-streaming chat-completions response.
#
# Arguments:
#
#   completion_id : Identifier for this completion.
#   text          : The assistant's response text.
#   stop_reason   : The loop's stop reason.
#   prompt_tokens : Prompt tokens consumed.
#   output_tokens : Tokens generated.
#   created       : Unix timestamp. The current time when omitted.
#
# Returns:
#
#   An OpenAI-shaped chat.completion object.
#
#-----------------------------------------------------------------------------------------------------------------------

def completion_response ( completion_id: str,
                          text: str,
                          stop_reason: str,
                          prompt_tokens: int = 0,
                          output_tokens: int = 0,
                          created: int | None = None ) -> dict [ str, Any ]:

    # Return data to caller.

    return {
        "id": completion_id,
        "object": OBJECT_COMPLETION,
        "created": created if created is not None else int ( time.time () ),
        "model": MODEL_IDENTIFIER,
        "choices": [
            {
                "index": 0,
                "message": { "role": "assistant", "content": text },
                "finish_reason": finish_reason_for ( stop_reason ),
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
        },
    }


#-----------------------------------------------------------------------------------------------------------------------
# Function: completion_chunk
#
# Description:
#
#   Build one streaming chat-completions chunk.
#
# Arguments:
#
#   completion_id : Identifier for this completion, identical across every chunk.
#   text          : Incremental text, or empty for the terminal chunk.
#   finish_reason : Finish reason, set only on the terminal chunk.
#   role          : Emit a role in the delta. True only for the first chunk, per the OpenAI schema.
#   created       : Unix timestamp. The current time when omitted.
#
# Returns:
#
#   An OpenAI-shaped chat.completion.chunk object.
#
#-----------------------------------------------------------------------------------------------------------------------

def completion_chunk ( completion_id: str,
                       text: str                 = "",
                       finish_reason: str | None = None,
                       role: bool                = False,
                       created: int | None = None ) -> dict [ str, Any ]:

    # The delta carries the role on the opening chunk and text thereafter.

    delta: dict [ str, Any ] = {}

    if role:
        delta [ "role" ] = "assistant"

    if text:
        delta [ "content" ] = text

    # Return data to caller.

    return {
        "id": completion_id,
        "object": OBJECT_COMPLETION_PART,
        "created": created if created is not None else int ( time.time () ),
        "model": MODEL_IDENTIFIER,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


#-----------------------------------------------------------------------------------------------------------------------
# Function: status_event
#
# Description:
#
#   Build a tool-progress status event (architecture 3.3.8).
#
#   This is Open WebUI's Functions API shape, not part of the OpenAI schema. It renders as a spinner with a caption
#   rather than as assistant text, which is what keeps internal tool chatter out of the conversation.
#
# Arguments:
#
#   description : What the agent is doing.
#   done        : Whether the described activity has finished.
#
# Returns:
#
#   A status event body.
#
#-----------------------------------------------------------------------------------------------------------------------

def status_event ( description: str, done: bool = False ) -> dict [ str, Any ]:

    # Return data to caller.

    return { "type": "status", "data": { "description": description, "done": done } }


#-----------------------------------------------------------------------------------------------------------------------
# Function: error_body
#
# Description:
#
#   Build an OpenAI-shaped error body.
#
# Arguments:
#
#   message    : Human-readable description. Must never carry a credential or a traceback.
#   error_type : Error class, e.g. "invalid_request_error".
#   code       : Machine-readable code, when one applies.
#
# Returns:
#
#   An error body clients already know how to parse.
#
#-----------------------------------------------------------------------------------------------------------------------

def error_body ( message: str,
                 error_type: str = "invalid_request_error",
                 code: str | None = None ) -> dict [ str, Any ]:

    # Return data to caller.

    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": code,
        }
    }
