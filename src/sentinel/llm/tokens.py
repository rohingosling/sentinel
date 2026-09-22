#-----------------------------------------------------------------------------------------------------------------------
# Module:  tokens.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   Local token estimation.
#
#   This is a fallback, not the primary counter. Anthropic's tokeniser is undistributed, and /count_tokens
#   is both authoritative and free, so AnthropicAdapter overrides count_tokens to call it. The agentic loop prefers, in
#   order: the exact usage figures the previous call already returned, then the provider's counter, then this estimate.
#
#   Accuracy, measured against claude-opus-5 /count_tokens across prose, Python, YAML, JSON, and Markdown:
#
#       characters per token   1.95 (dense YAML)  ..  2.99 (English prose),  median 2.61
#
#   A single divisor therefore cannot be within 5% of the real count for arbitrary content -- the honest band for this
#   function is roughly +/-30%. CHARACTERS_PER_TOKEN is set slightly below the median so the estimate biases high: for a
#   budget guard, over-counting costs one unnecessary compression, while under-counting overruns the context window.
#
#   Do not "improve" this by pointing it at tiktoken. That is OpenAI's tokeniser and is wrong for Claude by a wider
#   margin than the heuristic below.
#-----------------------------------------------------------------------------------------------------------------------

import json

from collections.abc import Sequence
from typing          import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sentinel.llm.adapter import PromptBlock

# Characters per token. Calibrated below the measured median so the estimate errs high.

CHARACTERS_PER_TOKEN = 2.5

# Per-message envelope overhead: role, delimiters, and the provider's own framing.

TOKENS_PER_MESSAGE = 4


#-----------------------------------------------------------------------------------------------------------------------
# Function: estimate_tokens
#
# Description:
#
#   Estimate the token count of a string.
#
# Arguments:
#
#   text : The text to measure.
#
# Returns:
#
#   An estimated token count, never negative and never zero for non-empty input. Accurate to roughly +/-30%; see the
#   module header before relying on it for anything but a guard.
#
#-----------------------------------------------------------------------------------------------------------------------

def estimate_tokens ( text: str ) -> int:

    # Empty text costs nothing.

    if not text:
        return 0

    # Round up, so any non-empty string costs at least one token.

    estimate = int ( len ( text ) / CHARACTERS_PER_TOKEN ) + 1

    # Return data to caller.

    return estimate


#-----------------------------------------------------------------------------------------------------------------------
# Function: estimate_content_tokens
#
# Description:
#
#   Estimate the token count of one message's content, which may be a string or a list of content blocks.
#
# Arguments:
#
#   content : Message content, as a string or a list of provider content blocks.
#
# Returns:
#
#   An estimated token count. Non-text blocks are measured by their serialised form, which is crude but bounded -- and
#   Phase 1 has no image or document blocks to get wrong.
#
#-----------------------------------------------------------------------------------------------------------------------

def estimate_content_tokens ( content: Any ) -> int:

    # A plain string is the common case.

    if isinstance ( content, str ):
        return estimate_tokens ( content )

    # A block list: measure text blocks directly, everything else by its serialisation.

    if isinstance ( content, list ):
        total = 0

        for block in content:

            if isinstance ( block, str ):
                total += estimate_tokens ( block )

            elif isinstance ( block, dict ):
                text = block.get ( "text" )

                if isinstance ( text, str ):
                    total += estimate_tokens ( text )
                else:
                    total += estimate_tokens ( json.dumps ( block, sort_keys = True ) )

            else:
                total += estimate_tokens ( str ( block ) )

        return total

    # Anything else: fall back to its string form rather than claiming zero.

    return estimate_tokens ( str ( content ) )


#-----------------------------------------------------------------------------------------------------------------------
# Function: estimate_message_tokens
#
# Description:
#
#   Estimate the token count of a conversation history.
#
# Arguments:
#
#   messages : Conversation history.
#
# Returns:
#
#   An estimated token count, including per-message envelope overhead.
#
#-----------------------------------------------------------------------------------------------------------------------

def estimate_message_tokens ( messages: Sequence [ dict [ str, Any ] ] ) -> int:

    # Sum the content, then add the envelope cost of each message.

    total = 0

    for message in messages:
        total += TOKENS_PER_MESSAGE
        total += estimate_content_tokens ( message.get ( "content", "" ) )

    # Return data to caller.

    return total


#-----------------------------------------------------------------------------------------------------------------------
# Function: estimate_request_tokens
#
# Description:
#
#   Estimate the prompt token count of a whole request.
#
# Arguments:
#
#   system   : System prompt blocks.
#   messages : Conversation history.
#   tools     : Tool schemas advertised to the model.
#
# Returns:
#
#   An estimated prompt token count covering the system prompt, the history, and the tool schemas. Tool schemas render
#   before the system prompt and are far from free, so omitting them would understate a full request badly.
#
#-----------------------------------------------------------------------------------------------------------------------

def estimate_request_tokens ( system: Sequence [ "PromptBlock" ],
                              messages: Sequence [ dict [ str, Any ] ],
                              tools: Sequence [ dict [ str, Any ] ] | None = None ) -> int:

    # System prompt blocks.

    total = sum ( estimate_tokens ( block.text ) for block in system )

    # Conversation history.

    total += estimate_message_tokens ( messages )

    # Tool schemas, measured by their JSON form because that is what is transmitted.

    if tools:
        total += sum (
            estimate_tokens ( json.dumps ( tool, sort_keys = True ) ) for tool in tools
        )

    # Return data to caller.

    return total
