#-----------------------------------------------------------------------------------------------------------------------
# Module:  schemas.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   The event taxonomy and the log entry shape (architecture 3.2.10).
#
#   Two vocabularies live here, and they are enforced differently on purpose:
#
#     * The CATEGORY is closed. Eight categories, fixed by the architecture, and an unknown one is refused. Category is
#       the axis every query and every index is built on, so a ninth invented at a call site would produce events that
#       no listing shows and no dashboard counts -- a silent hole in an audit trail, which is the one failure an audit
#       trail may not have.
#     * The EVENT NAME is open within its category. The architecture names thirty; Phases 6 to 13 will add more, and
#       requiring a taxonomy edit before a new event can be logged would mean the first thing a hurried author does is
#       reach for the nearest existing name instead. An unrecognised name is accepted and warned about once.
#
#   The one structural rule that IS enforced on event names is the prefix: an event in category "tool" must be named
#   "tool.something". That makes a row self-describing without its category column, and catches the specific mistake of
#   filing "tool.invoke" under "agent", which no amount of open vocabulary should permit.
#
#   Redaction runs on the way in, not on the way out. A secret that reaches the table has already been written to disk
#   in the clear, and no amount of careful reading afterwards undoes that.
#-----------------------------------------------------------------------------------------------------------------------

import json
import logging
import re
import uuid

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sentinel            import __version__
from sentinel.errors     import EventLogError
from sentinel.timestamps import to_iso_timestamp

logger = logging.getLogger ( __name__ )

# The eight categories of architecture 3.2.10. Closed set -- see the module header.

CATEGORY_HEARTBEAT = "heartbeat"
CATEGORY_USER      = "user"
CATEGORY_AGENT     = "agent"
CATEGORY_TOOL      = "tool"
CATEGORY_MEMORY    = "memory"
CATEGORY_IDENTITY  = "identity"
CATEGORY_SECURITY  = "security"
CATEGORY_SYSTEM    = "system"

# Every event the architecture names, grouped by its category. Open set -- a name absent from here is accepted with a
# warning, because the alternative is authors reaching for an approximate existing name to avoid editing this table.

EVENT_TAXONOMY: dict [ str, frozenset [ str ] ] = {
    CATEGORY_HEARTBEAT: frozenset ( { "heartbeat.start", "heartbeat.end", "heartbeat.error" } ),
    CATEGORY_USER: frozenset ( { "user.message", "user.feedback", "user.command" } ),
    CATEGORY_AGENT: frozenset ( { "agent.response", "agent.decision", "agent.plan" } ),
    CATEGORY_TOOL: frozenset ( { "tool.invoke", "tool.result", "tool.error", "tool.timeout" } ),
    CATEGORY_MEMORY: frozenset ( { "memory.store", "memory.retrieve", "memory.prune" } ),
    CATEGORY_IDENTITY: frozenset ( { "identity.evolve", "identity.snapshot", "identity.rollback" } ),
    CATEGORY_SECURITY: frozenset ( { "security.allow", "security.deny", "security.escalate" } ),
    CATEGORY_SYSTEM: frozenset ( { "system.startup", "system.shutdown", "system.health" } ),
}

CATEGORIES = frozenset ( EVENT_TAXONOMY )

# Named constants for the events Sentinel's own code emits. A constant rather than a literal at each call site, so a
# rename is one edit and a typo is an import error rather than a row nobody ever queries for.

EVENT_HEARTBEAT_START = "heartbeat.start"
EVENT_HEARTBEAT_END   = "heartbeat.end"
EVENT_HEARTBEAT_ERROR = "heartbeat.error"

EVENT_USER_MESSAGE = "user.message"
EVENT_USER_COMMAND = "user.command"

EVENT_AGENT_RESPONSE = "agent.response"
EVENT_AGENT_DECISION = "agent.decision"

EVENT_TOOL_INVOKE = "tool.invoke"
EVENT_TOOL_RESULT = "tool.result"
EVENT_TOOL_ERROR  = "tool.error"

EVENT_SECURITY_DENY = "security.deny"

EVENT_SYSTEM_STARTUP  = "system.startup"
EVENT_SYSTEM_SHUTDOWN = "system.shutdown"
EVENT_SYSTEM_HEALTH   = "system.health"

# What replaces a redacted value. A fixed marker rather than removal of the key: the shape of what was logged stays
# visible, so a reader can see that a credential was passed and was not recorded, rather than seeing nothing at all.

REDACTION_MARKER = "[redacted]"

# Words that mean "this value is a credential". Matched against the key's own words rather than as substrings of it,
# and every one of them is SINGULAR on purpose.
#
# That is not a stylistic choice. "token" names a credential; "tokens" names a count, and the event log is full of
# them -- input_tokens, output_tokens, cache_read_tokens, max_tokens. Substring matching redacted every one of those,
# which quietly destroyed the most useful numeric data in the trail while protecting nothing. Credential fields are
# singular in every API worth naming (api_token, refresh_token, bot_token); counts are plural. The distinction is
# reliable, and it is what the word list encodes.
#
# "key" is included knowingly. It will occasionally redact a cache_key or a sort_key, and that trade is deliberate: a
# false positive costs a line of detail, and a false negative writes a credential to disk in the clear.

SENSITIVE_KEY_WORDS = frozenset (
    {
        "password", "passwd", "secret", "token", "credential", "credentials",
        "authorization", "auth", "key", "keys", "passphrase", "signature",
    }
)

# Compounds that carry no separator and so do not split into words. Matched against the key with every separator
# removed, which is what catches "apikey" and "APIKey" alike.

SENSITIVE_KEY_COMPOUNDS = ( "apikey", "accesskey", "privatekey", "secretkey", "authtoken", "sessionkey" )

# How a key is split into words: snake_case, kebab-case, dotted, and camelCase all reduce to the same word list, so
# "api_key", "apiKey", and "API-Key" are one case rather than three.

KEY_WORD_PATTERN = re.compile ( r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|\d+" )

# Credential shapes that turn up inside otherwise innocent free text -- a key pasted into a task description, a header
# echoed into an error message. Key-based redaction cannot see these, because they are values with no key of their own.

CREDENTIAL_PATTERNS = (
    re.compile ( r"sk-ant-[A-Za-z0-9_\-]{16,}" ),
    re.compile ( r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}" ),
)

# How deep redaction walks a nested payload before giving up and rendering the remainder as text. Event payloads are
# shallow by construction; the bound exists so a structure that somehow contains itself cannot hang the logger.

MAX_REDACTION_DEPTH = 8

# Event names already warned about, so an unrecognised name logs once rather than on every occurrence. A tick emitting
# an unknown event every 240 seconds would otherwise fill the console with the same line.

_warned_events: set [ str ] = set ()


#-----------------------------------------------------------------------------------------------------------------------
# Validation
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Function: validate_category
#
# Description:
#
#   Check a category against the closed set of eight.
#
# Arguments:
#
#   category : The category to check.
#
# Returns:
#
#   The category unchanged.
#
#   Raises EventLogError when the category is not one of the eight. Refused rather than warned about: category is what
#   every query filters on, so an invented one produces events that nothing will ever surface.
#
#-----------------------------------------------------------------------------------------------------------------------

def validate_category ( category: str ) -> str:

    if category not in CATEGORIES:
        raise EventLogError (
            f"Unknown event category {category!r}. The taxonomy of architecture 3.2.10 is closed; "
            f"it must be one of {sorted ( CATEGORIES )}."
        )

    # Return data to caller.

    return category


#-----------------------------------------------------------------------------------------------------------------------
# Function: validate_event
#
# Description:
#
#   Check an event name against its category.
#
#   The prefix rule is enforced; membership of the taxonomy is not. See the module header for why the two are treated
#   differently.
#
# Arguments:
#
#   category : The category the event belongs to. Assumed already validated.
#   event    : The dotted event name.
#
# Returns:
#
#   The event name unchanged.
#
#   Raises EventLogError when the name is empty or does not begin with its own category, which is the mistake of filing
#   an event under the wrong heading.
#
#-----------------------------------------------------------------------------------------------------------------------

def validate_event ( category: str, event: str ) -> str:

    if not event.strip ():
        raise EventLogError ( f"An event in category {category!r} was logged with no name." )

    prefix = f"{category}."

    if not event.startswith ( prefix ):
        raise EventLogError (
            f"Event {event!r} is filed under category {category!r} but does not begin with {prefix!r}. "
            f"An event name always carries its own category, so a row is readable without its category column."
        )

    # An unrecognised name is legitimate -- later phases add events -- but is worth saying once, because the other
    # reason a name is unrecognised is that it was misspelled.

    if event not in EVENT_TAXONOMY [ category ] and event not in _warned_events:
        _warned_events.add ( event )

        logger.debug (
            "Event %r is not in the architecture 3.2.10 taxonomy for category %r. Accepted; "
            "add it to EVENT_TAXONOMY if it is here to stay.",
            event, category,
        )

    # Return data to caller.

    return event


#-----------------------------------------------------------------------------------------------------------------------
# Redaction
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Function: redact_text
#
# Description:
#
#   Mask credential shapes inside a free-text value.
#
# Arguments:
#
#   text : The text to scan.
#
# Returns:
#
#   The text with any recognised credential replaced by the redaction marker.
#
#-----------------------------------------------------------------------------------------------------------------------

def redact_text ( text: str ) -> str:

    scrubbed = text

    for pattern in CREDENTIAL_PATTERNS:
        scrubbed = pattern.sub ( REDACTION_MARKER, scrubbed )

    # Return data to caller.

    return scrubbed


#-----------------------------------------------------------------------------------------------------------------------
# Function: is_sensitive_key
#
# Description:
#
#   Report whether a mapping key names a credential.
#
#   Two passes, because keys are written two ways. Most separate their words -- api_key, bot-token, session.secret --
#   and are judged word by word, which is what keeps "output_tokens" out of it. The rest run together, and are judged
#   against the compound list with the separators stripped.
#
# Arguments:
#
#   key : The key to judge.
#
# Returns:
#
#   True when any of the key's words names a credential, or the key compacts to a known compound.
#
#-----------------------------------------------------------------------------------------------------------------------

def is_sensitive_key ( key: str ) -> bool:

    words = { word.lower () for word in KEY_WORD_PATTERN.findall ( key ) }

    if words & SENSITIVE_KEY_WORDS:
        return True

    compacted = "".join ( character for character in key.lower () if character.isalnum () )

    # Return data to caller.

    return any ( compound in compacted for compound in SENSITIVE_KEY_COMPOUNDS )


#-----------------------------------------------------------------------------------------------------------------------
# Function: redact
#
# Description:
#
#   Remove credentials from an event payload before it is written anywhere.
#
#   Applied on the way in rather than on the way out. A secret that reaches the table has already been written to disk
#   in the clear, and every later reader being careful does not undo that.
#
# Arguments:
#
#   value : The payload to scrub -- a mapping, a sequence, or a scalar.
#   depth : Current recursion depth. Callers leave this alone.
#
# Returns:
#
#   The payload with sensitive values replaced by the redaction marker. Structure is preserved so the shape of what was
#   logged remains readable; only the values go.
#
#-----------------------------------------------------------------------------------------------------------------------

def redact ( value: Any, depth: int = 0 ) -> Any:

    # Past the depth bound, render whatever is left as scrubbed text rather than descending further.

    if depth >= MAX_REDACTION_DEPTH:
        return redact_text ( str ( value ) )

    if isinstance ( value, dict ):
        return {
            str ( key ): REDACTION_MARKER if is_sensitive_key ( str ( key ) ) else redact ( item, depth + 1 )
            for key, item in value.items ()
        }

    if isinstance ( value, ( list, tuple ) ):
        return [ redact ( item, depth + 1 ) for item in value ]

    if isinstance ( value, str ):
        return redact_text ( value )

    # Return data to caller.

    return value


#-----------------------------------------------------------------------------------------------------------------------
# Types
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Class: EventMetadata
#
# Description:
#
#   The envelope carried by every event, from architecture 3.2.10.
#
#   Separate from `data` because these three answer "which agent produced this", not "what happened". Keeping them out
#   of the payload means a query for "everything the agent did before identity version 4" does not have to know the
#   shape of thirty different payloads.
#
# Attributes:
#
#   agent_version    : Sentinel's own version string.
#   identity_version : Identity generation in force. Zero until Phase 8 supplies one.
#   session_id       : Conversation this event belongs to, or None outside a session.
#-----------------------------------------------------------------------------------------------------------------------

class EventMetadata ( BaseModel ):

    model_config = ConfigDict ( extra = "allow" )

    agent_version:    str        = __version__
    identity_version: int        = 0
    session_id:       str | None = None


#-----------------------------------------------------------------------------------------------------------------------
# Class: LogEvent
#
# Description:
#
#   One entry in the append-only event log.
#
# Attributes:
#
#   id             : UUID.
#   timestamp      : ISO 8601 instant, timezone-aware.
#   category       : One of the eight categories.
#   event          : Dotted event name, prefixed by its category.
#   source         : Component that emitted it.
#   correlation_id : Ties every event of one turn together, or None outside a turn.
#   data           : Event-specific payload, already redacted.
#   metadata       : The envelope.
#-----------------------------------------------------------------------------------------------------------------------

class LogEvent ( BaseModel ):

    model_config = ConfigDict ( extra = "forbid" )

    id:             str
    timestamp:      str
    category:       str
    event:          str
    source:         str | None        = None
    correlation_id: str | None        = None
    data:           dict [ str, Any ] = Field ( default_factory = dict )
    metadata:       EventMetadata     = Field ( default_factory = EventMetadata )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: as_row
    #
    # Description:
    #
    #   Render the event as the eight column values the event_log table takes.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   The values in the order the INSERT expects. Both JSON columns are serialised here rather than at the call site,
    #   so the file exporter and the database can never disagree about what a payload looked like.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def as_row ( self ) -> tuple [ object, ... ]:

        # Return data to caller.

        return (
            self.id,
            self.timestamp,
            self.category,
            self.event,
            self.source,
            self.correlation_id,
            json.dumps ( self.data, default = str ),
            json.dumps ( self.metadata.model_dump (), default = str ),
        )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: as_dict
    #
    # Description:
    #
    #   Render the event as plain JSON-serialisable data.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   The event as a mapping, for an API response body or a file exporter.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def as_dict ( self ) -> dict [ str, Any ]:

        # Return data to caller.

        return self.model_dump ()

    #-------------------------------------------------------------------------------------------------------------------
    # Function: as_json
    #
    # Description:
    #
    #   Render the event as one line of JSON.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   A single-line JSON object, with no trailing newline. One event per line is what makes the exported files
    #   greppable and streamable without a parser that understands the whole file.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def as_json ( self ) -> str:

        # Return data to caller.

        return json.dumps ( self.as_dict (), default = str, separators = ( ",", ":" ) )


#-----------------------------------------------------------------------------------------------------------------------
# Construction
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Function: build_event
#
# Description:
#
#   Validate, redact, and assemble one log entry.
#
#   The single place an event comes into existence, so nothing can reach the table having skipped either check.
#
# Arguments:
#
#   category         : One of the eight categories.
#   event            : Dotted event name.
#   data             : Event-specific payload. Redacted here.
#   correlation_id   : Turn identifier, or None.
#   source           : Component emitting the event.
#   session_id       : Conversation this belongs to, or None.
#   identity_version : Identity generation in force.
#   timestamp        : Instant to record. Now when omitted.
#   event_id         : Identifier to use. Generated when omitted.
#
# Returns:
#
#   The assembled event.
#
#   Raises EventLogError when the category or the event name is not acceptable.
#
#-----------------------------------------------------------------------------------------------------------------------

def build_event ( category: str,
                  event: str,
                  data: dict [ str, Any ] | None = None,
                  correlation_id: str | None     = None,
                  source: str | None             = None,
                  session_id: str | None         = None,
                  identity_version: int          = 0,
                  timestamp: str | None          = None,
                  event_id: str | None = None ) -> LogEvent:

    validate_category ( category )
    validate_event ( category, event )

    scrubbed = redact ( data or {} )

    # Return data to caller.

    return LogEvent (
        id             = event_id if event_id is not None else str ( uuid.uuid4 () ),
        timestamp      = timestamp if timestamp is not None else to_iso_timestamp (),
        category       = category,
        event          = event,
        source         = source,
        correlation_id = correlation_id,
        data           = scrubbed if isinstance ( scrubbed, dict ) else { "value": scrubbed },
        metadata = EventMetadata (
            identity_version = identity_version,
            session_id       = session_id,
        ),
    )


#-----------------------------------------------------------------------------------------------------------------------
# Function: row_to_event
#
# Description:
#
#   Map one event_log row back onto a LogEvent.
#
# Arguments:
#
#   row : A row selected as id, timestamp, category, event, source, correlation_id, data, metadata.
#
# Returns:
#
#   The reconstructed event. A JSON column that will not decode yields an empty payload rather than raising, so one
#   corrupt row does not take down a whole listing -- the same treatment the heartbeat's JSON columns get.
#
#-----------------------------------------------------------------------------------------------------------------------

def row_to_event ( row: tuple [ object, ... ] ) -> LogEvent:

    # Return data to caller.

    return LogEvent (
        id             = str ( row [ 0 ] ),
        timestamp      = str ( row [ 1 ] ),
        category       = str ( row [ 2 ] ),
        event          = str ( row [ 3 ] ),
        source         = str ( row [ 4 ] ) if row [ 4 ] is not None else None,
        correlation_id = str ( row [ 5 ] ) if row [ 5 ] is not None else None,
        data           = decode_json_object ( row [ 6 ] ),
        metadata       = EventMetadata.model_validate ( decode_json_object ( row [ 7 ] ) ),
    )


#-----------------------------------------------------------------------------------------------------------------------
# Function: decode_json_object
#
# Description:
#
#   Decode a stored JSON object column.
#
# Arguments:
#
#   value : The stored column value.
#
# Returns:
#
#   The decoded mapping, or an empty mapping when the value is absent, unparseable, or not an object.
#
#-----------------------------------------------------------------------------------------------------------------------

def decode_json_object ( value: object ) -> dict [ str, Any ]:

    if not isinstance ( value, str ) or not value:
        return {}

    try:
        decoded = json.loads ( value )
    except json.JSONDecodeError:
        logger.warning ( "Ignoring an unparseable event_log JSON column: %r.", value [ : 80 ] )

        return {}

    # Return data to caller.

    return decoded if isinstance ( decoded, dict ) else {}
