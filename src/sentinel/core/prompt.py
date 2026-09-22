#-----------------------------------------------------------------------------------------------------------------------
# Module:  prompt.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   System prompt assembly (architecture 3.3.5).
#
#   Sections are ordered by volatility, not by topic, and the cache breakpoint sits at the boundary between the two
#   regions:
#
#       STABLE      1. base persona           identity, traits            changes on identity evolution (daily)
#                   2. permission boundaries  tools allowed and denied    changes on permission or skill change
#                   3. learned preferences    user communication style    changes on identity evolution (daily)
#                   -- cache_control breakpoint ------------------------------------------------------------------
#       VOLATILE    4. datetime + environment current time, timezone      changes every request
#                   5. memory context         retrieved memories          re-retrieved per query
#                   6. channel context        originating channel         only for a ChannelMessage trigger
#                   7. heartbeat context      pending scheduled work      only for a HeartbeatTick trigger
#
#   This ordering is Decision 3 and it is not cosmetic. Caching is a prefix match, so one volatile byte placed above
#   a stable section makes everything below it uncacheable. The original section order in the architecture document put
#   memory before permissions and datetime before preferences, which meant nothing after section 2 could ever cache.
#
#   Identity and memory are placeholders in Phase 1 -- Phases 3 and 8 populate them. The section *slots* exist now
#   because retrofitting the ordering after prompt assembly is written is exactly the cost this design avoids.
#-----------------------------------------------------------------------------------------------------------------------

import logging
import platform

from datetime import UTC, datetime, tzinfo
from typing   import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sentinel.config      import PromptConfig, SentinelConfig
from sentinel.llm.adapter import PromptBlock
from sentinel.llm.tokens  import CHARACTERS_PER_TOKEN, estimate_tokens

logger = logging.getLogger ( __name__ )

# Section headings, emitted verbatim so a prompt is legible in a log or a diff.

HEADING_PERSONA     = "# Identity"
HEADING_PERMISSIONS = "# Permissions"
HEADING_PREFERENCES = "# Learned preferences"
HEADING_ENVIRONMENT = "# Environment"
HEADING_MEMORY      = "# Retrieved memory"
HEADING_CHANNEL     = "# Channel context"
HEADING_HEARTBEAT   = "# Pending work"

# Marker appended to a section that had to be cut to fit its budget. Visible on purpose:
# silent truncation reads as "there was nothing more to say".

TRUNCATION_MARKER = "\n[... truncated to fit the section budget]"


#-----------------------------------------------------------------------------------------------------------------------
# Function: truncate_to_tokens
#
# Description:
#
#   Cut text to an approximate token budget.
#
# Arguments:
#
#   text         : The text to cut.
#   token_budget : Ceiling, in tokens.
#
# Returns:
#
#   The text unchanged when it already fits, otherwise a prefix carrying a visible truncation marker. The cut is made on
#   the local character-per-token estimate, so it is approximate -- a section budget is a guard against one source
#   swallowing the window, not an exact accounting.
#
#-----------------------------------------------------------------------------------------------------------------------

def truncate_to_tokens ( text: str, token_budget: int ) -> str:

    # Nothing to do when it already fits.

    if estimate_tokens ( text ) <= token_budget:
        return text

    # Leave room for the marker itself.

    marker_tokens    = estimate_tokens ( TRUNCATION_MARKER )
    character_budget = int ( max ( token_budget - marker_tokens, 1 ) * CHARACTERS_PER_TOKEN )

    # Return data to caller.

    return text [ : character_budget ].rstrip () + TRUNCATION_MARKER


#-----------------------------------------------------------------------------------------------------------------------
# Section builders
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Function: build_persona_section
#
# Description:
#
#   Build section 1: base persona.
#
#   Phase 1 emits the configured agent name and a default character. Phase 8 replaces the body with the evolving trait
#   vector, at which point this section starts changing daily rather than never.
#
# Arguments:
#
#   configuration : The loaded configuration.
#   identity      : Identity state from Phase 8. None until then.
#
# Returns:
#
#   The rendered section text.
#
#-----------------------------------------------------------------------------------------------------------------------

def build_persona_section ( configuration: SentinelConfig,
                            identity: dict [ str, Any ] | None = None ) -> str:

    # Start from the configured name and language.

    name = configuration.agent.name
    lines = [
        HEADING_PERSONA,
        "",
        f"You are {name}, an autonomous personal AI agent running as a desktop application "
        f"on the user's own machine.",
        "",
        f"Respond in {configuration.agent.language} unless the user writes in another language.",
    ]

    # Trait-influenced parameters, once Phase 8 supplies them.

    if identity:
        traits = identity.get ( "traits" ) or {}

        if traits:
            lines.append ( "" )
            lines.append ( "Calibrate your manner to these traits, each scored from 0.0 to 1.0:" )
            lines.append ( "" )

            for trait, value in sorted ( traits.items () ):
                lines.append ( f"  {trait}: {value}" )

        version = identity.get ( "version" )

        if version is not None:
            lines.append ( "" )
            lines.append ( f"Identity version {version}." )

    # Return data to caller.

    return "\n".join ( lines )


#-----------------------------------------------------------------------------------------------------------------------
# Function: build_permissions_section
#
# Description:
#
#   Build section 2: permission boundaries.
#
# Arguments:
#
#   allowed : Tool names the agent may use.
#   denied  : Tool names the agent must not use.
#
# Returns:
#
#   The rendered section text. States plainly that no tools exist yet when both lists are empty, rather than leaving the
#   model to infer its own capabilities.
#
#-----------------------------------------------------------------------------------------------------------------------

def build_permissions_section ( allowed: list [ str ] | None = None,
                                denied: list [ str ] | None = None ) -> str:

    # Render what is permitted and what is refused.

    allowed_names = sorted ( allowed or [] )
    denied_names  = sorted ( denied or [] )

    lines = [ HEADING_PERMISSIONS, "" ]

    if allowed_names:
        lines.append ( "You have access to the following tools: " + ", ".join ( allowed_names ) + "." )
    else:
        lines.append (
            "You currently have no tools. Answer from your own knowledge and the context you "
            "are given, and say so plainly when a request would need a capability you lack."
        )

    if denied_names:
        lines.append ( "You may not access: " + ", ".join ( denied_names ) + "." )

    # Return data to caller.

    return "\n".join ( lines )


#-----------------------------------------------------------------------------------------------------------------------
# Function: build_preferences_section
#
# Description:
#
#   Build section 3: learned preferences.
#
# Arguments:
#
#   preferences : Preference statements extracted from identity state.
#
# Returns:
#
#   The rendered section text, or an empty string when nothing has been learned yet. An empty section is omitted rather
#   than emitted as a heading with no body -- an empty heading costs cacheable tokens and says nothing.
#
#-----------------------------------------------------------------------------------------------------------------------

def build_preferences_section ( preferences: list [ str ] | None = None ) -> str:

    # Omit the section entirely when there is nothing to say.

    if not preferences:
        return ""

    lines = [ HEADING_PREFERENCES, "" ]

    for preference in preferences:
        lines.append ( f"- {preference}" )

    # Return data to caller.

    return "\n".join ( lines )


#-----------------------------------------------------------------------------------------------------------------------
# Function: build_environment_section
#
# Description:
#
#   Build section 4: datetime and environment.
#
#   This is the first volatile section: it changes on every single request, which is precisely why it sits below the
#   cache breakpoint.
#
# Arguments:
#
#   configuration : The loaded configuration, for the timezone.
#   moment        : The current time. Read from the clock when omitted.
#
# Returns:
#
#   The rendered section text.
#
#-----------------------------------------------------------------------------------------------------------------------

def build_environment_section ( configuration: SentinelConfig,
                                moment: datetime | None = None ) -> str:

    # Resolve the configured timezone, falling back to UTC rather than failing a turn.
    #
    # The fallback uses datetime.UTC, not ZoneInfo("UTC"): Windows ships no system zone
    # database, so on a machine without the tzdata package every ZoneInfo lookup fails --
    # including the fallback itself. tzdata is a declared dependency for exactly this
    # reason, but a stdlib constant is the one fallback that cannot fail.

    try:
        zone: tzinfo = ZoneInfo ( configuration.agent.timezone )
    except ( ZoneInfoNotFoundError, ValueError, KeyError ):
        logger.warning (
            "Timezone %r could not be resolved. Falling back to UTC. On Windows this usually "
            "means the tzdata package is missing.",
            configuration.agent.timezone,
        )

        zone = UTC

    now = moment.astimezone ( zone ) if moment is not None else datetime.now ( zone )

    lines = [
        HEADING_ENVIRONMENT,
        "",
        f"Current time: {now.isoformat ( timespec = 'seconds' )}",
        f"Timezone:     {configuration.agent.timezone}",
        f"Platform:     {platform.system ()} {platform.release ()}",
    ]

    # Return data to caller.

    return "\n".join ( lines )


#-----------------------------------------------------------------------------------------------------------------------
# Function: build_memory_section
#
# Description:
#
#   Build section 5: retrieved memory.
#
# Arguments:
#
#   memories : Retrieved memories, each carrying a summary and optionally a timestamp and score. Ordered by relevance
#              descending by the caller.
#
# Returns:
#
#   The rendered section text, or an empty string when nothing was retrieved.
#
#-----------------------------------------------------------------------------------------------------------------------

def build_memory_section ( memories: list [ dict [ str, Any ] ] | None = None ) -> str:

    # Omit the section entirely when retrieval found nothing.

    if not memories:
        return ""

    lines = [ HEADING_MEMORY, "" ]

    for memory in memories:
        summary   = str ( memory.get ( "summary", "" ) ).strip ()
        timestamp = memory.get ( "timestamp" )

        if not summary:
            continue

        lines.append ( f"[Memory] {summary} ({timestamp})" if timestamp else f"[Memory] {summary}" )

    # Return data to caller.

    return "\n".join ( lines )


#-----------------------------------------------------------------------------------------------------------------------
# Function: build_channel_section
#
# Description:
#
#   Build section 6: channel context.
#
# Arguments:
#
#   channel : Channel metadata: name, type, sender, and any constraints such as a message length cap.
#
# Returns:
#
#   The rendered section text, or an empty string for a non-channel trigger.
#
#-----------------------------------------------------------------------------------------------------------------------

def build_channel_section ( channel: dict [ str, Any ] | None = None ) -> str:

    # Omit the section entirely unless the turn arrived over a channel.

    if not channel:
        return ""

    lines = [ HEADING_CHANNEL, "" ]

    for label, key in ( ( "Channel", "name" ), ( "Type", "type" ), ( "Sender", "sender" ) ):
        value = channel.get ( key )

        if value:
            lines.append ( f"{label}: {value}" )

    # Channel constraints are the reason this section exists -- a 4096-character cap the
    # model does not know about produces a reply that gets silently cut in half.

    constraints = channel.get ( "constraints" )

    if constraints:
        lines.append ( "" )
        lines.append ( "Constraints:" )

        if isinstance ( constraints, dict ):
            for key, value in sorted ( constraints.items () ):
                lines.append ( f"  {key}: {value}" )
        else:
            lines.append ( f"  {constraints}" )

    # Return data to caller.

    return "\n".join ( lines )


#-----------------------------------------------------------------------------------------------------------------------
# Function: build_heartbeat_section
#
# Description:
#
#   Build section 7: heartbeat context.
#
# Arguments:
#
#   heartbeat : Heartbeat state: pending tasks, scheduled jobs, and signals.
#
# Returns:
#
#   The rendered section text, or an empty string for any trigger other than a heartbeat tick.
#
#-----------------------------------------------------------------------------------------------------------------------

def build_heartbeat_section ( heartbeat: dict [ str, Any ] | None = None ) -> str:

    # Omit the section entirely unless this turn is a heartbeat tick.

    if not heartbeat:
        return ""

    lines = [ HEADING_HEARTBEAT, "" ]

    lines.append (
        "This turn was triggered by an autonomous heartbeat tick, not by the user. "
        "Act only if something below genuinely needs doing; ending the turn without "
        "output is a valid and often correct outcome."
    )

    pending = heartbeat.get ( "pending_tasks" ) or []

    if pending:
        lines.append ( "" )
        lines.append ( "Pending tasks:" )

        for task in pending:
            lines.append ( f"- {task}" )

    scheduled = heartbeat.get ( "scheduled_jobs" ) or []

    if scheduled:
        lines.append ( "" )
        lines.append ( "Scheduled jobs:" )

        for job in scheduled:
            lines.append ( f"- {job}" )

    # Return data to caller.

    return "\n".join ( lines )


#-----------------------------------------------------------------------------------------------------------------------
# Assembly
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Function: assemble
#
# Description:
#
#   Assemble the system prompt, stable sections before volatile ones.
#
#   Every stable section is joined into one block carrying the cache breakpoint, and every volatile one into a second
#   block carrying none. Two blocks rather than seven is deliberate: the API allows only four breakpoints per request,
#   and one boundary is all this design needs.
#
# Arguments:
#
#   configuration : The loaded configuration.
#   identity      : Identity state (Phase 8).
#   allowed_tools : Tool names the agent may use.
#   denied_tools  : Tool names the agent must not use.
#   preferences   : Learned user preferences (Phase 8).
#   memories      : Retrieved memories (Phase 3).
#   channel       : Channel metadata, for a ChannelMessage trigger.
#   heartbeat     : Heartbeat state, for a HeartbeatTick trigger.
#   moment        : The current time. Read from the clock when omitted.
#
# Returns:
#
#   Prompt blocks in assembly order: the stable block first, carrying cached=True, then the volatile block. Two
#   guarantees hold and are tested:
#
#     * Exactly one block carries the breakpoint (T1.29).
#     * Two calls differing only in volatile inputs produce a byte-identical stable block (T1.28).
#
#-----------------------------------------------------------------------------------------------------------------------

def assemble ( configuration: SentinelConfig,
               identity: dict [ str, Any ] | None          = None,
               allowed_tools: list [ str ] | None          = None,
               denied_tools: list [ str ] | None           = None,
               preferences: list [ str ] | None            = None,
               memories: list [ dict [ str, Any ] ] | None = None,
               channel: dict [ str, Any ] | None           = None,
               heartbeat: dict [ str, Any ] | None         = None,
               moment: datetime | None = None ) -> list [ PromptBlock ]:

    budget: PromptConfig = configuration.prompt

    # Stable region: sections 1 to 3. Each is truncated to its own budget.

    stable_sections = [
        truncate_to_tokens ( build_persona_section ( configuration, identity ), budget.max_persona ),
        truncate_to_tokens (
            build_permissions_section ( allowed_tools, denied_tools ), budget.max_permissions
        ),
        truncate_to_tokens ( build_preferences_section ( preferences ), budget.max_preferences ),
    ]

    # Volatile region: sections 4 to 7.

    volatile_sections = [
        build_environment_section ( configuration, moment ),
        truncate_to_tokens ( build_memory_section ( memories ), budget.max_memory ),
        truncate_to_tokens ( build_channel_section ( channel ), budget.max_channel ),
        truncate_to_tokens ( build_heartbeat_section ( heartbeat ), budget.max_heartbeat ),
    ]

    stable_text   = "\n\n".join ( section for section in stable_sections if section )
    volatile_text = "\n\n".join ( section for section in volatile_sections if section )

    # The stable block carries the breakpoint. Everything above and including it caches;
    # everything below it never does.

    blocks = [ PromptBlock ( text = stable_text, cached = True ) ]

    if volatile_text:
        blocks.append ( PromptBlock ( text = volatile_text, cached = False ) )

    # Return data to caller.

    return blocks


#-----------------------------------------------------------------------------------------------------------------------
# Function: stable_prefix
#
# Description:
#
#   Return the text of the cacheable prefix of an assembled prompt.
#
#   Exists for testing and diagnosis: the cheapest way to find a silent cache invalidator is to diff this string across
#   two requests that should have shared it.
#
# Arguments:
#
#   blocks : Assembled prompt blocks.
#
# Returns:
#
#   The concatenated text of every block up to and including the one carrying the breakpoint.
#
#-----------------------------------------------------------------------------------------------------------------------

def stable_prefix ( blocks: list [ PromptBlock ] ) -> str:

    # Collect blocks up to and including the breakpoint.

    collected: list [ str ] = []

    for block in blocks:
        collected.append ( block.text )

        if block.cached:
            break

    # Return data to caller.

    return "\n\n".join ( collected )
