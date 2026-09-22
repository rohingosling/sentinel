#-----------------------------------------------------------------------------------------------------------------------
# Module:  timestamps.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   ISO 8601 encoding and decoding for every instant Sentinel persists.
#
#   Timestamps are stored as timezone-aware ISO 8601 strings throughout -- episodes, tasks, cron schedules, and
#   heartbeat ticks alike. That is a deliberate choice over an integer epoch:
#
#     * ISO 8601 sorts lexicographically in the same order it sorts chronologically, so a DESC index does real work
#       without a conversion function in the query.
#     * The stored value stays legible in a sqlite3 shell, which matters for a database the user owns and may inspect.
#
#   The cost is that every write must normalise, which is why these two functions live here rather than being written
#   out at each call site. Two copies of this logic that disagreed by a timezone would produce a database whose rows
#   sort correctly against their own table and incorrectly against every other.
#-----------------------------------------------------------------------------------------------------------------------

import logging

from datetime import UTC, datetime

logger = logging.getLogger ( __name__ )


#-----------------------------------------------------------------------------------------------------------------------
# Function: to_iso_timestamp
#
# Description:
#
#   Normalise a moment to a timezone-aware ISO 8601 string.
#
# Arguments:
#
#   moment : The instant to encode. The current UTC time when omitted.
#
# Returns:
#
#   The ISO 8601 representation. A naive datetime is assumed to be UTC rather than local: the agent runs unattended
#   across timezone changes and daylight-saving boundaries, and a wrong assumption that is consistent is recoverable
#   where one that follows the host's clock settings is not.
#
#-----------------------------------------------------------------------------------------------------------------------

def to_iso_timestamp ( moment: datetime | None = None ) -> str:

    instant = moment if moment is not None else datetime.now ( UTC )

    if instant.tzinfo is None:
        instant = instant.replace ( tzinfo = UTC )

    # Return data to caller.

    return instant.astimezone ( UTC ).isoformat ()


#-----------------------------------------------------------------------------------------------------------------------
# Function: from_iso_timestamp
#
# Description:
#
#   Parse a stored timestamp back into a datetime.
#
# Arguments:
#
#   text : The stored ISO 8601 string.
#
# Returns:
#
#   A timezone-aware datetime, or None when the value cannot be parsed. Returning None rather than raising is
#   deliberate: one unparseable row must not take down a whole retrieval, and the scorer treats an undated memory as
#   infinitely old, which is the safe direction.
#
#-----------------------------------------------------------------------------------------------------------------------

def from_iso_timestamp ( text: str | None ) -> datetime | None:

    if not text:
        return None

    try:
        instant = datetime.fromisoformat ( text )
    except ValueError:
        logger.warning ( "Ignoring an unparseable timestamp: %r.", text )

        return None

    # Return data to caller.

    return instant if instant.tzinfo is not None else instant.replace ( tzinfo = UTC )
