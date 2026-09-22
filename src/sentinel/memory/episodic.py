#-----------------------------------------------------------------------------------------------------------------------
# Module:  episodic.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   Tier 2: episodic memory (architecture 3.2.4).
#
#   What happened, and when. Conversation summaries, interaction events, tool outcomes, and heartbeat results, indexed
#   by timestamp and by category. This is the recency arm of retrieval -- the tier that answers "what were we just
#   doing", which semantic similarity alone cannot, because the most recent thing is rarely the most similar one.
#
#   Timestamps are timezone-aware ISO 8601 strings, encoded and decoded by sentinel.timestamps -- see that module for
#   why, and for what a naive datetime is taken to mean.
#
#   Tags are a JSON array queried through json_each(). SQLite has no array type and no native JSON index; extracting
#   tags into a side table would buy indexed lookup at the cost of a join on every read. At the scale of one person's
#   memories the scan is cheaper than the schema.
#-----------------------------------------------------------------------------------------------------------------------

import json
import logging
import uuid

from dataclasses import dataclass, field
from datetime    import UTC, datetime

import aiosqlite

from sentinel.database   import DATABASE_FAULTS
from sentinel.errors     import DatabaseError
from sentinel.timestamps import from_iso_timestamp, to_iso_timestamp

logger = logging.getLogger ( __name__ )

# Re-exported so `from sentinel.memory.episodic import to_iso_timestamp` keeps working. The definitions moved to
# sentinel.timestamps in Phase 4, when the task queue needed exactly the same normalisation and two copies that had to
# agree became the worse option.

__all__ = [
    "CATEGORY_CONVERSATION",
    "EPISODE_COLUMNS",
    "Episode",
    "EpisodicMemory",
    "from_iso_timestamp",
    "row_to_episode",
    "to_iso_timestamp",
]

# Every column a caller sees, in one place, so the row-to-Episode mapping cannot drift from the SELECT list.

EPISODE_COLUMNS = "id, timestamp, category, summary, tags, session_id, created_at"

# The default category. Named rather than defaulted inline so the value that ends up in the majority of rows is
# greppable.

CATEGORY_CONVERSATION = "conversation"


#-----------------------------------------------------------------------------------------------------------------------
# Class: Episode
#
# Description:
#
#   One stored episode.
#
# Attributes:
#
#   id         : UUID.
#   timestamp  : ISO 8601 instant the episode describes.
#   category   : Episode kind, e.g. "conversation" or "heartbeat".
#   summary    : What happened, in prose.
#   tags       : Free-form labels.
#   session_id : Session the episode belongs to, when it belongs to one.
#   created_at : When the row was written, which differs from timestamp for backfilled history.
#-----------------------------------------------------------------------------------------------------------------------

@dataclass ( frozen = True )
class Episode:

    id:         str
    timestamp:  str
    category:   str
    summary:    str
    tags:       list [ str ] = field ( default_factory = list )
    session_id: str | None   = None
    created_at: str | None   = None

    #-------------------------------------------------------------------------------------------------------------------
    # Function: moment
    #
    # Description:
    #
    #   The episode's timestamp as a datetime.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   A timezone-aware datetime, or None when the stored value cannot be parsed.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def moment ( self ) -> datetime | None:

        # Return data to caller.

        return from_iso_timestamp ( self.timestamp )


#-----------------------------------------------------------------------------------------------------------------------
# Function: row_to_episode
#
# Description:
#
#   Build an Episode from a database row selected with EPISODE_COLUMNS.
#
# Arguments:
#
#   row : The row, in EPISODE_COLUMNS order.
#
# Returns:
#
#   The Episode. A tags column that is not a JSON array decodes to an empty list rather than raising -- a malformed
#   tag list must not make the episode itself unreadable.
#
#-----------------------------------------------------------------------------------------------------------------------

def row_to_episode ( row: tuple [ object, ... ] | aiosqlite.Row ) -> Episode:

    # Decode the tag list defensively.

    tags: list [ str ] = []

    if row [ 4 ]:
        try:
            decoded = json.loads ( str ( row [ 4 ] ) )

            if isinstance ( decoded, list ):
                tags = [ str ( tag ) for tag in decoded ]

        except json.JSONDecodeError:
            logger.warning ( "Ignoring an unreadable tag list on episode %s.", row [ 0 ] )

    # Return data to caller.

    return Episode (
        id         = str ( row [ 0 ] ),
        timestamp  = str ( row [ 1 ] ),
        category   = str ( row [ 2 ] ),
        summary    = str ( row [ 3 ] ),
        tags       = tags,
        session_id = str ( row [ 5 ] ) if row [ 5 ] is not None else None,
        created_at = str ( row [ 6 ] ) if row [ 6 ] is not None else None,
    )


#-----------------------------------------------------------------------------------------------------------------------
# Class: EpisodicMemory
#
# Description:
#
#   Durable, time-ordered episode storage.
#
#   Holds a connection rather than opening one: the whole agent shares a single SQLite connection so that WAL mode's
#   single-writer rule is honoured by construction rather than by discipline.
#
# Attributes:
#
#   connection : The shared aiosqlite connection.
#-----------------------------------------------------------------------------------------------------------------------

class EpisodicMemory:

    #-------------------------------------------------------------------------------------------------------------------
    # Function: __init__
    #
    # Description:
    #
    #   Bind to an open connection.
    #
    # Arguments:
    #
    #   connection : An open aiosqlite connection whose schema has been migrated.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def __init__ ( self, connection: aiosqlite.Connection ) -> None:

        self.connection = connection

    #-------------------------------------------------------------------------------------------------------------------
    # Writes
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: store
    #
    # Description:
    #
    #   Record one episode.
    #
    # Arguments:
    #
    #   summary    : What happened.
    #   category   : Episode kind.
    #   tags       : Free-form labels.
    #   session_id : Session this belongs to.
    #   moment     : The instant the episode describes. Now when omitted.
    #   episode_id : Explicit identifier. A fresh UUID when omitted.
    #
    # Returns:
    #
    #   The episode's identifier.
    #
    #   Raises DatabaseError if the row could not be written.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def store ( self,
                      summary: str,
                      category: str             = CATEGORY_CONVERSATION,
                      tags: list [ str ] | None = None,
                      session_id: str | None    = None,
                      moment: datetime | None   = None,
                      episode_id: str | None = None ) -> str:

        identifier = episode_id if episode_id is not None else str ( uuid.uuid4 () )

        try:
            await self.connection.execute (
                "INSERT INTO episodes ( id, timestamp, category, summary, tags, session_id ) "
                "VALUES ( ?, ?, ?, ?, ?, ? )",
                (
                    identifier,
                    to_iso_timestamp ( moment ),
                    category,
                    summary,
                    json.dumps ( tags or [] ),
                    session_id,
                ),
            )

            await self.connection.commit ()

        except DATABASE_FAULTS as error:
            raise DatabaseError ( f"Could not store an episode: {error}" ) from error

        # Return data to caller.

        return identifier

    #-------------------------------------------------------------------------------------------------------------------
    # Function: prune
    #
    # Description:
    #
    #   Delete episodes older than the retention window.
    #
    # Arguments:
    #
    #   retention_days : Days of history to keep. Zero or negative keeps everything, because "retain nothing" is far
    #                    more likely to be an unset value than a deliberate instruction to erase the agent's past.
    #   moment         : The instant to measure from. Now when omitted.
    #
    # Returns:
    #
    #   The number of episodes removed.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def prune ( self, retention_days: int, moment: datetime | None = None ) -> int:

        if retention_days <= 0:
            return 0

        from datetime import timedelta

        now    = moment if moment is not None else datetime.now ( UTC )
        cutoff = to_iso_timestamp ( now - timedelta ( days = retention_days ) )

        try:
            cursor = await self.connection.execute ( "DELETE FROM episodes WHERE timestamp < ?", ( cutoff, ) )

            await self.connection.commit ()

        except DATABASE_FAULTS as error:
            raise DatabaseError ( f"Could not prune episodes older than {cutoff}: {error}" ) from error

        removed = cursor.rowcount if cursor.rowcount is not None and cursor.rowcount > 0 else 0

        if removed:
            logger.info ( "Pruned %d episode(s) older than %s.", removed, cutoff )

        # Return data to caller.

        return removed

    #-------------------------------------------------------------------------------------------------------------------
    # Reads
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: get
    #
    # Description:
    #
    #   Read one episode by identifier.
    #
    # Arguments:
    #
    #   episode_id : The identifier to look up.
    #
    # Returns:
    #
    #   The Episode, or None when no such row exists.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def get ( self, episode_id: str ) -> Episode | None:

        async with self.connection.execute (
            f"SELECT {EPISODE_COLUMNS} FROM episodes WHERE id = ?", ( episode_id, )
        ) as cursor:
            row = await cursor.fetchone ()

        # Return data to caller.

        return row_to_episode ( row ) if row else None

    #-------------------------------------------------------------------------------------------------------------------
    # Function: recent
    #
    # Description:
    #
    #   The most recent episodes, newest first.
    #
    #   The recency arm of the retrieval algorithm.
    #
    # Arguments:
    #
    #   limit    : Maximum rows to return.
    #   before   : Return only episodes strictly older than this instant. Unbounded when omitted.
    #   category : Restrict to one category. Every category when omitted.
    #
    # Returns:
    #
    #   Episodes ordered by timestamp descending.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def recent ( self,
                       limit: int              = 10,
                       before: datetime | None = None,
                       category: str | None = None ) -> list [ Episode ]:

        # Built as a fixed skeleton with appended predicates, so every value still travels as a bound parameter.

        query = f"SELECT {EPISODE_COLUMNS} FROM episodes"
        predicates: list [ str ]    = []
        values:     list [ object ] = []

        if before is not None:
            predicates.append ( "timestamp < ?" )
            values.append ( to_iso_timestamp ( before ) )

        if category is not None:
            predicates.append ( "category = ?" )
            values.append ( category )

        if predicates:
            query += " WHERE " + " AND ".join ( predicates )

        query += " ORDER BY timestamp DESC LIMIT ?"

        values.append ( max ( int ( limit ), 0 ) )

        async with self.connection.execute ( query, tuple ( values ) ) as cursor:
            rows = await cursor.fetchall ()

        # Return data to caller.

        return [ row_to_episode ( row ) for row in rows ]

    #-------------------------------------------------------------------------------------------------------------------
    # Function: between
    #
    # Description:
    #
    #   Episodes within a timestamp range, newest first.
    #
    # Arguments:
    #
    #   start : Inclusive lower bound. Unbounded when omitted.
    #   end   : Exclusive upper bound. Unbounded when omitted.
    #   limit : Maximum rows to return.
    #
    # Returns:
    #
    #   Episodes in the range, ordered by timestamp descending. The upper bound is exclusive so that adjacent ranges
    #   tile without double-counting the boundary instant.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def between ( self,
                        start: datetime | None = None,
                        end: datetime | None   = None,
                        limit: int = 100 ) -> list [ Episode ]:

        query = f"SELECT {EPISODE_COLUMNS} FROM episodes"
        predicates: list [ str ]    = []
        values:     list [ object ] = []

        if start is not None:
            predicates.append ( "timestamp >= ?" )
            values.append ( to_iso_timestamp ( start ) )

        if end is not None:
            predicates.append ( "timestamp < ?" )
            values.append ( to_iso_timestamp ( end ) )

        if predicates:
            query += " WHERE " + " AND ".join ( predicates )

        query += " ORDER BY timestamp DESC LIMIT ?"

        values.append ( max ( int ( limit ), 0 ) )

        async with self.connection.execute ( query, tuple ( values ) ) as cursor:
            rows = await cursor.fetchall ()

        # Return data to caller.

        return [ row_to_episode ( row ) for row in rows ]

    #-------------------------------------------------------------------------------------------------------------------
    # Function: by_category
    #
    # Description:
    #
    #   Episodes of one category, newest first.
    #
    # Arguments:
    #
    #   category : The category to match.
    #   limit    : Maximum rows to return.
    #
    # Returns:
    #
    #   Matching episodes, ordered by timestamp descending.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def by_category ( self, category: str, limit: int = 100 ) -> list [ Episode ]:

        # Return data to caller.

        return await self.recent ( limit = limit, category = category )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: by_tag
    #
    # Description:
    #
    #   Episodes carrying a given tag, newest first.
    #
    #   Matched through json_each() rather than a LIKE over the raw JSON, which would match "python" inside
    #   "python-debugging" and inside a summary that merely mentions it.
    #
    # Arguments:
    #
    #   tag   : The exact tag to match.
    #   limit : Maximum rows to return.
    #
    # Returns:
    #
    #   Matching episodes, ordered by timestamp descending.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def by_tag ( self, tag: str, limit: int = 100 ) -> list [ Episode ]:

        query = (
            f"SELECT {EPISODE_COLUMNS} FROM episodes "
            f"WHERE tags IS NOT NULL AND EXISTS ( "
            f"  SELECT 1 FROM json_each ( episodes.tags ) WHERE json_each.value = ? "
            f") "
            f"ORDER BY timestamp DESC LIMIT ?"
        )

        async with self.connection.execute ( query, ( tag, max ( int ( limit ), 0 ) ) ) as cursor:
            rows = await cursor.fetchall ()

        # Return data to caller.

        return [ row_to_episode ( row ) for row in rows ]

    #-------------------------------------------------------------------------------------------------------------------
    # Function: count
    #
    # Description:
    #
    #   How many episodes are stored.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   The row count.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def count ( self ) -> int:

        async with self.connection.execute ( "SELECT count(*) FROM episodes" ) as cursor:
            row = await cursor.fetchone ()

        # Return data to caller.

        return int ( row [ 0 ] ) if row else 0
