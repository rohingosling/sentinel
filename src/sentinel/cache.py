#-----------------------------------------------------------------------------------------------------------------------
# Module:  cache.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   Working-memory cache initialisation.
#
#   Tier 1 of the memory system (architecture 3.2.4) is a diskcache store: SQLite-backed, in-process, TTL-evicted. No
#   daemon, no second database engine, and it survives a restart, which an in-memory dict would not.
#
#   Phase 0 only opens and verifies the store. Tier-1 semantics -- TTL policy, eviction, the working-memory API --
#   arrive in Phase 3.
#-----------------------------------------------------------------------------------------------------------------------

import logging

from pathlib import Path

import diskcache

from sentinel.errors import DatabaseError

logger = logging.getLogger ( __name__ )

# Working-memory budget. diskcache's own default is 1 GiB; 256 MiB is the budget from architecture 3.2.4 and matches
# memory.working_size_limit in agent.yaml.

DEFAULT_SIZE_LIMIT_BYTES = 268435456


#-----------------------------------------------------------------------------------------------------------------------
# Function: open_cache
#
# Description:
#
#   Open the working-memory cache, creating its directory if absent.
#
# Arguments:
#
#   cache_directory : Directory holding the diskcache store.
#   size_limit      : Byte ceiling before eviction begins.
#
# Returns:
#
#   An open diskcache.Cache. The caller owns closing it.
#
#   Raises DatabaseError if the directory or store could not be created.
#
#-----------------------------------------------------------------------------------------------------------------------

def open_cache ( cache_directory: str | Path,
                 size_limit: int = DEFAULT_SIZE_LIMIT_BYTES ) -> diskcache.Cache:

    # Open the working-memory cache, creating its directory if absent.

    path = Path ( cache_directory ).expanduser ()

    try:
        path.mkdir ( parents = True, exist_ok = True )

        cache = diskcache.Cache ( directory = str ( path ), size_limit = size_limit )
    except ( OSError, diskcache.Timeout ) as error:
        raise DatabaseError ( f"Cannot open the cache at {path}: {error}" ) from error

    # Return data to caller.

    return cache


#-----------------------------------------------------------------------------------------------------------------------
# Function: initialise_cache
#
# Description:
#
#   Create and verify the working-memory cache.
#
#   Writes and reads back a probe key so a permissions or disk problem surfaces now, during `sentinel init`, rather
#   than on the first heartbeat tick.
#
# Arguments:
#
#   cache_directory : Directory holding the diskcache store.
#   size_limit      : Byte ceiling before eviction begins.
#
# Returns:
#
#   A report with keys "path" and "size_limit".
#
#   Raises DatabaseError if the cache could not be created or failed its round trip.
#
#-----------------------------------------------------------------------------------------------------------------------

def initialise_cache ( cache_directory: str | Path,
                       size_limit: int = DEFAULT_SIZE_LIMIT_BYTES ) -> dict [ str, str ]:

    # Create and verify the working-memory cache.

    cache      = open_cache ( cache_directory, size_limit )
    probe_key  = "sentinel:init:probe"
    probe_data = "ok"

    try:

        # Prove the store round-trips before anything depends on it.

        cache.set ( probe_key, probe_data )

        if cache.get ( probe_key ) != probe_data:
            raise DatabaseError (
                f"The cache at {cache.directory} accepted a write but returned the "
                f"wrong value on read back."
            )

        cache.delete ( probe_key )

        report = {
            "path": str ( Path ( cache.directory ).resolve () ),
            "size_limit": str ( size_limit ),
        }
    except diskcache.Timeout as error:
        raise DatabaseError (
            f"Cache round trip timed out at {cache.directory}: {error}"
        ) from error
    finally:
        cache.close ()

    logger.info ( "Cache ready at %s (size limit %d bytes).", report [ "path" ], size_limit )

    # Return data to caller.

    return report
