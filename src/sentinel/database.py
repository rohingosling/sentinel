#-----------------------------------------------------------------------------------------------------------------------
# Module:  database.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   SQLite initialisation, pragmas, and sqlite-vec loading.
#
#   All persistent state -- three memory tiers, identity snapshots, the append-only event log -- lives in one SQLite
#   file. This module owns opening it correctly: the version floor, the pragma set from architecture 6.6, and the
#   sqlite-vec extension.
#
#   The version assertion exists to catch a swapped interpreter, not an expected failure. Python 3.12 ships SQLite
#   3.49.1, comfortably above sqlite-vec's 3.41 floor, so if it ever fires the runtime is not the one Sentinel was
#   built against.
#-----------------------------------------------------------------------------------------------------------------------

import logging
import sqlite3

from pathlib import Path

import aiosqlite
import sqlite_vec

from sentinel.errors     import DatabaseError, UnsupportedEnvironmentError
from sentinel.migrations import apply_migrations, schema_version

logger = logging.getLogger ( __name__ )

# What a failed database operation raises, for every `except` clause that means "the database would not answer".
#
# sqlite3.Error is the obvious half. ValueError is the half that is easy to miss: aiosqlite guards every statement with
# its own "no active connection" check and raises ValueError -- NOT an sqlite3.Error -- once the connection has been
# closed. Catching only sqlite3.Error therefore lets a closed-connection failure escape the SentinelError hierarchy
# entirely, which is exactly the failure that happens during shutdown, and exactly where the heartbeat's contract that
# a tick never raises would be broken.

DATABASE_FAULTS: tuple [ type [ Exception ], ... ] = ( sqlite3.Error, ValueError )

# sqlite-vec's documented floor. Below this the extension will not load.

MINIMUM_SQLITE_VERSION = ( 3, 41, 0 )

# Page cache expressed in kibibytes, negated as SQLite requires (64 MB).

PAGE_CACHE_SIZE_KIB = -64000

# WAL checkpoint threshold in pages.

WAL_AUTOCHECKPOINT_PAGES = 1000


#-----------------------------------------------------------------------------------------------------------------------
# Environment assertions
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Function: parse_sqlite_version
#
# Description:
#
#   Parse an SQLite version string into a comparable tuple.
#
# Arguments:
#
#   version_text : A dotted version, e.g. "3.49.1".
#
# Returns:
#
#   The numeric components, e.g. ( 3, 49, 1 ). Non-numeric trailing components are dropped rather than raising -- the
#   comparison only needs the prefix.
#
#-----------------------------------------------------------------------------------------------------------------------

def parse_sqlite_version ( version_text: str ) -> tuple [ int, ... ]:

    # Parse an SQLite version string into a comparable tuple.

    components: list [ int ] = []

    for part in version_text.split ( "." ):
        if not part.isdigit ():
            break

        components.append ( int ( part ) )

    # Return data to caller.

    return tuple ( components )


#-----------------------------------------------------------------------------------------------------------------------
# Function: assert_sqlite_version
#
# Description:
#
#   Fail loudly if the interpreter's SQLite is older than sqlite-vec's floor.
#
# Arguments:
#
#   version_text : Version to check. The running interpreter's when omitted.
#
# Returns:
#
#   The parsed version tuple.
#
#   Raises UnsupportedEnvironmentError if the version is below MINIMUM_SQLITE_VERSION.
#
#-----------------------------------------------------------------------------------------------------------------------

def assert_sqlite_version ( version_text: str | None = None ) -> tuple [ int, ... ]:

    # Fail loudly if the interpreter's SQLite is older than sqlite-vec's floor.

    reported = version_text if version_text is not None else sqlite3.sqlite_version
    version  = parse_sqlite_version ( reported )
    minimum  = ".".join ( str ( component ) for component in MINIMUM_SQLITE_VERSION )

    if version < MINIMUM_SQLITE_VERSION:
        raise UnsupportedEnvironmentError (
            f"SQLite {reported} is too old -- sqlite-vec requires {minimum} or newer. "
            f"Sentinel targets Python 3.12, which bundles SQLite 3.49.1, so this "
            f"almost certainly means the wrong interpreter is running. Check which "
            f"python is on PATH and re-run with the bundled 3.12 distribution."
        )

    logger.debug ( "SQLite %s satisfies the %s floor.", reported, minimum )

    # Return data to caller.

    return version


#-----------------------------------------------------------------------------------------------------------------------
# Connection setup
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Function: apply_pragmas
#
# Description:
#
#   Apply Sentinel's standard pragma set (architecture 6.6).
#
# Arguments:
#
#   connection         : An open aiosqlite connection.
#   wal_mode           : Enable write-ahead logging for concurrent reads.
#   busy_timeout       : Milliseconds to retry on lock contention.
#   journal_size_limit : Byte cap on the WAL file.
#
# Returns:
#
#   None.
#
#   Raises DatabaseError if WAL was requested but the journal mode did not take.
#
#-----------------------------------------------------------------------------------------------------------------------

async def apply_pragmas ( connection: aiosqlite.Connection,
                          wal_mode: bool    = True,
                          busy_timeout: int = 5000,
                          journal_size_limit: int = 67108864 ) -> None:

    # Apply Sentinel's standard pragma set.

    # Journal mode first -- it is the only pragma here that can be refused, and the
    # remaining tuning is pointless if the storage engine is not in the mode we want.

    if wal_mode:
        async with connection.execute ( "PRAGMA journal_mode = WAL" ) as cursor:
            row = await cursor.fetchone ()

        applied_mode = str ( row [ 0 ] ).lower () if row else "unknown"

        if applied_mode != "wal":
            raise DatabaseError (
                f"Requested WAL journal mode but SQLite reported {applied_mode!r}. "
                f"WAL is unavailable on some network filesystems -- move the database "
                f"to local storage or set database.wal_mode to false."
            )

    # Remaining tuning. None of these can be refused.

    await connection.execute ( "PRAGMA synchronous = NORMAL" )
    await connection.execute ( f"PRAGMA busy_timeout = {int ( busy_timeout )}" )
    await connection.execute ( f"PRAGMA cache_size = {PAGE_CACHE_SIZE_KIB}" )
    await connection.execute ( "PRAGMA foreign_keys = ON" )
    await connection.execute ( f"PRAGMA wal_autocheckpoint = {WAL_AUTOCHECKPOINT_PAGES}" )
    await connection.execute ( f"PRAGMA journal_size_limit = {int ( journal_size_limit )}" )
    await connection.commit ()


#-----------------------------------------------------------------------------------------------------------------------
# Function: load_vector_extension
#
# Description:
#
#   Load sqlite-vec and confirm it answers.
#
#   Extension loading is disabled again immediately afterwards. Leaving it on would let any later SQL -- including
#   anything a skill ever manages to influence -- pull an arbitrary DLL into the process.
#
# Arguments:
#
#   connection : An open aiosqlite connection.
#
# Returns:
#
#   The value of vec_version().
#
#   Raises DatabaseError if the build has no loadable-extension support, or the extension loaded but did not register
#   its functions.
#
#-----------------------------------------------------------------------------------------------------------------------

async def load_vector_extension ( connection: aiosqlite.Connection ) -> str:

    # Load sqlite-vec and confirm it answers.

    try:
        await connection.enable_load_extension ( True )
        await connection.load_extension ( sqlite_vec.loadable_path () )
    except ( AttributeError, sqlite3.OperationalError, sqlite3.NotSupportedError ) as error:
        raise DatabaseError (
            f"Could not load the sqlite-vec extension: {error}. Sentinel needs a "
            f"Python built with loadable-extension support; the official Windows "
            f"3.12 distribution has it."
        ) from error
    finally:
        await connection.enable_load_extension ( False )

    # Confirm the extension registered its functions, not merely that the load call returned.

    try:
        async with connection.execute ( "SELECT vec_version()" ) as cursor:
            row = await cursor.fetchone ()
    except sqlite3.OperationalError as error:
        raise DatabaseError (
            f"sqlite-vec loaded but vec_version() is not registered: {error}."
        ) from error

    if not row:
        raise DatabaseError ( "sqlite-vec loaded but vec_version() returned no rows." )

    version = str ( row [ 0 ] )

    logger.debug ( "sqlite-vec %s loaded.", version )

    # Return data to caller.

    return version


#-----------------------------------------------------------------------------------------------------------------------
# Function: connect
#
# Description:
#
#   Open a fully configured connection to the Sentinel database.
#
#   Asserts the SQLite version, creates the parent directory, applies the pragma set, and loads sqlite-vec. The caller
#   owns closing the connection.
#
# Arguments:
#
#   database_path      : Path to the database file. Created if absent.
#   wal_mode           : Enable write-ahead logging.
#   busy_timeout       : Milliseconds to retry on lock contention.
#   journal_size_limit : Byte cap on the WAL file.
#   load_vectors       : Load sqlite-vec. False is for tests that do not need it.
#
# Returns:
#
#   An open aiosqlite connection.
#
#   Raises UnsupportedEnvironmentError if SQLite is older than sqlite-vec's floor, or DatabaseError if the file could
#   not be opened or configured.
#
#-----------------------------------------------------------------------------------------------------------------------

async def connect ( database_path: str | Path,
                    wal_mode: bool          = True,
                    busy_timeout: int       = 5000,
                    journal_size_limit: int = 67108864,
                    load_vectors: bool = True ) -> aiosqlite.Connection:

    # Open a fully configured connection to the Sentinel database.

    assert_sqlite_version ()

    path = Path ( database_path ).expanduser ()

    # Create the containing directory before SQLite is asked for the file.

    try:
        path.parent.mkdir ( parents = True, exist_ok = True )
    except OSError as error:
        raise DatabaseError (
            f"Cannot create the database directory {path.parent}: {error}"
        ) from error

    try:
        connection = await aiosqlite.connect ( path )
    except sqlite3.Error as error:
        raise DatabaseError ( f"Cannot open the database {path}: {error}" ) from error

    # A half-configured connection is worse than none, so anything that goes wrong
    # from here closes it. The flag rather than `except BaseException` keeps
    # cancellation and shutdown signals propagating untouched.

    configured = False

    try:
        await apply_pragmas (
            connection         = connection,
            wal_mode           = wal_mode,
            busy_timeout       = busy_timeout,
            journal_size_limit = journal_size_limit,
        )

        if load_vectors:
            await load_vector_extension ( connection )

        configured = True
    finally:
        if not configured:
            await connection.close ()

    # Return data to caller.

    return connection


#-----------------------------------------------------------------------------------------------------------------------
# Function: initialise_database
#
# Description:
#
#   Create the database file if absent, verify it is usable, and bring its schema up to date.
#
#   This is what `sentinel init` runs, and what `sentinel start` runs before anything else. The job is to prove the
#   file, the pragmas, and the vector extension all work on this machine, and then to apply any migration this database
#   has not yet seen -- which is what makes an upgrade over an existing installation a no-op rather than a manual step.
#
# Arguments:
#
#   database_path      : Path to the database file.
#   wal_mode           : Enable write-ahead logging.
#   busy_timeout       : Milliseconds to retry on lock contention.
#   journal_size_limit : Byte cap on the WAL file.
#
# Returns:
#
#   A report with keys "path", "sqlite_version", "journal_mode", "vec_version", "schema_version", and "migrations"
#   (a comma-separated list of what this call applied, or an empty string).
#
#   Raises UnsupportedEnvironmentError if SQLite is older than sqlite-vec's floor, MigrationError if the schema could
#   not be brought up to date, or DatabaseError if the database could not be created or configured.
#
#-----------------------------------------------------------------------------------------------------------------------

async def initialise_database ( database_path: str | Path,
                                wal_mode: bool    = True,
                                busy_timeout: int = 5000,
                                journal_size_limit: int = 67108864 ) -> dict [ str, str ]:

    # Create the database file if absent and verify it is usable.

    path = Path ( database_path ).expanduser ()

    connection = await connect (
        database_path      = path,
        wal_mode           = wal_mode,
        busy_timeout       = busy_timeout,
        journal_size_limit = journal_size_limit,
        load_vectors       = False,
    )

    # Load the extension and read back the journal mode, then let the connection go --
    # init proves the environment works; it does not hold a handle open.
    #
    # Migrations run on this same connection and after the extension is loaded, because 001 creates a vec0 virtual
    # table and would otherwise fail with "no such module: vec0".

    try:
        vector_version = await load_vector_extension ( connection )

        applied = await apply_migrations ( connection )
        version = await schema_version ( connection )

        async with connection.execute ( "PRAGMA journal_mode" ) as cursor:
            row = await cursor.fetchone ()

        journal_mode = str ( row [ 0 ] ).lower () if row else "unknown"
    finally:
        await connection.close ()

    report = {
        "path": str ( path.resolve () ),
        "sqlite_version": sqlite3.sqlite_version,
        "journal_mode": journal_mode,
        "vec_version": vector_version,
        "schema_version": str ( version ),
        "migrations": ", ".join ( applied ),
    }

    logger.info (
        "Database ready at %s (SQLite %s, journal_mode=%s, sqlite-vec %s, schema v%s).",
        report [ "path" ], report [ "sqlite_version" ],
        report [ "journal_mode" ], report [ "vec_version" ], report [ "schema_version" ],
    )

    # Return data to caller.

    return report
