#-----------------------------------------------------------------------------------------------------------------------
# Module:  migrations.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   Sequential schema migrations.
#
#   Every migration is a numbered .sql file in migrations/, applied once, in numeric order, and recorded in a
#   schema_migrations ledger so a second run does nothing. That property is what makes `sentinel init` and `sentinel
#   start` safe to run repeatedly against a database that already holds a year of the user's memories.
#
#   Three details are deliberate:
#
#     * The version is parsed from the leading digits of the filename, not from a manifest. A manifest is a second place
#       to forget to update, and the number is already in the name.
#     * Each file is applied inside its own transaction. A migration that fails half way leaves the database on the
#       previous version rather than in a state no version describes.
#     * The ledger records the file's SHA-256. Editing an applied migration in place is the one mistake this scheme
#       cannot recover from, so it is detected and reported instead of ignored.
#
#   Migrations run against a connection that already has sqlite-vec loaded -- 001 creates a vec0 virtual table, which
#   without the extension fails with "no such module".
#-----------------------------------------------------------------------------------------------------------------------

import hashlib
import logging
import re
import sqlite3

from dataclasses import dataclass
from pathlib     import Path

import aiosqlite

from sentinel.errors import MigrationError

logger = logging.getLogger ( __name__ )

# Where the .sql files can be found, most specific first. An installed wheel carries them inside the package (the build
# force-includes migrations/ as sentinel/_migrations/), because the repository's top-level migrations/ directory does
# not survive installation. A source checkout falls back to the repository path.

PACKAGE_MIGRATIONS_DIRECTORY    = Path ( __file__ ).resolve ().parent / "_migrations"
REPOSITORY_MIGRATIONS_DIRECTORY = Path ( __file__ ).resolve ().parents [ 2 ] / "migrations"

# Filenames look like 001_memory_schema.sql. Anything else in the directory is ignored rather than guessed at.

MIGRATION_FILE_PATTERN = re.compile ( r"^(\d+)_([A-Za-z0-9_\-]+)\.sql$" )

# The ledger. Created by the runner itself rather than by a migration, because a migration cannot record that it ran
# until the table recording it exists.

LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_migrations
(
    version     INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    checksum    TEXT NOT NULL,
    applied_at  TEXT NOT NULL DEFAULT ( datetime ( 'now' ) )
)
"""


#-----------------------------------------------------------------------------------------------------------------------
# Class: Migration
#
# Description:
#
#   One migration file on disk.
#
# Attributes:
#
#   version  : Numeric version parsed from the filename.
#   name     : Descriptive part of the filename, without the number or extension.
#   path     : Absolute path of the .sql file.
#   sql      : File contents.
#   checksum : SHA-256 of the contents, hex encoded.
#-----------------------------------------------------------------------------------------------------------------------

@dataclass ( frozen = True )
class Migration:

    version:  int
    name:     str
    path:     Path
    sql:      str
    checksum: str


#-----------------------------------------------------------------------------------------------------------------------
# Discovery
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Function: find_migrations_directory
#
# Description:
#
#   Locate the directory holding the .sql migration files.
#
# Arguments:
#
#   None.
#
# Returns:
#
#   The packaged copy if this is an installed wheel, the repository copy if this is a checkout, or None if neither is
#   present.
#
#-----------------------------------------------------------------------------------------------------------------------

def find_migrations_directory () -> Path | None:

    # Locate the migrations directory, most specific location first.

    for candidate in ( PACKAGE_MIGRATIONS_DIRECTORY, REPOSITORY_MIGRATIONS_DIRECTORY ):
        if candidate.is_dir ():
            return candidate

    # Return data to caller.

    return None


#-----------------------------------------------------------------------------------------------------------------------
# Function: load_migrations
#
# Description:
#
#   Read every migration file, ordered by version.
#
# Arguments:
#
#   directory : Directory to read. Resolved automatically when omitted.
#
# Returns:
#
#   Migrations in ascending version order. An empty list when no directory could be found -- a database with no schema
#   is a diagnosable state, whereas a crash inside `sentinel version` is not.
#
#   Raises MigrationError if a file cannot be read, or if two files claim the same version.
#
#-----------------------------------------------------------------------------------------------------------------------

def load_migrations ( directory: str | Path | None = None ) -> list [ Migration ]:

    # Resolve the directory, tolerating its absence.

    root = Path ( directory ) if directory is not None else find_migrations_directory ()

    if root is None or not root.is_dir ():
        logger.warning (
            "No migrations directory found. The database schema will not be created. "
            "This is expected only for a partial installation."
        )

        return []

    migrations: dict [ int, Migration ] = {}

    for path in sorted ( root.iterdir () ):

        match = MIGRATION_FILE_PATTERN.match ( path.name )

        if match is None:
            continue

        try:
            sql = path.read_text ( encoding = "utf-8" )
        except OSError as error:
            raise MigrationError ( f"Cannot read migration {path}: {error}" ) from error

        version = int ( match.group ( 1 ) )

        # Two files claiming one version is unresolvable -- the ledger has room for exactly one of them, and picking
        # either silently means the other never runs on some machines and does on others.

        if version in migrations:
            raise MigrationError (
                f"Two migrations claim version {version}: {migrations [ version ].path.name} and {path.name}. "
                f"Renumber one of them."
            )

        migrations [ version ] = Migration (
            version  = version,
            name     = match.group ( 2 ),
            path     = path,
            sql      = sql,
            checksum = hashlib.sha256 ( sql.encode ( "utf-8" ) ).hexdigest (),
        )

    # Return data to caller.

    return [ migrations [ version ] for version in sorted ( migrations ) ]


#-----------------------------------------------------------------------------------------------------------------------
# Function: split_statements
#
# Description:
#
#   Split a migration script into individual SQL statements.
#
#   Exists because executescript() cannot be used here. It issues an implicit COMMIT before running and performs no
#   transaction control of its own, so a script that fails half way leaves every statement before the failure already
#   committed -- the opposite of what a migration needs. Executing statement by statement inside one explicit
#   transaction restores the guarantee, and lets the ledger row be inserted with a parameterised statement rather than
#   interpolated into the script text.
#
#   The split is done with sqlite3.complete_statement rather than by splitting on semicolons, which is what makes it
#   safe against semicolons inside string literals and comments.
#
# Arguments:
#
#   sql : The full script text.
#
# Returns:
#
#   Executable statements in order. Comment-only trailing text is discarded.
#
#-----------------------------------------------------------------------------------------------------------------------

def split_statements ( sql: str ) -> list [ str ]:

    # Accumulate lines until SQLite agrees the accumulated text is one complete statement.

    statements: list [ str ] = []
    buffer = ""

    for line in sql.splitlines ( keepends = True ):

        buffer += line

        if sqlite3.complete_statement ( buffer ):
            statements.append ( buffer.strip () )

            buffer = ""

    # Whatever is left is either trailing comments or whitespace. A genuinely unterminated statement would have been
    # rejected by SQLite anyway, and reporting it here would only duplicate that message less precisely.

    # Return data to caller.

    return [ statement for statement in statements if statement ]


#-----------------------------------------------------------------------------------------------------------------------
# Application
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Function: applied_versions
#
# Description:
#
#   Read the ledger of migrations already applied to this database.
#
# Arguments:
#
#   connection : An open aiosqlite connection.
#
# Returns:
#
#   Version numbers mapped to the checksum recorded when they were applied.
#
#-----------------------------------------------------------------------------------------------------------------------

async def applied_versions ( connection: aiosqlite.Connection ) -> dict [ int, str ]:

    # Ensure the ledger exists before reading it.

    await connection.execute ( LEDGER_SCHEMA )
    await connection.commit ()

    async with connection.execute ( "SELECT version, checksum FROM schema_migrations" ) as cursor:
        rows = await cursor.fetchall ()

    # Return data to caller.

    return { int ( row [ 0 ] ): str ( row [ 1 ] ) for row in rows }


#-----------------------------------------------------------------------------------------------------------------------
# Function: apply_migrations
#
# Description:
#
#   Apply every migration this database has not yet seen.
#
#   Idempotent by construction: a database already at the newest version does no work and reports an empty list.
#
# Arguments:
#
#   connection : An open aiosqlite connection with sqlite-vec already loaded.
#   directory  : Directory holding the .sql files. Resolved automatically when omitted.
#
# Returns:
#
#   The names of the migrations applied by this call, in order. Empty when the schema was already current.
#
#   Raises MigrationError if a migration fails, or if an already-applied migration's file has been edited since.
#
#-----------------------------------------------------------------------------------------------------------------------

async def apply_migrations ( connection: aiosqlite.Connection,
                             directory: str | Path | None = None ) -> list [ str ]:

    # Read what exists on disk and what this database has already seen.

    migrations = load_migrations ( directory )
    applied    = await applied_versions ( connection )
    performed: list [ str ] = []

    for migration in migrations:

        recorded = applied.get ( migration.version )

        # Already applied. Confirm the file has not changed underneath the database, because re-running it is not an
        # option and leaving it undetected means two installations disagree about what their schema is.

        if recorded is not None:

            if recorded != migration.checksum:
                raise MigrationError (
                    f"Migration {migration.path.name} has changed since it was applied to this database. "
                    f"An applied migration is immutable -- add a new numbered migration for the change, and restore "
                    f"this file to the version recorded in schema_migrations."
                )

            continue

        # Not yet applied. Run it in its own transaction so a failure leaves the previous version intact.

        logger.info ( "Applying migration %03d %s.", migration.version, migration.name )

        try:
            await connection.execute ( "BEGIN" )

            for statement in split_statements ( migration.sql ):
                await connection.execute ( statement )

            # The ledger row joins the same transaction, so "schema changed" and "schema recorded as changed" cannot
            # come apart.

            await connection.execute (
                "INSERT INTO schema_migrations ( version, name, checksum ) VALUES ( ?, ?, ? )",
                ( migration.version, migration.name, migration.checksum ),
            )

            await connection.commit ()

        except sqlite3.Error as error:
            await connection.rollback ()

            raise MigrationError (
                f"Migration {migration.path.name} failed and was rolled back: {error}"
            ) from error

        performed.append ( migration.path.name )

    if performed:
        logger.info ( "Applied %d migration(s): %s.", len ( performed ), ", ".join ( performed ) )

    # Return data to caller.

    return performed


#-----------------------------------------------------------------------------------------------------------------------
# Function: schema_version
#
# Description:
#
#   Report the highest migration version applied to this database.
#
# Arguments:
#
#   connection : An open aiosqlite connection.
#
# Returns:
#
#   The highest applied version, or 0 when no migration has ever run.
#
#-----------------------------------------------------------------------------------------------------------------------

async def schema_version ( connection: aiosqlite.Connection ) -> int:

    # An absent ledger means nothing has been applied, which is version zero rather than an error.

    versions = await applied_versions ( connection )

    # Return data to caller.

    return max ( versions ) if versions else 0
