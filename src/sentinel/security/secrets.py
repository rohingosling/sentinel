#-----------------------------------------------------------------------------------------------------------------------
# Module:  secrets.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   Credential storage, backed by the OS keyring.
#
#   Sentinel never writes a secret to the config tree, the database, or a log. On Windows the keyring is the Credential
#   Manager, encrypted by DPAPI against the logged-in user account, which gives the property that matters: a secret
#   written by this user cannot be read by another user or on another machine.
#
#   Resolution order for every secret is keyring, then environment variable, then absent:
#
#     * The keyring is where the installed product keeps credentials.
#     * The environment variable is a development convenience -- ANTHROPIC_API_KEY is a near-universal convention and
#       refusing to honour it would make local work needlessly awkward.
#     * Absent is a normal state with an actionable message, not a crash.
#
#   The environment fallback is deliberately *not* how skills will receive credentials. A skill runs in a sandboxed
#   subprocess and inherits the parent environment, so a key placed there is readable by every skill. Phase 6 hands
#   credentials to skills over the SDK socket instead.
#-----------------------------------------------------------------------------------------------------------------------

import logging
import os
import secrets as secrets_module

import keyring

from keyring.errors import KeyringError

from sentinel.errors import ConfigurationError, SecretsError

logger = logging.getLogger ( __name__ )

# Keyring service namespace. Every Sentinel entry lives under this one service name, so
# `sentinel key status` can enumerate what it owns without touching anything else.

SERVICE_NAME = "sentinel"

# Entry names within the service.
#
# These are keyring *lookup names*, not credentials -- the values they name never appear in
# source. ruff's S105 flags any string assigned to a name ending in _KEY or _TOKEN, which is
# the right default and the wrong call here, so it is suppressed per line rather than for
# the file: a genuine hardcoded secret in this module must still be caught.

SECRET_ANTHROPIC_API_KEY = "anthropic_api_key"   # noqa: S105
SECRET_API_TOKEN         = "api_token"           # noqa: S105

# Environment fallbacks, consulted only when the keyring holds nothing. ANTHROPIC_API_KEY
# is the provider's own convention; the Sentinel token gets a namespaced name.

ENVIRONMENT_FALLBACKS = {
    SECRET_ANTHROPIC_API_KEY: "ANTHROPIC_API_KEY",
    SECRET_API_TOKEN: "SENTINEL_API_TOKEN",
}

# Every secret this subsystem manages, with a human label for the CLI.

MANAGED_SECRETS = {
    SECRET_ANTHROPIC_API_KEY: "Anthropic API key",
    SECRET_API_TOKEN: "Sentinel API token",
}

# Width of a generated inbound token, in random bytes before URL-safe encoding.

GENERATED_TOKEN_BYTES = 32


#-----------------------------------------------------------------------------------------------------------------------
# Function: read_secret
#
# Description:
#
#   Read a secret, preferring the keyring over the environment.
#
# Arguments:
#
#   name              : Entry name, one of the SECRET_* constants.
#   allow_environment : Consult the environment fallback when the keyring holds nothing.
#
# Returns:
#
#   The secret, or None when neither source has it. A keyring backend that fails outright raises SecretsError -- that is
#   a broken environment, not an absent credential.
#
#-----------------------------------------------------------------------------------------------------------------------

def read_secret ( name: str, allow_environment: bool = True ) -> str | None:

    # The keyring is the product's store, so it wins.

    try:
        stored = keyring.get_password ( SERVICE_NAME, name )
    except KeyringError as error:
        raise SecretsError (
            f"Cannot read {name!r} from the OS keyring: {error}"
        ) from error

    if stored:
        return stored

    # Fall back to the environment for development convenience.

    if allow_environment:
        variable = ENVIRONMENT_FALLBACKS.get ( name )

        if variable:
            from_environment = os.environ.get ( variable )

            if from_environment:
                logger.debug (
                    "Using %s from the environment. The installed product reads this from the OS keyring.",
                    variable,
                )

                return from_environment

    # Return data to caller.

    return None


#-----------------------------------------------------------------------------------------------------------------------
# Function: require_secret
#
# Description:
#
#   Read a secret, or fail with a message that says how to supply it.
#
# Arguments:
#
#   name              : Entry name, one of the SECRET_* constants.
#   allow_environment : Consult the environment fallback when the keyring holds nothing.
#
# Returns:
#
#   The secret.
#
#   Raises ConfigurationError when the secret is absent. The message names the command that fixes it, because "missing
#   API key" without that is a dead end for the user.
#
#-----------------------------------------------------------------------------------------------------------------------

def require_secret ( name: str, allow_environment: bool = True ) -> str:

    # Read the secret, then insist on it.

    value = read_secret ( name, allow_environment = allow_environment )

    if value is None:
        label    = MANAGED_SECRETS.get ( name, name )
        variable = ENVIRONMENT_FALLBACKS.get ( name )
        hint     = f" or set {variable}" if variable else ""

        raise ConfigurationError (
            f"No {label} configured. Run `sentinel key set {short_name ( name )}`{hint}."
        )

    # Return data to caller.

    return value


#-----------------------------------------------------------------------------------------------------------------------
# Function: write_secret
#
# Description:
#
#   Store a secret in the OS keyring.
#
# Arguments:
#
#   name  : Entry name, one of the SECRET_* constants.
#   value : The secret. Must be non-empty -- storing an empty string would read back as absent and mislead.
#
# Returns:
#
#   None.
#
#   Raises ValueError on an empty value, SecretsError when the keyring backend refuses the write.
#
#-----------------------------------------------------------------------------------------------------------------------

def write_secret ( name: str, value: str ) -> None:

    # Refuse an empty value: it would be indistinguishable from "not set".

    if not value:
        raise ValueError ( f"Refusing to store an empty value for {name!r}." )

    try:
        keyring.set_password ( SERVICE_NAME, name, value )
    except KeyringError as error:
        raise SecretsError (
            f"Cannot write {name!r} to the OS keyring: {error}"
        ) from error

    # Log the entry name only. The value never reaches a log record.

    logger.info ( "Stored %s in the OS keyring.", MANAGED_SECRETS.get ( name, name ) )


#-----------------------------------------------------------------------------------------------------------------------
# Function: delete_secret
#
# Description:
#
#   Remove a secret from the OS keyring.
#
# Arguments:
#
#   name : Entry name, one of the SECRET_* constants.
#
# Returns:
#
#   True if an entry was removed, False if there was nothing to remove. Deleting an absent secret is a no-op rather than
#   an error -- the caller's intent is satisfied either way.
#
#-----------------------------------------------------------------------------------------------------------------------

def delete_secret ( name: str ) -> bool:

    # Deleting what is not there is success, not failure.

    try:
        if keyring.get_password ( SERVICE_NAME, name ) is None:
            return False

        keyring.delete_password ( SERVICE_NAME, name )
    except KeyringError as error:
        raise SecretsError (
            f"Cannot delete {name!r} from the OS keyring: {error}"
        ) from error

    logger.info ( "Removed %s from the OS keyring.", MANAGED_SECRETS.get ( name, name ) )

    # Return data to caller.

    return True


#-----------------------------------------------------------------------------------------------------------------------
# Function: generate_token
#
# Description:
#
#   Generate a random bearer token for Sentinel's own API.
#
# Arguments:
#
#   None.
#
# Returns:
#
#   A URL-safe token with 256 bits of entropy, suitable for an Authorization header.
#
#-----------------------------------------------------------------------------------------------------------------------

def generate_token () -> str:

    # Return data to caller.

    return secrets_module.token_urlsafe ( GENERATED_TOKEN_BYTES )


#-----------------------------------------------------------------------------------------------------------------------
# Function: short_name
#
# Description:
#
#   Convert an entry name to the hyphenated form the CLI accepts.
#
# Arguments:
#
#   name : Entry name, one of the SECRET_* constants.
#
# Returns:
#
#   The CLI form: "anthropic_api_key" becomes "anthropic", "api_token" becomes "api-token".
#
#-----------------------------------------------------------------------------------------------------------------------

def short_name ( name: str ) -> str:

    # Return data to caller.

    if name == SECRET_ANTHROPIC_API_KEY:
        return "anthropic"

    return name.replace ( "_", "-" )


#-----------------------------------------------------------------------------------------------------------------------
# Function: resolve_name
#
# Description:
#
#   Convert a CLI name to its keyring entry name.
#
# Arguments:
#
#   requested : The name as typed on the command line.
#
# Returns:
#
#   The matching SECRET_* entry name.
#
#   Raises ValueError when the name matches nothing, listing what is accepted.
#
#-----------------------------------------------------------------------------------------------------------------------

def resolve_name ( requested: str ) -> str:

    # Accept either the CLI form or the entry name itself.

    for entry in MANAGED_SECRETS:
        if requested in ( entry, short_name ( entry ) ):
            return entry

    accepted = ", ".join ( short_name ( entry ) for entry in MANAGED_SECRETS )

    raise ValueError ( f"Unknown secret {requested!r}. Accepted names: {accepted}." )


#-----------------------------------------------------------------------------------------------------------------------
# Function: describe_secrets
#
# Description:
#
#   Report which managed secrets are present, without revealing any of them.
#
# Arguments:
#
#   None.
#
# Returns:
#
#   One entry per managed secret, each carrying the CLI name, the human label, and where the value came from -- keyring,
#   environment, or absent. Values are never included.
#
#-----------------------------------------------------------------------------------------------------------------------

def describe_secrets () -> list [ dict [ str, str ] ]:

    # Report presence and source, never the value itself.

    report = []

    for entry, label in MANAGED_SECRETS.items ():

        # Check the keyring first, then the environment, so the reported source is the
        # one that would actually be used.

        in_keyring = read_secret ( entry, allow_environment = False ) is not None

        if in_keyring:
            source = "keyring"
        elif read_secret ( entry ) is not None:
            source = "environment"
        else:
            source = "absent"

        report.append (
            {
                "name": short_name ( entry ),
                "label": label,
                "source": source,
            }
        )

    # Return data to caller.

    return report
