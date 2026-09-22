#-----------------------------------------------------------------------------------------------------------------------
# Module:  working.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   Tier 1: working memory (architecture 3.2.4).
#
#   Session-scoped, TTL-evicted, size-bounded key-value storage over diskcache. This is where the current conversation
#   context and any active task state live -- everything the agent needs across the turns of one session and nothing it
#   needs across sessions.
#
#   Two design points worth stating, because both look like accidents otherwise:
#
#     * Keys are tuples of (session id, name), not strings joined by a separator. A separator can appear inside a name;
#       a tuple cannot collide however the name is spelled, and it makes "every key in this session" a filter rather
#       than a prefix match that could catch a neighbouring session whose id is a prefix of this one.
#     * It is durable, despite being the volatile tier. diskcache is SQLite-backed, so a crash mid-conversation does
#       not lose the conversation. The TTL is what makes an entry temporary, not the storage medium.
#
#   Eviction is diskcache's own least-recently-stored policy, triggered when the store exceeds its byte ceiling. That
#   ceiling is memory.working_size_limit, 256 MB by default.
#-----------------------------------------------------------------------------------------------------------------------

import logging

from collections.abc import Iterator
from types           import TracebackType
from typing          import Any, Literal

import diskcache

from sentinel.cache  import open_cache
from sentinel.errors import DatabaseError

logger = logging.getLogger ( __name__ )

# Reserved key names, so the conversation context and task state cannot be overwritten by an ordinary set() from
# somewhere else. Namespaced rather than merely documented, because a collision here silently rewrites the agent's idea
# of what it is currently doing.

KEY_CONVERSATION = "__conversation__"
KEY_TASK_STATE   = "__task_state__"

# The session every caller lands in when none is named. A single-user desktop agent spends most of its life here.

DEFAULT_SESSION = "default"


#-----------------------------------------------------------------------------------------------------------------------
# Class: WorkingMemory
#
# Description:
#
#   Session-scoped key-value storage with expiry.
#
#   One instance is bound to one session. Reading another session's entries is possible only by constructing a second
#   instance for it, which is what makes cross-session leakage a deliberate act rather than a typo.
#
# Attributes:
#
#   cache       : The underlying diskcache store.
#   session_id  : Session this instance reads and writes.
#   default_ttl : Seconds an entry survives when set() is not given an explicit ttl. None means no expiry.
#   owns_cache  : Whether closing this instance should close the store.
#-----------------------------------------------------------------------------------------------------------------------

class WorkingMemory:

    #-------------------------------------------------------------------------------------------------------------------
    # Function: __init__
    #
    # Description:
    #
    #   Bind to a cache store and a session.
    #
    # Arguments:
    #
    #   cache       : An open diskcache store.
    #   session_id  : Session to scope every operation to.
    #   default_ttl : Default entry lifetime, in seconds. None disables expiry.
    #   owns_cache  : Close the store when this instance is closed.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def __init__ ( self,
                   cache: diskcache.Cache,
                   session_id: str           = DEFAULT_SESSION,
                   default_ttl: float | None = None,
                   owns_cache: bool = False ) -> None:

        self.cache       = cache
        self.session_id  = session_id
        self.default_ttl = default_ttl
        self.owns_cache  = owns_cache

    #-------------------------------------------------------------------------------------------------------------------
    # Function: open
    #
    # Description:
    #
    #   Open a store and bind to it in one step.
    #
    # Arguments:
    #
    #   cache_directory : Directory holding the diskcache store.
    #   session_id      : Session to scope every operation to.
    #   default_ttl     : Default entry lifetime, in seconds.
    #   size_limit      : Byte ceiling before eviction begins.
    #
    # Returns:
    #
    #   A WorkingMemory that owns the store it opened, so closing it closes the store.
    #
    #   Raises DatabaseError if the store could not be opened.
    #
    #-------------------------------------------------------------------------------------------------------------------

    @classmethod
    def open ( cls,
               cache_directory: str,
               session_id: str           = DEFAULT_SESSION,
               default_ttl: float | None = None,
               size_limit: int = 268435456 ) -> "WorkingMemory":

        # Return data to caller.

        return cls (
            cache       = open_cache ( cache_directory, size_limit ),
            session_id  = session_id,
            default_ttl = default_ttl,
            owns_cache  = True,
        )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: for_session
    #
    # Description:
    #
    #   A view of the same store scoped to a different session.
    #
    # Arguments:
    #
    #   session_id : The session to bind to.
    #
    # Returns:
    #
    #   A new WorkingMemory sharing this store. It does not own the store, so closing the view leaves the original
    #   usable.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def for_session ( self, session_id: str ) -> "WorkingMemory":

        # Return data to caller.

        return WorkingMemory (
            cache       = self.cache,
            session_id  = session_id,
            default_ttl = self.default_ttl,
            owns_cache  = False,
        )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: scoped_key
    #
    # Description:
    #
    #   The storage key for a name in this session.
    #
    # Arguments:
    #
    #   name : The caller's key name.
    #
    # Returns:
    #
    #   A (session id, name) tuple.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def scoped_key ( self, name: str ) -> tuple [ str, str ]:

        # Return data to caller.

        return ( self.session_id, name )

    #-------------------------------------------------------------------------------------------------------------------
    # Storage
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: set
    #
    # Description:
    #
    #   Store a value in this session.
    #
    # Arguments:
    #
    #   name  : Key name.
    #   value : Any picklable value.
    #   ttl   : Lifetime in seconds. The instance default when omitted; pass 0 or a negative number for no expiry.
    #
    # Returns:
    #
    #   True when the value was stored. False when diskcache refused it, which happens when a single value exceeds the
    #   whole store's size limit.
    #
    #   Raises DatabaseError if the store timed out.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def set ( self, name: str, value: Any, ttl: float | None = None ) -> bool:

        # An explicit ttl wins; otherwise the instance default, which may itself be None for no expiry.

        lifetime = self.default_ttl if ttl is None else ttl

        if lifetime is not None and lifetime <= 0:
            lifetime = None

        try:
            stored = bool ( self.cache.set ( self.scoped_key ( name ), value, expire = lifetime ) )
        except diskcache.Timeout as error:
            raise DatabaseError ( f"Working memory timed out storing {name!r}: {error}" ) from error

        if not stored:
            logger.warning (
                "Working memory refused %r in session %s -- the value is larger than the whole store's size limit.",
                name, self.session_id,
            )

        # Return data to caller.

        return stored

    #-------------------------------------------------------------------------------------------------------------------
    # Function: get
    #
    # Description:
    #
    #   Read a value from this session.
    #
    # Arguments:
    #
    #   name    : Key name.
    #   default : Value returned when the entry is absent or expired.
    #
    # Returns:
    #
    #   The stored value, or the default. An expired entry is indistinguishable from an absent one by design -- the
    #   caller's recovery is the same either way.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def get ( self, name: str, default: Any = None ) -> Any:

        try:

            # Return data to caller.

            return self.cache.get ( self.scoped_key ( name ), default = default )

        except diskcache.Timeout as error:
            raise DatabaseError ( f"Working memory timed out reading {name!r}: {error}" ) from error

    #-------------------------------------------------------------------------------------------------------------------
    # Function: delete
    #
    # Description:
    #
    #   Remove an entry from this session.
    #
    # Arguments:
    #
    #   name : Key name.
    #
    # Returns:
    #
    #   True when an entry was removed, False when there was nothing to remove.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def delete ( self, name: str ) -> bool:

        # Return data to caller.

        return bool ( self.cache.delete ( self.scoped_key ( name ) ) )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: names
    #
    # Description:
    #
    #   Every live key name in this session.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   The names, excluding the two reserved keys. Order is diskcache's, which is insertion order in practice but is
    #   not promised by it, so callers that care must sort.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def names ( self ) -> list [ str ]:

        reserved = { KEY_CONVERSATION, KEY_TASK_STATE }
        found: list [ str ] = []

        for key in self.iterate_keys ():
            if key [ 0 ] == self.session_id and key [ 1 ] not in reserved:
                found.append ( key [ 1 ] )

        # Return data to caller.

        return found

    #-------------------------------------------------------------------------------------------------------------------
    # Function: iterate_keys
    #
    # Description:
    #
    #   Every storage key in the whole store, across all sessions.
    #
    #   Separated from names() so the tuple-shape check lives in one place: diskcache will happily hold a key written by
    #   something that is not a WorkingMemory, and iterating would otherwise crash on it.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   The (session id, name) keys. Anything not of that shape is skipped.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def iterate_keys ( self ) -> Iterator [ tuple [ str, str ] ]:

        for key in self.cache:
            if isinstance ( key, tuple ) and len ( key ) == 2:
                yield ( str ( key [ 0 ] ), str ( key [ 1 ] ) )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: clear_session
    #
    # Description:
    #
    #   Remove every entry belonging to this session.
    #
    #   What ends a conversation. Other sessions are untouched, which is the whole reason the tier is scoped.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   The number of entries removed.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def clear_session ( self ) -> int:

        # Collected before deleting: mutating the store while iterating it is undefined.

        doomed  = [ key for key in self.iterate_keys () if key [ 0 ] == self.session_id ]
        removed = 0

        for key in doomed:
            if self.cache.delete ( key ):
                removed += 1

        logger.debug ( "Cleared %d working-memory entries from session %s.", removed, self.session_id )

        # Return data to caller.

        return removed

    #-------------------------------------------------------------------------------------------------------------------
    # Conversation and task state
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: set_conversation
    #
    # Description:
    #
    #   Store the current conversation context.
    #
    # Arguments:
    #
    #   messages : The conversation so far, in the adapter's message shape.
    #   ttl      : Lifetime in seconds. The instance default when omitted.
    #
    # Returns:
    #
    #   True when stored.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def set_conversation ( self, messages: list [ dict [ str, Any ] ], ttl: float | None = None ) -> bool:

        # Return data to caller.

        return self.set ( KEY_CONVERSATION, messages, ttl )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: get_conversation
    #
    # Description:
    #
    #   Read the current conversation context.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   The stored messages, or an empty list when the session is new or has expired.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def get_conversation ( self ) -> list [ dict [ str, Any ] ]:

        messages = self.get ( KEY_CONVERSATION, default = [] )

        # Return data to caller.

        return list ( messages ) if isinstance ( messages, list ) else []

    #-------------------------------------------------------------------------------------------------------------------
    # Function: set_task_state
    #
    # Description:
    #
    #   Store the active task state.
    #
    # Arguments:
    #
    #   state : Whatever the caller needs to resume: a task name, a step index, partial results.
    #   ttl   : Lifetime in seconds. The instance default when omitted.
    #
    # Returns:
    #
    #   True when stored.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def set_task_state ( self, state: dict [ str, Any ], ttl: float | None = None ) -> bool:

        # Return data to caller.

        return self.set ( KEY_TASK_STATE, state, ttl )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: get_task_state
    #
    # Description:
    #
    #   Read the active task state.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   The stored state, or an empty mapping when nothing is in flight.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def get_task_state ( self ) -> dict [ str, Any ]:

        state = self.get ( KEY_TASK_STATE, default = {} )

        # Return data to caller.

        return dict ( state ) if isinstance ( state, dict ) else {}

    #-------------------------------------------------------------------------------------------------------------------
    # Lifecycle
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: volume
    #
    # Description:
    #
    #   Bytes the store currently occupies on disk.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   The size in bytes, across every session.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def volume ( self ) -> int:

        # Return data to caller.

        return int ( self.cache.volume () )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: expire
    #
    # Description:
    #
    #   Remove expired entries now rather than on next access.
    #
    #   diskcache evicts lazily, so an expired entry still occupies its bytes until something reads it. Housekeeping
    #   between turns calls this so the size ceiling is not spent on entries that are already dead.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   The number of entries removed.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def expire ( self ) -> int:

        # Return data to caller.

        return int ( self.cache.expire () )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: close
    #
    # Description:
    #
    #   Release the store, if this instance owns it.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def close ( self ) -> None:

        if self.owns_cache:
            self.cache.close ()

    #-------------------------------------------------------------------------------------------------------------------
    # Function: __enter__
    #
    # Description:
    #
    #   Enter a context manager.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   This instance.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def __enter__ ( self ) -> "WorkingMemory":

        # Return data to caller.

        return self

    #-------------------------------------------------------------------------------------------------------------------
    # Function: __exit__
    #
    # Description:
    #
    #   Leave a context manager, closing an owned store.
    #
    # Arguments:
    #
    #   exception_type      : Type of any exception being propagated.
    #   exception_value     : The exception being propagated.
    #   exception_traceback  : Its traceback.
    #
    # Returns:
    #
    #   False, so any exception continues to propagate. Typed as Literal[False] rather than bool because a bool return
    #   annotation tells the type checker this context manager might swallow exceptions, which it must never do.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def __exit__ ( self,
                   exception_type: type [ BaseException ] | None,
                   exception_value: BaseException | None,
                   exception_traceback: TracebackType | None ) -> Literal [ False ]:

        self.close ()

        # Return data to caller.

        return False
