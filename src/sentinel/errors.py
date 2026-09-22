#-----------------------------------------------------------------------------------------------------------------------
# Module:  errors.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   Exception hierarchy.
#
#   Every exception Sentinel raises deliberately derives from SentinelError, so callers can distinguish "the agent
#   decided this is wrong" from "something in the runtime broke". Subsystems add their own leaf types as they land;
#   keep the root shallow.
#-----------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------
# Class: SentinelError
#
# Description:
#
#   Root of the Sentinel exception hierarchy.
#-----------------------------------------------------------------------------------------------------------------------

class SentinelError ( Exception ):

    pass


#-----------------------------------------------------------------------------------------------------------------------
# Class: ConfigurationError
#
# Description:
#
#   Configuration could not be located, parsed, or validated.
#-----------------------------------------------------------------------------------------------------------------------

class ConfigurationError ( SentinelError ):

    pass


#-----------------------------------------------------------------------------------------------------------------------
# Class: DatabaseError
#
# Description:
#
#   The database could not be opened, migrated, or configured.
#-----------------------------------------------------------------------------------------------------------------------

class DatabaseError ( SentinelError ):

    pass


#-----------------------------------------------------------------------------------------------------------------------
# Class: MigrationError
#
# Description:
#
#   A schema migration could not be located, read, or applied.
#
#   A DatabaseError rather than a sibling of it: to every caller above the storage layer, a database whose schema is
#   half-applied is simply a database that could not be opened.
#-----------------------------------------------------------------------------------------------------------------------

class MigrationError ( DatabaseError ):

    pass


#-----------------------------------------------------------------------------------------------------------------------
# Class: MemorySystemError
#
# Description:
#
#   Root of the memory subsystem failures.
#
#   Named MemorySystemError, not MemoryError: the latter is a Python builtin for heap exhaustion, and shadowing it in a
#   module that other modules star-nothing but import freely would turn an out-of-memory condition into something the
#   agent's own except clauses quietly swallow.
#-----------------------------------------------------------------------------------------------------------------------

class MemorySystemError ( SentinelError ):

    pass


#-----------------------------------------------------------------------------------------------------------------------
# Class: EmbeddingError
#
# Description:
#
#   The embedding model could not be loaded, or a text could not be embedded.
#
#   Covers the absent package, the absent model file, and a model that loaded but produced nothing.
#-----------------------------------------------------------------------------------------------------------------------

class EmbeddingError ( MemorySystemError ):

    pass


#-----------------------------------------------------------------------------------------------------------------------
# Class: VectorDimensionError
#
# Description:
#
#   The configured vector width, the embedding model's output width, and the stored schema's width do not all agree.
#
#   Raised at startup rather than tolerated, because sqlite-vec stores whatever width it is handed and a disagreement
#   corrupts search silently instead of failing (Decision 2).
#-----------------------------------------------------------------------------------------------------------------------

class VectorDimensionError ( MemorySystemError ):

    pass


#-----------------------------------------------------------------------------------------------------------------------
# Class: EmbeddingModelMismatchError
#
# Description:
#
#   Stored vectors were produced by a different embedding model than the one now configured.
#
#   Vectors from two models share a coordinate space only by coincidence, so searching across them degrades recall
#   without erroring. Blocking the search and asking for a re-embed is the only honest response.
#-----------------------------------------------------------------------------------------------------------------------

class EmbeddingModelMismatchError ( MemorySystemError ):

    pass


#-----------------------------------------------------------------------------------------------------------------------
# Class: SecretsError
#
# Description:
#
#   A credential could not be read from or written to the OS keyring.
#
#   Distinct from a credential that is simply absent -- that is a ConfigurationError, because the fix is to supply the
#   key. This is the keyring backend itself failing.
#-----------------------------------------------------------------------------------------------------------------------

class SecretsError ( SentinelError ):

    pass


#-----------------------------------------------------------------------------------------------------------------------
# Class: LlmError
#
# Description:
#
#   Root of the LLM adapter failures.
#-----------------------------------------------------------------------------------------------------------------------

class LlmError ( SentinelError ):

    pass


#-----------------------------------------------------------------------------------------------------------------------
# Class: LlmAuthenticationError
#
# Description:
#
#   The provider rejected the credential. Never retried -- a bad key stays bad.
#-----------------------------------------------------------------------------------------------------------------------

class LlmAuthenticationError ( LlmError ):

    pass


#-----------------------------------------------------------------------------------------------------------------------
# Class: LlmTransportError
#
# Description:
#
#   The provider call failed in a way that may succeed on retry: a 5xx, a rate limit, a dropped connection.
#-----------------------------------------------------------------------------------------------------------------------

class LlmTransportError ( LlmError ):

    pass


#-----------------------------------------------------------------------------------------------------------------------
# Class: LlmRequestError
#
# Description:
#
#   The provider rejected the request itself: an unknown model, a malformed parameter, an unsupported option.
#
#   Never retried. The request will be rejected identically on every attempt, so retrying delays an unchanging answer
#   and buries the cause under a retry summary. A misspelled model name is the common case.
#-----------------------------------------------------------------------------------------------------------------------

class LlmRequestError ( LlmError ):

    pass


#-----------------------------------------------------------------------------------------------------------------------
# Class: LoopError
#
# Description:
#
#   The agentic loop could not complete a turn.
#
#   Raised only for conditions the loop cannot itself report as a response to the user -- a guard firing is normal
#   operation and produces an error *response*, not an exception.
#-----------------------------------------------------------------------------------------------------------------------

class LoopError ( SentinelError ):

    pass


#-----------------------------------------------------------------------------------------------------------------------
# Class: HeartbeatError
#
# Description:
#
#   The autonomous tick could not be scheduled, started, or stopped.
#
#   Not raised for a tick that ran and went badly: a failing action is recorded in the tick's errors column and the
#   loop carries on to the next one, because an agent that stops waking up after one bad task is worse than one that
#   occasionally does nothing useful.
#-----------------------------------------------------------------------------------------------------------------------

class HeartbeatError ( SentinelError ):

    pass


#-----------------------------------------------------------------------------------------------------------------------
# Class: ScheduleError
#
# Description:
#
#   A cron expression could not be parsed, or a scheduled job could not be stored.
#
#   A HeartbeatError rather than a sibling: to every caller above the scheduler, a job whose expression is unusable and
#   a tick that will not start are the same failure -- autonomous work will not happen.
#-----------------------------------------------------------------------------------------------------------------------

class ScheduleError ( HeartbeatError ):

    pass


#-----------------------------------------------------------------------------------------------------------------------
# Class: EventLogError
#
# Description:
#
#   An event could not be assembled because it does not fit the taxonomy.
#
#   Raised for a bad category or a malformed event name -- both authoring mistakes, which is why they raise rather than
#   being tolerated. A *storage* failure is deliberately not in this class: the event logger swallows those, because
#   losing an audit row is bad and taking down the operation being audited over it is worse.
#-----------------------------------------------------------------------------------------------------------------------

class EventLogError ( SentinelError ):

    pass


#-----------------------------------------------------------------------------------------------------------------------
# Class: UnsupportedEnvironmentError
#
# Description:
#
#   The host environment cannot run Sentinel.
#
#   Raised for conditions that no amount of retrying will fix -- an interpreter with an SQLite older than the
#   sqlite-vec floor, a missing loadable-extension build.
#-----------------------------------------------------------------------------------------------------------------------

class UnsupportedEnvironmentError ( SentinelError ):

    pass
