#-----------------------------------------------------------------------------------------------------------------------
# Module:  config.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   Configuration model and loader.
#
#   Maps config/agent.yaml onto a validated Pydantic tree. Two behaviours matter and are covered by tests:
#
#     * A missing config file is not an error. Defaults load and a warning is logged -- a fresh install should start
#       and be useful before the user has written any YAML.
#     * A present-but-wrong config file IS an error, raised as a ValidationError that names the offending field.
#       Silently substituting a default for a field the user tried to set would hide their mistake.
#
#   Path resolution follows the installed layout in architecture 6.5: user data lives in %LOCALAPPDATA%\Sentinel,
#   overridable with SENTINEL_DATA so a test or a portable install can redirect the whole tree with one variable.
#-----------------------------------------------------------------------------------------------------------------------

import logging
import os

from pathlib import Path
from typing  import Any

import yaml

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sentinel.errors import ConfigurationError

logger = logging.getLogger ( __name__ )

# Environment variables. SENTINEL_DATA relocates the whole user-data tree;
# SENTINEL_CONFIG overrides just the agent.yaml lookup.

ENV_DATA_DIRECTORY = "SENTINEL_DATA"
ENV_CONFIG_FILE    = "SENTINEL_CONFIG"

# Default file and directory names beneath the data directory.

CONFIG_FILE_NAME   = "agent.yaml"
CONFIG_DIRECTORY   = "config"
DATA_DIRECTORY     = "data"
CACHE_DIRECTORY    = "cache"
FILES_DIRECTORY    = "files"
LOGS_DIRECTORY     = "logs"
MODELS_DIRECTORY   = "models"
SKILLS_DIRECTORY   = "skills"
BACKUPS_DIRECTORY  = "backups"
WEBUI_DIRECTORY    = "webui"
DATABASE_FILE_NAME = "sentinel.db"
APPLICATION_FOLDER = "Sentinel"

# The embedding model and its output width are a matched pair (Decision 2). A mismatch
# corrupts vector search silently rather than loudly, so the width is validated here.

DEFAULT_EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
DEFAULT_VECTOR_DIMENSION = 384


#-----------------------------------------------------------------------------------------------------------------------
# Path resolution
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Function: resolve_data_directory
#
# Description:
#
#   Resolve the root of Sentinel's user-data tree.
#
# Arguments:
#
#   None.
#
# Returns:
#
#   SENTINEL_DATA if set, otherwise %LOCALAPPDATA%\Sentinel on Windows or ~/.local/share/sentinel elsewhere. The path
#   is absolute but need not exist.
#
#-----------------------------------------------------------------------------------------------------------------------

def resolve_data_directory () -> Path:

    # Resolve the root of Sentinel's user-data tree.

    # An explicit override wins outright -- tests and portable installs depend on it.

    override = os.environ.get ( ENV_DATA_DIRECTORY )

    if override:
        return Path ( override ).expanduser ().resolve ()

    # Windows: the installed layout in architecture 6.5.

    if os.name == "nt":
        local_application_data = os.environ.get ( "LOCALAPPDATA" )

        if local_application_data:
            return Path ( local_application_data ).joinpath ( APPLICATION_FOLDER ).resolve ()

    # Everything else follows the XDG convention (see architecture 6.7).

    return Path.home ().joinpath ( ".local", "share", "sentinel" ).resolve ()


#-----------------------------------------------------------------------------------------------------------------------
# Function: resolve_config_file
#
# Description:
#
#   Resolve the path of agent.yaml.
#
# Arguments:
#
#   data_directory : Data-tree root. Resolved from the environment when omitted.
#
# Returns:
#
#   SENTINEL_CONFIG if set, otherwise <data directory>/config/agent.yaml. The path need not exist -- an absent file is
#   a supported state.
#
#-----------------------------------------------------------------------------------------------------------------------

def resolve_config_file ( data_directory: Path | None = None ) -> Path:

    # Resolve the path of agent.yaml.

    override = os.environ.get ( ENV_CONFIG_FILE )

    if override:
        return Path ( override ).expanduser ().resolve ()

    root = data_directory if data_directory is not None else resolve_data_directory ()

    # Return data to caller.

    return root.joinpath ( CONFIG_DIRECTORY, CONFIG_FILE_NAME )


#-----------------------------------------------------------------------------------------------------------------------
# Configuration sections
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Class: StrictModel
#
# Description:
#
#   Base for every config section.
#
#   `extra = "forbid"` turns a typo into a named error instead of a setting that silently does nothing -- the single
#   most common way a config file lies to its author.
#-----------------------------------------------------------------------------------------------------------------------

class StrictModel ( BaseModel ):

    model_config = ConfigDict ( extra = "forbid", validate_assignment = True )


#-----------------------------------------------------------------------------------------------------------------------
# Class: AgentConfig
#
# Description:
#
#   Identity and locale of the agent itself.
#
# Attributes:
#
#   name     : Display name the agent answers to.
#   timezone : Timezone used when stamping the prompt with the current datetime.
#   language : Preferred response language.
#-----------------------------------------------------------------------------------------------------------------------

class AgentConfig ( StrictModel ):

    name:     str = "Sentinel"
    timezone: str = "UTC"
    language: str = "en"


#-----------------------------------------------------------------------------------------------------------------------
# Class: LlmConfig
#
# Description:
#
#   Primary LLM and its optional local fallback.
#
# Attributes:
#
#   provider          : Primary LLM provider.
#   model             : Primary model identifier.
#   max_tokens        : Ceiling on generated tokens per call.
#   temperature       : Sampling temperature, 0.0 to 2.0.
#   request_timeout   : Seconds before an LLM call is abandoned.
#   fallback_enabled  : Fall back to a local model when the primary is unavailable.
#   fallback_provider : Fallback provider, used only when fallback_enabled is true.
#   fallback_model    : Fallback model identifier.
#-----------------------------------------------------------------------------------------------------------------------

class LlmConfig ( StrictModel ):

    provider:          str   = "anthropic"
    model:             str   = "claude-opus-5"
    max_tokens:        int   = Field ( default = 4096, gt = 0 )
    temperature:       float = Field ( default = 1.0, ge = 0.0, le = 2.0 )
    request_timeout:   int   = Field ( default = 120, gt = 0 )
    fallback_enabled:  bool  = False
    fallback_provider: str   = "ollama"
    fallback_model:    str   = "qwen2.5:7b-instruct"


#-----------------------------------------------------------------------------------------------------------------------
# Class: RateLimitConfig
#
# Description:
#
#   Fixed-window rate limiting on the gateway.
#
#   A local agent cannot be flooded from the network, but it can be flooded by a runaway loop in a client. The limit
#   exists to make that failure loud rather than expensive.
#
# Attributes:
#
#   enabled  : Enforce the limit. Disabling it removes the 429 path entirely.
#   requests : Requests permitted per window, per client.
#   window   : Window length, in seconds.
#-----------------------------------------------------------------------------------------------------------------------

class RateLimitConfig ( StrictModel ):

    enabled:  bool = True
    requests: int  = Field ( default = 60, gt = 0 )
    window:   int  = Field ( default = 60, gt = 0 )


#-----------------------------------------------------------------------------------------------------------------------
# Class: ApiConfig
#
# Description:
#
#   The FastAPI gateway that Open WebUI talks to (architecture 4.2).
#
# Attributes:
#
#   host          : Interface the gateway binds to.
#   port          : Port the gateway listens on.
#   webui_port    : Port the managed Open WebUI subprocess listens on.
#   auth_required : Reject calls without a valid bearer token. The token itself lives in the OS keyring, never here.
#   rate_limit    : Fixed-window request limit.
#-----------------------------------------------------------------------------------------------------------------------

class ApiConfig ( StrictModel ):

    host:          str             = "127.0.0.1"
    port:          int             = Field ( default = 8000, ge = 1, le = 65535 )
    webui_port:    int             = Field ( default = 8080, ge = 1, le = 65535 )
    auth_required: bool            = True
    rate_limit:    RateLimitConfig = RateLimitConfig ()


#-----------------------------------------------------------------------------------------------------------------------
# Class: UiConfig
#
# Description:
#
#   The user-facing surface: the Open WebUI subprocess and the native window (architecture 4.2, 4.3.6).
#
#   Every switch here can turn a piece of the interface off, because each has a legitimate reason to be off. A
#   developer wants the API alone; a clean-room verification must not spawn a browser; a headless machine has no
#   window to show. What none of them may do is change what the interface *is* when it does run -- that lives in
#   config/webui.env.
#
# Attributes:
#
#   webui_enabled  : Start the managed Open WebUI subprocess.
#   native_window  : Open the pywebview desktop window once the interface is ready.
#   browser_fallback : Fall back to a browser in app mode when the native window cannot be shown.
#   window_title   : Window and taskbar title. This is the name the user sees, so it defaults to the product name.
#   window_width   : Initial window width, in pixels.
#   window_height  : Initial window height, in pixels.
#-----------------------------------------------------------------------------------------------------------------------

class UiConfig ( StrictModel ):

    webui_enabled:    bool = True
    native_window:    bool = True
    browser_fallback: bool = True
    window_title:     str  = "Sentinel"
    window_width:     int  = Field ( default = 1200, gt = 0 )
    window_height:    int  = Field ( default = 800, gt = 0 )


#-----------------------------------------------------------------------------------------------------------------------
# Class: ProcessConfig
#
# Description:
#
#   Recovery policy for the process lifecycle FSM (architecture 4.3.5).
#
#   These are the numbers that decide when a flapping component stops being restarted and starts being reported as
#   broken. Configurable because the right answer depends on the machine: a slow laptop needs a longer startup timeout
#   than the default, and being unable to raise it would look like a Sentinel fault.
#
# Attributes:
#
#   max_retries           : Recovery attempts per component before the system goes FATAL.
#   retry_window          : Seconds of sustained health after which the failure counter resets.
#   health_check_interval : Seconds between steady-state health probes.
#   startup_timeout       : Seconds a component has to become healthy during startup. 120 rather than the 60 the
#                           architecture first specified: Open WebUI downloads its own embedding model on first run,
#                           and a measured cold start took ~50 s on a fast connection.
#   shutdown_timeout      : Seconds a component has to exit gracefully before it is killed.
#-----------------------------------------------------------------------------------------------------------------------

class ProcessConfig ( StrictModel ):

    max_retries:           int = Field ( default = 3, ge = 0 )
    retry_window:          int = Field ( default = 300, gt = 0 )
    health_check_interval: int = Field ( default = 10, gt = 0 )
    startup_timeout:       int = Field ( default = 120, gt = 0 )
    shutdown_timeout:      int = Field ( default = 10, gt = 0 )


#-----------------------------------------------------------------------------------------------------------------------
# Class: DatabaseBackupConfig
#
# Description:
#
#   Hourly file-copy snapshots of the database (architecture 6.6).
#
# Attributes:
#
#   enabled  : Take periodic snapshots.
#   interval : Seconds between snapshots.
#   retain   : Number of snapshots kept before the oldest is discarded.
#-----------------------------------------------------------------------------------------------------------------------

class DatabaseBackupConfig ( StrictModel ):

    enabled:  bool = True
    interval: int  = Field ( default = 3600, gt = 0 )
    retain:   int  = Field ( default = 168, ge = 0 )


#-----------------------------------------------------------------------------------------------------------------------
# Class: DatabaseConfig
#
# Description:
#
#   SQLite tuning and vector-store geometry (architecture 6.6).
#
# Attributes:
#
#   path               : Database file path. A relative path resolves against the data directory.
#   wal_mode           : Enable write-ahead logging for concurrent reads during writes.
#   journal_size_limit : Byte cap on the WAL file.
#   busy_timeout       : Milliseconds to retry on lock contention.
#   vector_dimensions  : Embedding width. MUST match the embedding model.
#   embedding_model    : Embedding model name, recorded per vector so a change can trigger a re-embed.
#   backup             : Snapshot policy.
#-----------------------------------------------------------------------------------------------------------------------

class DatabaseConfig ( StrictModel ):

    path:               str                  = "./data/sentinel.db"
    wal_mode:           bool                 = True
    journal_size_limit: int                  = Field ( default = 67108864, gt = 0 )
    busy_timeout:       int                  = Field ( default = 5000, ge = 0 )
    vector_dimensions:  int                  = Field ( default = DEFAULT_VECTOR_DIMENSION, gt = 0 )
    embedding_model:    str                  = DEFAULT_EMBEDDING_MODEL
    backup:             DatabaseBackupConfig = DatabaseBackupConfig ()

    #-------------------------------------------------------------------------------------------------------------------
    # Function: check_known_dimension
    #
    # Description:
    #
    #   Warn when the width departs from the bundled model's 384 (Decision 2).
    #
    # Arguments:
    #
    #   value : The configured vector width.
    #
    # Returns:
    #
    #   The value unchanged. This validator warns; it does not reject, because a deliberate change of embedding model
    #   is legitimate.
    #
    #-------------------------------------------------------------------------------------------------------------------

    @field_validator ( "vector_dimensions" )
    @classmethod
    def check_known_dimension ( cls, value: int ) -> int:

        # Warn when the width departs from the bundled model's 384.

        if value != DEFAULT_VECTOR_DIMENSION:
            logger.warning (
                "database.vector_dimensions is %d, not the bundled %s width of %d. "
                "A mismatch between this value and the embedding model corrupts "
                "vector search silently.",
                value, DEFAULT_EMBEDDING_MODEL, DEFAULT_VECTOR_DIMENSION,
            )

        # Return data to caller.

        return value


#-----------------------------------------------------------------------------------------------------------------------
# Class: LoopConfig
#
# Description:
#
#   Hard limits on the agentic loop (architecture 3.3). These are the stop conditions for a runaway agent, not
#   performance tuning.
#
# Attributes:
#
#   max_iterations   : Ceiling on tool-use cycles per turn.
#   loop_timeout     : Wall-clock seconds for the whole loop.
#   tool_timeout     : Seconds for any single tool call.
#   max_total_tokens : Context budget before compression is attempted.
#   max_tool_result  : Tokens per tool result before truncation.
#   retry_attempts   : LLM call attempts before the loop reports failure.
#   retry_backoff    : Seconds before the first retry. Doubles per attempt (1/2/4).
#-----------------------------------------------------------------------------------------------------------------------

class LoopConfig ( StrictModel ):

    max_iterations:   int   = Field ( default = 10, gt = 0 )
    loop_timeout:     int   = Field ( default = 120, gt = 0 )
    tool_timeout:     int   = Field ( default = 30, gt = 0 )
    max_total_tokens: int   = Field ( default = 180000, gt = 0 )
    max_tool_result:  int   = Field ( default = 16000, gt = 0 )
    retry_attempts:   int   = Field ( default = 3, gt = 0 )
    retry_backoff:    float = Field ( default = 1.0, gt = 0.0 )


#-----------------------------------------------------------------------------------------------------------------------
# Class: PromptConfig
#
# Description:
#
#   Per-section token budgets for the assembled system prompt (architecture 3.3.5).
#
#   Each section is truncated to its own ceiling so no single source -- most plausibly retrieved memory -- can crowd out
#   the rest of the prompt.
#
# Attributes:
#
#   max_total       : Ceiling on the whole system prompt.
#   max_persona     : Stable section 1: identity and traits.
#   max_permissions : Stable section 2: permission boundaries.
#   max_preferences : Stable section 3: learned preferences.
#   max_memory      : Volatile section 5: retrieved memories.
#   max_channel     : Volatile section 6: originating channel.
#   max_heartbeat   : Volatile section 7: pending scheduled work.
#-----------------------------------------------------------------------------------------------------------------------

class PromptConfig ( StrictModel ):

    max_total:       int = Field ( default = 4500, gt = 0 )
    max_persona:     int = Field ( default = 500, gt = 0 )
    max_permissions: int = Field ( default = 200, gt = 0 )
    max_preferences: int = Field ( default = 500, gt = 0 )
    max_memory:      int = Field ( default = 2000, gt = 0 )
    max_channel:     int = Field ( default = 200, gt = 0 )
    max_heartbeat:   int = Field ( default = 300, gt = 0 )


#-----------------------------------------------------------------------------------------------------------------------
# Class: HeartbeatConfig
#
# Description:
#
#   Autonomous wake-up tick.
#
#   The 240 s default sits deliberately below the 300 s prompt-cache TTL so each tick re-uses the cached stable prefix
#   rather than paying to rebuild it (Decision 3).
#
#   The last three fields are the guardrails on unsupervised work, and they are the reason autonomy is safe to leave
#   switched on. Architecture 3.3.9 gives heartbeat mode a lower iteration ceiling and a restricted tool set than an
#   interactive turn, on the reasoning that nobody is watching: a runaway interactive turn is interrupted by the person
#   reading it, and a runaway autonomous one is not.
#
# Attributes:
#
#   enabled                : Run the heartbeat loop.
#   interval               : Seconds between ticks.
#   max_autonomous_actions : Ceiling on actions per tick.
#   max_iterations         : Tool-use cycles allowed in one autonomous turn. Lower than loop.max_iterations by design.
#   allowed_tools          : Tools an autonomous turn may call. Empty denies all -- the model is default-deny.
#-----------------------------------------------------------------------------------------------------------------------

class HeartbeatConfig ( StrictModel ):

    enabled:                bool         = True
    interval:               int          = Field ( default = 240, gt = 0 )
    max_autonomous_actions: int          = Field ( default = 5, gt = 0 )
    max_iterations:         int          = Field ( default = 3, gt = 0 )
    allowed_tools:          list [ str ] = Field ( default_factory = list )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: check_cache_ttl
    #
    # Description:
    #
    #   Warn when the interval exceeds the prompt-cache TTL it was chosen to beat.
    #
    # Arguments:
    #
    #   value : The configured interval, in seconds.
    #
    # Returns:
    #
    #   The value unchanged. A longer interval is wasteful, not invalid.
    #
    #-------------------------------------------------------------------------------------------------------------------

    @field_validator ( "interval" )
    @classmethod
    def check_cache_ttl ( cls, value: int ) -> int:

        # Warn when the interval exceeds the prompt-cache TTL it was chosen to beat.

        prompt_cache_ttl_seconds = 300

        if value >= prompt_cache_ttl_seconds:
            logger.warning (
                "heartbeat.interval is %d s, at or above the %d s prompt-cache TTL. "
                "Every tick will rebuild the cached prefix instead of re-using it.",
                value, prompt_cache_ttl_seconds,
            )

        # Return data to caller.

        return value


#-----------------------------------------------------------------------------------------------------------------------
# Class: MemoryConfig
#
# Description:
#
#   Three-tier memory sizing and the unified retrieval parameters (architecture 3.2.4).
#
#   The last four fields are the algorithm's k, theta, lambda, and tau_half. They are configurable because the right
#   answer depends on how the agent is used: a user who talks to it all day wants a shorter half-life than one who
#   consults it weekly, and neither can be guessed from here.
#
# Attributes:
#
#   working_ttl         : Tier 1 entry lifetime, in seconds.
#   working_size_limit  : Tier 1 cache ceiling, in bytes.
#   episodic_retention  : Tier 2 retention, in days.
#   retrieval_limit     : Memories injected per prompt. The algorithm's k.
#   relevance_threshold : Minimum cosine similarity for a semantic match. The algorithm's theta.
#   recency_weight      : Share of the blended score given to recency, 0.0 to 1.0. The algorithm's lambda.
#   recency_half_life   : Days after which a memory's recency contribution has halved. The algorithm's tau_half.
#-----------------------------------------------------------------------------------------------------------------------

class MemoryConfig ( StrictModel ):

    working_ttl:         int   = Field ( default = 3600, gt = 0 )
    working_size_limit:  int   = Field ( default = 268435456, gt = 0 )
    episodic_retention:  int   = Field ( default = 90, ge = 0 )
    retrieval_limit:     int   = Field ( default = 10, gt = 0 )
    relevance_threshold: float = Field ( default = 0.72, ge = 0.0, le = 1.0 )
    recency_weight:      float = Field ( default = 0.3, ge = 0.0, le = 1.0 )
    recency_half_life:   float = Field ( default = 7.0, gt = 0.0 )


#-----------------------------------------------------------------------------------------------------------------------
# Class: LoggingConfig
#
# Description:
#
#   Structured event logging (architecture 3.2.10, 4.4).
#
#   Two logs are configured from here and they are not the same thing. `level` and `json_output` govern the ordinary
#   operator log -- the console lines that say what the process is doing. Everything below them governs the *event*
#   log: the append-only audit trail of what the agent did, its rotating file copy, and its optional metrics.
#
#   `retention` applies to the rotated files only. The event_log table is append-only at the storage layer -- migration
#   003 installs triggers that refuse UPDATE and DELETE -- so its rows are never pruned. That is deliberate: an audit
#   trail the audited process can quietly shorten is not one.
#
# Attributes:
#
#   level              : Minimum severity emitted by the operator log. Normalised to upper case.
#   json_output        : Emit structured JSON rather than plain text.
#   retention          : Rotated event-log file retention, in days. Zero keeps them for ever.
#   event_log_enabled  : Persist events to the event_log table. False exports without persisting.
#   file_export        : Write the rotating JSON Lines copy under the logs directory.
#   stdout_export      : Echo every event to the console. A development switch; it doubles the console output.
#   max_file_size      : Bytes after which the live event file is rotated and stamped. Zero disables rotation.
#   prometheus_enabled : Count events by category and serve them at GET /metrics.
#   query_limit        : Ceiling on how many rows one GET /api/logs call may return.
#-----------------------------------------------------------------------------------------------------------------------

class LoggingConfig ( StrictModel ):

    level:              str  = "INFO"
    json_output:        bool = True
    retention:          int  = Field ( default = 30, ge = 0 )
    event_log_enabled:  bool = True
    file_export:        bool = True
    stdout_export:      bool = False
    max_file_size:      int  = Field ( default = 10485760, ge = 0 )
    prometheus_enabled: bool = False
    query_limit:        int  = Field ( default = 1000, gt = 0 )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: check_level
    #
    # Description:
    #
    #   Reject a level name the stdlib logging module will not recognise.
    #
    # Arguments:
    #
    #   value : The configured level name, in any case.
    #
    # Returns:
    #
    #   The level name in upper case.
    #
    #   Raises ValueError if the name is not one of DEBUG, INFO, WARNING, ERROR, CRITICAL.
    #
    #-------------------------------------------------------------------------------------------------------------------

    @field_validator ( "level" )
    @classmethod
    def check_level ( cls, value: str ) -> str:

        # Reject a level name the stdlib logging module will not recognise.

        normalised = value.upper ()
        permitted  = { "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL" }

        if normalised not in permitted:
            raise ValueError ( f"must be one of {sorted ( permitted )}, got {value!r}" )

        # Return data to caller.

        return normalised


#-----------------------------------------------------------------------------------------------------------------------
# Class: ChannelConfig
#
# Description:
#
#   One configured communication channel.
#
#   Channels are the one place a required field lives: the list is empty by default, but declaring an entry means
#   committing to a name and a type. There is no sane default for either, and guessing would attach the agent to the
#   wrong service.
#
# Attributes:
#
#   name    : Operator-chosen label for this channel. Required.
#   type    : Channel kind, e.g. "email" or "telegram". Required.
#   enabled : Whether the router should serve this channel.
#   options : Channel-specific settings, passed through untouched.
#-----------------------------------------------------------------------------------------------------------------------

class ChannelConfig ( StrictModel ):

    name:    str
    type:    str
    enabled: bool              = True
    options: dict [ str, Any ] = Field ( default_factory = dict )


#-----------------------------------------------------------------------------------------------------------------------
# Class: SentinelConfig
#
# Description:
#
#   Root configuration -- the parsed form of config/agent.yaml.
#
# Attributes:
#
#   agent          : Identity and locale.
#   llm            : Primary model and fallback.
#   api            : Gateway and Open WebUI ports.
#   ui             : Open WebUI subprocess and native window.
#   process        : Component recovery policy.
#   database       : SQLite tuning and vector geometry.
#   loop           : Agentic loop stop conditions.
#   prompt         : System prompt section budgets.
#   heartbeat      : Autonomous tick.
#   memory         : Three-tier memory sizing.
#   logging        : Event logging.
#   channels       : Configured communication channels. Empty by default.
#   data_directory : Root of the user-data tree. Populated by the loader, not by the YAML file.
#   source_file    : The agent.yaml this was read from, or None when defaults were used.
#-----------------------------------------------------------------------------------------------------------------------

class SentinelConfig ( StrictModel ):

    agent:     AgentConfig            = AgentConfig ()
    llm:       LlmConfig              = LlmConfig ()
    api:       ApiConfig              = ApiConfig ()
    ui:        UiConfig               = UiConfig ()
    process:   ProcessConfig          = ProcessConfig ()
    database:  DatabaseConfig         = DatabaseConfig ()
    loop:      LoopConfig             = LoopConfig ()
    prompt:    PromptConfig           = PromptConfig ()
    heartbeat: HeartbeatConfig        = HeartbeatConfig ()
    memory:    MemoryConfig           = MemoryConfig ()
    logging:   LoggingConfig          = LoggingConfig ()
    channels:  list [ ChannelConfig ] = Field ( default_factory = list )

    # Populated by the loader, not by the YAML file itself.

    data_directory: Path        = Field ( default_factory = resolve_data_directory )
    source_file:    Path | None = None

    #-------------------------------------------------------------------------------------------------------------------
    # Derived paths
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: database_path
    #
    # Description:
    #
    #   Absolute path of the SQLite database.
    #
    #   A relative database.path is resolved against the data directory, so the shipped "./data/sentinel.db" default
    #   lands inside the user's tree rather than beside whatever directory the process happened to start in.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   The absolute database path.
    #
    #-------------------------------------------------------------------------------------------------------------------

    @property
    def database_path ( self ) -> Path:

        # Resolve the database path, honouring an absolute setting as given.

        configured = Path ( self.database.path ).expanduser ()

        if configured.is_absolute ():
            return configured

        # Return data to caller.

        return ( self.data_directory / configured ).resolve ()

    #-------------------------------------------------------------------------------------------------------------------
    # Function: cache_directory
    #
    # Description:
    #
    #   Absolute path of the diskcache directory.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   The absolute cache directory path.
    #
    #-------------------------------------------------------------------------------------------------------------------

    @property
    def cache_directory ( self ) -> Path:

        # Return data to caller.

        return self.data_directory / CACHE_DIRECTORY

    #-------------------------------------------------------------------------------------------------------------------
    # Function: files_directory
    #
    # Description:
    #
    #   Absolute path of the skill artifact and upload directory.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   The absolute files directory path.
    #
    #-------------------------------------------------------------------------------------------------------------------

    @property
    def files_directory ( self ) -> Path:

        # Return data to caller.

        return self.data_directory / FILES_DIRECTORY

    #-------------------------------------------------------------------------------------------------------------------
    # Function: logs_directory
    #
    # Description:
    #
    #   Absolute path of the rotating event-log directory.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   The absolute logs directory path.
    #
    #-------------------------------------------------------------------------------------------------------------------

    @property
    def logs_directory ( self ) -> Path:

        # Return data to caller.

        return self.data_directory / LOGS_DIRECTORY

    #-------------------------------------------------------------------------------------------------------------------
    # Function: models_directory
    #
    # Description:
    #
    #   Absolute path of the local model cache.
    #
    #   Holds the ONNX embedding weights (Decision 2), and later the Whisper and Piper voice models. Kept inside
    #   Sentinel's own tree rather than in fastembed's default temporary directory, so the ~90 MB download survives a
    #   reboot and one backup captures every piece of user state.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   The absolute model cache path.
    #
    #-------------------------------------------------------------------------------------------------------------------

    @property
    def models_directory ( self ) -> Path:

        # Return data to caller.

        return self.data_directory / MODELS_DIRECTORY

    #-------------------------------------------------------------------------------------------------------------------
    # Function: webui_directory
    #
    # Description:
    #
    #   Absolute path of Open WebUI's own data directory.
    #
    #   Co-located inside Sentinel's tree rather than left in Open WebUI's default location, so one directory holds
    #   every piece of user state and one backup captures all of it (architecture 4.2).
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   The absolute Open WebUI data directory path.
    #
    #-------------------------------------------------------------------------------------------------------------------

    @property
    def webui_directory ( self ) -> Path:

        # Return data to caller.

        return self.data_directory / WEBUI_DIRECTORY

    #-------------------------------------------------------------------------------------------------------------------
    # Function: skills_directory
    #
    # Description:
    #
    #   Absolute path of the user-installed skills directory.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   The absolute skills directory path.
    #
    #-------------------------------------------------------------------------------------------------------------------

    @property
    def skills_directory ( self ) -> Path:

        # Return data to caller.

        return self.data_directory / SKILLS_DIRECTORY

    #-------------------------------------------------------------------------------------------------------------------
    # Function: backups_directory
    #
    # Description:
    #
    #   Absolute path of the database snapshot directory.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   The absolute backups directory path, alongside the database file.
    #
    #-------------------------------------------------------------------------------------------------------------------

    @property
    def backups_directory ( self ) -> Path:

        # Return data to caller.

        return self.database_path.parent / BACKUPS_DIRECTORY

    #-------------------------------------------------------------------------------------------------------------------
    # Function: managed_directories
    #
    # Description:
    #
    #   Every directory `sentinel init` is responsible for creating.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   Absolute paths, parents before children.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def managed_directories ( self ) -> list [ Path ]:

        # Return data to caller.

        return [
            self.data_directory,
            self.data_directory / CONFIG_DIRECTORY,
            self.database_path.parent,
            self.backups_directory,
            self.cache_directory,
            self.files_directory,
            self.logs_directory,
            self.models_directory,
            self.skills_directory,
            self.webui_directory,
        ]


#-----------------------------------------------------------------------------------------------------------------------
# Loader
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Function: load_config
#
# Description:
#
#   Load and validate configuration.
#
# Arguments:
#
#   config_path    : Explicit agent.yaml path. Resolved from the environment when omitted.
#   data_directory : Explicit data-tree root. Resolved from the environment when omitted.
#
# Returns:
#
#   A validated SentinelConfig. Every field carries a default, so an absent file yields a fully usable configuration
#   and logs a warning.
#
#   Raises ConfigurationError if the file exists but is not readable, is not valid YAML, or does not parse to a
#   mapping. Raises pydantic.ValidationError if the file parsed but a value is missing, unknown, or of the wrong type;
#   the error names the offending field.
#
#-----------------------------------------------------------------------------------------------------------------------

def load_config ( config_path: str | Path | None = None,
                  data_directory: str | Path | None = None ) -> SentinelConfig:

    # Load and validate configuration.

    root = (
        Path ( data_directory ).expanduser ().resolve ()
        if data_directory is not None
        else resolve_data_directory ()
    )

    path = (
        Path ( config_path ).expanduser ()
        if config_path is not None
        else resolve_config_file ( root )
    )

    # An absent file is a supported state, not a failure. Warn and take the defaults.

    if not path.is_file ():
        logger.warning (
            "No configuration file at %s. Using built-in defaults. "
            "Run `sentinel init` to write a starter agent.yaml.",
            path,
        )

        return SentinelConfig ( data_directory = root, source_file = None )

    # From here the file exists, so the user meant something by it. Parse strictly.

    try:
        raw_text = path.read_text ( encoding = "utf-8" )
    except OSError as error:
        raise ConfigurationError ( f"Cannot read configuration file {path}: {error}" ) from error

    try:
        document = yaml.safe_load ( raw_text )
    except yaml.YAMLError as error:
        raise ConfigurationError (
            f"Configuration file {path} is not valid YAML: {error}"
        ) from error

    # An empty file is legitimate -- it means "defaults, deliberately".

    if document is None:
        document = {}

    if not isinstance ( document, dict ):
        raise ConfigurationError (
            f"Configuration file {path} must contain a mapping at the top level, "
            f"got {type ( document ).__name__}."
        )

    # Loader-owned fields override anything the file tries to say about them.

    document [ "data_directory" ] = root
    document [ "source_file" ]    = path

    configuration = SentinelConfig.model_validate ( document )

    logger.info ( "Loaded configuration from %s.", path )

    # Return data to caller.

    return configuration
