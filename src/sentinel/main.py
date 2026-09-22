#-----------------------------------------------------------------------------------------------------------------------
# Program: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
#
# Description:
#
#   Application entry point and command-line interface.
#
#   Commands:
#
#     sentinel version    Print the version string.
#     sentinel init       Create the data tree, config, database, and cache.
#     sentinel start      Verify the environment, then run the agent and its interface.
#     sentinel stop       Ask a running agent to shut down.
#     sentinel key        Manage credentials in the OS keyring.
#
#   From Phase 2, `start` is the whole application: the API gateway, the managed Open WebUI subprocess, and a native
#   desktop window, coordinated by the process lifecycle FSM. `--no-serve` remains the verification-only path.
#
# Usage:
#
#   python -m sentinel version
#   python -m sentinel init
#   python -m sentinel key set anthropic
#   python -m sentinel key set api-token --generate
#   python -m sentinel key status
#   python -m sentinel start
#   python -m sentinel stop
#   python -m sentinel --data-dir D:\sentinel-data start
#-----------------------------------------------------------------------------------------------------------------------

import argparse
import asyncio
import getpass
import logging
import shutil
import sys

from collections.abc import Sequence
from pathlib         import Path
from typing          import TYPE_CHECKING, Any

from pydantic import ValidationError

if TYPE_CHECKING:
    from sentinel.core.process import ProcessManager
    from sentinel.llm.adapter  import LlmAdapter

from sentinel          import __version__
from sentinel.cache    import initialise_cache
from sentinel.config   import SentinelConfig, load_config, resolve_config_file
from sentinel.database import assert_sqlite_version, initialise_database
from sentinel.errors   import ConfigurationError, SentinelError
from sentinel.security.secrets import (
    SECRET_API_TOKEN,
    delete_secret,
    describe_secrets,
    generate_token,
    resolve_name,
    write_secret,
)

logger = logging.getLogger ( "sentinel" )

# Exit codes. Distinct values so a wrapper script can tell "you configured it wrong"
# apart from "this machine cannot run it".

EXIT_SUCCESS      = 0
EXIT_FAILURE      = 1
EXIT_CONFIG_ERROR = 2
EXIT_INTERRUPTED  = 130

# Where the starter agent.yaml can be found, most specific first.
#
# An installed wheel carries the file inside the package (the build force-includes
# config/agent.yaml as sentinel/_templates/agent.yaml), because the repository's
# top-level config/ directory does not survive installation. A source checkout or an
# editable install has no such copy and falls back to the repository path.

PACKAGE_CONFIG_TEMPLATE    = Path ( __file__ ).resolve ().parent / "_templates" / "agent.yaml"
REPOSITORY_CONFIG_TEMPLATE = Path ( __file__ ).resolve ().parents [ 2 ] / "config" / "agent.yaml"


#-----------------------------------------------------------------------------------------------------------------------
# Logging
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Function: configure_logging
#
# Description:
#
#   Configure console logging for the CLI.
#
#   This is the operator log -- the plain-text record of what the process is doing -- and it is a different thing from
#   the structured event log of architecture 3.2.10, which records what the *agent* does and lives in SQLite and the
#   rotating files under logs\. It must exist before configuration loads, because load_config warns through it when no
#   config file is found, which is also why it cannot be driven by logging.level from that same file.
#
# Arguments:
#
#   verbose : Emit DEBUG records instead of INFO.
#
# Returns:
#
#   None.
#
#-----------------------------------------------------------------------------------------------------------------------

def configure_logging ( verbose: bool = False ) -> None:

    # Configure console logging for the CLI.

    logging.basicConfig (
        level  = logging.DEBUG if verbose else logging.INFO,
        format = "%(levelname)-8s %(name)s: %(message)s",
        stream = sys.stderr,
    )

    # Quieten APScheduler's per-execution chatter.
    #
    # It logs two INFO records for every heartbeat tick -- one starting, one finishing -- whether or not the tick did
    # anything. At the 240 s default that is around 720 lines a day of "executed successfully" saying nothing, and it
    # buries Sentinel's own records. Ticks that actually do something report it themselves; a tick that does not is
    # not news. WARNING and above still comes through, so a missed run or a job that overran is still visible.
    #
    # Raised back to the configured level under --verbose, where the whole point is to see everything.

    if not verbose:
        logging.getLogger ( "apscheduler" ).setLevel ( logging.WARNING )


#-----------------------------------------------------------------------------------------------------------------------
# Output
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Function: print_block
#
# Description:
#
#   Print a block of terminal output, padded with one blank line above and below.
#
#   Every command's output goes through here, so the spacing rule is structural rather than a convention each new
#   command has to remember. A command that prints directly will look subtly different from every other one, which is
#   exactly the drift this prevents.
#
# Arguments:
#
#   lines : The lines to print, in order. An empty string prints a blank line, so a block can carry its own internal
#           spacing without the caller reaching for print directly.
#
# Returns:
#
#   None.
#
#-----------------------------------------------------------------------------------------------------------------------

def print_block ( *lines: str ) -> None:

    # One blank line above, the content, one blank line below.

    print ()

    for line in lines:
        print ( line )

    print ()


#-----------------------------------------------------------------------------------------------------------------------
# Class: PaddedVersionAction
#
# Description:
#
#   argparse action that prints the version through print_block.
#
#   argparse's own "version" action writes straight to stdout with no padding, which would leave `sentinel --version`
#   as the one command in the program that does not follow the spacing rule. Replacing the action is cheaper than
#   tolerating the exception.
#-----------------------------------------------------------------------------------------------------------------------

class PaddedVersionAction ( argparse.Action ):

    #-------------------------------------------------------------------------------------------------------------------
    # Function: __init__
    #
    # Description:
    #
    #   Construct the action, defaulting to consuming no arguments.
    #
    # Arguments:
    #
    #   option_strings : The flags this action responds to.
    #   dest           : Destination attribute. Unused; suppressed so it never reaches the namespace.
    #   keywords       : Remaining argparse keyword arguments.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def __init__ ( self, option_strings: Sequence [ str ], dest: str = argparse.SUPPRESS, **keywords: Any ) -> None:

        keywords.setdefault ( "nargs", 0 )
        keywords.setdefault ( "help", "Print the version string and exit." )

        super ().__init__ ( option_strings = list ( option_strings ), dest = dest, **keywords )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: __call__
    #
    # Description:
    #
    #   Print the version and exit.
    #
    # Arguments:
    #
    #   parser        : The parser invoking the action.
    #   namespace     : Parsed arguments so far. Unused.
    #   values        : Consumed values. Always empty.
    #   option_string : The flag that triggered the action. Unused.
    #
    # Returns:
    #
    #   None. Exits the process with a success code, as argparse's own version action does.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def __call__ ( self,
                   parser: argparse.ArgumentParser,
                   namespace: argparse.Namespace,
                   values: str | Sequence [ Any ] | None,
                   option_string: str | None = None ) -> None:

        print_block ( f"sentinel {__version__}" )

        parser.exit ()


#-----------------------------------------------------------------------------------------------------------------------
# Commands
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Function: command_version
#
# Description:
#
#   Print the version string.
#
# Arguments:
#
#   None.
#
# Returns:
#
#   Process exit code.
#
#-----------------------------------------------------------------------------------------------------------------------

def command_version () -> int:

    # Print the version string.

    print_block ( f"sentinel {__version__}" )

    # Return data to caller.

    return EXIT_SUCCESS


#-----------------------------------------------------------------------------------------------------------------------
# Function: find_config_template
#
# Description:
#
#   Locate the starter agent.yaml.
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

def find_config_template () -> Path | None:

    # Locate the starter agent.yaml, most specific location first.

    for candidate in ( PACKAGE_CONFIG_TEMPLATE, REPOSITORY_CONFIG_TEMPLATE ):
        if candidate.is_file ():
            return candidate

    # Return data to caller.

    return None


#-----------------------------------------------------------------------------------------------------------------------
# Function: write_starter_config
#
# Description:
#
#   Write a starter agent.yaml, never overwriting an existing one.
#
# Arguments:
#
#   destination : Path the config should occupy.
#
# Returns:
#
#   True if a file was written, False if one was already there or no template could be found.
#
#-----------------------------------------------------------------------------------------------------------------------

def write_starter_config ( destination: Path ) -> bool:

    # Write a starter agent.yaml, never overwriting an existing one.

    if destination.exists ():
        logger.info ( "Configuration already present at %s -- left untouched.", destination )

        return False

    template = find_config_template ()

    # No template is survivable -- every setting has a default, so the agent runs
    # regardless. Say so plainly rather than writing a stub that claims to document
    # the defaults while listing two of them.

    if template is None:
        logger.warning (
            "No starter configuration template found. Sentinel will run on built-in "
            "defaults; write %s by hand to change any of them.",
            destination,
        )

        return False

    destination.parent.mkdir ( parents = True, exist_ok = True )

    shutil.copyfile ( template, destination )

    logger.info ( "Wrote starter configuration to %s.", destination )

    # Return data to caller.

    return True


#-----------------------------------------------------------------------------------------------------------------------
# Function: initialise
#
# Description:
#
#   Create every directory, the database, and the cache.
#
# Arguments:
#
#   configuration : The loaded configuration.
#
# Returns:
#
#   A merged report from the database and cache initialisers.
#
#-----------------------------------------------------------------------------------------------------------------------

async def initialise ( configuration: SentinelConfig ) -> dict [ str, str ]:

    # Create every managed directory before anything tries to write inside one.

    for directory in configuration.managed_directories ():
        directory.mkdir ( parents = True, exist_ok = True )

    # Stand up the database, then the working-memory cache.

    database_report = await initialise_database (
        database_path      = configuration.database_path,
        wal_mode           = configuration.database.wal_mode,
        busy_timeout       = configuration.database.busy_timeout,
        journal_size_limit = configuration.database.journal_size_limit,
    )

    cache_report = initialise_cache (
        cache_directory = configuration.cache_directory,
        size_limit      = configuration.memory.working_size_limit,
    )

    # Return data to caller.

    return {
        "data_directory": str ( configuration.data_directory ),
        "database": database_report [ "path" ],
        "sqlite_version": database_report [ "sqlite_version" ],
        "journal_mode": database_report [ "journal_mode" ],
        "vec_version": database_report [ "vec_version" ],
        "schema_version": database_report [ "schema_version" ],
        "cache": cache_report [ "path" ],
    }


#-----------------------------------------------------------------------------------------------------------------------
# Function: command_init
#
# Description:
#
#   Create the data tree, starter configuration, database, and cache.
#
#   Idempotent: running it twice changes nothing and overwrites no existing file.
#
# Arguments:
#
#   configuration : The loaded configuration.
#
# Returns:
#
#   Process exit code.
#
#-----------------------------------------------------------------------------------------------------------------------

def command_init ( configuration: SentinelConfig ) -> int:

    # Create the data tree, starter configuration, database, and cache.

    configuration.data_directory.mkdir ( parents = True, exist_ok = True )

    write_starter_config ( resolve_config_file ( configuration.data_directory ) )

    report = asyncio.run ( initialise ( configuration ) )

    # Report what was built, so a failed install is diagnosable from the console alone.

    sqlite_summary = f"{report [ 'sqlite_version' ]} (journal_mode={report [ 'journal_mode' ]})"

    print_block (
        f"Sentinel initialised in {report [ 'data_directory' ]}",
        f"  database      {report [ 'database' ]}",
        f"  sqlite        {sqlite_summary}",
        f"  sqlite-vec    {report [ 'vec_version' ]}",
        f"  schema        v{report [ 'schema_version' ]}",
        f"  cache         {report [ 'cache' ]}",
    )

    # Return data to caller.

    return EXIT_SUCCESS


#-----------------------------------------------------------------------------------------------------------------------
# Function: command_start
#
# Description:
#
#   Verify the environment and start the agent.
#
#   Phase 0 stops after verification -- there is no runtime to hand off to until the API gateway and agentic loop land
#   in Phase 1.
#
# Arguments:
#
#   configuration : The loaded configuration.
#
# Returns:
#
#   Process exit code.
#
#-----------------------------------------------------------------------------------------------------------------------

def command_start ( configuration: SentinelConfig, serve: bool = True ) -> int:

    # Verify the environment before claiming the agent is ready.

    assert_sqlite_version ()

    if not configuration.database_path.exists ():
        logger.warning (
            "No database at %s. Running initialisation first.", configuration.database_path
        )

    report = asyncio.run ( initialise ( configuration ) )

    # Build the LLM adapter BEFORE announcing readiness.
    #
    # A missing credential surfaces here as a ConfigurationError naming the command that
    # fixes it, rather than as a 401 on the first chat message. It has to happen before the
    # report below, or a first-time user is told "Sentinel is ready" and then watched it die
    # on the next line -- which is worse than either message alone.
    #
    # --no-serve deliberately skips this: it is the verification-only path, and requiring a
    # provider credential to verify the database and data tree would make the check useless
    # on exactly the fresh install that needs it most.

    adapter = build_adapter ( configuration ) if serve else None

    # Report the resolved runtime before binding a socket, so a failure to serve still
    # leaves the user with a diagnosable picture of what was configured.
    #
    # Assembled as one block rather than printed in two parts: the closing line differs
    # between the serving and verification paths, and emitting two blocks would put a
    # double blank line through the middle of what is really one report.

    lines = [
        f"{configuration.agent.name} is ready.",
        f"  data          {report [ 'data_directory' ]}",
        f"  config        {configuration.source_file or '(defaults -- no file)'}",
        f"  model         {configuration.llm.model} via {configuration.llm.provider}",
        f"  api           http://{configuration.api.host}:{configuration.api.port}",
        f"  auth          {'required' if configuration.api.auth_required else 'disabled'}",
        "",
    ]

    # --no-serve stops here. It exists because everything above is verification, and a
    # verification step that can only be exercised by starting a server that never
    # returns is a step that cannot be tested at all.

    if not serve or adapter is None:
        lines.append ( "Verification only (--no-serve). Nothing is listening." )

        print_block ( *lines )

        return EXIT_SUCCESS

    # Imported here rather than at module scope: `sentinel version` and `sentinel key`
    # must work on a machine where the web stack is broken or absent.

    from sentinel.api.gateway  import create_application
    from sentinel.core         import webui as webui_support
    from sentinel.core.process import ProcessManager

    # The manager exists before the application because the application needs its shutdown
    # hook, and the manager needs the application to serve. One of the two has to be wired
    # up after construction; the manager is the one that can be.

    manager = ProcessManager ( configuration, adapter = adapter )

    # The gateway is handed a callable rather than the heartbeat itself. The heartbeat does not exist yet -- its
    # database connection has to be opened on the runtime's event loop, which starts below -- so the control
    # endpoints resolve it per request and answer 503 until the runtime has finished coming up.

    application = create_application (
        configuration = configuration,
        adapter       = adapter,
        on_shutdown   = manager.request_shutdown,
        heartbeat     = lambda: manager.heartbeat,
        events        = lambda: manager.events,
        orchestrator  = lambda: manager.orchestrator,
    )

    manager.application = application

    # Claim the port before claiming to serve.
    #
    # uvicorn binds after its own startup messages, so a port clash prints "The API is
    # serving now" followed by a winerror 10048 -- the same false promise as announcing
    # readiness before checking the credential. Testing the bind here turns that into one
    # actionable message, and the window between this check and uvicorn's own bind is not
    # worth closing: losing that race produces uvicorn's error, which is the honest outcome.

    if is_port_in_use ( configuration.api.host, configuration.api.port ):
        raise ConfigurationError (
            f"Port {configuration.api.port} on {configuration.api.host} is already in use. "
            f"Another Sentinel may still be running -- stop it, or change api.port in agent.yaml."
        )

    if configuration.ui.webui_enabled:
        lines.append ( f"  interface     {webui_support.webui_url ( configuration )}" )

    lines.append ( "Starting. The window opens once the interface is ready." )

    print_block ( *lines )

    # Return data to caller.

    return run_application ( configuration, manager )


#-----------------------------------------------------------------------------------------------------------------------
# Function: run_application
#
# Description:
#
#   Run the agent until the user closes the window or stops it.
#
#   The threading arrangement is forced by pywebview, not chosen: its window must own the main thread (see window.py),
#   so the asyncio loop that runs every other component is put on a background thread and the main thread is left free
#   to block in the window. Every call the main thread makes into the manager from here is one of its documented
#   cross-thread entry points.
#
# Arguments:
#
#   configuration : The loaded configuration.
#   manager       : The process manager, already holding the application it should serve.
#
# Returns:
#
#   Process exit code. A failure to start every required component is a failure of the command, so it exits non-zero
#   even though the teardown itself was clean.
#
#-----------------------------------------------------------------------------------------------------------------------

def run_application ( configuration: SentinelConfig, manager: "ProcessManager" ) -> int:

    import threading

    from sentinel.core        import webui as webui_support
    from sentinel.core.window import show_window

    started = False

    #-------------------------------------------------------------------------------------------------------------------
    # Function: run_runtime
    #
    # Description:
    #
    #   Body of the background runtime thread.
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

    def run_runtime () -> None:

        nonlocal started

        started = asyncio.run ( manager.run_until_shutdown () )

    runtime = threading.Thread ( target = run_runtime, name = "sentinel-runtime" )

    runtime.start ()

    try:

        # Wait for readiness with a margin over the component startup timeout, so a slow start is reported by the
        # manager -- which knows which component was slow -- rather than by a bare timeout here that does not.

        ready = manager.wait_until_ready ( timeout = configuration.process.startup_timeout + 5.0 )

        if ready and configuration.ui.native_window and configuration.ui.webui_enabled:

            manager.notify_window_opened ()

            show_window (
                url            = webui_support.webui_url ( configuration ),
                title          = configuration.ui.window_title,
                width          = configuration.ui.window_width,
                height         = configuration.ui.window_height,
                allow_fallback = configuration.ui.browser_fallback,
                on_closed      = manager.notify_window_closed,
            )

        else:

            # No window to block in -- either the interface is off, or startup failed. Wait for the runtime instead.
            #
            # wait_until_stopped waits in slices precisely so this stays interruptible: an unbounded
            # threading.Event.wait() on Windows cannot be broken by Ctrl+C, and this is the main thread, so the agent
            # would have no way to stop from its own terminal. See ProcessManager.wait_until_stopped.

            manager.wait_until_stopped ()

    except KeyboardInterrupt:
        logger.info ( "Interrupted. Shutting down." )

        manager.request_shutdown ()

    finally:
        manager.request_shutdown ()

        runtime.join ()

    # Return data to caller.

    return EXIT_SUCCESS if started else EXIT_FAILURE


#-----------------------------------------------------------------------------------------------------------------------
# Function: command_stop
#
# Description:
#
#   Stop a running Sentinel.
#
#   Reaches the running instance through its own authenticated control endpoint rather than through a PID file and a
#   signal: Windows has no portable graceful signal, and a killed process skips the FSM's teardown -- which is what
#   flushes the event log, checkpoints the WAL, and gives Open WebUI time to close its own database.
#
# Arguments:
#
#   configuration : The loaded configuration.
#
# Returns:
#
#   Process exit code. Nothing running is reported and treated as success: `stop` asks for a state, and that state
#   already holds.
#
#-----------------------------------------------------------------------------------------------------------------------

def command_stop ( configuration: SentinelConfig ) -> int:

    import httpx

    from sentinel.security.secrets import read_secret

    url   = f"http://{configuration.api.host}:{configuration.api.port}/control/shutdown"
    token = read_secret ( SECRET_API_TOKEN )

    headers = { "Authorization": f"Bearer {token}" } if token else {}

    try:
        response = httpx.post ( url, headers = headers, timeout = 10.0 )

    except httpx.HTTPError:
        print_block ( f"Nothing is listening on {configuration.api.host}:{configuration.api.port}." )

        return EXIT_SUCCESS

    if response.status_code == 401:
        logger.error (
            "The running agent refused the stop request: the stored API token does not match the "
            "one it started with. Restart it, or re-issue the token with `sentinel key set api-token --generate`."
        )

        return EXIT_CONFIG_ERROR

    if response.status_code == 404:
        logger.error (
            "The process listening on %s:%d is not a Sentinel that can be stopped this way.",
            configuration.api.host, configuration.api.port,
        )

        return EXIT_FAILURE

    if response.status_code >= 400:
        logger.error ( "The stop request failed with HTTP %d.", response.status_code )

        return EXIT_FAILURE

    print_block ( "Sentinel is shutting down." )

    # Return data to caller.

    return EXIT_SUCCESS


#-----------------------------------------------------------------------------------------------------------------------
# Function: is_port_in_use
#
# Description:
#
#   Report whether a TCP port is already accepting connections.
#
#   Tested by connecting rather than by binding. A bind test on Windows can succeed against a port another process holds
#   without SO_EXCLUSIVEADDRUSE, which would report "free" for a port that then fails to bind for real.
#
# Arguments:
#
#   host : Interface to test.
#   port : Port to test.
#
# Returns:
#
#   True when something is listening. False when the connection is refused, or when the check itself cannot be made --
#   an inconclusive probe must not block startup, since uvicorn will report the clash properly a moment later.
#
#-----------------------------------------------------------------------------------------------------------------------

def is_port_in_use ( host: str, port: int ) -> bool:

    # A successful connection means something is already there.

    import socket

    try:
        with socket.socket ( socket.AF_INET, socket.SOCK_STREAM ) as probe:
            probe.settimeout ( 0.5 )

            return probe.connect_ex ( ( host, port ) ) == 0

    except OSError:

        # Return data to caller.

        return False


#-----------------------------------------------------------------------------------------------------------------------
# Function: build_adapter
#
# Description:
#
#   Construct the LLM adapter the configuration asks for.
#
# Arguments:
#
#   configuration : The loaded configuration.
#
# Returns:
#
#   An adapter for the configured provider.
#
#   Raises ConfigurationError for an unknown provider, or when the provider's credential is absent.
#
#-----------------------------------------------------------------------------------------------------------------------

def build_adapter ( configuration: SentinelConfig ) -> "LlmAdapter":

    # Imported lazily so the CLI's credential and version commands do not need the
    # provider SDK to be importable.

    from sentinel.llm.anthropic import AnthropicAdapter
    from sentinel.llm.ollama    import OllamaAdapter

    provider = configuration.llm.provider.lower ()

    # Passed explicitly rather than as a **kwargs dict: a dict of mixed value types erases
    # the per-argument types, and mypy --strict then cannot check the call at all.

    if provider == "anthropic":
        return AnthropicAdapter (
            model          = configuration.llm.model,
            max_tokens     = configuration.llm.max_tokens,
            timeout        = float ( configuration.llm.request_timeout ),
            retry_attempts = configuration.loop.retry_attempts,
            retry_backoff  = configuration.loop.retry_backoff,
        )

    if provider == "ollama":
        return OllamaAdapter (
            model          = configuration.llm.model,
            max_tokens     = configuration.llm.max_tokens,
            timeout        = float ( configuration.llm.request_timeout ),
            retry_attempts = configuration.loop.retry_attempts,
            retry_backoff  = configuration.loop.retry_backoff,
        )

    raise ConfigurationError (
        f"Unknown LLM provider {configuration.llm.provider!r}. Supported providers: anthropic, ollama."
    )


#-----------------------------------------------------------------------------------------------------------------------
# Function: command_key
#
# Description:
#
#   Manage credentials held in the OS keyring.
#
#   A secret is read from a prompt, never from a command-line argument: argv lands in shell history and is visible in
#   process listings, so `sentinel key set anthropic sk-ant-...` would leak the key twice over.
#
# Arguments:
#
#   arguments : Parsed command-line arguments.
#
# Returns:
#
#   Process exit code.
#
#-----------------------------------------------------------------------------------------------------------------------

def command_key ( arguments: argparse.Namespace ) -> int:

    action = arguments.key_action

    # status: report presence and source, never a value.

    if action == "status":

        rows = [
            f"  {entry [ 'name' ]:<12} {entry [ 'source' ]:<12} {entry [ 'label' ]}"
            for entry in describe_secrets ()
        ]

        print_block (
            "Credentials:",
            *rows,
            "",
            "Secrets live in the OS keyring. 'environment' means a development fallback is",
            "in use; the installed product reads from the keyring only.",
        )

        return EXIT_SUCCESS

    # set and clear both need a named secret.

    try:
        name = resolve_name ( arguments.name )
    except ValueError as error:
        logger.error ( "%s", error )

        return EXIT_CONFIG_ERROR

    if action == "clear":

        removed = delete_secret ( name )

        print_block ( f"{'Removed' if removed else 'Nothing stored for'} {arguments.name}." )

        return EXIT_SUCCESS

    # set: generate a token, or prompt for the value.

    if arguments.generate:

        if name != SECRET_API_TOKEN:
            logger.error (
                "--generate applies only to the Sentinel API token. An Anthropic key must be "
                "issued by Anthropic and pasted in."
            )

            return EXIT_CONFIG_ERROR

        value = generate_token ()

        write_secret ( name, value )

        print_block (
            f"Generated and stored a new Sentinel API token ({len ( value )} characters).",
            "Open WebUI will be configured with it automatically in Phase 2.",
            "",
            "Token (shown once -- it cannot be retrieved later):",
            f"  {value}",
        )

        return EXIT_SUCCESS

    # Prompt, twice for confirmation, without echoing.

    value = getpass.getpass ( f"Paste the value for {arguments.name} (input hidden): " ).strip ()

    if not value:
        logger.error ( "No value entered. Nothing was stored." )

        return EXIT_CONFIG_ERROR

    confirmation = getpass.getpass ( "Repeat it to confirm: " ).strip ()

    if value != confirmation:
        logger.error ( "The two entries differ. Nothing was stored." )

        return EXIT_CONFIG_ERROR

    write_secret ( name, value )

    print_block ( f"Stored {arguments.name} in the OS keyring." )

    # Return data to caller.

    return EXIT_SUCCESS


#-----------------------------------------------------------------------------------------------------------------------
# Argument parsing
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Function: build_parser
#
# Description:
#
#   Build the command-line parser.
#
# Arguments:
#
#   None.
#
# Returns:
#
#   A configured ArgumentParser.
#
#-----------------------------------------------------------------------------------------------------------------------

def build_parser () -> argparse.ArgumentParser:

    # Build the command-line parser.

    parser = argparse.ArgumentParser (
        prog        = "sentinel",
        description = "Sentinel -- a secure, self-evolving personal AI agent.",
    )

    parser.add_argument ( "--version", action = PaddedVersionAction )

    parser.add_argument (
        "--config",
        metavar = "PATH",
        default = None,
        help    = "Path to agent.yaml. Defaults to <data directory>/config/agent.yaml.",
    )

    parser.add_argument (
        "--data-dir",
        metavar = "PATH",
        default = None,
        help    = "Root of the user-data tree. Overrides the SENTINEL_DATA variable.",
    )

    parser.add_argument (
        "-v", "--verbose",
        action = "store_true",
        help   = "Emit debug-level logging.",
    )

    # Subcommands.

    subparsers = parser.add_subparsers ( dest = "command", metavar = "COMMAND" )

    subparsers.add_parser ( "version", help = "Print the version string." )
    subparsers.add_parser ( "init", help = "Create the data directory, database, and cache." )
    start_parser = subparsers.add_parser (
        "start",
        help = "Verify the environment and start the agent.",
    )

    start_parser.add_argument (
        "--no-serve",
        action = "store_true",
        help   = "Verify the environment and exit without binding the API port.",
    )

    subparsers.add_parser (
        "stop",
        help = "Ask a running Sentinel to shut down gracefully.",
    )

    # Credential management. Secrets are prompted for, never passed as arguments.

    key_parser = subparsers.add_parser (
        "key",
        help = "Manage credentials in the OS keyring.",
    )

    key_actions = key_parser.add_subparsers ( dest = "key_action", metavar = "ACTION", required = True )

    key_set = key_actions.add_parser ( "set", help = "Store a credential, prompting for the value." )

    key_set.add_argument ( "name", help = "Credential name: anthropic or api-token." )

    key_set.add_argument (
        "--generate",
        action = "store_true",
        help   = "Generate a random value instead of prompting. Applies to api-token only.",
    )

    key_clear = key_actions.add_parser ( "clear", help = "Remove a stored credential." )

    key_clear.add_argument ( "name", help = "Credential name: anthropic or api-token." )

    key_actions.add_parser ( "status", help = "Report which credentials are present." )

    # Return data to caller.

    return parser


#-----------------------------------------------------------------------------------------------------------------------
# Entry point
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Function: main
#
# Description:
#
#   Command-line entry point.
#
# Arguments:
#
#   argv : Arguments excluding the program name. sys.argv[1:] when omitted.
#
# Returns:
#
#   Process exit code.
#
#-----------------------------------------------------------------------------------------------------------------------

def main ( argv: list [ str ] | None = None ) -> int:

    # Parse arguments and bring up logging before anything can need to report.

    parser    = build_parser ()
    arguments = parser.parse_args ( argv )

    configure_logging ( verbose = arguments.verbose )

    # No subcommand is a usage question, not an error to run something by guesswork.

    if arguments.command is None:

        # argparse writes the help text itself, so the surrounding blank lines are added
        # here rather than through print_block.

        print ()

        parser.print_help ()

        print ()

        return EXIT_SUCCESS

    # version reads nothing, so it must work even when the config is broken. This is
    # the command a user reaches for while diagnosing exactly that.

    if arguments.command == "version":
        return command_version ()

    # key touches only the keyring, so it must work before any config file exists -- it is
    # how the user supplies the credential that everything else needs.

    if arguments.command == "key":
        try:
            return command_key ( arguments )
        except SentinelError as error:
            logger.error ( "%s", error )

            return EXIT_FAILURE

    # Load configuration. A bad file is reported as a distinct exit code, never a traceback.

    try:
        configuration = load_config (
            config_path    = arguments.config,
            data_directory = arguments.data_dir,
        )
    except ValidationError as error:
        logger.error ( "Invalid configuration:\n%s", error )

        return EXIT_CONFIG_ERROR
    except ConfigurationError as error:
        logger.error ( "%s", error )

        return EXIT_CONFIG_ERROR

    # Dispatch the requested command.

    try:
        if arguments.command == "init":
            return command_init ( configuration )

        if arguments.command == "start":
            return command_start ( configuration, serve = not arguments.no_serve )

        if arguments.command == "stop":
            return command_stop ( configuration )

    # A missing credential or an unknown provider is a configuration problem, and the exit
    # codes exist precisely so a wrapper script can tell that apart from "this machine
    # cannot run it". Caught before SentinelError, which is the broader case.

    except ConfigurationError as error:
        logger.error ( "%s", error )

        return EXIT_CONFIG_ERROR
    except SentinelError as error:
        logger.error ( "%s", error )

        return EXIT_FAILURE
    except KeyboardInterrupt:
        logger.info ( "Interrupted." )

        return EXIT_INTERRUPTED

    # Every subparser above is handled, but argparse will happily accept a new one
    # that nobody wired up. parser.error exits, so there is nothing to return.

    parser.error ( f"Unhandled command {arguments.command!r}." )


#-----------------------------------------------------------------------------------------------------------------------
# Program Entry Point
#-----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit ( main () )
