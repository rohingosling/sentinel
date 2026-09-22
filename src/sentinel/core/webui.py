#-----------------------------------------------------------------------------------------------------------------------
# Module:  webui.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   Open WebUI launch configuration (architecture 4.2).
#
#   Two things only: the environment the managed subprocess is given, and the command line that starts it. Lifecycle --
#   starting, health, restart, teardown -- belongs to ProcessManager, which is where the FSM can see it.
#
#   Branding note: Open WebUI appends " (Open WebUI)" to any custom WEBUI_NAME, so the interface reports
#   "Sentinel (Open WebUI)". Accepted permanently (architecture Decision 5). Nothing in this module should ever try to
#   defeat that -- not through the environment, and not by importing and patching open_webui from the launcher.
#
#   The variable list lives in config/webui.env rather than in this file. That keeps one authoritative, readable
#   statement of how the interface is configured, and makes a change to the branding or the data location an edit to a
#   config file rather than to code. Four placeholders are expanded here from live configuration; the bearer token is
#   one of them, and is read from the OS keyring at launch so it never appears in a published file.
#-----------------------------------------------------------------------------------------------------------------------

import base64
import logging
import os
import sys

from pathlib import Path

from sentinel.config           import SentinelConfig
from sentinel.security.secrets import SECRET_API_TOKEN, read_secret

logger = logging.getLogger ( __name__ )

# Open WebUI's only entry point. `open-webui` on PATH is a shim around `open_webui:app`; calling that object is what
# both forms of build_command ultimately do.

CONSOLE_SCRIPT_NAME = "open-webui.exe" if os.name == "nt" else "open-webui"
CONSOLE_ENTRY_POINT = "import sys; from open_webui import app; sys.exit ( app () )"

# Where the environment template can be found, most specific first. An installed wheel carries a copy inside the
# package; a source checkout or editable install falls back to the repository's config directory. Same arrangement as
# agent.yaml, for the same reason: the repository's top-level config/ does not survive installation.

PACKAGE_ENVIRONMENT_TEMPLATE    = Path ( __file__ ).resolve ().parents [ 1 ] / "_templates" / "webui.env"
REPOSITORY_ENVIRONMENT_TEMPLATE = Path ( __file__ ).resolve ().parents [ 3 ] / "config" / "webui.env"

# Sentinel's mark, shown beside the agent's replies and in the model picker. Same two-location arrangement as the
# templates above: packaged in an installed wheel, in assets/ for a checkout.

PACKAGE_AVATAR    = Path ( __file__ ).resolve ().parents [ 1 ] / "_templates" / "sentinel-avatar.png"
REPOSITORY_AVATAR = Path ( __file__ ).resolve ().parents [ 3 ] / "assets" / "images" / "sentinel-avatar.png"

# Ceiling on the encoded avatar, in characters. Windows caps an environment block at 32,767 characters in total, and
# this variable is carried inside one; a mark that needs more than 8 KB encoded is a mark that has stopped being an
# icon. The 96 px PNG this ships with encodes to roughly 4.5 KB.

MAXIMUM_AVATAR_CHARACTERS = 8192

# Fallback environment, used only when no template file can be found. Deliberately the same values as the shipped
# template: a missing template must degrade to a working interface, not to an unbranded one with a login screen.

FALLBACK_ENVIRONMENT: dict [ str, str ] = {
    "WEBUI_NAME": "${SENTINEL_NAME}",
    "WEBUI_AUTH": "false",
    "ENABLE_SIGNUP": "false",
    "ENABLE_LOGIN_FORM": "false",
    "SHOW_ADMIN_DETAILS": "false",
    "OPENAI_API_BASE_URL": "${SENTINEL_API_URL}",
    "OPENAI_API_KEY": "${SENTINEL_API_TOKEN}",
    "ENABLE_OPENAI_API": "true",
    "DEFAULT_MODELS": "sentinel",
    "DEFAULT_MODEL_METADATA": '{"profile_image_url":"${SENTINEL_AVATAR}"}',
    "ENABLE_OLLAMA_API": "false",
    "DATA_DIR": "${SENTINEL_WEBUI_DATA}",
    "SCARF_NO_ANALYTICS": "true",
    "DO_NOT_TRACK": "true",
    "ANONYMIZED_TELEMETRY": "false",
}


#-----------------------------------------------------------------------------------------------------------------------
# Function: find_environment_template
#
# Description:
#
#   Locate config/webui.env.
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

def find_environment_template () -> Path | None:

    # Most specific location first.

    for candidate in ( PACKAGE_ENVIRONMENT_TEMPLATE, REPOSITORY_ENVIRONMENT_TEMPLATE ):
        if candidate.is_file ():
            return candidate

    # Return data to caller.

    return None


#-----------------------------------------------------------------------------------------------------------------------
# Function: parse_environment_template
#
# Description:
#
#   Parse a KEY=VALUE environment file.
#
#   Deliberately not a full dotenv implementation: no export keyword, no quoting rules, no interpolation beyond the four
#   Sentinel placeholders handled by the caller. The file is ours, and a parser that accepts more than the format we
#   write is a parser that can disagree with it.
#
# Arguments:
#
#   text : The file's contents.
#
# Returns:
#
#   Variable names mapped to their raw values, placeholders unexpanded. Blank lines and # comments are skipped; a line
#   with no '=' is skipped with a warning rather than aborting the launch.
#
#-----------------------------------------------------------------------------------------------------------------------

def parse_environment_template ( text: str ) -> dict [ str, str ]:

    variables: dict [ str, str ] = {}

    for number, raw_line in enumerate ( text.splitlines (), start = 1 ):

        line = raw_line.strip ()

        if not line or line.startswith ( "#" ):
            continue

        name, separator, value = line.partition ( "=" )

        if not separator:
            logger.warning ( "Ignoring line %d of the Open WebUI environment template: no '=' present.", number )

            continue

        variables [ name.strip () ] = value.strip ()

    # Return data to caller.

    return variables


#-----------------------------------------------------------------------------------------------------------------------
# Function: find_avatar
#
# Description:
#
#   Locate Sentinel's avatar PNG.
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

def find_avatar () -> Path | None:

    # Most specific location first.

    for candidate in ( PACKAGE_AVATAR, REPOSITORY_AVATAR ):
        if candidate.is_file ():
            return candidate

    # Return data to caller.

    return None


#-----------------------------------------------------------------------------------------------------------------------
# Function: avatar_data_uri
#
# Description:
#
#   Encode Sentinel's avatar as a data URI.
#
#   A data URI rather than a URL to Sentinel's own gateway, for one reason that decides it: an <img> tag cannot present
#   a bearer token, so serving the image would mean opening an unauthenticated path through a gateway whose whole
#   design is that there are only three of them. Embedding the bytes costs a few kilobytes of environment and adds no
#   endpoint.
#
#   PNG rather than SVG because Open WebUI rejects SVG data URIs outright -- they can carry script.
#
# Arguments:
#
#   None.
#
# Returns:
#
#   A `data:image/png;base64,...` URI, or an empty string when no avatar can be found or the encoding would be too
#   large. Empty is a supported value: Open WebUI falls back to its own default icon, which is the behaviour Sentinel
#   had before this existed.
#
#-----------------------------------------------------------------------------------------------------------------------

def avatar_data_uri () -> str:

    avatar = find_avatar ()

    if avatar is None:
        logger.warning ( "No Sentinel avatar found. Open WebUI will show its own default icon for the model." )

        return ""

    encoded = base64.b64encode ( avatar.read_bytes () ).decode ( "ascii" )

    if len ( encoded ) > MAXIMUM_AVATAR_CHARACTERS:
        logger.warning (
            "The Sentinel avatar encodes to %d characters, above the %d ceiling, and has not been applied. "
            "Re-render it smaller with tools/generate_icon.py.",
            len ( encoded ), MAXIMUM_AVATAR_CHARACTERS,
        )

        return ""

    # Return data to caller.

    return f"data:image/png;base64,{encoded}"


#-----------------------------------------------------------------------------------------------------------------------
# Function: build_environment
#
# Description:
#
#   Build the environment for the Open WebUI subprocess.
#
# Arguments:
#
#   configuration : The loaded configuration.
#   api_token     : Bearer token the interface should present to the gateway. Read from the keyring when omitted.
#   base          : Environment to inherit from. os.environ when omitted.
#
# Returns:
#
#   The parent environment with Sentinel's variables layered on top. Inheriting rather than starting empty is
#   deliberate: the child is a Python process that needs PATH, SystemRoot, and the rest of the platform's own
#   variables to start at all.
#
#   An absent token yields an empty OPENAI_API_KEY and a warning. Failing the launch outright would leave the user with
#   no interface at all when the fix -- `sentinel key set api-token --generate` -- is one command away, and the gateway
#   already refuses the unauthenticated calls that would follow.
#
#-----------------------------------------------------------------------------------------------------------------------

def build_environment ( configuration: SentinelConfig,
                        api_token: str | None = None,
                        base: dict [ str, str ] | None = None ) -> dict [ str, str ]:

    # Read the template, or fall back to the same values in code.

    template = find_environment_template ()

    if template is not None:
        variables = parse_environment_template ( template.read_text ( encoding = "utf-8" ) )
    else:
        logger.warning (
            "No Open WebUI environment template found. Falling back to built-in defaults; "
            "the interface will still be branded and unauthenticated."
        )

        variables = dict ( FALLBACK_ENVIRONMENT )

    # Resolve the four placeholders. The token is fetched last and never logged.

    token = api_token if api_token is not None else read_secret ( SECRET_API_TOKEN )

    if not token:
        logger.warning (
            "No Sentinel API token configured. Open WebUI will be unable to authenticate against "
            "the gateway. Run `sentinel key set api-token --generate`."
        )

    substitutions = {
        "${SENTINEL_API_URL}": api_base_url ( configuration ),
        "${SENTINEL_API_TOKEN}": token or "",
        "${SENTINEL_WEBUI_DATA}": str ( configuration.webui_directory ),
        "${SENTINEL_NAME}": configuration.agent.name,
        "${SENTINEL_AVATAR}": avatar_data_uri (),
    }

    environment = dict ( os.environ ) if base is None else dict ( base )

    for name, value in variables.items ():

        for placeholder, replacement in substitutions.items ():
            value = value.replace ( placeholder, replacement )

        environment [ name ] = value

    # Port and host are Sentinel's to decide, not the template's -- ProcessManager health-checks the value it set, so a
    # template that could disagree with agent.yaml would health-check the wrong port.

    environment [ "PORT" ] = str ( configuration.api.webui_port )
    environment [ "HOST" ] = configuration.api.host

    # Return data to caller.

    return environment


#-----------------------------------------------------------------------------------------------------------------------
# Function: build_command
#
# Description:
#
#   Build the command line that starts Open WebUI.
#
# Arguments:
#
#   configuration : The loaded configuration.
#   executable    : Python interpreter to run it under. sys.executable when omitted.
#
# Returns:
#
#   An argument list for the subprocess.
#
#   `-m open_webui` does NOT work and must not be reintroduced: the package ships no __main__, so the module form fails
#   with "'open_webui' is a package and cannot be directly executed". Its only entry point is the console script, which
#   maps to `open_webui:app`.
#
#   The script is located beside the running interpreter rather than searched for on PATH, so the interface always comes
#   from the same environment as the agent. When it cannot be found -- an install layout with no Scripts directory --
#   the entry point is called directly through `-c`, which needs nothing but an importable package.
#
#-----------------------------------------------------------------------------------------------------------------------

def build_command ( configuration: SentinelConfig, executable: str | None = None ) -> list [ str ]:

    interpreter = executable if executable is not None else sys.executable

    arguments = [
        "serve",
        "--host", configuration.api.host,
        "--port", str ( configuration.api.webui_port ),
    ]

    script = find_console_script ( interpreter )

    if script is not None:
        return [ str ( script ), *arguments ]

    # Return data to caller.

    return [ interpreter, "-c", CONSOLE_ENTRY_POINT, *arguments ]


#-----------------------------------------------------------------------------------------------------------------------
# Function: find_console_script
#
# Description:
#
#   Locate the open-webui console script belonging to a given interpreter.
#
# Arguments:
#
#   interpreter : Path of the Python interpreter the interface should run under.
#
# Returns:
#
#   The script path, or None when this environment has no console script -- which is the normal case for the embedded
#   Python distribution the installer ships.
#
#-----------------------------------------------------------------------------------------------------------------------

def find_console_script ( interpreter: str ) -> Path | None:

    root = Path ( interpreter ).resolve ().parent

    # A venv puts scripts beside python.exe on Windows and in bin/ elsewhere; a base install puts them in a Scripts
    # subdirectory. Both are checked rather than guessed at from the platform.

    for candidate in ( root / CONSOLE_SCRIPT_NAME, root / "Scripts" / CONSOLE_SCRIPT_NAME ):

        if candidate.is_file ():
            return candidate

    # Return data to caller.

    return None


#-----------------------------------------------------------------------------------------------------------------------
# Function: api_base_url
#
# Description:
#
#   The gateway URL Open WebUI should call.
#
# Arguments:
#
#   configuration : The loaded configuration.
#
# Returns:
#
#   The OpenAI-compatible base URL, including the /v1 prefix Open WebUI appends its paths to.
#
#-----------------------------------------------------------------------------------------------------------------------

def api_base_url ( configuration: SentinelConfig ) -> str:

    # Return data to caller.

    return f"http://{configuration.api.host}:{configuration.api.port}/v1"


#-----------------------------------------------------------------------------------------------------------------------
# Function: webui_url
#
# Description:
#
#   The URL the native window should display.
#
# Arguments:
#
#   configuration : The loaded configuration.
#
# Returns:
#
#   The Open WebUI root URL.
#
#-----------------------------------------------------------------------------------------------------------------------

def webui_url ( configuration: SentinelConfig ) -> str:

    # Return data to caller.

    return f"http://{configuration.api.host}:{configuration.api.webui_port}/"
