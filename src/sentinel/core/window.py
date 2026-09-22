#-----------------------------------------------------------------------------------------------------------------------
# Module:  window.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   Native desktop window, with a browser fallback (architecture 4.3.6 step 4).
#
#   The product's claim is that the user sees one application, not a browser pointed at a local port. pywebview delivers
#   that by hosting Edge WebView2 in an ordinary OS window: no address bar, no tabs, no navigation buttons.
#
#   Two properties of this module drive its shape:
#
#     * pywebview must own the main thread. Its event loop is the platform's, not asyncio's, and on Windows a WebView2
#       window created off the main thread will not pump messages. Everything else therefore runs on a background
#       thread and this call blocks until the window closes -- see ProcessManager.
#
#     * pywebview is imported inside the function, not at module scope. A machine without WebView2, or an install
#       without the `ui` extra, must fall back to a browser rather than fail to start; an import at module scope would
#       make that failure unreachable by turning it into an ImportError at startup.
#-----------------------------------------------------------------------------------------------------------------------

import logging
import os
import shutil
import subprocess

from collections.abc import Callable
from enum            import StrEnum
from pathlib         import Path

logger = logging.getLogger ( __name__ )

# Chromium-family browsers, in preference order. Edge first: it is present on every supported Windows install and uses
# the same rendering engine as the native window, so the fallback looks like the real thing.

BROWSER_EXECUTABLES: tuple [ str, ... ] = ( "msedge", "chrome", "chromium", "brave" )

# Well-known install locations, searched when the executable is not on PATH -- which, for browsers on Windows, is the
# normal case rather than the exception.

WINDOWS_BROWSER_PATHS: tuple [ str, ... ] = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
)


#-----------------------------------------------------------------------------------------------------------------------
# Class: WindowMode
#
# Description:
#
#   How the interface was ultimately shown.
#
#   Reported rather than inferred, because "no browser process was launched" is a test criterion in its own right
#   (T2.10) and the caller should not have to guess which path ran.
#-----------------------------------------------------------------------------------------------------------------------

class WindowMode ( StrEnum ):

    NATIVE      = "native"
    BROWSER     = "browser"
    UNAVAILABLE = "unavailable"


#-----------------------------------------------------------------------------------------------------------------------
# Function: show_window
#
# Description:
#
#   Show the interface and block until the user closes it.
#
#   Must be called on the main thread. Returns when the window is closed, which the caller treats as a WindowClosed
#   event rather than as a failure.
#
# Arguments:
#
#   url            : Address to display.
#   title          : Window title. The user sees this in the title bar and the taskbar, so it is the application name.
#   width          : Initial window width, in pixels.
#   height         : Initial window height, in pixels.
#   allow_fallback : Launch a browser in app mode when the native window is unavailable.
#   on_closed      : Called once, after the window closes. Invoked for both the native and browser paths.
#
# Returns:
#
#   Which path actually ran. UNAVAILABLE means neither a native window nor a browser could be shown -- the agent keeps
#   running headless, since the window is an optional component.
#
#-----------------------------------------------------------------------------------------------------------------------

def show_window ( url: str,
                  title: str           = "Sentinel",
                  width: int           = 1200,
                  height: int          = 800,
                  allow_fallback: bool = True,
                  on_closed: Callable [ [], None ] | None = None ) -> WindowMode:

    # Try the native window first. Anything that goes wrong here is a fallback condition, not a failure: a missing
    # pywebview raises ImportError, a missing WebView2 runtime raises at start(), and a headless session raises
    # something platform-specific that is not worth enumerating.

    mode = WindowMode.UNAVAILABLE

    try:
        show_native_window ( url = url, title = title, width = width, height = height )

        mode = WindowMode.NATIVE

    except ImportError as error:
        logger.warning ( "pywebview is not installed (%s). Falling back to a browser window.", error )

    except Exception as error:
        logger.warning ( "The native window could not be shown (%s). Falling back to a browser window.", error )

    # Fallback: a Chromium browser in app mode, which at least has no address bar or tabs.

    if mode is WindowMode.UNAVAILABLE and allow_fallback:

        process = launch_browser_app_mode ( url )

        if process is not None:

            mode = WindowMode.BROWSER

            # Wait for it, so this call blocks until the window closes on both paths. A fallback that returned
            # immediately would look to the caller exactly like a window the user had just closed.

            try:
                process.wait ()
            except KeyboardInterrupt:
                process.terminate ()

    if mode is WindowMode.UNAVAILABLE:
        logger.warning (
            "No window could be shown. Sentinel is still running; open %s in a browser to reach it.", url
        )

    if on_closed is not None:
        on_closed ()

    # Return data to caller.

    return mode


#-----------------------------------------------------------------------------------------------------------------------
# Function: show_native_window
#
# Description:
#
#   Create and run a chromeless pywebview window.
#
# Arguments:
#
#   url    : Address to display.
#   title  : Window title.
#   width  : Initial window width, in pixels.
#   height : Initial window height, in pixels.
#
# Returns:
#
#   None. Blocks until the window closes.
#
#   Raises ImportError when pywebview is absent, and whatever pywebview raises when no renderer is available.
#
#-----------------------------------------------------------------------------------------------------------------------

def show_native_window ( url: str, title: str, width: int, height: int ) -> None:

    # Imported here so its absence is a fallback condition rather than a startup failure.

    import webview

    logger.info ( "Opening the native window on %s.", url )

    webview.create_window ( title, url, width = width, height = height )

    webview.start ()


#-----------------------------------------------------------------------------------------------------------------------
# Function: launch_browser_app_mode
#
# Description:
#
#   Launch a Chromium browser in app mode as a fallback window.
#
#   `--app=` suppresses the address bar, tabs, and navigation buttons. It is not the native window -- the browser's own
#   window frame and process remain -- but it is the closest available approximation on a machine with no WebView2.
#
# Arguments:
#
#   url        : Address to display.
#   executable : Browser to run. Discovered when omitted.
#
# Returns:
#
#   The launched process, or None when no browser could be found.
#
#-----------------------------------------------------------------------------------------------------------------------

def launch_browser_app_mode ( url: str, executable: str | None = None ) -> subprocess.Popen [ bytes ] | None:

    browser = executable if executable is not None else find_browser ()

    if browser is None:
        logger.error ( "No Chromium-family browser found for the fallback window." )

        return None

    logger.info ( "Opening %s in %s app mode.", url, Path ( browser ).stem )

    try:

        # The argument list is fixed and the URL is Sentinel's own loopback address, never user input, so there is no
        # injection surface here. shell=False throughout, as the project's rules require.

        return subprocess.Popen (   # noqa: S603
            [ browser, f"--app={url}" ],
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
        )

    except OSError as error:
        logger.error ( "Could not launch %s: %s", browser, error )

        # Return data to caller.

        return None


#-----------------------------------------------------------------------------------------------------------------------
# Function: find_browser
#
# Description:
#
#   Locate a Chromium-family browser.
#
# Arguments:
#
#   None.
#
# Returns:
#
#   An absolute path to the first browser found, or None. PATH is searched first, then the well-known Windows install
#   locations -- browsers are rarely on PATH on Windows, so the second search is the one that usually succeeds.
#
#-----------------------------------------------------------------------------------------------------------------------

def find_browser () -> str | None:

    for name in BROWSER_EXECUTABLES:

        located = shutil.which ( name )

        if located:
            return located

    if os.name == "nt":

        for candidate in WINDOWS_BROWSER_PATHS:

            if Path ( candidate ).is_file ():
                return candidate

    # Return data to caller.

    return None
