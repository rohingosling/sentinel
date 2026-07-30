#-----------------------------------------------------------------------------------------------------------------------
# Module:  tray.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   System tray icon -- Phase 2 stub (architecture 4.3.3).
#
#   The tray is an optional component: its absence keeps the system in RUNNING rather than degrading it. That property
#   is what makes a stub honest here. ProcessManager already treats the tray as a component with a lifecycle, so
#   Phase 9 replaces the body of start() and changes nothing else.
#
#   Deliberately not implemented yet, because the tray only becomes useful alongside the behaviour it enables: in
#   Phase 9, closing the window stops shutting the agent down and starts minimising it here. Shipping an icon in
#   Phase 2 that can only quit -- the one thing closing the window already does -- would be a menu with no purpose.
#-----------------------------------------------------------------------------------------------------------------------

import logging

from collections.abc import Callable

logger = logging.getLogger ( __name__ )


#-----------------------------------------------------------------------------------------------------------------------
# Class: TrayIcon
#
# Description:
#
#   The system tray icon and its menu.
#
#   Phase 2 stub: start() reports that no icon was shown, which ProcessManager records as an unavailable optional
#   component. Phase 9 supplies the pystray implementation behind the same three methods.
#
# Attributes:
#
#   title       : Tooltip shown on hover.
#   on_open     : Called when the user asks for the window. Wired in Phase 9.
#   on_quit     : Called when the user quits from the menu. Wired in Phase 9.
#   is_visible  : Whether an icon is currently shown.
#-----------------------------------------------------------------------------------------------------------------------

class TrayIcon:

    #-------------------------------------------------------------------------------------------------------------------
    # Function: __init__
    #
    # Description:
    #
    #   Construct the tray icon.
    #
    # Arguments:
    #
    #   title   : Tooltip shown on hover.
    #   on_open : Called when the user asks for the window.
    #   on_quit : Called when the user quits from the menu.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def __init__ ( self,
                   title: str                            = "Sentinel",
                   on_open: Callable [ [], None ] | None = None,
                   on_quit: Callable [ [], None ] | None = None ) -> None:

        self.title      = title
        self.on_open    = on_open
        self.on_quit    = on_quit
        self.is_visible = False

    #-------------------------------------------------------------------------------------------------------------------
    # Function: start
    #
    # Description:
    #
    #   Show the tray icon.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   True when an icon is now visible. Always False in Phase 2 -- the caller must treat a missing tray as normal,
    #   which is exactly the code path Phase 9 will need to keep working on a machine with no notification area.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def start ( self ) -> bool:

        logger.debug ( "System tray is not implemented until Phase 9; continuing without an icon." )

        # Return data to caller.

        return False

    #-------------------------------------------------------------------------------------------------------------------
    # Function: stop
    #
    # Description:
    #
    #   Remove the tray icon.
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

    def stop ( self ) -> None:

        self.is_visible = False
