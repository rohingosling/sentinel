#-----------------------------------------------------------------------------------------------------------------------
# Module:  __main__.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
#
# Description:
#
#   Module execution entry point. Delegates to sentinel.main so that `python -m sentinel <command>` and the installed
#   `sentinel` console script share one code path.
#
# Usage:
#
#   python -m sentinel version
#   python -m sentinel init
#   python -m sentinel start
#-----------------------------------------------------------------------------------------------------------------------

import sys

from sentinel.main import main


#-----------------------------------------------------------------------------------------------------------------------
# Program Entry Point
#-----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit ( main () )
