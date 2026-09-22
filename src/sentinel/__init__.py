#-----------------------------------------------------------------------------------------------------------------------
# Module:  __init__.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   Package root for Sentinel, a secure, self-evolving personal AI agent.
#
#   Sentinel is an autonomous agent that runs as a native Windows desktop application. It wraps the Claude API behind an
#   OpenAI-compatible API, reasons over a three-tier memory system, executes sandboxed skills, and communicates across
#   several channels.
#
#   The package is importable and runnable:
#
#     python -m sentinel start
#-----------------------------------------------------------------------------------------------------------------------

# Package metadata. __version__ is the single source of truth for the version string reported by `sentinel version`.

__version__ = "0.1.0"
__author__  = "Rohin Gosling"

__all__ = [ "__version__", "__author__" ]
