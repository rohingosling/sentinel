#-----------------------------------------------------------------------------------------------------------------------
# Package: sentinel.security
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only package; not executable directly.
#
# Description:
#
#   Security subsystem.
#
#   Phase 1 lands only the credential store: secrets have to go somewhere the moment the first API key exists, and the
#   OS keyring is that somewhere. The sandbox, permission engine, and tool-use hooks arrive in Phase 6 alongside it.
#-----------------------------------------------------------------------------------------------------------------------
