#-----------------------------------------------------------------------------------------------------------------------
# Package: sentinel.logging
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only package; not executable directly.
#
# Description:
#
#   Structured event logging (architecture 3.2.10, 4.4).
#
#   schemas.py holds the taxonomy and the entry shape, logger.py the reader and writer, exporters.py the destinations
#   besides the database.
#
#   The package deliberately shares a name with the standard library's `logging`. That is safe -- Python 3 imports are
#   absolute, so `import logging` inside these modules still reaches the stdlib -- and it is what architecture 5 names.
#   The two are used side by side throughout: the stdlib logger reports to the operator, and this one records what the
#   agent did.
#-----------------------------------------------------------------------------------------------------------------------

from sentinel.logging.exporters import (
    EventExporter,
    PrometheusExporter,
    RotatingFileExporter,
    StdoutExporter,
)
from sentinel.logging.logger import (
    EventLogger,
    build_exporters,
    correlation_scope,
    current_correlation_id,
    new_correlation_id,
    open_event_logger,
)
from sentinel.logging.schemas import (
    CATEGORIES,
    EVENT_TAXONOMY,
    EventMetadata,
    LogEvent,
    build_event,
)

__all__ = [
    "CATEGORIES",
    "EVENT_TAXONOMY",
    "EventExporter",
    "EventLogger",
    "EventMetadata",
    "LogEvent",
    "PrometheusExporter",
    "RotatingFileExporter",
    "StdoutExporter",
    "build_event",
    "build_exporters",
    "correlation_scope",
    "current_correlation_id",
    "new_correlation_id",
    "open_event_logger",
]
