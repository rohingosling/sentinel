#-----------------------------------------------------------------------------------------------------------------------
# Module:  exporters.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   Where events go besides the database (architecture 4.4).
#
#   Three destinations, each answering a different question:
#
#     * RotatingFileExporter   what happened, readable without SQLite. One JSON object per line, so `findstr`, `jq`, and
#                              a text editor all work on it, and a partially written last line costs one event rather
#                              than the file.
#     * StdoutExporter         what is happening, while developing. Off by default -- it would double every console line
#                              in normal use.
#     * PrometheusExporter     how much is happening. Counters only, rendered in the text exposition format.
#
#   Exporters are synchronous. An async interface would buy nothing -- these are an appended line and a dictionary
#   increment -- and would force every emitting site, including the heartbeat tick whose contract is that it never
#   raises, to await something that can fail. The stdlib logging module makes the same trade for the same reason.
#
#   No exporter may raise. An event log that can take down the thing it observes is worse than one that quietly misses
#   an entry, so a failure is reported through the ordinary logger once and the caller carries on.
#
#   Prometheus is implemented in about forty lines rather than pulled in as a dependency. `prometheus_client` would add
#   a package to the bundleability audit (architecture 6.8) to provide a counter dictionary and a text format that is
#   two lines per metric -- and this exporter is optional and off by default, so the dependency would be paid for by
#   every install and used by almost none.
#-----------------------------------------------------------------------------------------------------------------------

import logging
import shutil
import sys

from collections import Counter
from datetime    import UTC, datetime, timedelta
from pathlib     import Path
from typing      import Protocol, TextIO, runtime_checkable

from sentinel.logging.schemas import CATEGORIES, LogEvent

logger = logging.getLogger ( __name__ )

# Default rotation geometry. Ten megabytes is roughly seventy thousand events -- large enough that a busy day does not
# produce a directory full of stubs, small enough that a rotated file still opens in an editor.

DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024

# The live file's name, and the stamp appended to a rotated one. Colons are illegal in Windows filenames, so the
# rotation stamp is the compact ISO 8601 basic format rather than the extended one used inside the events themselves.

LIVE_FILE_NAME        = "events.jsonl"
ROTATION_STAMP_FORMAT = "%Y%m%dT%H%M%S%f"

# How many same-stamp rotations to disambiguate before giving up.
#
# The stamp carries microseconds, which looks unique and is not: datetime.now() on Windows advances in ticks of about
# half a millisecond, so a burst of rotations yields the same text many times over -- measured at 2000 consecutive
# calls producing 8 distinct values. A colliding destination is not cosmetic. On Windows the rename fails outright and
# the live file goes on growing past its ceiling; on POSIX os.rename REPLACES the destination silently, destroying the
# file rotated a moment earlier. A taken name therefore gets a sequence suffix rather than a retry or an overwrite.

ROTATION_SEQUENCE_LIMIT = 1000

# Metric names for the Prometheus rendering.

METRIC_EVENTS_TOTAL = "sentinel_events_total"


#-----------------------------------------------------------------------------------------------------------------------
# Interface
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Class: EventExporter
#
# Description:
#
#   Something an event can be sent to besides the database.
#
#   A Protocol rather than a base class, so a test double or a future channel exporter satisfies it without importing
#   anything from here.
#-----------------------------------------------------------------------------------------------------------------------

@runtime_checkable
class EventExporter ( Protocol ):

    #-------------------------------------------------------------------------------------------------------------------
    # Function: export
    #
    # Description:
    #
    #   Send one event.
    #
    # Arguments:
    #
    #   event : The event to send.
    #
    # Returns:
    #
    #   None. Never raises -- see the module header.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def export ( self, event: LogEvent ) -> None: ...

    #-------------------------------------------------------------------------------------------------------------------
    # Function: close
    #
    # Description:
    #
    #   Release whatever the exporter holds open.
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

    def close ( self ) -> None: ...


#-----------------------------------------------------------------------------------------------------------------------
# File export
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Class: RotatingFileExporter
#
# Description:
#
#   Appends events to a JSON Lines file, rotating it by size and expiring old ones by age.
#
#   Rotation is checked before each write rather than after, using a running byte count rather than a stat() call. A
#   stat per event would put a filesystem round trip on the path of every logged event, and the count is exact because
#   this exporter is the only writer of the live file.
#
#   Not built on logging.handlers.RotatingFileHandler, which numbers its backups (events.1, events.2) and renames every
#   one of them on each rotation. The test plan asks for a timestamped name, and a stamped file is what makes "what was
#   the agent doing last Tuesday" answerable from the directory listing alone.
#
# Attributes:
#
#   directory     : Where the files live.
#   max_bytes     : Size at which the live file is rotated. Zero disables rotation.
#   retention     : Days a rotated file is kept. Zero keeps them for ever.
#   path          : The live file.
#   rotations     : Number of rotations performed, for tests and diagnostics.
#-----------------------------------------------------------------------------------------------------------------------

class RotatingFileExporter:

    #-------------------------------------------------------------------------------------------------------------------
    # Function: __init__
    #
    # Description:
    #
    #   Open the live file for appending, creating the directory if needed.
    #
    # Arguments:
    #
    #   directory : Where the files live.
    #   max_bytes : Size at which the live file rotates. Zero disables rotation.
    #   retention : Days a rotated file is kept. Zero keeps them for ever.
    #   file_name : Name of the live file.
    #
    # Returns:
    #
    #   None. A directory that cannot be created leaves the exporter disabled rather than raising -- the agent must
    #   still run on a machine whose log directory is read-only.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def __init__ ( self,
                   directory: str | Path,
                   max_bytes: int = DEFAULT_MAX_FILE_SIZE,
                   retention: int = 30,
                   file_name: str = LIVE_FILE_NAME ) -> None:

        self.directory = Path ( directory )
        self.max_bytes = max ( 0, max_bytes )
        self.retention = max ( 0, retention )
        self.path      = self.directory / file_name
        self.rotations = 0

        self._handle: TextIO | None = None
        self._written = 0

        self._open ()

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _open
    #
    # Description:
    #
    #   Open the live file for appending and take its current size.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   None. A failure leaves the handle None, which disables the exporter for the rest of its life; every later export
    #   is a no-op rather than a repeated attempt to open a path that has already refused once.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def _open ( self ) -> None:

        try:
            self.directory.mkdir ( parents = True, exist_ok = True )

            # The existing size matters: appending to a file already at the limit must rotate on the next write, not
            # grow unbounded until the process happens to restart.

            self._written = self.path.stat ().st_size if self.path.exists () else 0
            self._handle  = self.path.open ( "a", encoding = "utf-8" )

        except OSError as error:
            logger.warning (
                "Cannot open the event log file %s: %s. Events will still reach the database; "
                "file export is disabled for this run.",
                self.path, error,
            )

            self._handle = None

    #-------------------------------------------------------------------------------------------------------------------
    # Function: export
    #
    # Description:
    #
    #   Append one event as a line of JSON, rotating first if the file is full.
    #
    # Arguments:
    #
    #   event : The event to write.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def export ( self, event: LogEvent ) -> None:

        if self._handle is None:
            return

        line = f"{event.as_json ()}\n"
        size = len ( line.encode ( "utf-8" ) )

        # Rotate before writing, so the limit is a ceiling on the file rather than a threshold it is allowed to cross.
        # An empty file is never rotated, or an event larger than max_bytes would rotate for ever and never be written.

        if self.max_bytes and self._written and self._written + size > self.max_bytes:
            self.rotate ()

        try:
            self._handle.write ( line )
            self._handle.flush ()

            self._written += size

        except OSError as error:
            logger.warning ( "Could not write to the event log file %s: %s", self.path, error )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: rotate
    #
    # Description:
    #
    #   Copy the live file aside under a timestamped name, then truncate it in place.
    #
    #   Copy-and-truncate rather than rename, which is the obvious implementation and the wrong one on Windows. A
    #   rename fails outright while ANY other process holds the file open -- an editor, a `Get-Content -Wait`, a
    #   backup agent, a virus scanner mid-scan -- and the log would then grow without bound for as long as that
    #   handle lived. Measured: a plain read handle held by another process blocked every rotation, and the file
    #   reached twelve times its ceiling. Truncation needs only the handle this exporter already owns, so it cannot
    #   be refused by a reader.
    #
    #   The cost is copying the file's bytes instead of moving a directory entry -- O(size) rather than O(1), paid
    #   once per max_bytes. At the shipped 10 MB that is a 10 MB copy roughly once per seventy thousand events, which
    #   is not a rate worth optimising against correctness.
    #
    #   A reader tailing the live file sees it reset to empty rather than vanish, which is the friendlier of the two
    #   behaviours anyway.
    #
    # Arguments:
    #
    #   moment : Instant to stamp the rotated file with. Now when omitted.
    #
    # Returns:
    #
    #   The rotated file's path, or None when nothing was rotated.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def rotate ( self, moment: datetime | None = None ) -> Path | None:

        if self._handle is None:
            return None

        rotated = self.rotated_path ( moment if moment is not None else datetime.now ( UTC ) )

        if rotated is None:
            logger.warning (
                "Could not find a free name to rotate the event log file %s into after %d attempts. "
                "The live file will keep growing.",
                self.path, ROTATION_SEQUENCE_LIMIT,
            )

            return None

        try:

            # Flush first: the copy reads through the filesystem, so anything still sitting in this handle's buffer
            # would be left out of the rotated file and then destroyed by the truncate below.

            self._handle.flush ()

            shutil.copyfile ( self.path, rotated )

            # Truncate through the handle already held, never by reopening. Reopening would leave a window in which
            # another process could take the file, which is the failure this whole approach exists to avoid.

            self._handle.truncate ( 0 )
            self._handle.seek ( 0 )
            self._handle.flush ()

        except OSError as error:
            logger.warning ( "Could not rotate the event log file %s: %s", self.path, error )

            return None

        self.rotations += 1
        self._written = 0

        self.expire ()

        logger.info ( "Rotated the event log to %s.", rotated.name )

        # Return data to caller.

        return rotated

    #-------------------------------------------------------------------------------------------------------------------
    # Function: rotated_path
    #
    # Description:
    #
    #   Choose a free filename for the file being rotated out.
    #
    # Arguments:
    #
    #   moment : Instant to stamp the name with.
    #
    # Returns:
    #
    #   A path that does not yet exist, or None when ROTATION_SEQUENCE_LIMIT names were all taken. The bare stamp is
    #   preferred; a sequence suffix is added only when it collides, which keeps the common case readable and the
    #   burst case correct. See ROTATION_SEQUENCE_LIMIT for why collisions are routine rather than theoretical.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def rotated_path ( self, moment: datetime ) -> Path | None:

        stamp     = moment.strftime ( ROTATION_STAMP_FORMAT )
        candidate = self.path.with_name ( f"{self.path.stem}-{stamp}{self.path.suffix}" )

        if not candidate.exists ():
            return candidate

        for sequence in range ( 1, ROTATION_SEQUENCE_LIMIT ):

            candidate = self.path.with_name ( f"{self.path.stem}-{stamp}-{sequence:03d}{self.path.suffix}" )

            if not candidate.exists ():
                return candidate

        # Return data to caller.

        return None

    #-------------------------------------------------------------------------------------------------------------------
    # Function: expire
    #
    # Description:
    #
    #   Delete rotated files older than the retention window.
    #
    #   Only rotated files are considered. The live file is never a candidate however old it is, because its age is the
    #   age of the oldest event still in it, not of the newest.
    #
    # Arguments:
    #
    #   moment : Instant to measure age from. Now when omitted.
    #
    # Returns:
    #
    #   The number of files deleted. Zero when retention is disabled.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def expire ( self, moment: datetime | None = None ) -> int:

        if not self.retention:
            return 0

        now     = moment if moment is not None else datetime.now ( UTC )
        cut_off = now - timedelta ( days = self.retention )
        removed = 0

        try:
            candidates = sorted ( self.directory.glob ( f"{self.path.stem}-*{self.path.suffix}" ) )
        except OSError as error:
            logger.warning ( "Could not list the event log directory %s: %s", self.directory, error )

            return 0

        for candidate in candidates:

            try:
                modified = datetime.fromtimestamp ( candidate.stat ().st_mtime, UTC )

                if modified < cut_off:
                    candidate.unlink ()

                    removed += 1

            except OSError as error:
                logger.warning ( "Could not expire the rotated event log %s: %s", candidate, error )

        if removed:
            logger.info ( "Removed %d rotated event log(s) older than %d day(s).", removed, self.retention )

        # Return data to caller.

        return removed

    #-------------------------------------------------------------------------------------------------------------------
    # Function: close
    #
    # Description:
    #
    #   Close the live file.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   None. Closing twice is harmless, because shutdown paths run without knowing whether the exporter was ever
    #   successfully opened.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def close ( self ) -> None:

        if self._handle is None:
            return

        try:
            self._handle.close ()
        except OSError as error:
            logger.warning ( "The event log file %s did not close cleanly: %s", self.path, error )

        self._handle = None


#-----------------------------------------------------------------------------------------------------------------------
# Console export
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Class: StdoutExporter
#
# Description:
#
#   Writes each event to a stream, for development.
#
#   Off by default. With it on, every event appears twice in a terminal -- once here and once through whatever ordinary
#   log line the emitting code also writes -- which is useful while building a subsystem and noise everywhere else.
#
# Attributes:
#
#   stream : Where events are written.
#   json_output : Emit the full JSON object rather than a one-line summary.
#-----------------------------------------------------------------------------------------------------------------------

class StdoutExporter:

    #-------------------------------------------------------------------------------------------------------------------
    # Function: __init__
    #
    # Description:
    #
    #   Construct the exporter.
    #
    # Arguments:
    #
    #   stream      : Where events are written. sys.stdout when omitted, resolved per write so a test that replaces the
    #                 stream after construction still sees its own.
    #   json_output : Emit the full JSON object rather than a readable one-line summary.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def __init__ ( self, stream: TextIO | None = None, json_output: bool = True ) -> None:

        self.stream      = stream
        self.json_output = json_output

    #-------------------------------------------------------------------------------------------------------------------
    # Function: export
    #
    # Description:
    #
    #   Write one event.
    #
    # Arguments:
    #
    #   event : The event to write.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def export ( self, event: LogEvent ) -> None:

        target = self.stream if self.stream is not None else sys.stdout

        if self.json_output:
            line = event.as_json ()
        else:
            correlation = event.correlation_id [ : 8 ] if event.correlation_id else "--------"
            line        = f"{event.timestamp} {correlation} {event.event:<20} {event.data}"

        try:
            target.write ( f"{line}\n" )
            target.flush ()

        except ( OSError, ValueError ) as error:

            # ValueError is the closed-stream case, which a test that swaps streams around will reach before any real
            # deployment does.

            logger.warning ( "Could not write an event to the console: %s", error )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: close
    #
    # Description:
    #
    #   Release the stream.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   None. The stream is not closed -- it belongs to the caller, and closing sys.stdout would take the console with
    #   it.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def close ( self ) -> None:

        pass


#-----------------------------------------------------------------------------------------------------------------------
# Metrics export
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Class: PrometheusExporter
#
# Description:
#
#   Counts events by category and event name, and renders them in the Prometheus text exposition format.
#
#   Counters only. A histogram would need bucket configuration and a duration to observe, and no event carries one --
#   durations live inside `data`, where they are queryable through the log API instead.
#
#   Every one of the eight categories is initialised to zero at construction, so a category that has produced nothing
#   still appears in the output. A missing series and a zero series look identical to an alerting rule right up to the
#   moment the alert should have fired.
#
# Attributes:
#
#   counts : Events seen, keyed by (category, event).
#-----------------------------------------------------------------------------------------------------------------------

class PrometheusExporter:

    #-------------------------------------------------------------------------------------------------------------------
    # Function: __init__
    #
    # Description:
    #
    #   Construct the exporter with every category at zero.
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

    def __init__ ( self ) -> None:

        self.counts: Counter [ tuple [ str, str ] ] = Counter ()

        for category in sorted ( CATEGORIES ):
            self.counts [ ( category, "" ) ] = 0

    #-------------------------------------------------------------------------------------------------------------------
    # Function: export
    #
    # Description:
    #
    #   Count one event.
    #
    # Arguments:
    #
    #   event : The event to count.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def export ( self, event: LogEvent ) -> None:

        self.counts [ ( event.category, event.event ) ] += 1

    #-------------------------------------------------------------------------------------------------------------------
    # Function: total_for_category
    #
    # Description:
    #
    #   Sum every event counted in one category.
    #
    # Arguments:
    #
    #   category : The category to total.
    #
    # Returns:
    #
    #   The number of events counted in that category.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def total_for_category ( self, category: str ) -> int:

        # Return data to caller.

        return sum ( count for ( seen, _ ), count in self.counts.items () if seen == category )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: render
    #
    # Description:
    #
    #   Render the counters in the Prometheus text exposition format.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   The exposition text, ending in a newline as the format requires. Series are sorted so two scrapes of an
    #   unchanged process produce byte-identical output, which makes the endpoint testable.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def render ( self ) -> str:

        lines = [
            f"# HELP {METRIC_EVENTS_TOTAL} Events written to the Sentinel event log.",
            f"# TYPE {METRIC_EVENTS_TOTAL} counter",
        ]

        # The zero-initialised placeholder rows carry an empty event name; they exist to make a silent category
        # visible, and their total is the category's own.

        for category in sorted ( CATEGORIES ):
            lines.append (
                f'{METRIC_EVENTS_TOTAL}{{category="{category}"}} {self.total_for_category ( category )}'
            )

        for ( category, event ), count in sorted ( self.counts.items () ):

            if not event:
                continue

            lines.append (
                f'{METRIC_EVENTS_TOTAL}{{category="{category}",event="{event}"}} {count}'
            )

        # Return data to caller.

        return "\n".join ( lines ) + "\n"

    #-------------------------------------------------------------------------------------------------------------------
    # Function: close
    #
    # Description:
    #
    #   Release whatever the exporter holds.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   None. The counters are kept rather than cleared: a scrape arriving during shutdown should see the run's totals,
    #   not zeros.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def close ( self ) -> None:

        pass
