"""Timestamped logging helper and a tee-writer for mirroring stdout to a log file."""

import sys
from datetime import datetime

LEVEL_PREFIXES = {
    "INFO": "ℹ",
    "ERROR": "✗",
    "SUCCESS": "✓",
    "WARN": "⚠",
}


def _force_utf8_streams() -> None:
    """Make stdout/stderr able to carry the characters this pipeline actually emits.

    Every level prefix above is non-ASCII and timings are reported in µs, so a
    locale-derived ASCII stream raises UnicodeEncodeError on the first log line and
    takes the process with it. ``errors="replace"``: a mangled character beats a
    crashed run.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            encoding = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
            if encoding != "utf8":
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            # Not a reconfigurable stream (already wrapped, or redirected). Logging
            # must never be the thing that breaks, so this stays best-effort.
            pass


# Runs at import. utils.log is imported by effectively every module, so this is the
# earliest single point that covers the CLI, the API server, and the workers alike.
_force_utf8_streams()


def log(msg: str, level: str = "INFO"):
    """Print a timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = LEVEL_PREFIXES.get(level, "•")
    print(f"[{timestamp}] {prefix} {msg}", flush=True)


class TeeWriter:
    """Write to both an original stream and a log file."""

    def __init__(self, original, log_file):
        self._original = original
        self._log_file = log_file

    def write(self, data: str) -> int:
        self._original.write(data)
        self._log_file.write(data)
        self._log_file.flush()
        return len(data)

    def flush(self):
        self._original.flush()
        self._log_file.flush()

    def fileno(self):
        return self._original.fileno()
