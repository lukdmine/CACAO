"""Timestamped logging helper and a tee-writer for mirroring stdout to a log file."""

from datetime import datetime

LEVEL_PREFIXES = {
    "INFO": "ℹ",
    "ERROR": "✗",
    "SUCCESS": "✓",
    "WARN": "⚠",
}


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
