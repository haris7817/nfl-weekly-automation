"""Logging setup shared by every entry point.

Timestamps are UTC (implementation guide section 19). Output goes to the
console so GitHub Actions shows it live, and additionally to
``logs/nfl_weekly.log`` for local debugging.

Nothing in this module ever formats a credential: callers are responsible
for passing redacted values, and :func:`redact` is provided for the few
places where an identifier has to appear at all.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Optional

from .paths import LOG_DIR, LOG_FILE

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

_CONFIGURED = False


class _UtcFormatter(logging.Formatter):
    converter = time.gmtime


def setup_logging(
    level: str = "INFO",
    to_file: bool = True,
    force: bool = False,
) -> logging.Logger:
    """Configure the root logger once and return the project logger."""
    global _CONFIGURED

    root = logging.getLogger()
    if _CONFIGURED and not force:
        return logging.getLogger("nfl")

    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()

    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    root.setLevel(numeric_level)

    formatter = _UtcFormatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if to_file:
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError as exc:  # read-only filesystem should not kill the run
            root.warning("Could not open log file %s (%s).", LOG_FILE, exc)

    # Third-party libraries are chatty at DEBUG/INFO; keep the run log readable.
    for noisy in ("googleapiclient", "google_auth_httplib2", "urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)

    _CONFIGURED = True
    return logging.getLogger("nfl")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a namespaced child logger (``nfl.<name>``)."""
    return logging.getLogger("nfl" if not name else f"nfl.{name}")


def redact(value: Optional[str], keep: int = 4) -> str:
    """Render an identifier safe for logs, e.g. ``1a2b...<len=57>``.

    Used for Drive folder / spreadsheet IDs, which are not secrets but are
    still not worth pasting in full into a public build log. Never call this
    on a refresh token or client secret - those must not be logged at all.
    """
    if not value:
        return "<unset>"
    if len(value) <= keep:
        return "*" * len(value)
    return f"{value[:keep]}...<len={len(value)}>"
