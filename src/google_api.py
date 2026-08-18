"""Shared execution helper for Google API requests.

Google surfaces rate limiting and backend hiccups as retryable HTTP status
codes; everything else (bad permission, missing file, invalid argument) is
deterministic and must fail immediately rather than burn three attempts.
"""

from __future__ import annotations

from typing import Any

from .logging_utils import get_logger
from .retry import with_retries

log = get_logger("google.api")

#: Status codes worth another attempt.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class GoogleApiError(RuntimeError):
    """Raised when a Google API call fails in a non-recoverable way."""


class _RetryableHttpError(Exception):
    """Internal marker so with_retries only retries transient statuses."""


def _status_of(exc: Exception):
    response = getattr(exc, "resp", None)
    status = getattr(response, "status", None)
    if status is None:
        status = getattr(exc, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def execute(request: Any, description: str) -> Any:
    """Execute a googleapiclient request with bounded retries."""

    def _call():
        try:
            return request.execute()
        except Exception as exc:
            status = _status_of(exc)
            if status in RETRYABLE_STATUS:
                raise _RetryableHttpError(f"HTTP {status}: {exc}") from exc
            raise

    try:
        return with_retries(
            _call,
            description=description,
            retry_on=(_RetryableHttpError, ConnectionError, TimeoutError),
        )
    except Exception as exc:
        status = _status_of(exc)
        hint = ""
        text = str(exc)
        if "invalid_grant" in text:
            if "Bad Request" in text:
                hint = (
                    " Google rejected the stored refresh token as MALFORMED "
                    "(invalid_grant: Bad Request). This almost always means "
                    "the GOOGLE_REFRESH_TOKEN secret was damaged when it was "
                    "copied: a line break picked up from a wrapped console "
                    "window, a truncated selection, or the "
                    "'GOOGLE_REFRESH_TOKEN=' prefix pasted into the value. "
                    "Re-enter the secret as ONE continuous line (it starts "
                    "with '1//' and contains no spaces). If the token is no "
                    "longer available, rerun scripts/google_auth_setup.py - "
                    "it is safe to repeat and reuses the existing folder and "
                    "Sheet."
                )
            else:
                hint = (
                    " The refresh token has been revoked or has expired "
                    "(invalid_grant). If the OAuth consent screen is still "
                    "in 'Testing' status, Google revokes its tokens after 7 "
                    "days - publish the app ('In production'), then rerun "
                    "scripts/google_auth_setup.py and update the "
                    "GOOGLE_REFRESH_TOKEN secret."
                )
        elif status == 403:
            hint = (
                " This usually means the OAuth scopes do not cover the "
                "operation, or the account cannot write to this resource."
            )
        elif status == 404:
            hint = (
                " The folder or spreadsheet ID was not found. With the "
                "drive.file scope the automation can only touch files it "
                "created itself - check GOOGLE_DRIVE_FOLDER_ID and "
                "GOOGLE_SHEET_ID were produced by scripts/google_auth_setup.py."
            )
        elif status == 401:
            hint = (
                " The refresh token was rejected; the client needs to "
                "re-authorise via scripts/google_auth_setup.py."
            )
        raise GoogleApiError(f"{description} failed.{hint} Details: {exc}") from exc
