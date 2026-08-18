"""Bounded retry helper for transient external calls.

Implements the schedule from the implementation guide (section 20.1):
attempt 1 -> wait 2s -> attempt 2 -> wait 5s -> attempt 3 -> raise.

Deterministic problems (a missing column, an empty schedule, a bad week
number) must *not* be retried, so callers pass the narrow set of exception
types that genuinely represent transient failures.
"""

from __future__ import annotations

import time
from typing import Callable, Iterable, Sequence, Type, TypeVar

from .logging_utils import get_logger

T = TypeVar("T")

log = get_logger("retry")

DEFAULT_BACKOFF: Sequence[float] = (2.0, 5.0)


class RetryExhaustedError(RuntimeError):
    """Raised when every attempt of a retried operation failed."""


def with_retries(
    func: Callable[[], T],
    description: str,
    backoff: Sequence[float] = DEFAULT_BACKOFF,
    retry_on: Iterable[Type[BaseException]] = (Exception,),
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``func`` up to ``len(backoff) + 1`` times.

    Parameters
    ----------
    func:
        Zero-argument callable performing the transient operation.
    description:
        Human-readable operation name used in log messages and the final error.
    backoff:
        Delays in seconds between attempts. ``(2, 5)`` gives three attempts.
    retry_on:
        Exception types that should trigger another attempt. Anything else
        propagates immediately - validation errors must fail fast.
    """
    retry_on = tuple(retry_on)
    attempts = len(backoff) + 1
    last_error: BaseException | None = None

    for attempt in range(1, attempts + 1):
        try:
            return func()
        except retry_on as exc:  # noqa: PERF203 - retry loop is the point
            last_error = exc
            if attempt == attempts:
                break
            delay = backoff[attempt - 1]
            log.warning(
                "%s failed on attempt %d/%d (%s: %s). Retrying in %.0fs.",
                description,
                attempt,
                attempts,
                type(exc).__name__,
                exc,
                delay,
            )
            sleep(delay)

    raise RetryExhaustedError(
        f"{description} failed after {attempts} attempts. "
        f"Last error: {type(last_error).__name__}: {last_error}"
    ) from last_error
