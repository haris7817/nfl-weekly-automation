"""Retry policy: transient failures get another go, validation errors do not."""

from __future__ import annotations

import pytest

from src.retry import DEFAULT_BACKOFF, RetryExhaustedError, with_retries


class Recorder:
    def __init__(self):
        self.delays = []

    def __call__(self, delay):
        self.delays.append(delay)


class TestWithRetries:
    def test_returns_immediately_on_success(self):
        sleeps = Recorder()

        result = with_retries(lambda: "ok", "op", sleep=sleeps)

        assert result == "ok"
        assert sleeps.delays == []

    def test_recovers_on_a_later_attempt(self):
        sleeps = Recorder()
        attempts = {"n": 0}

        def flaky():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ConnectionError("network blip")
            return "recovered"

        result = with_retries(
            flaky, "op", retry_on=(ConnectionError,), sleep=sleeps
        )

        assert result == "recovered"
        assert attempts["n"] == 3

    def test_uses_the_documented_two_then_five_second_backoff(self):
        sleeps = Recorder()

        def always_fails():
            raise ConnectionError("down")

        with pytest.raises(RetryExhaustedError):
            with_retries(
                always_fails, "op", retry_on=(ConnectionError,), sleep=sleeps
            )

        assert sleeps.delays == [2.0, 5.0]
        assert tuple(sleeps.delays) == tuple(DEFAULT_BACKOFF)

    def test_gives_up_after_three_attempts(self):
        sleeps = Recorder()
        attempts = {"n": 0}

        def always_fails():
            attempts["n"] += 1
            raise ConnectionError("down")

        with pytest.raises(RetryExhaustedError):
            with_retries(
                always_fails, "op", retry_on=(ConnectionError,), sleep=sleeps
            )

        assert attempts["n"] == 3

    def test_deterministic_errors_are_not_retried(self):
        """A missing column or bad week must fail fast, not burn attempts."""
        sleeps = Recorder()
        attempts = {"n": 0}

        def bad_input():
            attempts["n"] += 1
            raise ValueError("week 99 does not exist")

        with pytest.raises(ValueError):
            with_retries(
                bad_input, "op", retry_on=(ConnectionError,), sleep=sleeps
            )

        assert attempts["n"] == 1
        assert sleeps.delays == []

    def test_error_message_names_the_operation_and_cause(self):
        def always_fails():
            raise ConnectionError("connection reset")

        with pytest.raises(RetryExhaustedError) as exc:
            with_retries(
                always_fails,
                "nflreadpy.load_pbp([2025])",
                retry_on=(ConnectionError,),
                sleep=Recorder(),
            )

        message = str(exc.value)
        assert "nflreadpy.load_pbp([2025])" in message
        assert "connection reset" in message
        assert "3 attempts" in message

    def test_custom_backoff_controls_the_attempt_count(self):
        sleeps = Recorder()
        attempts = {"n": 0}

        def always_fails():
            attempts["n"] += 1
            raise ConnectionError("down")

        with pytest.raises(RetryExhaustedError):
            with_retries(
                always_fails,
                "op",
                backoff=(1.0,),
                retry_on=(ConnectionError,),
                sleep=sleeps,
            )

        assert attempts["n"] == 2
        assert sleeps.delays == [1.0]
