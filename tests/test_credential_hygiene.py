"""Copy/paste damage to credentials must heal or diagnose itself.

Regression tests for the live incident where the GOOGLE_REFRESH_TOKEN
GitHub secret carried invisible damage from being copied out of a console
window, and every Google call failed with the unhelpful
``invalid_grant: Bad Request``.
"""

from __future__ import annotations

import pytest

from src.config import Config
from src.google_api import GoogleApiError, execute

TOKEN = "1//0abcDEFghiJKLmnoPQRstuVWXyz1234567890abcdefghijklmnop"


def make_env(monkeypatch, **overrides):
    values = {
        "GOOGLE_CLIENT_ID": "id-123.apps.googleusercontent.com",
        "GOOGLE_CLIENT_SECRET": "GOCSPX-secret",
        "GOOGLE_REFRESH_TOKEN": TOKEN,
        "GOOGLE_DRIVE_FOLDER_ID": "folder-abc",
        "GOOGLE_SHEET_ID": "sheet-xyz",
    }
    values.update(overrides)
    for name, value in values.items():
        monkeypatch.setenv(name, value)


class TestCredentialHealing:
    def test_clean_values_pass_through_unchanged(self, monkeypatch):
        make_env(monkeypatch)

        config = Config.from_env(load_env_file=False)

        assert config.google_refresh_token == TOKEN

    def test_embedded_line_break_is_removed(self, monkeypatch):
        """A console window wraps the token line; copying it inserts a hard
        line break in the middle of the value."""
        broken = TOKEN[:40] + "\r\n" + TOKEN[40:]
        make_env(monkeypatch, GOOGLE_REFRESH_TOKEN=broken)

        config = Config.from_env(load_env_file=False)

        assert config.google_refresh_token == TOKEN

    def test_wrap_with_indent_spaces_is_removed(self, monkeypatch):
        broken = TOKEN[:40] + "\n    " + TOKEN[40:]
        make_env(monkeypatch, GOOGLE_REFRESH_TOKEN=broken)

        config = Config.from_env(load_env_file=False)

        assert config.google_refresh_token == TOKEN

    def test_pasted_name_prefix_is_stripped(self, monkeypatch):
        """The whole GOOGLE_REFRESH_TOKEN=... line pasted as the value."""
        make_env(monkeypatch, GOOGLE_REFRESH_TOKEN=f"GOOGLE_REFRESH_TOKEN={TOKEN}")

        config = Config.from_env(load_env_file=False)

        assert config.google_refresh_token == TOKEN

    def test_prefix_and_wrap_together_are_healed(self, monkeypatch):
        broken = f"GOOGLE_REFRESH_TOKEN={TOKEN[:30]}\n{TOKEN[30:]}"
        make_env(monkeypatch, GOOGLE_REFRESH_TOKEN=broken)

        config = Config.from_env(load_env_file=False)

        assert config.google_refresh_token == TOKEN

    def test_all_five_settings_are_healed(self, monkeypatch):
        make_env(
            monkeypatch,
            GOOGLE_CLIENT_ID="id-123.apps\n.googleusercontent.com",
            GOOGLE_CLIENT_SECRET="GOCSPX-\nsecret",
            GOOGLE_DRIVE_FOLDER_ID="GOOGLE_DRIVE_FOLDER_ID=folder-abc",
            GOOGLE_SHEET_ID=" sheet-xyz \n",
        )

        config = Config.from_env(load_env_file=False)

        assert config.google_client_id == "id-123.apps.googleusercontent.com"
        assert config.google_client_secret == "GOCSPX-secret"
        assert config.google_drive_folder_id == "folder-abc"
        assert config.google_sheet_id == "sheet-xyz"


class FailingRequest:
    def __init__(self, error):
        self._error = error
        self.calls = 0

    def execute(self):
        self.calls += 1
        raise self._error


class TestInvalidGrantDiagnostics:
    def test_bad_request_names_the_copy_paste_cause(self):
        request = FailingRequest(
            Exception("('invalid_grant: Bad Request', {'error': 'invalid_grant'})")
        )

        with pytest.raises(GoogleApiError) as exc:
            execute(request, "Drive folder lookup for 'Model'")

        message = str(exc.value)
        assert "MALFORMED" in message
        assert "line break" in message
        assert "ONE continuous line" in message
        assert "google_auth_setup.py" in message

    def test_revoked_token_points_at_testing_status(self):
        request = FailingRequest(
            Exception(
                "('invalid_grant: Token has been expired or revoked.', {})"
            )
        )

        with pytest.raises(GoogleApiError) as exc:
            execute(request, "Drive folder lookup")

        message = str(exc.value)
        assert "revoked" in message
        assert "In production" in message

    def test_invalid_grant_is_not_retried(self):
        """A rejected token is deterministic - burning the 2s/5s retry
        schedule on it would just delay the failure."""
        request = FailingRequest(Exception("invalid_grant: Bad Request"))

        with pytest.raises(GoogleApiError):
            execute(request, "op")

        assert request.calls == 1
