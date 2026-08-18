"""OAuth scope policy - the production workflow requests drive.file ONLY.

These tests pin the least-privilege decision from the scope audit: one
non-sensitive scope covers the whole workflow (the Sheets API accepts
drive.file for every method this project calls on app-created files), so
neither the sensitive ``spreadsheets`` scope nor the restricted full
``drive`` scope may ever appear in the default path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from src.config import DEFAULT_GOOGLE_SCOPES, Config

DRIVE_FILE = "https://www.googleapis.com/auth/drive.file"
DRIVE_FULL = "https://www.googleapis.com/auth/drive"
SPREADSHEETS = "https://www.googleapis.com/auth/spreadsheets"

ROOT = Path(__file__).resolve().parent.parent


def load_setup_helper():
    """Import scripts/google_auth_setup.py as a module."""
    spec = importlib.util.spec_from_file_location(
        "google_auth_setup", ROOT / "scripts" / "google_auth_setup.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDefaultScopes:
    def test_default_is_exactly_drive_file(self):
        assert DEFAULT_GOOGLE_SCOPES == (DRIVE_FILE,)

    def test_no_sensitive_or_restricted_scope_by_default(self):
        assert SPREADSHEETS not in DEFAULT_GOOGLE_SCOPES
        assert DRIVE_FULL not in DEFAULT_GOOGLE_SCOPES

    def test_config_from_env_uses_the_default(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_SCOPES", raising=False)

        config = Config.from_env(load_env_file=False)

        assert config.google_scopes == (DRIVE_FILE,)

    def test_env_override_is_still_possible(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_SCOPES", f"{DRIVE_FULL}, {DRIVE_FILE}")

        config = Config.from_env(load_env_file=False)

        assert config.google_scopes == (DRIVE_FULL, DRIVE_FILE)


class TestSetupHelperScopes:
    def test_normal_mode_requests_exactly_drive_file(self):
        helper = load_setup_helper()

        assert helper.select_scopes(scope_drive_full=False) == [DRIVE_FILE]

    def test_normal_mode_never_requests_spreadsheets(self):
        helper = load_setup_helper()

        scopes = helper.select_scopes(scope_drive_full=False)

        assert SPREADSHEETS not in scopes
        assert DRIVE_FULL not in scopes

    def test_full_drive_flag_is_isolated_from_the_default(self):
        helper = load_setup_helper()

        full = helper.select_scopes(scope_drive_full=True)

        assert full == [DRIVE_FULL]
        assert DRIVE_FILE not in full  # never both scopes at once
        assert SPREADSHEETS not in full  # full drive already covers Sheets

    def test_default_argument_matches_normal_mode(self):
        helper = load_setup_helper()

        assert helper.select_scopes() == [DRIVE_FILE]


class TestRuntimeCredentials:
    def full_config(self):
        return Config(
            google_client_id="client-id",
            google_client_secret="client-secret",
            google_refresh_token="refresh-token",
            google_drive_folder_id="folder-id",
            google_sheet_id="sheet-id",
        )

    def test_credentials_carry_only_drive_file(self):
        from src.google_auth import build_credentials

        credentials = build_credentials(self.full_config())

        assert credentials.scopes == [DRIVE_FILE]

    def test_missing_credentials_name_the_missing_settings(self):
        from src.google_auth import GoogleAuthError, build_credentials

        with pytest.raises(GoogleAuthError) as exc:
            build_credentials(Config())

        message = str(exc.value)
        assert "GOOGLE_CLIENT_ID" in message
        assert "GOOGLE_REFRESH_TOKEN" in message
        # The secret values themselves are never in the message.
        assert "refresh-token" not in message
