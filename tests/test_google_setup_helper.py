"""One-time setup helper: offline consent, refresh-token guard, rerun safety.

Regression tests for the incident where David's refresh token was not
captured: the flow must always request offline access with forced consent,
a missing refresh token must be a loud failure (never a false SUCCESS),
the token must be handed off in copy-ready form, and re-running the helper
must reuse the Drive folder and Sheet created by the first authorization
instead of duplicating them.
"""

from __future__ import annotations

import importlib.util
import itertools
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def load_helper():
    spec = importlib.util.spec_from_file_location(
        "google_auth_setup_under_test", ROOT / "scripts" / "google_auth_setup.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------
# OAuth flow fakes
# ---------------------------------------------------------------------

class FakeCredentials:
    def __init__(self, refresh_token):
        self.refresh_token = refresh_token


class FakeFlow:
    captured_kwargs = None
    next_refresh_token = "refresh-token-value"

    def __init__(self, path, scopes):
        self.path = path
        self.scopes = scopes

    @classmethod
    def from_client_secrets_file(cls, path, scopes):
        return cls(path, scopes)

    def run_local_server(self, **kwargs):
        FakeFlow.captured_kwargs = kwargs
        return FakeCredentials(FakeFlow.next_refresh_token)


@pytest.fixture
def fake_flow(monkeypatch, tmp_path):
    import google_auth_oauthlib.flow as real_flow_module

    FakeFlow.captured_kwargs = None
    FakeFlow.next_refresh_token = "refresh-token-value"
    monkeypatch.setattr(real_flow_module, "InstalledAppFlow", FakeFlow)

    secrets = tmp_path / "client_secret.json"
    secrets.write_text('{"installed": {"client_id": "x", "client_secret": "y"}}')
    return secrets


class TestAuthorise:
    def test_requests_offline_access(self, fake_flow):
        helper = load_helper()

        helper.authorise(fake_flow, helper.select_scopes(), use_console=False)

        assert FakeFlow.captured_kwargs["access_type"] == "offline"

    def test_forces_the_consent_screen(self, fake_flow):
        """prompt=consent is what makes Google reissue a refresh token on a
        rerun without the earlier grant being revoked."""
        helper = load_helper()

        helper.authorise(fake_flow, helper.select_scopes(), use_console=False)

        assert FakeFlow.captured_kwargs["prompt"] == "consent"

    def test_console_mode_does_not_launch_a_browser(self, fake_flow):
        """run_console() no longer exists in google-auth-oauthlib; the
        console fallback must use the no-browser local server instead."""
        helper = load_helper()

        helper.authorise(fake_flow, helper.select_scopes(), use_console=True)

        assert FakeFlow.captured_kwargs["open_browser"] is False

    def test_returns_credentials_with_the_refresh_token(self, fake_flow):
        helper = load_helper()

        credentials = helper.authorise(
            fake_flow, helper.select_scopes(), use_console=False
        )

        assert credentials.refresh_token == "refresh-token-value"

    def test_missing_refresh_token_is_a_loud_failure(self, fake_flow):
        helper = load_helper()
        FakeFlow.next_refresh_token = None

        with pytest.raises(SystemExit) as exc:
            helper.authorise(fake_flow, helper.select_scopes(), use_console=False)

        message = str(exc.value)
        assert "FAILED" in message
        assert "refresh token" in message
        assert "DID NOT SUCCEED" in message
        assert "repeat" in message.lower()

    def test_empty_refresh_token_also_fails(self, fake_flow):
        helper = load_helper()
        FakeFlow.next_refresh_token = ""

        with pytest.raises(SystemExit):
            helper.authorise(fake_flow, helper.select_scopes(), use_console=False)


class TestSecretsHandoff:
    def test_block_contains_the_refresh_token_in_env_format(self):
        helper = load_helper()

        block = helper.format_secrets_block(
            client_id="cid",
            client_secret="csec",
            refresh_token="tok-123",
            folder_id="folder-abc",
            sheet_id="sheet-xyz",
        )

        assert "GOOGLE_REFRESH_TOKEN=tok-123" in block
        assert block.count("tok-123") == 1  # exactly once, nowhere else

    def test_block_contains_all_five_secret_names(self):
        helper = load_helper()

        block = helper.format_secrets_block("a", "b", "c", "d", "e")

        for name in helper.SECRET_NAMES:
            assert f"{name}=" in block

    def test_block_warns_that_values_are_shown_once(self):
        helper = load_helper()

        block = helper.format_secrets_block("a", "b", "c", "d", "e")

        assert "ONLY ONCE" in block
        assert "DO NOT CLOSE" in block


# ---------------------------------------------------------------------
# Rerun safety: provision() must reuse, never duplicate
# ---------------------------------------------------------------------

class FakeRequest:
    def __init__(self, result):
        self._result = result

    def execute(self):
        return self._result


class World:
    """Shared Drive/Sheets state behind both fake services."""

    def __init__(self):
        self.files = {}  # id -> {name, mime, parents:set}
        self.tabs = {}  # spreadsheet id -> [tab titles]
        self._ids = itertools.count(1)

    def new_id(self):
        return f"id-{next(self._ids)}"

    def count(self, mime):
        return sum(1 for meta in self.files.values() if meta["mime"] == mime)


class FakeFiles:
    def __init__(self, world):
        self.world = world

    def list(self, q, **_kwargs):
        name = re.search(r"name = '([^']*)'", q)
        mime = re.search(r"mimeType = '([^']*)'", q)
        parent = re.search(r"'([^']*)' in parents", q)
        matches = []
        for file_id, meta in self.world.files.items():
            if name and meta["name"] != name.group(1):
                continue
            if mime and meta["mime"] != mime.group(1):
                continue
            if parent and parent.group(1) not in meta["parents"]:
                continue
            matches.append({"id": file_id, "name": meta["name"]})
        return FakeRequest({"files": matches})

    def create(self, body, **_kwargs):
        file_id = self.world.new_id()
        self.world.files[file_id] = {
            "name": body["name"],
            "mime": body.get("mimeType", "file"),
            "parents": set(body.get("parents", ["root"])),
        }
        return FakeRequest({"id": file_id})

    def get(self, fileId, **_kwargs):
        return FakeRequest({"parents": sorted(self.world.files[fileId]["parents"])})

    def update(self, fileId, addParents=None, removeParents=None, **_kwargs):
        meta = self.world.files[fileId]
        if removeParents:
            for parent in removeParents.split(","):
                meta["parents"].discard(parent)
        if addParents:
            meta["parents"].add(addParents)
        return FakeRequest({"id": fileId})


class FakeDrive:
    def __init__(self, world):
        self.world = world

    def files(self):
        return FakeFiles(self.world)


class FakeSpreadsheets:
    def __init__(self, world):
        self.world = world

    def create(self, body, fields=None):
        sheet_id = self.world.new_id()
        self.world.files[sheet_id] = {
            "name": body["properties"]["title"],
            "mime": "application/vnd.google-apps.spreadsheet",
            "parents": {"root"},
        }
        self.world.tabs[sheet_id] = [
            sheet["properties"]["title"] for sheet in body.get("sheets", [])
        ]
        return FakeRequest({"spreadsheetId": sheet_id})

    def get(self, spreadsheetId):
        return FakeRequest(
            {
                "sheets": [
                    {"properties": {"title": title}}
                    for title in self.world.tabs.get(spreadsheetId, [])
                ]
            }
        )

    def batchUpdate(self, spreadsheetId, body):
        for request in body["requests"]:
            self.world.tabs[spreadsheetId].append(
                request["addSheet"]["properties"]["title"]
            )
        return FakeRequest({})


class FakeSheets:
    def __init__(self, world):
        self.world = world

    def spreadsheets(self):
        return FakeSpreadsheets(self.world)


FOLDER_MIME = "application/vnd.google-apps.folder"
SHEET_MIME = "application/vnd.google-apps.spreadsheet"


class TestProvisionRerunSafety:
    def setup_world(self):
        world = World()
        return world, FakeDrive(world), FakeSheets(world)

    def test_first_run_creates_folder_model_subfolder_and_sheet(self):
        helper = load_helper()
        world, drive, sheets = self.setup_world()

        result = helper.provision(drive, sheets)

        assert result["folder_created"] is True
        assert result["sheet_created"] is True
        assert world.count(FOLDER_MIME) == 2  # root folder + Model/
        assert world.count(SHEET_MIME) == 1
        assert sorted(world.tabs[result["sheet_id"]]) == sorted(
            load_helper_tabs()
        )

    def test_rerun_reuses_everything_and_creates_nothing(self):
        helper = load_helper()
        world, drive, sheets = self.setup_world()

        first = helper.provision(drive, sheets)
        second = helper.provision(drive, sheets)

        assert second["folder_id"] == first["folder_id"]
        assert second["sheet_id"] == first["sheet_id"]
        assert second["folder_created"] is False
        assert second["sheet_created"] is False
        assert world.count(FOLDER_MIME) == 2  # still just folder + Model/
        assert world.count(SHEET_MIME) == 1  # still one spreadsheet

    def test_rerun_does_not_duplicate_tabs(self):
        helper = load_helper()
        world, drive, sheets = self.setup_world()

        first = helper.provision(drive, sheets)
        helper.provision(drive, sheets)

        titles = world.tabs[first["sheet_id"]]
        assert len(titles) == len(set(titles)) == 5

    def test_sheet_orphaned_by_a_crashed_run_is_adopted_not_duplicated(self):
        """First run crashed after creating the Sheet but before moving it
        into the folder: the rerun must find it anyway and move it."""
        helper = load_helper()
        world, drive, sheets = self.setup_world()

        orphan_id = world.new_id()
        world.files[orphan_id] = {
            "name": helper.DEFAULT_SHEET_NAME,
            "mime": SHEET_MIME,
            "parents": {"root"},
        }
        world.tabs[orphan_id] = ["Latest Predictions"]

        result = helper.provision(drive, sheets)

        assert result["sheet_id"] == orphan_id
        assert result["sheet_created"] is False
        assert world.count(SHEET_MIME) == 1
        # adopted into the project folder and missing tabs added
        assert result["folder_id"] in world.files[orphan_id]["parents"]
        assert len(world.tabs[orphan_id]) == 5

    def test_explicit_ids_short_circuit_discovery(self):
        helper = load_helper()
        world, drive, sheets = self.setup_world()
        existing = helper.provision(drive, sheets)

        result = helper.provision(
            drive,
            sheets,
            folder_id=existing["folder_id"],
            sheet_id=existing["sheet_id"],
        )

        assert result["folder_created"] is False
        assert result["sheet_created"] is False
        assert world.count(SHEET_MIME) == 1


def load_helper_tabs():
    from src.google_sheets import ALL_TABS

    return list(ALL_TABS)
