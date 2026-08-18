"""Drive archive behaviour, exercised against a fake Drive API.

The rules that matter on a rerun are that folders are reused rather than
duplicated, and that a file of the same name is updated in place rather
than added alongside (implementation guide 23 / 31.7).
"""

from __future__ import annotations

import pytest

from src.google_drive import (
    FOLDER_MIME,
    ensure_model_folder,
    ensure_week_folder,
    find_file,
    get_or_create_folder,
    persist_model_to_drive,
    restore_model_from_drive,
    upload_file,
    upload_week_outputs,
)


class FakeRequest:
    def __init__(self, result):
        self._result = result

    def execute(self):
        return self._result


class FakeFiles:
    def __init__(self, store, calls, next_id):
        self._store = store
        self._calls = calls
        # Shared with the service: files() hands out a new wrapper each
        # call, so the id counter must not live on the wrapper.
        self._next_id = next_id

    def list(self, q, spaces, fields, pageSize, **kwargs):
        self._calls.append(("list", q))
        name = q.split("name = '")[1].split("'")[0]
        parent = q.split(" in parents")[0].split("'")[-2]
        wants_folder = FOLDER_MIME in q
        matches = [
            {"id": file_id, "name": meta["name"]}
            for file_id, meta in self._store.items()
            if meta["name"] == name
            and meta.get("parent") == parent
            and (meta.get("mimeType") == FOLDER_MIME) == wants_folder
        ]
        return FakeRequest({"files": matches})

    def create(self, body, fields=None, media_body=None, **kwargs):
        file_id = f"id-{self._next_id[0]}"
        self._next_id[0] += 1
        parents = body.get("parents") or [None]
        self._store[file_id] = {
            "name": body["name"],
            "parent": parents[0],
            "mimeType": body.get("mimeType"),
            "revisions": 1,
        }
        self._calls.append(("create", body["name"]))
        return FakeRequest({"id": file_id})

    def update(self, fileId, media_body=None, **kwargs):
        self._store[fileId]["revisions"] += 1
        self._calls.append(("update", self._store[fileId]["name"]))
        return FakeRequest({"id": fileId})

    def get_media(self, fileId, **kwargs):
        return FakeRequest(b"")


class FakeDriveService:
    def __init__(self):
        self.store = {}
        self.calls = []
        self._next_id = [1]

    def files(self):
        return FakeFiles(self.store, self.calls, self._next_id)

    def names(self):
        return sorted(meta["name"] for meta in self.store.values())

    def folders(self):
        return sorted(
            meta["name"]
            for meta in self.store.values()
            if meta.get("mimeType") == FOLDER_MIME
        )


@pytest.fixture
def drive():
    return FakeDriveService()


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "epa_splits.csv"
    path.write_text("team,net_epa_play\nKC,0.2\n", encoding="utf-8")
    return path


class TestFolders:
    def test_creates_a_folder_when_absent(self, drive):
        folder_id = get_or_create_folder(drive, "CSV Archive", "root")

        assert folder_id in drive.store
        assert drive.folders() == ["CSV Archive"]

    def test_reuses_an_existing_folder(self, drive):
        first = get_or_create_folder(drive, "CSV Archive", "root")
        second = get_or_create_folder(drive, "CSV Archive", "root")

        assert first == second
        assert drive.folders() == ["CSV Archive"]

    def test_same_name_under_different_parents_are_distinct(self, drive):
        a = get_or_create_folder(drive, "2026", "parent-a")
        b = get_or_create_folder(drive, "2026", "parent-b")

        assert a != b

    def test_week_path_is_built_once(self, drive):
        first = ensure_week_folder(drive, "root", 2026, 1)
        second = ensure_week_folder(drive, "root", 2026, 1)

        assert first == second
        assert drive.folders() == ["2026", "CSV Archive", "Week 01"]

    def test_week_number_is_zero_padded(self, drive):
        ensure_week_folder(drive, "root", 2026, 3)

        assert "Week 03" in drive.folders()

    def test_different_weeks_get_different_folders(self, drive):
        one = ensure_week_folder(drive, "root", 2026, 1)
        two = ensure_week_folder(drive, "root", 2026, 2)

        assert one != two
        assert "Week 01" in drive.folders()
        assert "Week 02" in drive.folders()


class TestUploads:
    def test_first_upload_creates(self, drive, csv_file):
        upload_file(drive, csv_file, "folder-1")

        assert [kind for kind, _ in drive.calls if kind == "create"] == ["create"]
        assert drive.names() == ["epa_splits.csv"]

    def test_second_upload_updates_in_place(self, drive, csv_file):
        upload_file(drive, csv_file, "folder-1")
        upload_file(drive, csv_file, "folder-1")

        assert drive.names() == ["epa_splits.csv"]
        assert [meta["revisions"] for meta in drive.store.values()] == [2]

    def test_missing_local_file_raises(self, drive, tmp_path):
        from src.google_api import GoogleApiError

        with pytest.raises(GoogleApiError):
            upload_file(drive, tmp_path / "nope.csv", "folder-1")

    def test_week_outputs_land_in_the_week_folder(self, drive, tmp_path):
        paths = []
        for name in ("epa_splits.csv", "fair_lines.csv", "run_summary.json"):
            path = tmp_path / name
            path.write_text("x", encoding="utf-8")
            paths.append(path)

        uploaded = upload_week_outputs(drive, "root", 2026, 1, paths)

        assert set(uploaded) == {
            "epa_splits.csv",
            "fair_lines.csv",
            "run_summary.json",
        }

    def test_rerunning_a_week_does_not_duplicate(self, drive, tmp_path):
        path = tmp_path / "fair_lines.csv"
        path.write_text("x", encoding="utf-8")

        upload_week_outputs(drive, "root", 2026, 1, [path])
        upload_week_outputs(drive, "root", 2026, 1, [path])

        assert drive.names().count("fair_lines.csv") == 1

    def test_absent_output_is_skipped_not_fatal(self, drive, tmp_path):
        present = tmp_path / "fair_lines.csv"
        present.write_text("x", encoding="utf-8")

        uploaded = upload_week_outputs(
            drive, "root", 2026, 1, [present, tmp_path / "missing.csv"]
        )

        assert list(uploaded) == ["fair_lines.csv"]


class TestModelStore:
    def test_restore_reports_false_when_nothing_is_stored(self, drive, tmp_path):
        restored = restore_model_from_drive(
            drive,
            "root",
            tmp_path / "fair_line_model.pkl",
            tmp_path / "model_metadata.json",
        )

        assert restored is False

    def test_persist_uploads_both_files(self, drive, tmp_path):
        model = tmp_path / "fair_line_model.pkl"
        metadata = tmp_path / "model_metadata.json"
        model.write_bytes(b"pickle")
        metadata.write_text("{}", encoding="utf-8")

        persist_model_to_drive(drive, "root", model, metadata)

        assert drive.names() == [
            "Model",
            "fair_line_model.pkl",
            "model_metadata.json",
        ]

    def test_persisting_twice_updates_rather_than_duplicates(self, drive, tmp_path):
        model = tmp_path / "fair_line_model.pkl"
        metadata = tmp_path / "model_metadata.json"
        model.write_bytes(b"pickle")
        metadata.write_text("{}", encoding="utf-8")

        persist_model_to_drive(drive, "root", model, metadata)
        persist_model_to_drive(drive, "root", model, metadata)

        assert drive.names().count("fair_line_model.pkl") == 1

    def test_model_folder_is_reused(self, drive):
        first = ensure_model_folder(drive, "root")
        second = ensure_model_folder(drive, "root")

        assert first == second

    def test_find_file_returns_none_when_absent(self, drive):
        assert find_file(drive, "nothing.csv", "folder-1") is None
