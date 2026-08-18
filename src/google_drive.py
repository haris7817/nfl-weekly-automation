"""Google Drive archive and model store.

Layout created and maintained by the automation:

    <root folder>/                 GOOGLE_DRIVE_FOLDER_ID
      CSV Archive/
        2026/
          Week 01/
            epa_splits.csv
            fair_lines.csv
            run_summary.json
      Model/
        fair_line_model.pkl
        model_metadata.json

Two behaviours matter for reruns (implementation guide 23 / 31.7):

* folders are looked up by name before being created, so a second run for
  the same week reuses the existing folder instead of creating a duplicate;
* a file with the same name in the same folder is *updated in place*
  (new revision) rather than added alongside, so the archive keeps one
  canonical file per week.

The ``Model`` folder is what makes retraining survive between runs. GitHub
runners are ephemeral, so a model trained this Friday would be gone by the
next one; storing it in Drive gives it a home the client already owns
(implementation guide 26).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .google_api import GoogleApiError, execute
from .logging_utils import get_logger

log = get_logger("google.drive")

FOLDER_MIME = "application/vnd.google-apps.folder"

ARCHIVE_FOLDER_NAME = "CSV Archive"
MODEL_FOLDER_NAME = "Model"

_MIME_BY_SUFFIX = {
    ".csv": "text/csv",
    ".json": "application/json",
    ".pkl": "application/octet-stream",
    ".log": "text/plain",
}


def _escape(name: str) -> str:
    return name.replace("\\", "\\\\").replace("'", "\\'")


def get_or_create_folder(service, name: str, parent_id: str) -> str:
    """Return the ID of ``name`` under ``parent_id``, creating it if needed."""
    query = (
        f"name = '{_escape(name)}' and "
        f"'{parent_id}' in parents and "
        f"mimeType = '{FOLDER_MIME}' and trashed = false"
    )
    response = execute(
        service.files().list(
            q=query,
            spaces="drive",
            fields="files(id, name)",
            pageSize=10,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ),
        f"Drive folder lookup for {name!r}",
    )
    files = response.get("files", [])
    if files:
        if len(files) > 1:
            log.warning(
                "Drive has %d folders named %r in the same parent; using the "
                "first. Consider tidying this up in the Drive UI.",
                len(files),
                name,
            )
        return files[0]["id"]

    created = execute(
        service.files().create(
            body={"name": name, "mimeType": FOLDER_MIME, "parents": [parent_id]},
            fields="id",
            supportsAllDrives=True,
        ),
        f"Drive folder creation for {name!r}",
    )
    log.info("Created Drive folder %r.", name)
    return created["id"]


def ensure_week_folder(service, root_folder_id: str, season: int, week: int) -> str:
    """Create/reuse ``CSV Archive/<season>/Week <NN>`` and return its ID."""
    archive_id = get_or_create_folder(service, ARCHIVE_FOLDER_NAME, root_folder_id)
    season_id = get_or_create_folder(service, str(season), archive_id)
    return get_or_create_folder(service, f"Week {int(week):02d}", season_id)


def ensure_model_folder(service, root_folder_id: str) -> str:
    return get_or_create_folder(service, MODEL_FOLDER_NAME, root_folder_id)


def upload_file(
    service,
    local_path: Path,
    parent_id: str,
    name: Optional[str] = None,
) -> str:
    """Create or update ``name`` inside ``parent_id``. Returns the file ID."""
    from googleapiclient.http import MediaFileUpload

    local_path = Path(local_path)
    if not local_path.is_file():
        raise GoogleApiError(f"Cannot upload {local_path}: the file does not exist.")

    name = name or local_path.name
    mime = _MIME_BY_SUFFIX.get(local_path.suffix.lower(), "application/octet-stream")
    media = MediaFileUpload(str(local_path), mimetype=mime, resumable=False)

    query = (
        f"name = '{_escape(name)}' and '{parent_id}' in parents and trashed = false"
    )
    existing = execute(
        service.files().list(
            q=query,
            spaces="drive",
            fields="files(id, name)",
            pageSize=10,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ),
        f"Drive lookup for {name!r}",
    ).get("files", [])

    if existing:
        file_id = existing[0]["id"]
        execute(
            service.files().update(
                fileId=file_id, media_body=media, supportsAllDrives=True
            ),
            f"Drive update of {name!r}",
        )
        log.info("Updated %s in Drive.", name)
        return file_id

    created = execute(
        service.files().create(
            body={"name": name, "parents": [parent_id]},
            media_body=media,
            fields="id",
            supportsAllDrives=True,
        ),
        f"Drive upload of {name!r}",
    )
    log.info("Uploaded %s to Drive.", name)
    return created["id"]


def download_file(service, file_id: str, destination: Path) -> Path:
    """Download a Drive file to ``destination``."""
    import io

    from googleapiclient.http import MediaIoBaseDownload

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    tmp = destination.with_suffix(destination.suffix + ".part")
    tmp.write_bytes(buffer.getvalue())
    tmp.replace(destination)
    return destination


def find_file(service, name: str, parent_id: str) -> Optional[str]:
    query = (
        f"name = '{_escape(name)}' and '{parent_id}' in parents and trashed = false"
    )
    files = execute(
        service.files().list(
            q=query,
            spaces="drive",
            fields="files(id, name)",
            pageSize=10,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ),
        f"Drive lookup for {name!r}",
    ).get("files", [])
    return files[0]["id"] if files else None


# ---------------------------------------------------------------------
# Model persistence across ephemeral cloud runs
# ---------------------------------------------------------------------

def restore_model_from_drive(
    service,
    root_folder_id: str,
    model_path: Path,
    metadata_path: Path,
) -> bool:
    """Download the stored model + metadata. Returns True when restored."""
    model_folder = ensure_model_folder(service, root_folder_id)

    model_id = find_file(service, model_path.name, model_folder)
    if not model_id:
        log.info("No model is stored in Drive yet; one will be trained.")
        return False

    download_file(service, model_id, model_path)
    log.info("Restored %s from the Drive model store.", model_path.name)

    metadata_id = find_file(service, metadata_path.name, model_folder)
    if metadata_id:
        download_file(service, metadata_id, metadata_path)
        log.info("Restored %s from the Drive model store.", metadata_path.name)
    else:
        log.warning(
            "The Drive model store has %s but no %s; provenance will be "
            "rebuilt on the next retrain.",
            model_path.name,
            metadata_path.name,
        )
    return True


def persist_model_to_drive(
    service,
    root_folder_id: str,
    model_path: Path,
    metadata_path: Path,
) -> None:
    """Upload the current model + metadata to the Drive model store."""
    model_folder = ensure_model_folder(service, root_folder_id)
    if model_path.is_file():
        upload_file(service, model_path, model_folder)
    if metadata_path.is_file():
        upload_file(service, metadata_path, model_folder)
    log.info("Model and metadata persisted to the Drive model store.")


def upload_week_outputs(
    service,
    root_folder_id: str,
    season: int,
    week: int,
    paths: list,
) -> dict:
    """Upload this week's CSV/JSON outputs. Returns ``{name: file_id}``."""
    week_folder = ensure_week_folder(service, root_folder_id, season, week)
    uploaded = {}
    for path in paths:
        path = Path(path)
        if not path.is_file():
            log.warning("Skipping %s - it was not produced by this run.", path)
            continue
        uploaded[path.name] = upload_file(service, path, week_folder)
    log.info(
        "Drive upload success: %d file(s) into CSV Archive/%s/Week %02d.",
        len(uploaded),
        season,
        week,
    )
    return uploaded


def create_root_folder(service, name: str) -> str:
    """Create a new top-level folder (used by the one-time setup helper)."""
    created = execute(
        service.files().create(
            body={"name": name, "mimeType": FOLDER_MIME},
            fields="id",
            supportsAllDrives=True,
        ),
        f"Drive root folder creation for {name!r}",
    )
    return created["id"]


def find_file_by_name(
    service,
    name: str,
    parent_id: Optional[str] = None,
    mime_type: Optional[str] = None,
) -> Optional[str]:
    """Find a file the app can see, by name and optionally parent/type.

    Under the drive.file scope this can only ever match files this app
    created - which is exactly what makes it safe for the setup helper to
    reuse resources from an earlier authorization instead of duplicating
    them: the file<->app association is bound to the OAuth client, not to
    any individual token, so it survives re-consent.
    """
    clauses = [f"name = '{_escape(name)}'", "trashed = false"]
    if parent_id:
        clauses.append(f"'{parent_id}' in parents")
    if mime_type:
        clauses.append(f"mimeType = '{mime_type}'")
    files = execute(
        service.files().list(
            q=" and ".join(clauses),
            spaces="drive",
            fields="files(id, name)",
            pageSize=10,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ),
        f"Drive lookup for {name!r}",
    ).get("files", [])
    if len(files) > 1:
        log.warning(
            "Drive has %d files named %r visible to this app; using the "
            "first.",
            len(files),
            name,
        )
    return files[0]["id"] if files else None


def get_or_create_root_folder(service, name: str) -> tuple:
    """Reuse the app's existing top-level folder or create it.

    Returns ``(folder_id, created)``. Rerunning the setup helper must not
    produce 'NFL Weekly Model (2)' next to the original.
    """
    existing = find_file_by_name(service, name, mime_type=FOLDER_MIME)
    if existing:
        log.info("Reusing the existing Drive folder %r.", name)
        return existing, False
    return create_root_folder(service, name), True


def ensure_file_in_folder(service, file_id: str, folder_id: str) -> None:
    """Idempotently make ``folder_id`` the file's parent."""
    current = execute(
        service.files().get(
            fileId=file_id, fields="parents", supportsAllDrives=True
        ),
        "Drive parent lookup",
    )
    parents = current.get("parents", [])
    if folder_id in parents:
        return
    execute(
        service.files().update(
            fileId=file_id,
            addParents=folder_id,
            removeParents=",".join(parents),
            fields="id, parents",
            supportsAllDrives=True,
        ),
        "Drive move into the project folder",
    )
    log.info("Moved the file into the project folder.")


def write_json_tempfile(data: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=str)
    return path
