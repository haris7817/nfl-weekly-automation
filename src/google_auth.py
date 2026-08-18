"""Google credential construction for unattended runs.

The workflow has to run while the client's PC is off, so there is no
browser available to complete an OAuth prompt. Instead the client
authorises once (see ``scripts/google_auth_setup.py``) and the resulting
*refresh token* is stored as a GitHub Actions secret. Every run exchanges
that refresh token for a short-lived access token.

Nothing here reads or writes a credentials file on disk, and no secret is
ever logged - only whether each one is present.
"""

from __future__ import annotations

from .config import TOKEN_URI, Config
from .logging_utils import get_logger, redact

log = get_logger("google.auth")


class GoogleAuthError(RuntimeError):
    """Raised when Google credentials are missing or rejected."""


def build_credentials(config: Config):
    """Build refreshable OAuth credentials from configuration."""
    missing = [
        name
        for name, value in (
            ("GOOGLE_CLIENT_ID", config.google_client_id),
            ("GOOGLE_CLIENT_SECRET", config.google_client_secret),
            ("GOOGLE_REFRESH_TOKEN", config.google_refresh_token),
        )
        if not value
    ]
    if missing:
        raise GoogleAuthError(
            "Google credentials are not configured. Missing: "
            + ", ".join(missing)
            + ". Add them as GitHub Actions secrets (or to a local .env), or "
            "run with --skip-google."
        )

    try:
        from google.oauth2.credentials import Credentials
    except ImportError as exc:  # pragma: no cover
        raise GoogleAuthError(
            "google-auth is not installed. Run: pip install -r requirements.txt"
        ) from exc

    return Credentials(
        token=None,
        refresh_token=config.google_refresh_token,
        client_id=config.google_client_id,
        client_secret=config.google_client_secret,
        token_uri=TOKEN_URI,
        scopes=list(config.google_scopes),
    )


def _build_service(name: str, version: str, config: Config):
    from .net import enable_system_trust_store

    enable_system_trust_store()

    try:
        from googleapiclient.discovery import build
    except ImportError as exc:  # pragma: no cover
        raise GoogleAuthError(
            "google-api-python-client is not installed. "
            "Run: pip install -r requirements.txt"
        ) from exc

    credentials = build_credentials(config)
    try:
        service = build(
            name,
            version,
            credentials=credentials,
            cache_discovery=False,
        )
    except Exception as exc:
        raise GoogleAuthError(
            f"Could not create the Google {name} client "
            f"({type(exc).__name__}: {exc}). If this mentions invalid_grant, "
            "the refresh token has been revoked or expired and the client "
            "needs to re-authorise via scripts/google_auth_setup.py."
        ) from exc

    log.info("Google %s v%s client ready.", name, version)
    return service


def build_drive_service(config: Config):
    return _build_service("drive", "v3", config)


def build_sheets_service(config: Config):
    return _build_service("sheets", "v4", config)


def describe_targets(config: Config) -> str:
    """Log-safe description of which Google resources will be used."""
    return (
        f"drive_folder={redact(config.google_drive_folder_id)} "
        f"spreadsheet={redact(config.google_sheet_id)}"
    )
