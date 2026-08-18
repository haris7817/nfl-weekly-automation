"""Environment-driven configuration.

Analytics defaults live here so that the client's agreed behaviour
(trailing 5 games, minimum 3 games of history, LinearRegression) is stated
in exactly one place. Infrastructure values (Google IDs, OAuth secrets)
come from environment variables / GitHub Actions secrets and are never
written to disk or logged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

try:  # python-dotenv is optional at runtime (GitHub Actions injects real env vars)
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised only when dotenv is absent
    def load_dotenv(*_args, **_kwargs):  # type: ignore[misc]
        return False

from .paths import BASE_DIR

# Client-agreed analytics defaults. Changing any of these is a scope change
# (implementation guide section 43) and must be approved by the client.
DEFAULT_TRAILING_GAMES_N = 5
DEFAULT_MIN_TRAILING_GAMES = 3
DEFAULT_TRAINING_START_SEASON = 2021
DEFAULT_RETRAIN_EVERY_WEEKS = 4

TOKEN_URI = "https://oauth2.googleapis.com/token"

# The single scope the production workflow requests (implementation guide
# 22.4). `drive.file` is non-sensitive and limits the app to files it created
# itself - and because the Sheets API accepts it for every method this
# project calls (create, get, batchUpdate, values.*) on app-created
# spreadsheets, the separate sensitive `spreadsheets` scope is unnecessary.
# This is why the setup helper must create the Drive folder and the Sheet
# through the API: hand-made files are invisible under drive.file.
DEFAULT_GOOGLE_SCOPES = (
    "https://www.googleapis.com/auth/drive.file",
)


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name}={raw!r} is not a valid integer."
        ) from exc


def _get_str(name: str) -> Optional[str]:
    raw = os.getenv(name)
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


@dataclass
class Config:
    """Runtime configuration for a single weekly run."""

    # --- analytics -----------------------------------------------------
    trailing_games_n: int = DEFAULT_TRAILING_GAMES_N
    min_trailing_games: int = DEFAULT_MIN_TRAILING_GAMES
    training_start_season: int = DEFAULT_TRAINING_START_SEASON
    retrain_every_weeks: int = DEFAULT_RETRAIN_EVERY_WEEKS
    include_completed_week_games: bool = False

    # --- logging -------------------------------------------------------
    log_level: str = "INFO"

    # --- Google --------------------------------------------------------
    google_client_id: Optional[str] = None
    google_client_secret: Optional[str] = None
    google_refresh_token: Optional[str] = None
    google_drive_folder_id: Optional[str] = None
    google_sheet_id: Optional[str] = None
    google_scopes: tuple = field(default=DEFAULT_GOOGLE_SCOPES)

    # --- run switches (set from the CLI, not the environment) ----------
    skip_google: bool = False
    dry_run: bool = False
    force_retrain: bool = False

    @classmethod
    def from_env(cls, load_env_file: bool = True) -> "Config":
        if load_env_file:
            load_dotenv(BASE_DIR / ".env")

        scopes_raw = _get_str("GOOGLE_SCOPES")
        scopes = (
            tuple(s for s in (part.strip() for part in scopes_raw.split(",")) if s)
            if scopes_raw
            else DEFAULT_GOOGLE_SCOPES
        )

        return cls(
            trailing_games_n=_get_int("TRAILING_GAMES_N", DEFAULT_TRAILING_GAMES_N),
            min_trailing_games=_get_int(
                "MIN_TRAILING_GAMES", DEFAULT_MIN_TRAILING_GAMES
            ),
            training_start_season=_get_int(
                "TRAINING_START_SEASON", DEFAULT_TRAINING_START_SEASON
            ),
            retrain_every_weeks=_get_int(
                "RETRAIN_EVERY_WEEKS", DEFAULT_RETRAIN_EVERY_WEEKS
            ),
            include_completed_week_games=_get_bool(
                "INCLUDE_COMPLETED_WEEK_GAMES", False
            ),
            log_level=(_get_str("LOG_LEVEL") or "INFO").upper(),
            google_client_id=_get_str("GOOGLE_CLIENT_ID"),
            google_client_secret=_get_str("GOOGLE_CLIENT_SECRET"),
            google_refresh_token=_get_str("GOOGLE_REFRESH_TOKEN"),
            google_drive_folder_id=_get_str("GOOGLE_DRIVE_FOLDER_ID"),
            google_sheet_id=_get_str("GOOGLE_SHEET_ID"),
            google_scopes=scopes,
        )

    # ------------------------------------------------------------------
    @property
    def has_google_credentials(self) -> bool:
        return all(
            (
                self.google_client_id,
                self.google_client_secret,
                self.google_refresh_token,
            )
        )

    def missing_google_settings(self) -> list:
        """Names of the Google settings that are absent (never their values)."""
        required = {
            "GOOGLE_CLIENT_ID": self.google_client_id,
            "GOOGLE_CLIENT_SECRET": self.google_client_secret,
            "GOOGLE_REFRESH_TOKEN": self.google_refresh_token,
            "GOOGLE_DRIVE_FOLDER_ID": self.google_drive_folder_id,
            "GOOGLE_SHEET_ID": self.google_sheet_id,
        }
        return sorted(name for name, value in required.items() if not value)

    def describe(self) -> str:
        """Human-readable summary that deliberately contains no secrets."""
        return (
            "config("
            f"trailing_games_n={self.trailing_games_n}, "
            f"min_trailing_games={self.min_trailing_games}, "
            f"training_start_season={self.training_start_season}, "
            f"retrain_every_weeks={self.retrain_every_weeks}, "
            f"include_completed_week_games={self.include_completed_week_games}, "
            f"google_credentials_present={self.has_google_credentials}, "
            f"drive_folder_configured={bool(self.google_drive_folder_id)}, "
            f"sheet_configured={bool(self.google_sheet_id)}"
            ")"
        )
