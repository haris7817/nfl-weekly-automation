"""Project-relative paths.

Everything is derived from the repository root rather than the current
working directory, so the code behaves identically whether it is started
from an IDE, a Windows Scheduled Task, or a GitHub Actions runner
(implementation guide section 5.4).
"""

from __future__ import annotations

from pathlib import Path

# src/paths.py -> src/ -> project root
BASE_DIR: Path = Path(__file__).resolve().parent.parent

MODEL_DIR: Path = BASE_DIR / "models"
OUTPUT_DIR: Path = BASE_DIR / "outputs"
LOG_DIR: Path = BASE_DIR / "logs"
ORIGINAL_DIR: Path = BASE_DIR / "original"

MODEL_PATH: Path = MODEL_DIR / "fair_line_model.pkl"
MODEL_TMP_PATH: Path = MODEL_DIR / "fair_line_model.tmp.pkl"
MODEL_METADATA_PATH: Path = MODEL_DIR / "model_metadata.json"

LOG_FILE: Path = LOG_DIR / "nfl_weekly.log"


def ensure_dirs() -> None:
    """Create the writable working directories if they do not exist."""
    for directory in (MODEL_DIR, OUTPUT_DIR, LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def week_output_dir(season: int, week: int) -> Path:
    """Return ``outputs/<season>/week_<NN>/`` and make sure it exists."""
    directory = OUTPUT_DIR / str(season) / f"week_{int(week):02d}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory
