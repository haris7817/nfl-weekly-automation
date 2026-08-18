"""Shared fixtures.

The unit tests run entirely on synthetic frames so the suite is fast,
deterministic and works with no network access. Tests that genuinely need
nflverse are marked ``integration`` and skipped unless
``RUN_INTEGRATION_TESTS=1`` is set.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TEAMS = ["AAA", "BBB", "CCC", "DDD"]


def make_pbp(
    season: int = 2025,
    weeks: int = 6,
    teams=None,
    plays_per_side: int = 8,
    seed: int = 7,
    season_type: str = "REG",
) -> pd.DataFrame:
    """Build a synthetic play-by-play frame.

    Each week every team plays exactly one game (pairs rotate), and each
    offence runs ``plays_per_side`` pass plays and ``plays_per_side`` run
    plays with reproducible EPA values.
    """
    teams = list(teams or TEAMS)
    rng = np.random.default_rng(seed)
    rows = []

    # Circle method: team 0 stays put, the rest rotate. Guarantees every
    # team plays exactly one game per week.
    rotation = list(teams)
    for week in range(1, weeks + 1):
        size = len(rotation)
        pairs = [(rotation[i], rotation[size - 1 - i]) for i in range(size // 2)]
        rotation = [rotation[0], rotation[-1]] + rotation[1:-1]
        for away, home in pairs:
            game_id = f"{season}_{week:02d}_{away}_{home}"
            home_score = int(rng.integers(10, 35))
            away_score = int(rng.integers(10, 35))
            for offence, defence in ((home, away), (away, home)):
                for play_type in ("pass", "run"):
                    for _ in range(plays_per_side):
                        rows.append(
                            {
                                "game_id": game_id,
                                "season": season,
                                "season_type": season_type,
                                "week": week,
                                "play_type": play_type,
                                "epa": float(rng.normal(0.05, 0.5)),
                                "posteam": offence,
                                "defteam": defence,
                                "home_team": home,
                                "away_team": away,
                                "home_score": home_score,
                                "away_score": away_score,
                            }
                        )
    return pd.DataFrame(rows)


def make_schedule(
    season: int = 2026,
    weeks: int = 4,
    teams=None,
    completed_through_week: int = 0,
    completed_games_in_current_week: int = 0,
    start_date: str = "2026-09-10",
) -> pd.DataFrame:
    """Build a synthetic schedule frame matching the nflverse column set."""
    teams = list(teams or TEAMS)
    base = pd.Timestamp(start_date)
    rows = []

    rotation = list(teams)
    for week in range(1, weeks + 1):
        size = len(rotation)
        pairs = [(rotation[i], rotation[size - 1 - i]) for i in range(size // 2)]
        rotation = [rotation[0], rotation[-1]] + rotation[1:-1]
        for index, (away, home) in enumerate(pairs):
            # game 0 = Thursday, game 1 = Sunday (three days later)
            offset = 0 if index == 0 else 3
            gameday = (base + pd.Timedelta(days=7 * (week - 1) + offset)).strftime(
                "%Y-%m-%d"
            )
            weekday = "Thursday" if index == 0 else "Sunday"
            finished = week <= completed_through_week or (
                week == completed_through_week + 1
                and index < completed_games_in_current_week
            )
            rows.append(
                {
                    "game_id": f"{season}_{week:02d}_{away}_{home}",
                    "season": season,
                    "game_type": "REG",
                    "week": week,
                    "gameday": gameday,
                    "weekday": weekday,
                    "gametime": "20:15" if index == 0 else "13:00",
                    "away_team": away,
                    "home_team": home,
                    "away_score": 20.0 if finished else np.nan,
                    "home_score": 24.0 if finished else np.nan,
                    "result": 4.0 if finished else np.nan,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def pbp():
    return make_pbp()


@pytest.fixture
def schedule():
    return make_schedule()


@pytest.fixture
def utc_now():
    return datetime(2026, 9, 11, 1, 0, tzinfo=timezone.utc)


@pytest.fixture
def tmp_model_dir(tmp_path, monkeypatch):
    """Redirect the model paths at module level for isolation."""
    from src import model_manager, paths

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_path = model_dir / "fair_line_model.pkl"
    tmp_model_path = model_dir / "fair_line_model.tmp.pkl"
    metadata_path = model_dir / "model_metadata.json"

    monkeypatch.setattr(paths, "MODEL_DIR", model_dir)
    monkeypatch.setattr(paths, "MODEL_PATH", model_path)
    monkeypatch.setattr(paths, "MODEL_TMP_PATH", tmp_model_path)
    monkeypatch.setattr(paths, "MODEL_METADATA_PATH", metadata_path)
    monkeypatch.setattr(model_manager, "MODEL_DIR", model_dir)
    monkeypatch.setattr(model_manager, "MODEL_PATH", model_path)
    monkeypatch.setattr(model_manager, "MODEL_TMP_PATH", tmp_model_path)
    monkeypatch.setattr(model_manager, "MODEL_METADATA_PATH", metadata_path)

    return {
        "dir": model_dir,
        "model": model_path,
        "tmp": tmp_model_path,
        "metadata": metadata_path,
    }


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: requires live nflverse network access"
    )
