"""Nothing broken may reach the client's Drive or Sheet."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fair_line import FAIR_LINE_COLUMNS, STATUS_INSUFFICIENT, STATUS_PROJECTED
from src.validation import (
    ValidationError,
    validate_epa_splits,
    validate_fair_lines,
)


def epa_frame(teams=("AAA", "BBB", "CCC"), season=2025, week=6):
    return pd.DataFrame(
        {
            "season": [season] * len(teams),
            "analysis_week": [week] * len(teams),
            "run_timestamp_utc": ["2026-01-01T00:00:00Z"] * len(teams),
            "team": list(teams),
            "games_included": [5] * len(teams),
            "weeks": ["1, 2, 3, 4, 5"] * len(teams),
            "off_epa_pass": [0.1] * len(teams),
            "off_epa_rush": [0.0] * len(teams),
            "def_epa_pass_allowed": [0.0] * len(teams),
            "def_epa_rush_allowed": [0.0] * len(teams),
            "net_epa_play": [0.1 - 0.01 * i for i in range(len(teams))],
        }
    )


def fair_frame(rows=None, season=2025, week=6):
    # NB: `rows=[]` must stay empty, so test explicitly against None.
    if rows is None:
        rows = [("BBB", "AAA", -3.0, STATUS_PROJECTED)]
    records = []
    for away, home, spread, status in rows:
        record = {column: "" for column in FAIR_LINE_COLUMNS}
        record.update(
            {
                "run_timestamp_utc": "2026-01-01T00:00:00Z",
                "season": season,
                "week": week,
                "away_team": away,
                "home_team": home,
                "fair_spread_home": spread,
                "predicted_home_margin": None if spread is None else -spread,
                "status": status,
                "model_n": 5,
            }
        )
        records.append(record)
    return pd.DataFrame(records, columns=list(FAIR_LINE_COLUMNS))


FULL_SLATE = tuple(f"T{i:02d}" for i in range(32))


class TestEpaValidation:
    def test_accepts_a_good_frame(self):
        assert validate_epa_splits(epa_frame(teams=FULL_SLATE), 2025, 6) == []

    def test_empty_is_fatal_by_default(self):
        with pytest.raises(ValidationError):
            validate_epa_splits(epa_frame(teams=[]), 2025, 6)

    def test_empty_is_allowed_when_permitted(self):
        warnings = validate_epa_splits(epa_frame(teams=[]), 2025, 1, allow_empty=True)

        assert any("week 1" in w for w in warnings)

    def test_duplicate_teams_are_fatal(self):
        df = epa_frame(teams=("AAA", "AAA", "BBB"))

        with pytest.raises(ValidationError) as exc:
            validate_epa_splits(df, 2025, 6)

        assert "duplicate" in str(exc.value).lower()

    def test_missing_team_name_is_fatal(self):
        df = epa_frame()
        df.loc[0, "team"] = None

        with pytest.raises(ValidationError):
            validate_epa_splits(df, 2025, 6)

    def test_season_mismatch_is_fatal(self):
        with pytest.raises(ValidationError) as exc:
            validate_epa_splits(epa_frame(season=2024), 2025, 6)

        assert "2024" in str(exc.value)

    def test_all_blank_net_epa_is_fatal(self):
        df = epa_frame()
        df["net_epa_play"] = np.nan

        with pytest.raises(ValidationError):
            validate_epa_splits(df, 2025, 6)

    def test_small_slate_only_warns(self):
        warnings = validate_epa_splits(epa_frame(), 2025, 6)

        assert any("Only 3 teams" in w for w in warnings)


class TestFairLineValidation:
    def test_accepts_a_good_frame(self):
        assert validate_fair_lines(fair_frame(), 2025, 6) == []

    def test_empty_is_fatal(self):
        with pytest.raises(ValidationError):
            validate_fair_lines(fair_frame(rows=[]), 2025, 6)

    def test_same_team_both_sides_is_fatal(self):
        df = fair_frame(rows=[("AAA", "AAA", -3.0, STATUS_PROJECTED)])

        with pytest.raises(ValidationError) as exc:
            validate_fair_lines(df, 2025, 6)

        assert "same team" in str(exc.value)

    def test_duplicate_matchups_are_fatal(self):
        df = fair_frame(
            rows=[
                ("BBB", "AAA", -3.0, STATUS_PROJECTED),
                ("BBB", "AAA", -4.0, STATUS_PROJECTED),
            ]
        )

        with pytest.raises(ValidationError) as exc:
            validate_fair_lines(df, 2025, 6)

        assert "Duplicate" in str(exc.value)

    def test_week_mismatch_is_fatal(self):
        with pytest.raises(ValidationError) as exc:
            validate_fair_lines(fair_frame(week=7), 2025, 6)

        assert "week" in str(exc.value).lower()

    def test_projected_row_without_a_spread_is_fatal(self):
        df = fair_frame(rows=[("BBB", "AAA", None, STATUS_PROJECTED)])

        with pytest.raises(ValidationError) as exc:
            validate_fair_lines(df, 2025, 6)

        assert "blank" in str(exc.value)

    def test_missing_team_code_is_fatal(self):
        df = fair_frame()
        df.loc[0, "home_team"] = None

        with pytest.raises(ValidationError):
            validate_fair_lines(df, 2025, 6)

    def test_insufficient_history_rows_only_warn(self):
        df = fair_frame(rows=[("BBB", "AAA", None, STATUS_INSUFFICIENT)])

        warnings = validate_fair_lines(df, 2025, 6)

        assert any("insufficient trailing history" in w for w in warnings)

    def test_extreme_spread_warns_but_publishes(self):
        df = fair_frame(rows=[("BBB", "AAA", -45.0, STATUS_PROJECTED)])

        warnings = validate_fair_lines(df, 2025, 6)

        assert any("28 points" in w or "+/-28" in w for w in warnings)

    def test_row_count_mismatch_warns(self):
        warnings = validate_fair_lines(fair_frame(), 2025, 6, expected_games=16)

        assert any("scheduled games" in w for w in warnings)
