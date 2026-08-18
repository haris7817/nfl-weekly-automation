"""EPA split tests - the client's math must not drift."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import TEAMS, make_pbp
from src.epa import (
    InsufficientDataError,
    build_epa_splits,
    last_n_splits,
    require_columns,
    team_game_epa,
)
from src.nfl_data import filter_plays


class TestPlayFiltering:
    def test_keeps_only_pass_and_run(self):
        df = make_pbp(weeks=2)
        df.loc[df.index[:10], "play_type"] = "punt"
        df.loc[df.index[10:20], "play_type"] = "kickoff"

        filtered = filter_plays(df)

        assert set(filtered["play_type"].unique()) == {"pass", "run"}

    def test_drops_rows_with_missing_epa(self):
        df = make_pbp(weeks=2)
        df.loc[df.index[:15], "epa"] = np.nan

        filtered = filter_plays(df)

        assert filtered["epa"].notna().all()
        assert len(filtered) == len(df) - 15

    def test_drops_postseason(self):
        reg = make_pbp(season=2025, weeks=2)
        post = make_pbp(season=2025, weeks=1, season_type="POST")

        filtered = filter_plays(pd.concat([reg, post], ignore_index=True))

        assert set(filtered["season_type"].unique()) == {"REG"}


class TestTeamGameEpa:
    def test_one_row_per_team_per_game(self, pbp):
        result = team_game_epa(pbp)

        assert not result.duplicated(subset=["team", "game_id"]).any()
        # 4 teams x 6 weeks = 24 team-games
        assert len(result) == 24

    def test_offence_matches_manual_mean(self, pbp):
        result = team_game_epa(pbp)
        game_id = pbp["game_id"].iloc[0]
        team = pbp[pbp["game_id"] == game_id]["posteam"].iloc[0]

        expected = pbp[
            (pbp["game_id"] == game_id)
            & (pbp["posteam"] == team)
            & (pbp["play_type"] == "pass")
        ]["epa"].mean()

        row = result[(result["game_id"] == game_id) & (result["team"] == team)]
        assert row["off_epa_pass"].iloc[0] == pytest.approx(expected)

    def test_defence_is_epa_allowed_not_generated(self, pbp):
        result = team_game_epa(pbp)
        game_id = pbp["game_id"].iloc[0]
        team = pbp[pbp["game_id"] == game_id]["defteam"].iloc[0]

        expected = pbp[
            (pbp["game_id"] == game_id)
            & (pbp["defteam"] == team)
            & (pbp["play_type"] == "run")
        ]["epa"].mean()

        row = result[(result["game_id"] == game_id) & (result["team"] == team)]
        assert row["def_epa_rush_allowed"].iloc[0] == pytest.approx(expected)

    def test_empty_input_raises(self):
        empty = make_pbp(weeks=1).iloc[0:0]
        with pytest.raises(InsufficientDataError):
            team_game_epa(empty)


class TestLastNSplits:
    def test_excludes_the_target_week_and_later(self, pbp):
        game_epa = team_game_epa(pbp)

        splits = last_n_splits(game_epa, through_week=4, n=5)

        # weeks 1-3 only
        for weeks in splits["weeks"]:
            assert all(int(w) < 4 for w in weeks.split(", "))

    def test_trailing_window_caps_at_n(self, pbp):
        game_epa = team_game_epa(pbp)

        splits = last_n_splits(game_epa, through_week=7, n=3)

        assert (splits["games_included"] == 3).all()
        for weeks in splits["weeks"]:
            assert weeks == "4, 5, 6"

    def test_uses_most_recent_games(self, pbp):
        game_epa = team_game_epa(pbp)

        splits = last_n_splits(game_epa, through_week=6, n=2)

        for weeks in splits["weeks"]:
            assert weeks == "4, 5"

    def test_net_epa_is_offence_minus_defence(self, pbp):
        game_epa = team_game_epa(pbp)

        splits = last_n_splits(game_epa, through_week=6, n=5)

        expected = (
            (splits["off_epa_pass"] + splits["off_epa_rush"])
            - (splits["def_epa_pass_allowed"] + splits["def_epa_rush_allowed"])
        ).round(3)
        pd.testing.assert_series_equal(
            splits["net_epa_play"], expected, check_names=False
        )

    def test_sorted_by_net_epa_descending(self, pbp):
        game_epa = team_game_epa(pbp)

        splits = last_n_splits(game_epa, through_week=6, n=5)

        assert splits["net_epa_play"].is_monotonic_decreasing

    def test_component_means_rounded_to_three_decimals(self, pbp):
        game_epa = team_game_epa(pbp)

        splits = last_n_splits(game_epa, through_week=6, n=5)

        for column in ("off_epa_pass", "def_epa_rush_allowed"):
            assert (splits[column].round(3) == splits[column]).all()

    def test_week_one_returns_empty_not_an_error(self, pbp):
        game_epa = team_game_epa(pbp)

        splits = last_n_splits(game_epa, through_week=1, n=5)

        assert splits.empty
        assert "net_epa_play" in splits.columns

    def test_teams_are_unique(self, pbp):
        game_epa = team_game_epa(pbp)

        splits = last_n_splits(game_epa, through_week=6, n=5)

        assert splits["team"].is_unique
        assert set(splits["team"]) == set(TEAMS)


class TestBuildEpaSplits:
    def test_adds_metadata_columns(self, pbp):
        result = build_epa_splits(
            pbp, season=2025, week=6, n=5, run_timestamp_utc="2026-01-01T00:00:00Z"
        )

        assert (result["season"] == 2025).all()
        assert (result["analysis_week"] == 6).all()
        assert (result["run_timestamp_utc"] == "2026-01-01T00:00:00Z").all()

    def test_column_order_is_stable(self, pbp):
        result = build_epa_splits(pbp, season=2025, week=6)

        assert list(result.columns) == [
            "season",
            "analysis_week",
            "run_timestamp_utc",
            "team",
            "games_included",
            "weeks",
            "off_epa_pass",
            "off_epa_rush",
            "def_epa_pass_allowed",
            "def_epa_rush_allowed",
            "net_epa_play",
        ]

    def test_strict_mode_raises_on_week_one(self, pbp):
        with pytest.raises(InsufficientDataError):
            build_epa_splits(pbp, season=2025, week=1, strict=True)

    def test_non_strict_mode_returns_empty_frame_on_week_one(self, pbp):
        result = build_epa_splits(pbp, season=2025, week=1, strict=False)

        assert result.empty
        assert "team" in result.columns

    def test_weeks_column_has_no_float_artifacts(self, pbp):
        result = build_epa_splits(pbp, season=2025, week=6)

        for weeks in result["weeks"]:
            assert "." not in weeks


class TestRequireColumns:
    def test_names_every_missing_column(self):
        df = pd.DataFrame({"a": [1]})

        with pytest.raises(RuntimeError) as exc:
            require_columns(df, ["a", "b", "c"], "test source")

        assert "b" in str(exc.value)
        assert "c" in str(exc.value)

    def test_passes_when_all_present(self):
        df = pd.DataFrame({"a": [1], "b": [2]})

        require_columns(df, ["a", "b"], "test source")
