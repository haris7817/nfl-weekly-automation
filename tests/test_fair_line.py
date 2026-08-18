"""Fair-line model tests - the client's regression must not drift."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_pbp
from src.fair_line import (
    FAIR_LINE_COLUMNS,
    MIN_TRAILING_GAMES,
    STATUS_INSUFFICIENT,
    STATUS_PROJECTED,
    TrailingEpaIndex,
    TrainingError,
    build_training_set,
    fit_model,
    format_projection_table,
    game_scores,
    project_matchups,
    team_game_epa,
    trailing_net_epa,
)
from src.schedule import PROJECTION_STATUS_COMPLETED


class FakeModel:
    """Stand-in with the client's coefficient shape."""

    def __init__(self, intercept=2.0, coefficient=20.0):
        self.intercept_ = intercept
        self.coef_ = np.array([coefficient])

    def predict(self, X):
        return self.intercept_ + self.coef_[0] * np.asarray(X).ravel()


@pytest.fixture
def game_epa():
    return team_game_epa(make_pbp(season=2025, weeks=8))


class TestTrailingNetEpa:
    def test_requires_a_minimum_of_three_games(self, game_epa):
        assert trailing_net_epa(game_epa, "AAA", 2025, 3, n=5) is None
        assert trailing_net_epa(game_epa, "AAA", 2025, 4, n=5) is not None

    def test_uses_only_games_before_the_target_week(self, game_epa):
        manual = game_epa[
            (game_epa["team"] == "AAA") & (game_epa["week"] < 6)
        ].sort_values("week").tail(5)["net_epa"].mean()

        assert trailing_net_epa(game_epa, "AAA", 2025, 6, n=5) == pytest.approx(manual)

    def test_window_is_capped_at_n(self, game_epa):
        capped = game_epa[
            (game_epa["team"] == "AAA") & (game_epa["week"] < 8)
        ].sort_values("week").tail(3)["net_epa"].mean()

        assert trailing_net_epa(game_epa, "AAA", 2025, 8, n=3) == pytest.approx(capped)

    def test_unknown_team_returns_none(self, game_epa):
        assert trailing_net_epa(game_epa, "ZZZ", 2025, 6, n=5) is None

    def test_previous_season_history_is_used_early_in_a_season(self):
        prior = make_pbp(season=2024, weeks=6)
        current = make_pbp(season=2025, weeks=1, seed=11)
        combined = team_game_epa(pd.concat([prior, current], ignore_index=True))

        # Week 1 of 2025 has no same-season history, but 2024 supplies it.
        assert trailing_net_epa(combined, "AAA", 2025, 1, n=5) is not None


class TestTrailingEpaIndex:
    """The fast index must be arithmetically identical to the original."""

    def test_matches_the_naive_implementation_everywhere(self, game_epa):
        index = TrailingEpaIndex(game_epa)
        teams = sorted(game_epa["team"].dropna().unique())

        for week in range(1, 10):
            for team in teams:
                for n in (3, 5, 7):
                    expected = trailing_net_epa(game_epa, team, 2025, week, n)
                    actual = index.get(team, 2025, week, n)
                    if expected is None:
                        assert actual is None
                    else:
                        assert actual == pytest.approx(expected, abs=1e-12)

    def test_matches_across_a_season_boundary(self):
        combined = team_game_epa(
            pd.concat(
                [make_pbp(season=2024, weeks=6), make_pbp(season=2025, weeks=3, seed=3)],
                ignore_index=True,
            )
        )
        index = TrailingEpaIndex(combined)

        for team in sorted(combined["team"].unique()):
            for week in (1, 2, 3):
                expected = trailing_net_epa(combined, team, 2025, week, 5)
                actual = index.get(team, 2025, week, 5)
                if expected is None:
                    assert actual is None
                else:
                    assert actual == pytest.approx(expected, abs=1e-12)


class TestGameScores:
    def test_one_row_per_game(self):
        pbp = make_pbp(weeks=4)

        games = game_scores(pbp)

        assert games["game_id"].is_unique
        assert len(games) == 8  # 2 games per week x 4 weeks


class TestBuildTrainingSet:
    def test_produces_feature_and_target_columns(self):
        data = build_training_set(make_pbp(weeks=8))

        assert {"epa_diff", "actual_margin"}.issubset(data.columns)
        assert len(data) > 0

    def test_target_is_home_minus_away(self):
        pbp = make_pbp(weeks=8)

        data = build_training_set(pbp)

        row = data.iloc[0]
        game = pbp[
            (pbp["home_team"] == row["home_team"])
            & (pbp["away_team"] == row["away_team"])
            & (pbp["week"] == row["week"])
        ].iloc[0]
        assert row["actual_margin"] == game["home_score"] - game["away_score"]

    def test_leakage_guard_excludes_the_target_week_and_later(self):
        pbp = make_pbp(weeks=10)

        data = build_training_set(pbp, target_season=2025, target_week=6)

        assert (data["week"] < 6).all()

    def test_without_a_guard_all_completed_games_are_used(self):
        pbp = make_pbp(weeks=10)

        data = build_training_set(pbp)

        assert data["week"].max() == 10

    def test_earlier_seasons_are_unrestricted_by_the_guard(self):
        pbp = pd.concat(
            [make_pbp(season=2024, weeks=10), make_pbp(season=2025, weeks=10, seed=5)],
            ignore_index=True,
        )

        data = build_training_set(pbp, target_season=2025, target_week=4)

        assert data[data["season"] == 2024]["week"].max() == 10
        assert data[data["season"] == 2025]["week"].max() < 4


class TestFitModel:
    def test_returns_the_client_metrics(self):
        data = build_training_set(make_pbp(weeks=18, teams=[f"T{i:02d}" for i in range(16)]))

        fit = fit_model(data)

        assert {"coefficient", "intercept", "mae", "r2", "model"}.issubset(fit)
        assert np.isfinite(fit["mae"])

    def test_empty_training_set_raises(self):
        with pytest.raises(TrainingError):
            fit_model(pd.DataFrame(columns=["epa_diff", "actual_margin"]))

    def test_tiny_training_set_raises(self):
        data = pd.DataFrame(
            {"epa_diff": np.linspace(-1, 1, 5), "actual_margin": np.arange(5)}
        )

        with pytest.raises(TrainingError):
            fit_model(data)

    def test_recovers_a_known_linear_relationship(self):
        rng = np.random.default_rng(0)
        diffs = rng.normal(0, 0.3, 400)
        data = pd.DataFrame(
            {"epa_diff": diffs, "actual_margin": 3.0 + 20.0 * diffs}
        )

        fit = fit_model(data)

        assert fit["coefficient"] == pytest.approx(20.0, abs=1e-6)
        assert fit["intercept"] == pytest.approx(3.0, abs=1e-6)


class TestProjectMatchups:
    @pytest.fixture
    def pbp(self):
        return make_pbp(season=2025, weeks=8)

    def test_sign_convention_home_favoured_is_negative(self, pbp):
        model = FakeModel(intercept=2.0, coefficient=20.0)
        matchups = [{"away_team": "BBB", "home_team": "AAA", "status": None}]

        df = project_matchups(pbp, matchups, model, 2025, 8, n=5)

        row = df.iloc[0]
        assert row["fair_spread_home"] == pytest.approx(-row["predicted_home_margin"])
        if row["predicted_home_margin"] > 0:
            assert row["fair_spread_home"] < 0

    def test_prediction_matches_the_formula(self, pbp):
        model = FakeModel(intercept=2.5, coefficient=18.0)
        matchups = [{"away_team": "BBB", "home_team": "AAA", "status": None}]

        df = project_matchups(pbp, matchups, model, 2025, 8, n=5)

        row = df.iloc[0]
        expected = 2.5 + 18.0 * row["epa_diff"]
        # Output columns are rounded for readability: epa_diff to 4dp and
        # predicted_home_margin to 2dp, so the reconstructed value can differ
        # by up to (5e-5 * coefficient) + 5e-3.
        assert row["predicted_home_margin"] == pytest.approx(expected, abs=0.01)

    def test_prediction_is_exact_before_output_rounding(self, pbp):
        """Guard the formula itself, independent of display rounding."""
        model = FakeModel(intercept=2.5, coefficient=18.0)
        index = TrailingEpaIndex(team_game_epa(pbp))
        home = index.get("AAA", 2025, 8, 5)
        away = index.get("BBB", 2025, 8, 5)

        df = project_matchups(
            pbp,
            [{"away_team": "BBB", "home_team": "AAA", "status": None}],
            model,
            2025,
            8,
            n=5,
        )

        exact = 2.5 + 18.0 * (home - away)
        assert df.iloc[0]["predicted_home_margin"] == pytest.approx(
            round(exact, 2), abs=1e-9
        )
        assert df.iloc[0]["fair_spread_home"] == pytest.approx(
            round(-exact, 2), abs=1e-9
        )

    def test_epa_diff_is_home_minus_away(self, pbp):
        model = FakeModel()
        matchups = [{"away_team": "BBB", "home_team": "AAA", "status": None}]

        df = project_matchups(pbp, matchups, model, 2025, 8, n=5)

        row = df.iloc[0]
        assert row["epa_diff"] == pytest.approx(
            row["home_trailing_net_epa"] - row["away_trailing_net_epa"], abs=1e-6
        )

    def test_insufficient_history_leaves_numbers_blank(self, pbp):
        model = FakeModel()
        matchups = [{"away_team": "ZZZ", "home_team": "AAA", "status": None}]

        df = project_matchups(pbp, matchups, model, 2025, 8, n=5)

        row = df.iloc[0]
        assert row["status"] == STATUS_INSUFFICIENT
        assert pd.isna(row["fair_spread_home"])
        assert pd.isna(row["predicted_home_margin"])
        assert "ZZZ" in row["notes"]

    def test_completed_games_are_not_projected(self, pbp):
        model = FakeModel()
        matchups = [
            {
                "away_team": "BBB",
                "home_team": "AAA",
                "status": PROJECTION_STATUS_COMPLETED,
            }
        ]

        df = project_matchups(pbp, matchups, model, 2025, 8, n=5)

        row = df.iloc[0]
        assert row["status"] == PROJECTION_STATUS_COMPLETED
        assert pd.isna(row["fair_spread_home"])

    def test_output_columns_are_stable(self, pbp):
        model = FakeModel()
        matchups = [{"away_team": "BBB", "home_team": "AAA", "status": None}]

        df = project_matchups(pbp, matchups, model, 2025, 8, n=5)

        assert list(df.columns) == list(FAIR_LINE_COLUMNS)

    def test_metadata_is_carried_through(self, pbp):
        model = FakeModel()
        matchups = [{"away_team": "BBB", "home_team": "AAA", "status": None}]

        df = project_matchups(
            pbp, matchups, model, 2025, 8, n=5,
            model_version=7, model_trained_at="2026-01-01T00:00:00Z",
            run_timestamp_utc="2026-02-02T00:00:00Z",
        )

        row = df.iloc[0]
        assert row["model_version"] == 7
        assert row["model_trained_at"] == "2026-01-01T00:00:00Z"
        assert row["run_timestamp_utc"] == "2026-02-02T00:00:00Z"
        assert row["model_n"] == 5

    def test_every_scheduled_game_appears_once(self, pbp):
        model = FakeModel()
        matchups = [
            {"away_team": "BBB", "home_team": "AAA", "status": None},
            {"away_team": "DDD", "home_team": "CCC", "status": None},
        ]

        df = project_matchups(pbp, matchups, model, 2025, 8, n=5)

        assert len(df) == 2
        assert not df.duplicated(subset=["away_team", "home_team"]).any()

    def test_table_renders_without_error_for_mixed_statuses(self, pbp):
        model = FakeModel()
        matchups = [
            {"away_team": "BBB", "home_team": "AAA", "status": None},
            {"away_team": "ZZZ", "home_team": "CCC", "status": None},
        ]

        text = format_projection_table(
            project_matchups(pbp, matchups, model, 2025, 8, n=5)
        )

        assert "Fair Spread (home)" in text
        assert "Negative = home team favoured" in text
