"""End-to-end runner tests with nflverse mocked out.

These drive ``weekly_runner.main`` through the whole pipeline - detection,
EPA, retraining, projection, validation, file writing - using synthetic
data, so the workflow's wiring is covered without touching the network.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

import weekly_runner
from conftest import make_pbp, make_schedule
from src import nfl_data, schedule as schedule_module
from src.config import Config

TEAMS = [f"T{i:02d}" for i in range(16)]


@pytest.fixture
def fake_nflverse(monkeypatch, tmp_path):
    """Replace every nflverse call with deterministic synthetic data."""
    history = pd.concat(
        [
            make_pbp(season=2024, weeks=18, teams=TEAMS, seed=1),
            make_pbp(season=2025, weeks=18, teams=TEAMS, seed=2),
        ],
        ignore_index=True,
    )
    sched = make_schedule(
        season=2025, weeks=18, teams=TEAMS, start_date="2025-09-04"
    )

    def fake_load_pbp_available(seasons, apply_filter=True, columns=None):
        wanted = [seasons] if isinstance(seasons, int) else list(seasons)
        usable = [s for s in wanted if s in (2024, 2025)]
        subset = history[history["season"].isin(usable)].copy()
        return subset, usable

    def fake_load_schedules(seasons):
        wanted = [seasons] if isinstance(seasons, int) else list(seasons)
        subset = sched[sched["season"].isin(wanted)].copy()
        if subset.empty:
            raise nfl_data.NflDataError(f"no schedule for {wanted}")
        return subset

    monkeypatch.setattr(weekly_runner, "load_pbp_available", fake_load_pbp_available)
    monkeypatch.setattr(weekly_runner, "load_schedules", fake_load_schedules)
    monkeypatch.setattr(weekly_runner, "load_team_codes", lambda: set(TEAMS))
    return {"pbp": history, "schedule": sched}


@pytest.fixture
def isolated_outputs(monkeypatch, tmp_path):
    """Point outputs and model storage at a temp directory."""
    from src import model_manager, paths

    out = tmp_path / "outputs"
    models = tmp_path / "models"
    out.mkdir()
    models.mkdir()

    monkeypatch.setattr(paths, "OUTPUT_DIR", out)
    monkeypatch.setattr(paths, "MODEL_DIR", models)
    monkeypatch.setattr(paths, "MODEL_PATH", models / "fair_line_model.pkl")
    monkeypatch.setattr(paths, "MODEL_TMP_PATH", models / "fair_line_model.tmp.pkl")
    monkeypatch.setattr(paths, "MODEL_METADATA_PATH", models / "model_metadata.json")
    monkeypatch.setattr(model_manager, "MODEL_PATH", models / "fair_line_model.pkl")
    monkeypatch.setattr(
        model_manager, "MODEL_TMP_PATH", models / "fair_line_model.tmp.pkl"
    )
    monkeypatch.setattr(
        model_manager, "MODEL_METADATA_PATH", models / "model_metadata.json"
    )
    monkeypatch.setattr(weekly_runner, "MODEL_PATH", models / "fair_line_model.pkl")
    monkeypatch.setattr(
        weekly_runner, "MODEL_METADATA_PATH", models / "model_metadata.json"
    )

    def fake_week_output_dir(season, week):
        directory = out / str(season) / f"week_{int(week):02d}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    monkeypatch.setattr(weekly_runner, "week_output_dir", fake_week_output_dir)
    monkeypatch.setattr(weekly_runner, "ensure_dirs", lambda: None)
    return {"outputs": out, "models": models}


@pytest.fixture
def clean_env(monkeypatch):
    for name in (
        "GOOGLE_CLIENT_ID",
        "GOOGLE_CLIENT_SECRET",
        "GOOGLE_REFRESH_TOKEN",
        "GOOGLE_DRIVE_FOLDER_ID",
        "GOOGLE_SHEET_ID",
        "TRAILING_GAMES_N",
        "RETRAIN_EVERY_WEEKS",
        "TRAINING_START_SEASON",
        "INCLUDE_COMPLETED_WEEK_GAMES",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(Config, "from_env", classmethod(lambda cls, **kw: cls()))


@pytest.mark.usefixtures("fake_nflverse", "isolated_outputs", "clean_env")
class TestHistoricalRun:
    def test_completes_successfully(self, isolated_outputs):
        code = weekly_runner.main(
            ["--season", "2025", "--week", "6", "--skip-google", "--include-completed"]
        )

        assert code == weekly_runner.EXIT_OK

    def test_writes_all_three_outputs(self, isolated_outputs):
        weekly_runner.main(
            ["--season", "2025", "--week", "6", "--skip-google", "--include-completed"]
        )

        week_dir = isolated_outputs["outputs"] / "2025" / "week_06"
        assert (week_dir / "epa_splits.csv").is_file()
        assert (week_dir / "fair_lines.csv").is_file()
        assert (week_dir / "run_summary.json").is_file()

    def test_outputs_are_for_the_requested_week_only(self, isolated_outputs):
        weekly_runner.main(
            ["--season", "2025", "--week", "6", "--skip-google", "--include-completed"]
        )

        week_dir = isolated_outputs["outputs"] / "2025" / "week_06"
        fair = pd.read_csv(week_dir / "fair_lines.csv")
        epa = pd.read_csv(week_dir / "epa_splits.csv")

        assert set(fair["week"]) == {6}
        assert set(fair["season"]) == {2025}
        assert set(epa["analysis_week"]) == {6}

    def test_no_future_data_leaks_into_epa(self, isolated_outputs):
        weekly_runner.main(
            ["--season", "2025", "--week", "6", "--skip-google", "--include-completed"]
        )

        epa = pd.read_csv(
            isolated_outputs["outputs"] / "2025" / "week_06" / "epa_splits.csv"
        )
        for weeks in epa["weeks"]:
            assert all(int(w) < 6 for w in str(weeks).split(", "))

    def test_no_future_data_leaks_into_training(self, isolated_outputs):
        weekly_runner.main(
            ["--season", "2025", "--week", "6", "--skip-google", "--include-completed"]
        )

        metadata = json.loads(
            (isolated_outputs["models"] / "model_metadata.json").read_text()
        )
        assert metadata["target_week"] == 6
        assert metadata["training_last_week"] < 6

    def test_no_duplicate_teams_or_matchups(self, isolated_outputs):
        weekly_runner.main(
            ["--season", "2025", "--week", "6", "--skip-google", "--include-completed"]
        )

        week_dir = isolated_outputs["outputs"] / "2025" / "week_06"
        epa = pd.read_csv(week_dir / "epa_splits.csv")
        fair = pd.read_csv(week_dir / "fair_lines.csv")

        assert epa["team"].is_unique
        assert not fair.duplicated(subset=["away_team", "home_team"]).any()

    def test_summary_records_the_run(self, isolated_outputs):
        weekly_runner.main(
            ["--season", "2025", "--week", "6", "--skip-google", "--include-completed"]
        )

        summary = json.loads(
            (
                isolated_outputs["outputs"] / "2025" / "week_06" / "run_summary.json"
            ).read_text()
        )
        assert summary["season"] == 2025
        assert summary["week"] == 6
        assert summary["status"] == "success"
        assert summary["projections"] == 8  # 16 teams -> 8 games

    def test_rerunning_the_same_week_is_stable(self, isolated_outputs):
        args = ["--season", "2025", "--week", "6", "--skip-google", "--include-completed"]
        weekly_runner.main(args)
        fair_path = isolated_outputs["outputs"] / "2025" / "week_06" / "fair_lines.csv"
        first = pd.read_csv(fair_path)

        weekly_runner.main(args)
        second = pd.read_csv(fair_path)

        assert len(first) == len(second)
        pd.testing.assert_series_equal(
            first["fair_spread_home"], second["fair_spread_home"]
        )


@pytest.mark.usefixtures("fake_nflverse", "isolated_outputs", "clean_env")
class TestRunnerSwitches:
    def test_dry_run_writes_nothing(self, isolated_outputs):
        code = weekly_runner.main(
            [
                "--season", "2025", "--week", "6",
                "--skip-google", "--include-completed", "--dry-run",
            ]
        )

        assert code == weekly_runner.EXIT_OK
        assert not (isolated_outputs["outputs"] / "2025").exists()
        assert not (isolated_outputs["models"] / "fair_line_model.pkl").exists()

    def test_forced_retrain_bumps_the_model_version(self, isolated_outputs):
        base = ["--season", "2025", "--week", "6", "--skip-google", "--include-completed"]
        weekly_runner.main(base)
        metadata_path = isolated_outputs["models"] / "model_metadata.json"
        first = json.loads(metadata_path.read_text())["model_version"]

        weekly_runner.main(base + ["--retrain"])
        second = json.loads(metadata_path.read_text())["model_version"]

        assert second == first + 1

    def test_model_is_reused_within_the_retrain_window(self, isolated_outputs):
        weekly_runner.main(
            ["--season", "2025", "--week", "6", "--skip-google", "--include-completed"]
        )
        metadata_path = isolated_outputs["models"] / "model_metadata.json"
        first = json.loads(metadata_path.read_text())["model_version"]

        weekly_runner.main(
            ["--season", "2025", "--week", "8", "--skip-google", "--include-completed"]
        )

        assert json.loads(metadata_path.read_text())["model_version"] == first

    def test_four_week_rule_retrains(self, isolated_outputs):
        weekly_runner.main(
            ["--season", "2025", "--week", "6", "--skip-google", "--include-completed"]
        )
        metadata_path = isolated_outputs["models"] / "model_metadata.json"
        first = json.loads(metadata_path.read_text())["model_version"]

        weekly_runner.main(
            ["--season", "2025", "--week", "10", "--skip-google", "--include-completed"]
        )

        assert json.loads(metadata_path.read_text())["model_version"] == first + 1

    def test_completed_games_are_not_projected_by_default(self, isolated_outputs):
        weekly_runner.main(["--season", "2025", "--week", "6", "--skip-google"])

        fair = pd.read_csv(
            isolated_outputs["outputs"] / "2025" / "week_06" / "fair_lines.csv"
        )
        assert (fair["status"] == "already_completed").all()
        assert fair["fair_spread_home"].isna().all()

    def test_missing_google_config_reports_a_distinct_exit_code(self, isolated_outputs):
        code = weekly_runner.main(
            ["--season", "2025", "--week", "6", "--include-completed"]
        )

        assert code == weekly_runner.EXIT_GOOGLE_FAILED
        # ...but the analytics still landed on disk.
        week_dir = isolated_outputs["outputs"] / "2025" / "week_06"
        assert (week_dir / "fair_lines.csv").is_file()


class TestArgumentValidation:
    def test_season_without_week_is_rejected(self, clean_env):
        assert weekly_runner.main(["--season", "2025"]) == weekly_runner.EXIT_BAD_ARGS

    def test_week_without_season_is_rejected(self, clean_env):
        assert weekly_runner.main(["--week", "6"]) == weekly_runner.EXIT_BAD_ARGS

    @pytest.mark.parametrize("week", ["0", "23", "-1"])
    def test_out_of_range_week_is_rejected(self, clean_env, week):
        assert (
            weekly_runner.main(["--season", "2025", "--week", week])
            == weekly_runner.EXIT_BAD_ARGS
        )


@pytest.mark.usefixtures("fake_nflverse", "isolated_outputs", "clean_env")
class TestAutomaticDetection:
    def test_auto_mode_needs_no_arguments(self, monkeypatch, isolated_outputs):
        """Simulate a Friday during week 6 and let detection do the work."""
        friday = datetime(2025, 10, 10, 22, 0, tzinfo=timezone.utc)
        real_resolve = weekly_runner.resolve_season_week

        def fake_resolve(season=None, week=None, **kwargs):
            return real_resolve(
                season, week, now_utc=friday,
                schedule_loader=kwargs.get("schedule_loader"),
            )

        monkeypatch.setattr(weekly_runner, "resolve_season_week", fake_resolve)
        monkeypatch.setattr(
            schedule_module, "detect_target_season_and_week",
            lambda now_utc=None, schedule_loader=None: (2025, 6, {
                "detected_by": "test", "reason": "simulated friday"}),
        )

        code = weekly_runner.main(["--skip-google", "--include-completed"])

        assert code == weekly_runner.EXIT_OK
        assert (
            isolated_outputs["outputs"] / "2025" / "week_06" / "fair_lines.csv"
        ).is_file()
