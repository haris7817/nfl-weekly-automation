"""Model persistence, metadata and the retraining policy."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

from conftest import make_pbp
from src import model_manager as mm
from src.fair_line import TrainingError


class FakeModel:
    def __init__(self, intercept=2.0, coefficient=20.0):
        self.intercept_ = intercept
        self.coef_ = np.array([coefficient])

    def predict(self, X):
        return self.intercept_ + self.coef_[0] * np.asarray(X).ravel()


def write_model(paths, model=None, n=5, metadata=None):
    with open(paths["model"], "wb") as handle:
        pickle.dump({"model": model or FakeModel(), "n": n}, handle)
    if metadata is not None:
        with open(paths["metadata"], "w", encoding="utf-8") as handle:
            json.dump(metadata, handle)


def base_metadata(season=2025, week=6, version=1, mae=10.5):
    return {
        "model_version": version,
        "trained_at_utc": "2026-01-01T00:00:00Z",
        "trailing_games_n": 5,
        "training_seasons": [2021, 2022, 2023, 2024, 2025],
        "training_last_season": season,
        "training_last_week": week - 1,
        "target_season": season,
        "target_week": week,
        "mae": mae,
        "r2": 0.1,
        "coefficient": 20.0,
        "intercept": 2.0,
    }


class TestLoadModel:
    def test_missing_file_raises_not_found(self, tmp_model_dir):
        with pytest.raises(mm.ModelNotFoundError):
            mm.load_model(path=tmp_model_dir["model"])

    def test_loads_the_original_pickle_shape(self, tmp_model_dir):
        write_model(tmp_model_dir)

        saved = mm.load_model(path=tmp_model_dir["model"])

        assert saved["n"] == 5
        assert hasattr(saved["model"], "coef_")

    def test_corrupt_file_raises_model_error(self, tmp_model_dir):
        tmp_model_dir["model"].write_bytes(b"this is not a pickle")

        with pytest.raises(mm.ModelError) as exc:
            mm.load_model(path=tmp_model_dir["model"])

        assert "corrupt" in str(exc.value).lower()

    def test_wrong_structure_raises(self, tmp_model_dir):
        with open(tmp_model_dir["model"], "wb") as handle:
            pickle.dump({"something": "else"}, handle)

        with pytest.raises(mm.ModelError):
            mm.load_model(path=tmp_model_dir["model"])

    def test_object_that_is_not_a_model_raises(self, tmp_model_dir):
        with open(tmp_model_dir["model"], "wb") as handle:
            pickle.dump({"model": "not a model", "n": 5}, handle)

        with pytest.raises(mm.ModelError):
            mm.load_model(path=tmp_model_dir["model"])

    def test_n_mismatch_is_refused(self, tmp_model_dir):
        """Guide 5.3: never project with an n the model was not trained on."""
        write_model(tmp_model_dir, n=5)

        with pytest.raises(mm.ModelError) as exc:
            mm.load_model(path=tmp_model_dir["model"], requested_n=7)

        assert "n=5" in str(exc.value)

    def test_matching_n_is_accepted(self, tmp_model_dir):
        write_model(tmp_model_dir, n=5)

        assert mm.load_model(path=tmp_model_dir["model"], requested_n=5)["n"] == 5


class TestAtomicSave:
    def test_saves_model_and_metadata(self, tmp_model_dir):
        mm.save_model_atomically(
            FakeModel(), n=5, metadata=base_metadata(),
            path=tmp_model_dir["model"], tmp_path=tmp_model_dir["tmp"],
        )

        assert tmp_model_dir["model"].is_file()
        assert tmp_model_dir["metadata"].is_file()
        assert not tmp_model_dir["tmp"].exists()

    def test_failed_serialisation_leaves_the_old_model_intact(self, tmp_model_dir):
        write_model(tmp_model_dir, model=FakeModel(intercept=1.0))
        original = tmp_model_dir["model"].read_bytes()

        class Unpicklable:
            coef_ = np.array([1.0])
            intercept_ = 1.0

            def __reduce__(self):
                raise RuntimeError("cannot pickle this")

        with pytest.raises(RuntimeError):
            mm.save_model_atomically(
                Unpicklable(), n=5, metadata=base_metadata(),
                path=tmp_model_dir["model"], tmp_path=tmp_model_dir["tmp"],
            )

        assert tmp_model_dir["model"].read_bytes() == original
        assert not tmp_model_dir["tmp"].exists()


class TestWeeksSinceTraining:
    def test_same_season_difference(self):
        assert mm.weeks_since_training(base_metadata(2025, 6), 2025, 9) == 3

    def test_crosses_a_season_boundary(self):
        assert mm.weeks_since_training(base_metadata(2025, 16), 2026, 2) == 4

    def test_missing_metadata_returns_none(self):
        assert mm.weeks_since_training(None, 2025, 6) is None
        assert mm.weeks_since_training({}, 2025, 6) is None


class TestShouldRetrain:
    def test_force_always_wins(self, tmp_model_dir):
        write_model(tmp_model_dir, metadata=base_metadata())

        retrain, reason = mm.should_retrain(2025, 6, force=True)

        assert retrain
        assert "forced" in reason

    def test_missing_model_triggers_training(self, tmp_model_dir):
        retrain, reason = mm.should_retrain(2025, 6)

        assert retrain
        assert "no trained model" in reason

    def test_corrupt_model_triggers_training(self, tmp_model_dir):
        tmp_model_dir["model"].write_bytes(b"garbage")

        retrain, reason = mm.should_retrain(2025, 6)

        assert retrain
        assert "unusable" in reason

    def test_missing_metadata_triggers_training(self, tmp_model_dir):
        write_model(tmp_model_dir)

        retrain, reason = mm.should_retrain(2025, 6)

        assert retrain
        assert "metadata" in reason

    def test_within_the_window_reuses_the_model(self, tmp_model_dir):
        write_model(tmp_model_dir, metadata=base_metadata(2025, 6))

        retrain, reason = mm.should_retrain(2025, 8, retrain_every_weeks=4)

        assert not retrain
        assert "2 week(s)" in reason

    def test_four_week_rule_triggers_training(self, tmp_model_dir):
        write_model(tmp_model_dir, metadata=base_metadata(2025, 6))

        retrain, reason = mm.should_retrain(2025, 10, retrain_every_weeks=4)

        assert retrain
        assert "4 weeks since" in reason

    def test_boundary_is_inclusive(self, tmp_model_dir):
        write_model(tmp_model_dir, metadata=base_metadata(2025, 6))

        assert mm.should_retrain(2025, 9, retrain_every_weeks=4)[0] is False
        assert mm.should_retrain(2025, 10, retrain_every_weeks=4)[0] is True

    def test_new_season_triggers_training(self, tmp_model_dir):
        write_model(tmp_model_dir, metadata=base_metadata(2025, 18))

        retrain, _ = mm.should_retrain(2026, 1, retrain_every_weeks=4)

        assert retrain

    def test_model_trained_for_a_later_week_is_reused(self, tmp_model_dir):
        write_model(tmp_model_dir, metadata=base_metadata(2025, 10))

        retrain, reason = mm.should_retrain(2025, 8, retrain_every_weeks=4)

        assert not retrain
        assert "later week" in reason


class TestTrainingSeasons:
    def test_spans_start_through_target(self):
        assert mm.training_seasons(2026, 2021) == [2021, 2022, 2023, 2024, 2025, 2026]

    def test_can_exclude_the_target_season(self):
        assert mm.training_seasons(2026, 2021, include_target_season=False) == [
            2021, 2022, 2023, 2024, 2025
        ]

    def test_handles_a_start_after_the_target(self):
        assert mm.training_seasons(2020, 2021) == [2020]


class TestTrainAndSave:
    @pytest.fixture
    def big_pbp(self):
        teams = [f"T{i:02d}" for i in range(16)]
        return make_pbp(season=2025, weeks=18, teams=teams)

    def test_trains_and_writes_metadata(self, tmp_model_dir, big_pbp):
        metadata = mm.train_and_save(
            big_pbp, seasons=[2025], target_season=2025, target_week=18, n=5
        )

        assert tmp_model_dir["model"].is_file()
        assert metadata["model_version"] == 1
        assert metadata["trailing_games_n"] == 5
        assert metadata["estimator"] == "sklearn.linear_model.LinearRegression"
        assert np.isfinite(metadata["mae"])

    def test_version_increments_on_retrain(self, tmp_model_dir, big_pbp):
        first = mm.train_and_save(
            big_pbp, seasons=[2025], target_season=2025, target_week=18, n=5
        )
        second = mm.train_and_save(
            big_pbp, seasons=[2025], target_season=2025, target_week=18, n=5
        )

        assert second["model_version"] == first["model_version"] + 1

    def test_dry_run_writes_nothing(self, tmp_model_dir, big_pbp):
        mm.train_and_save(
            big_pbp, seasons=[2025], target_season=2025, target_week=18, n=5,
            dry_run=True,
        )

        assert not tmp_model_dir["model"].exists()

    def test_leakage_guard_is_recorded(self, tmp_model_dir, big_pbp):
        metadata = mm.train_and_save(
            big_pbp, seasons=[2025], target_season=2025, target_week=10, n=5
        )

        assert metadata["target_week"] == 10
        assert metadata["training_last_week"] < 10

    def test_failed_training_preserves_the_previous_model(self, tmp_model_dir, big_pbp):
        mm.train_and_save(
            big_pbp, seasons=[2025], target_season=2025, target_week=18, n=5
        )
        good = tmp_model_dir["model"].read_bytes()

        # Week 2 leaves almost nothing after the leakage guard.
        with pytest.raises(TrainingError):
            mm.train_and_save(
                big_pbp, seasons=[2025], target_season=2025, target_week=2, n=5
            )

        assert tmp_model_dir["model"].read_bytes() == good


class TestMetadataRows:
    def test_renders_label_value_pairs(self):
        rows = mm.metadata_rows(base_metadata())

        labels = [row[0] for row in rows]
        assert "Model Version" in labels
        assert "MAE (points)" in labels
        assert all(len(row) == 2 for row in rows)

    def test_handles_absent_metadata(self):
        rows = mm.metadata_rows(None)

        assert rows == [["Status", "No model metadata available"]]
