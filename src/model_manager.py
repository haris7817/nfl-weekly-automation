"""Model persistence, metadata and the retraining policy.

Design points from the implementation guide:

* the pickle keeps the client's original ``{"model": ..., "n": ...}`` shape,
  so a model trained by the original script still loads here (5.3);
* the trailing-game ``n`` stored at training time is authoritative - a
  projection may not silently run with a different ``n`` (5.3 / step 6);
* a sidecar ``model_metadata.json`` records provenance and metrics so the
  client can see what produced a number without unpickling anything (15);
* training writes to a temporary file and only replaces the live model once
  serialisation succeeded, so a failed retrain can never destroy the last
  known-good model (16.3);
* retraining happens when forced, when no usable model exists, or every
  ``RETRAIN_EVERY_WEEKS`` weeks (16).
"""

from __future__ import annotations

import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .fair_line import (
    DEFAULT_N,
    MIN_TRAILING_GAMES,
    build_training_set,
    fit_model,
)
from .logging_utils import get_logger
from .paths import MODEL_METADATA_PATH, MODEL_PATH, MODEL_TMP_PATH, MODEL_DIR

log = get_logger("model")

REGULAR_SEASON_WEEKS = 18


class ModelError(RuntimeError):
    """Raised when the stored model is missing, unreadable or inconsistent."""


class ModelNotFoundError(ModelError):
    """Raised when no model file exists yet."""


# ---------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------

def model_exists(path: Optional[Path] = None) -> bool:
    return (path or MODEL_PATH).is_file()


def load_model(
    path: Optional[Path] = None,
    requested_n: Optional[int] = None,
) -> dict:
    """Load and validate the pickled model.

    Returns a dict with ``model``, ``n`` and (when available) ``metadata``.
    Raises :class:`ModelNotFoundError` if there is no file and
    :class:`ModelError` if the file is corrupt or internally inconsistent -
    never a silently wrong object (implementation guide 31.8).
    """
    path = path or MODEL_PATH
    if not path.is_file():
        raise ModelNotFoundError(
            f"No trained model found at {path}. Run with --retrain, or let "
            "the weekly runner train one automatically."
        )

    try:
        with open(path, "rb") as handle:
            saved = pickle.load(handle)
    except Exception as exc:
        raise ModelError(
            f"The model file at {path} could not be unpickled "
            f"({type(exc).__name__}: {exc}). It is corrupt and must be "
            "retrained before it can be used."
        ) from exc

    if not isinstance(saved, dict) or "model" not in saved or "n" not in saved:
        raise ModelError(
            f"The model file at {path} does not have the expected structure "
            "({'model': ..., 'n': ...}). Retrain to rebuild it."
        )

    model = saved["model"]
    for attribute in ("coef_", "intercept_", "predict"):
        if not hasattr(model, attribute):
            raise ModelError(
                f"The object stored in {path} is not a fitted regression "
                f"model (missing {attribute!r}). Retrain to rebuild it."
            )

    try:
        saved_n = int(saved["n"])
    except (TypeError, ValueError) as exc:
        raise ModelError(
            f"The trailing-game count stored in {path} is not an integer "
            f"({saved['n']!r})."
        ) from exc

    if requested_n is not None and int(requested_n) != saved_n:
        raise ModelError(
            f"Model was trained with n={saved_n}, but the projection "
            f"requested n={requested_n}. Either project with n={saved_n}, "
            f"or retrain for the new value first "
            f"(python nfl_fair_line.py train --seasons ... --n {requested_n}). "
            "Mixing the two would misstate the fair line."
        )

    saved["n"] = saved_n
    saved.setdefault("metadata", read_metadata())
    return saved


def read_metadata(path: Optional[Path] = None) -> Optional[dict]:
    """Return the metadata sidecar, or ``None`` when it is absent/unreadable."""
    path = path or MODEL_METADATA_PATH
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Model metadata at %s is unreadable (%s).", path, exc)
        return None


def write_metadata(metadata: dict, path: Optional[Path] = None) -> None:
    path = path or MODEL_METADATA_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def save_model_atomically(
    model,
    n: int,
    metadata: dict,
    path: Optional[Path] = None,
    tmp_path: Optional[Path] = None,
) -> None:
    """Serialise to a temp file, then swap it in.

    If pickling raises, the previous production model is still in place.
    """
    path = path or MODEL_PATH
    tmp_path = tmp_path or MODEL_TMP_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(tmp_path, "wb") as handle:
            pickle.dump({"model": model, "n": int(n), "metadata": metadata}, handle)
        # Read it straight back: a file that cannot be loaded must never
        # be promoted over a working model.
        with open(tmp_path, "rb") as handle:
            reloaded = pickle.load(handle)
        if "model" not in reloaded or "n" not in reloaded:
            raise ModelError("Round-trip verification of the new model failed.")
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    os.replace(tmp_path, path)
    write_metadata(metadata)
    log.info("Model saved to %s (version %s).", path, metadata.get("model_version"))


# ---------------------------------------------------------------------
# Retraining policy
# ---------------------------------------------------------------------

def weeks_since_training(
    metadata: Optional[dict],
    season: int,
    week: int,
) -> Optional[int]:
    """Approximate elapsed NFL weeks since the model was last trained."""
    if not metadata:
        return None
    last_season = metadata.get("target_season") or metadata.get(
        "training_last_season"
    )
    last_week = metadata.get("target_week") or metadata.get("training_last_week")
    if last_season is None or last_week is None:
        return None
    try:
        return (int(season) - int(last_season)) * REGULAR_SEASON_WEEKS + (
            int(week) - int(last_week)
        )
    except (TypeError, ValueError):
        return None


def should_retrain(
    season: int,
    week: int,
    retrain_every_weeks: int = 4,
    force: bool = False,
    requested_n: Optional[int] = None,
) -> tuple:
    """Decide whether to retrain. Returns ``(should_retrain, reason)``."""
    if force:
        return True, "forced by --retrain"

    try:
        saved = load_model(requested_n=requested_n)
    except ModelNotFoundError:
        return True, "no trained model exists yet"
    except ModelError as exc:
        return True, f"existing model is unusable ({exc})"

    metadata = saved.get("metadata") or read_metadata()
    if not metadata:
        return True, "model metadata is missing, provenance cannot be verified"

    last_season = metadata.get("target_season") or metadata.get(
        "training_last_season"
    )
    if last_season is not None and int(season) > int(last_season):
        # Week arithmetic compresses the offseason: week 18 -> next season
        # week 1 scores as a single week even though months have passed and
        # an entire completed season of new data is now available.
        return True, (
            f"a new season has started ({last_season} -> {season}); the "
            "completed season is now available as training data"
        )

    elapsed = weeks_since_training(metadata, season, week)
    if elapsed is None:
        return True, "model metadata does not record when it was trained"
    if elapsed < 0:
        return (
            False,
            f"model was trained for a later week "
            f"({metadata.get('target_season')} week "
            f"{metadata.get('target_week')}); reusing it",
        )
    if elapsed >= retrain_every_weeks:
        return True, (
            f"{elapsed} weeks since the last training "
            f"(policy: every {retrain_every_weeks})"
        )

    return False, (
        f"last trained {elapsed} week(s) ago; next retrain due in "
        f"{retrain_every_weeks - elapsed} week(s)"
    )


# ---------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------

def training_seasons(
    target_season: int,
    start_season: int,
    include_target_season: bool = True,
) -> list:
    """Seasons to pull for training, oldest first."""
    last = target_season if include_target_season else target_season - 1
    if last < start_season:
        return [last]
    return list(range(start_season, last + 1))


def train_model(
    pbp: pd.DataFrame,
    seasons: Sequence[int],
    target_season: int,
    target_week: int,
    n: int = DEFAULT_N,
    min_games: int = MIN_TRAILING_GAMES,
    previous_metadata: Optional[dict] = None,
) -> tuple:
    """Build the training set, fit and validate. Returns ``(model, metadata)``.

    Nothing is written to disk here, so the caller decides whether the new
    model is promoted. That keeps ``--dry-run`` able to produce real
    projections from a freshly fitted model without touching the stored one.
    """
    log.info(
        "Model retraining started for target %s week %s using seasons %s.",
        target_season,
        target_week,
        list(seasons),
    )

    data = build_training_set(
        pbp,
        n=n,
        min_games=min_games,
        target_season=target_season,
        target_week=target_week,
    )
    log.info("%d games with sufficient trailing history.", len(data))

    fit = fit_model(data)

    previous = previous_metadata or read_metadata()
    version = 1
    if previous and isinstance(previous.get("model_version"), int):
        version = previous["model_version"] + 1

    completed = data[["season", "week"]]
    metadata = {
        "model_version": version,
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "trailing_games_n": int(n),
        "min_trailing_games": int(min_games),
        "training_seasons": [int(s) for s in seasons],
        "training_last_season": int(completed["season"].max()),
        "training_last_week": int(
            completed[completed["season"] == completed["season"].max()]["week"].max()
        ),
        "training_rows": fit["n_rows"],
        "holdout_rows": fit["n_test"],
        "coefficient": round(fit["coefficient"], 6),
        "intercept": round(fit["intercept"], 6),
        "mae": round(fit["mae"], 4),
        "r2": round(fit["r2"], 6),
        "target_season": int(target_season),
        "target_week": int(target_week),
        "estimator": "sklearn.linear_model.LinearRegression",
        "feature": "home_trailing_net_epa - away_trailing_net_epa",
        "target": "home_score - away_score",
    }

    if previous and isinstance(previous.get("mae"), (int, float)):
        change = fit["mae"] - previous["mae"]
        log.info(
            "Model MAE %.3f (previous %.3f, change %+.3f points).",
            fit["mae"],
            previous["mae"],
            change,
        )
        if previous["mae"] > 0 and fit["mae"] > previous["mae"] * 1.25:
            log.warning(
                "New model MAE (%.2f) is more than 25%% worse than the "
                "previous model (%.2f). It is still being saved, but this "
                "is worth investigating.",
                fit["mae"],
                previous["mae"],
            )

    return fit["model"], metadata


def train_and_save(
    pbp: pd.DataFrame,
    seasons: Sequence[int],
    target_season: int,
    target_week: int,
    n: int = DEFAULT_N,
    min_games: int = MIN_TRAILING_GAMES,
    previous_metadata: Optional[dict] = None,
    dry_run: bool = False,
) -> dict:
    """Train and persist the model. Returns the new metadata."""
    model, metadata = train_model(
        pbp,
        seasons=seasons,
        target_season=target_season,
        target_week=target_week,
        n=n,
        min_games=min_games,
        previous_metadata=previous_metadata,
    )

    if dry_run:
        log.info("--dry-run: trained model was NOT written to disk.")
        return metadata

    save_model_atomically(model, n=n, metadata=metadata)
    return metadata


def metadata_rows(metadata: Optional[dict]) -> list:
    """Flatten metadata into ``[[label, value], ...]`` for the Sheet tab."""
    if not metadata:
        return [["Status", "No model metadata available"]]
    ordered = [
        ("Model Version", "model_version"),
        ("Last Trained (UTC)", "trained_at_utc"),
        ("Trailing Games (n)", "trailing_games_n"),
        ("Minimum Games Required", "min_trailing_games"),
        ("Training Start Season", None),
        ("Training Through Season", "training_last_season"),
        ("Training Through Week", "training_last_week"),
        ("Trained For Season", "target_season"),
        ("Trained For Week", "target_week"),
        ("Training Rows", "training_rows"),
        ("Hold-out Rows", "holdout_rows"),
        ("Coefficient", "coefficient"),
        ("Intercept (home-field points)", "intercept"),
        ("MAE (points)", "mae"),
        ("R^2", "r2"),
        ("Estimator", "estimator"),
        ("Feature", "feature"),
        ("Target", "target"),
    ]
    rows = []
    for label, key in ordered:
        if label == "Training Start Season":
            seasons = metadata.get("training_seasons") or []
            rows.append([label, min(seasons) if seasons else ""])
            continue
        rows.append([label, metadata.get(key, "")])
    return rows
