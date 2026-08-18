"""Fair-line model - the client's original regression, preserved.

The methodology is deliberately untouched (implementation guide 10.1 / 43):

* one feature: ``home trailing net EPA - away trailing net EPA``;
* target: ``home_score - away_score`` (home perspective);
* estimator: ``sklearn.linear_model.LinearRegression``;
* trailing window: last ``n`` games strictly before the game, default 5;
* a team needs at least 3 prior games or it is skipped as insufficient
  history;
* evaluation: ``train_test_split(test_size=0.2, random_state=42)`` scored
  with MAE and R-squared;
* sign convention: ``fair_spread_home = -predicted_home_margin``, so a
  negative number means the home team is favoured.

Note that this EPA definition is intentionally *not* the same as the one in
``src/epa.py``: this module averages EPA across all qualifying plays, while
the splits script averages pass and rush separately and sums the two. The
client's two scripts have always differed in this way and the implementation
guide (section 5.2) explicitly says not to silently reconcile them.

What the migration added: a trailing-EPA index that makes training roughly
two orders of magnitude faster without changing a single computed value,
structured DataFrame output instead of ``print``, and leakage control so a
projection for week N never trains on week N or later.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from .logging_utils import get_logger

log = get_logger("fair_line")

DEFAULT_N = 5
MIN_TRAILING_GAMES = 3

#: Column order for the fair-line CSV / Sheet tab.
FAIR_LINE_COLUMNS: tuple = (
    "run_timestamp_utc",
    "season",
    "week",
    "game_date",
    "away_team",
    "home_team",
    "away_trailing_net_epa",
    "home_trailing_net_epa",
    "epa_diff",
    "predicted_home_margin",
    "fair_spread_home",
    "model_n",
    "model_version",
    "model_trained_at",
    "status",
    "notes",
)

STATUS_PROJECTED = "projection_generated"
STATUS_INSUFFICIENT = "insufficient_history"
STATUS_COMPLETED = "already_completed"


class TrainingError(RuntimeError):
    """Raised when a training set cannot be built or the fit is unusable."""


# ---------------------------------------------------------------------
# Shared aggregation logic (mirrors the client's original module)
# ---------------------------------------------------------------------

def team_game_epa(df: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game with overall offensive/defensive EPA.

    Byte-for-byte the client's original aggregation.
    """
    off = (
        df.groupby(["posteam", "season", "week", "game_id"])["epa"]
        .mean()
        .reset_index()
        .rename(columns={"posteam": "team", "epa": "off_epa"})
    )
    deff = (
        df.groupby(["defteam", "season", "week", "game_id"])["epa"]
        .mean()
        .reset_index()
        .rename(columns={"defteam": "team", "epa": "def_epa_allowed"})
    )
    merged = off.merge(
        deff, on=["team", "season", "week", "game_id"], how="outer"
    )
    merged["net_epa"] = merged["off_epa"] - merged["def_epa_allowed"]
    return merged.sort_values(["team", "season", "week"])


def trailing_net_epa(
    game_epa: pd.DataFrame,
    team: str,
    season: int,
    week: int,
    n: int = DEFAULT_N,
    min_games: int = MIN_TRAILING_GAMES,
) -> Optional[float]:
    """Trailing ``n``-game net EPA using only games strictly before ``week``.

    This is the client's original implementation, kept as the reference
    definition and as the oracle the fast index below is tested against.
    """
    hist = (
        game_epa[
            (game_epa["team"] == team)
            & (
                (game_epa["season"] < season)
                | ((game_epa["season"] == season) & (game_epa["week"] < week))
            )
        ]
        .sort_values(["season", "week"])
        .tail(n)
    )
    if len(hist) < min_games:  # early season = less reliable
        return None
    return hist["net_epa"].mean()


class TrailingEpaIndex:
    """Pre-grouped view of team-game EPA for fast repeated lookups.

    ``build_training_set`` asks for two trailing averages per game, so a
    four-season training run performs several thousand lookups. Filtering
    the whole frame each time is O(rows) per call; grouping once up front
    makes each lookup a slice of that team's own rows.

    The arithmetic is identical to :func:`trailing_net_epa` - same ordering,
    same window, same minimum-sample rule - and ``tests/test_fair_line.py``
    asserts the two agree on real data.
    """

    def __init__(self, game_epa: pd.DataFrame):
        ordered = game_epa.sort_values(["season", "week"], kind="mergesort")
        self._teams = {}
        for team, grp in ordered.groupby("team", sort=False):
            self._teams[team] = (
                grp["season"].to_numpy(dtype="int64"),
                grp["week"].to_numpy(dtype="int64"),
                grp["net_epa"].to_numpy(dtype="float64"),
            )

    def get(
        self,
        team: str,
        season: int,
        week: int,
        n: int = DEFAULT_N,
        min_games: int = MIN_TRAILING_GAMES,
    ) -> Optional[float]:
        entry = self._teams.get(team)
        if entry is None:
            return None
        seasons, weeks, values = entry
        eligible = (seasons < season) | ((seasons == season) & (weeks < week))
        selected = values[eligible]
        if selected.size > n:
            selected = selected[-n:]
        if selected.size < min_games:
            return None
        return float(np.nanmean(selected))


def game_scores(df: pd.DataFrame) -> pd.DataFrame:
    """One row per game with home/away teams, scores, season, week."""
    return (
        df[
            [
                "game_id",
                "season",
                "week",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
            ]
        ]
        .drop_duplicates("game_id")
        .dropna(subset=["home_score", "away_score"])
    )


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def build_training_set(
    pbp: pd.DataFrame,
    n: int = DEFAULT_N,
    min_games: int = MIN_TRAILING_GAMES,
    target_season: Optional[int] = None,
    target_week: Optional[int] = None,
) -> pd.DataFrame:
    """Build ``epa_diff`` / ``actual_margin`` rows from play-by-play.

    ``target_season`` / ``target_week`` implement the leakage guard from
    implementation guide 16.1: when retraining ahead of a projection for
    week N, games from week N onward of that season are excluded from the
    training rows. Completed earlier seasons are used in full. Feature
    values were already leak-free (the trailing window is strictly
    backward-looking); this restricts which *games* become training rows.
    """
    ge = team_game_epa(pbp)
    index = TrailingEpaIndex(ge)
    games = game_scores(pbp)

    if target_season is not None and target_week is not None:
        before = len(games)
        games = games[
            (games["season"] < target_season)
            | (
                (games["season"] == target_season)
                & (games["week"] < target_week)
            )
        ]
        log.info(
            "Leakage guard: %d of %d completed games kept "
            "(excluded season %s week >= %s).",
            len(games),
            before,
            target_season,
            target_week,
        )

    rows = []
    for _, g in games.iterrows():
        home_epa = index.get(g["home_team"], g["season"], g["week"], n, min_games)
        away_epa = index.get(g["away_team"], g["season"], g["week"], n, min_games)
        if home_epa is None or away_epa is None:
            continue
        rows.append(
            {
                "season": g["season"],
                "week": g["week"],
                "home_team": g["home_team"],
                "away_team": g["away_team"],
                "epa_diff": home_epa - away_epa,  # feature
                "actual_margin": g["home_score"] - g["away_score"],  # target
            }
        )
    return pd.DataFrame(rows)


def fit_model(data: pd.DataFrame) -> dict:
    """Fit the client's LinearRegression and return the model plus metrics."""
    if data.empty:
        raise TrainingError(
            "The training set is empty - no games had sufficient trailing "
            "history. Check that the requested seasons downloaded correctly."
        )
    if len(data) < 20:
        raise TrainingError(
            f"Only {len(data)} training rows were built; that is too few to "
            "fit a meaningful model. Widen TRAINING_START_SEASON."
        )

    X = data[["epa_diff"]].values
    y = data["actual_margin"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    coefficient = float(model.coef_[0])
    intercept = float(model.intercept_)

    if not all(np.isfinite([mae, r2, coefficient, intercept])):
        raise TrainingError(
            "Training produced non-finite coefficients or metrics; the new "
            "model was rejected and the previous one is unchanged."
        )

    return {
        "model": model,
        "n_rows": int(len(data)),
        "n_test": int(len(y_test)),
        "coefficient": coefficient,
        "intercept": intercept,
        "mae": mae,
        "r2": r2,
    }


def format_fit_report(fit: dict) -> str:
    """The explanatory block the client's original script printed."""
    return (
        "=== Model fit ===\n"
        f"Predicted margin = {fit['intercept']:.2f} + "
        f"{fit['coefficient']:.2f} * (home_epa - away_epa)\n"
        f"  Intercept ({fit['intercept']:.2f}) is roughly the "
        "home-field-advantage term in points.\n"
        f"\nHold-out test set (n={fit['n_test']}):\n"
        f"  Mean Absolute Error: {fit['mae']:.2f} points\n"
        f"  R^2: {fit['r2']:.3f}\n"
        "\nContext: Vegas closing lines typically have an MAE of roughly "
        "10-10.5 points against actual margins league-wide, since final "
        "scores are inherently noisy. Your model beating that bar isn't "
        "realistic or the goal -- the goal is a transparent baseline to "
        "compare against the market number, not to replace it."
    )


# ---------------------------------------------------------------------
# Projecting upcoming games
# ---------------------------------------------------------------------

def project_matchups(
    pbp: pd.DataFrame,
    matchups: Iterable[dict],
    model,
    season: int,
    week: int,
    n: int = DEFAULT_N,
    min_games: int = MIN_TRAILING_GAMES,
    model_version: Optional[int] = None,
    model_trained_at: Optional[str] = None,
    run_timestamp_utc: Optional[str] = None,
) -> pd.DataFrame:
    """Project every matchup and return a structured DataFrame.

    ``matchups`` is an iterable of dicts with at least ``away_team`` and
    ``home_team``; ``game_date`` and ``status`` are used when present.

    Games that already finished are carried through with
    ``status='already_completed'`` and blank projection fields, so the
    week's schedule stays visible without a completed Thursday game being
    presented as an upcoming pick (implementation guide 31.9).
    """
    run_timestamp = run_timestamp_utc or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    ge = team_game_epa(pbp)
    index = TrailingEpaIndex(ge)

    intercept = float(model.intercept_)
    coefficient = float(model.coef_[0])

    rows = []
    for matchup in matchups:
        away = matchup["away_team"]
        home = matchup["home_team"]
        status = matchup.get("status") or STATUS_PROJECTED
        notes = ""

        home_epa = index.get(home, season, week, n, min_games)
        away_epa = index.get(away, season, week, n, min_games)

        if status == STATUS_COMPLETED:
            predicted_margin = None
            fair_spread = None
            diff = None
            notes = "game already completed; no upcoming projection"
        elif home_epa is None or away_epa is None:
            status = STATUS_INSUFFICIENT
            predicted_margin = None
            fair_spread = None
            diff = None
            missing = [
                label
                for label, value in ((away, away_epa), (home, home_epa))
                if value is None
            ]
            notes = (
                f"need >={min_games} prior games for {', '.join(missing)}"
            )
        else:
            diff = home_epa - away_epa
            predicted_margin = intercept + coefficient * diff
            # Convention: negative = home favoured by that many points.
            fair_spread = -predicted_margin
            status = STATUS_PROJECTED
            notes = "projection generated"

        rows.append(
            {
                "run_timestamp_utc": run_timestamp,
                "season": int(season),
                "week": int(week),
                "game_date": matchup.get("game_date", ""),
                "away_team": away,
                "home_team": home,
                "away_trailing_net_epa": _round(away_epa, 4),
                "home_trailing_net_epa": _round(home_epa, 4),
                "epa_diff": _round(diff, 4),
                "predicted_home_margin": _round(predicted_margin, 2),
                "fair_spread_home": _round(fair_spread, 2),
                "model_n": int(n),
                "model_version": model_version,
                "model_trained_at": model_trained_at or "",
                "status": status,
                "notes": notes,
            }
        )

    df = pd.DataFrame(rows, columns=list(FAIR_LINE_COLUMNS))
    projected = int((df["status"] == STATUS_PROJECTED).sum())
    log.info(
        "Generated %d projections from %d scheduled games "
        "(%d insufficient history, %d already completed).",
        projected,
        len(df),
        int((df["status"] == STATUS_INSUFFICIENT).sum()),
        int((df["status"] == STATUS_COMPLETED).sum()),
    )
    return df


def format_projection_table(df: pd.DataFrame) -> str:
    """Render projections the way the client's original terminal output did."""
    lines = [
        f"\n{'Matchup':<12}{'Fair Spread (home)':<22}{'Note'}",
        "-" * 78,
    ]
    for _, row in df.iterrows():
        matchup = f"{row['away_team']}@{row['home_team']}"
        if pd.isna(row["fair_spread_home"]) or row["fair_spread_home"] is None:
            lines.append(f"{matchup:<12}{row['status']:<22}{row['notes']}")
        else:
            detail = (
                f"home_epa={row['home_trailing_net_epa']:.3f}, "
                f"away_epa={row['away_trailing_net_epa']:.3f}"
            )
            lines.append(
                f"{matchup:<12}{row['fair_spread_home']:+.1f}{'':<17}{detail}"
            )
    lines.append(
        "\nRead as: your model's projected home-team spread. Compare directly "
        "to the market spread for the same game (Step 5 of the framework).\n"
        "Negative = home team favoured. Positive = home team underdog."
    )
    return "\n".join(lines)


def _round(value, digits: int):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return round(float(value), digits)
