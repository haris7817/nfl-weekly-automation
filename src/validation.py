"""Output sanity checks run before anything is published.

The goal (implementation guide 32) is that a run either publishes correct
data or fails loudly - it must never quietly upload an empty or malformed
"successful" result to the client's Drive and Sheet.

Checks are split into hard failures (raise :class:`ValidationError`) and
soft warnings (returned for logging), because some conditions are
legitimate: a week where every game lacks trailing history produces no
projections but is not a bug.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .fair_line import STATUS_INSUFFICIENT, STATUS_PROJECTED
from .logging_utils import get_logger

log = get_logger("validation")


class ValidationError(RuntimeError):
    """Raised when output is broken enough that it must not be published."""


def validate_epa_splits(
    df: pd.DataFrame,
    season: int,
    week: int,
    allow_empty: bool = False,
) -> list:
    """Validate the EPA splits frame. Returns non-fatal warnings."""
    warnings = []

    if df.empty:
        if allow_empty:
            warnings.append(
                f"EPA splits are empty for {season} week {week}. This is "
                "expected in week 1, where the client's single-season method "
                "has no completed games to average."
            )
            return warnings
        raise ValidationError(
            f"EPA splits are empty for {season} week {week}."
        )

    required = {"team", "net_epa_play", "games_included"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValidationError(
            f"EPA splits are missing required columns: {', '.join(missing)}."
        )

    if df["team"].isna().any():
        raise ValidationError("EPA splits contain a row with no team code.")

    if not df["team"].is_unique:
        duplicates = sorted(df.loc[df["team"].duplicated(), "team"].unique())
        raise ValidationError(
            f"EPA splits contain duplicate teams: {', '.join(map(str, duplicates))}."
        )

    if "season" in df.columns:
        wrong = df.loc[df["season"] != season, "season"].unique()
        if len(wrong):
            raise ValidationError(
                f"EPA splits contain rows for season(s) {list(wrong)} but the "
                f"run targets {season}."
            )

    if df["net_epa_play"].isna().all():
        raise ValidationError("Every net_epa_play value in the EPA splits is blank.")

    blank = int(df["net_epa_play"].isna().sum())
    if blank:
        warnings.append(f"{blank} team(s) have a blank net_epa_play value.")

    if len(df) < 20:
        warnings.append(
            f"Only {len(df)} teams appear in the EPA splits (32 is a full "
            "slate). This is normal early in a season."
        )

    return warnings


def validate_fair_lines(
    df: pd.DataFrame,
    season: int,
    week: int,
    expected_games: Optional[int] = None,
) -> list:
    """Validate the fair-line frame. Returns non-fatal warnings."""
    warnings = []

    if df.empty:
        raise ValidationError(
            f"No fair-line rows were produced for {season} week {week}."
        )

    required = {"away_team", "home_team", "fair_spread_home", "status", "season", "week"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValidationError(
            f"Fair-line output is missing required columns: {', '.join(missing)}."
        )

    if df["away_team"].isna().any() or df["home_team"].isna().any():
        raise ValidationError("A fair-line row has a missing team code.")

    same = df[df["away_team"] == df["home_team"]]
    if len(same):
        raise ValidationError(
            f"{len(same)} fair-line row(s) list the same team home and away."
        )

    duplicated = df.duplicated(subset=["season", "week", "away_team", "home_team"])
    if duplicated.any():
        pairs = df.loc[duplicated, ["away_team", "home_team"]].apply(
            lambda r: f"{r['away_team']}@{r['home_team']}", axis=1
        )
        raise ValidationError(
            f"Duplicate matchup rows: {', '.join(sorted(set(pairs)))}."
        )

    wrong_season = df.loc[df["season"] != season, "season"].unique()
    if len(wrong_season):
        raise ValidationError(
            f"Fair-line rows reference season(s) {list(wrong_season)} but the "
            f"run targets {season}."
        )
    wrong_week = df.loc[df["week"] != week, "week"].unique()
    if len(wrong_week):
        raise ValidationError(
            f"Fair-line rows reference week(s) {list(wrong_week)} but the run "
            f"targets {week}."
        )

    projected = df[df["status"] == STATUS_PROJECTED]
    if not projected.empty:
        if projected["fair_spread_home"].isna().any():
            raise ValidationError(
                "A row marked as a generated projection has a blank "
                "fair_spread_home value."
            )
        # A linear model on EPA differentials should not produce absurd
        # numbers; anything beyond four touchdowns signals a data problem.
        extreme = projected[projected["fair_spread_home"].abs() > 28]
        if len(extreme):
            warnings.append(
                f"{len(extreme)} projection(s) exceed +/-28 points, which is "
                "unusually large for this model - worth a look."
            )
    else:
        warnings.append(
            f"No live projections were generated for {season} week {week}; "
            "every game was already completed or lacked trailing history."
        )

    insufficient = int((df["status"] == STATUS_INSUFFICIENT).sum())
    if insufficient:
        warnings.append(
            f"{insufficient} game(s) had insufficient trailing history and "
            "were left unprojected."
        )

    if expected_games is not None and len(df) != expected_games:
        warnings.append(
            f"{len(df)} fair-line rows were produced for {expected_games} "
            "scheduled games."
        )

    return warnings


def log_warnings(warnings: list, context: str) -> None:
    for warning in warnings:
        log.warning("[%s] %s", context, warning)
