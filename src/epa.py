"""EPA splits - the client's original `nfl_epa_splits.py` maths, preserved.

Nothing in this module changes how EPA is computed. The functions are the
same aggregations the client already ran; they were moved here so that both
the standalone script and ``weekly_runner.py`` share one implementation
instead of two drifting copies.

Deliberately preserved behaviours (implementation guide section 43):

* regular-season pass/run plays with a non-null EPA only;
* offence = mean EPA on plays where the team had the ball, split pass/rush;
* defence = mean EPA allowed on plays where the team was defending;
* the trailing window is the last ``n`` games *strictly before* the analysis
  week, default ``n = 5``;
* every component mean is rounded to 3 decimals **before** ``net_epa_play``
  is derived from those rounded values, exactly as the original did;
* ``net_epa_play`` is a sum-based differential
  ``(off_pass + off_rush) - (def_pass_allowed + def_rush_allowed)``;
* output is sorted by ``net_epa_play`` descending.
"""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from .logging_utils import get_logger

log = get_logger("epa")

#: Columns the EPA aggregation cannot run without.
REQUIRED_PBP_COLUMNS: tuple = (
    "season",
    "season_type",
    "week",
    "game_id",
    "play_type",
    "epa",
    "posteam",
    "defteam",
)

#: Column order used for the EPA splits CSV / Sheet tab.
EPA_SPLIT_COLUMNS: tuple = (
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
)

_BASE_SPLIT_COLUMNS: tuple = (
    "team",
    "games_included",
    "weeks",
    "off_epa_pass",
    "off_epa_rush",
    "def_epa_pass_allowed",
    "def_epa_rush_allowed",
    "net_epa_play",
)


class InsufficientDataError(RuntimeError):
    """Raised when there is not enough play-by-play data to produce output."""


def empty_splits_frame(with_metadata: bool = False) -> pd.DataFrame:
    """An empty frame carrying the right columns, for early-season weeks."""
    columns = EPA_SPLIT_COLUMNS if with_metadata else _BASE_SPLIT_COLUMNS
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def team_game_epa(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse play-by-play into one row per team per game.

    Identical to the client's original aggregation, with ``season`` added to
    the grouping key. For the single-season input the original script used,
    adding ``season`` cannot change the result (every row shares one season);
    it simply makes the same function safe to reuse across seasons.
    """
    if df.empty:
        raise InsufficientDataError(
            "No qualifying NFL play-by-play data was available for EPA splits."
        )

    off = (
        df.groupby(["posteam", "season", "week", "game_id", "play_type"])["epa"]
        .mean()
        .unstack("play_type")
        .rename(columns={"pass": "off_epa_pass", "run": "off_epa_rush"})
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    deff = (
        df.groupby(["defteam", "season", "week", "game_id", "play_type"])["epa"]
        .mean()
        .unstack("play_type")
        .rename(
            columns={"pass": "def_epa_pass_allowed", "run": "def_epa_rush_allowed"}
        )
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    merged = off.merge(deff, on=["team", "season", "week", "game_id"], how="outer")

    # A play_type that never occurs in the slice leaves the column missing
    # entirely after unstack(); create it as NaN so downstream means work.
    for column in (
        "off_epa_pass",
        "off_epa_rush",
        "def_epa_pass_allowed",
        "def_epa_rush_allowed",
    ):
        if column not in merged.columns:
            merged[column] = pd.NA

    return merged.sort_values(["team", "season", "week"])


def last_n_splits(
    game_epa: pd.DataFrame,
    through_week: int,
    n: int = 5,
) -> pd.DataFrame:
    """Average each team's last ``n`` games before ``through_week``.

    This is the client's original ``last_n_splits`` with two additions that
    do not alter any computed value: an empty-input guard, and sorting by
    ``(season, week)`` rather than ``week`` alone so that the function stays
    correct if it is ever handed multi-season input.
    """
    if game_epa.empty:
        return empty_splits_frame()

    eligible = game_epa[game_epa["week"] < through_week]
    if eligible.empty:
        return empty_splits_frame()

    sort_keys = ["season", "week"] if "season" in eligible.columns else ["week"]

    results = []
    for team, grp in eligible.groupby("team"):
        grp = grp.sort_values(sort_keys, kind="mergesort").tail(n)
        if grp.empty:
            continue
        results.append(
            {
                "team": team,
                "games_included": len(grp),
                "weeks": ", ".join(str(w) for w in grp["week"]),
                "off_epa_pass": round(grp["off_epa_pass"].mean(), 3),
                "off_epa_rush": round(grp["off_epa_rush"].mean(), 3),
                "def_epa_pass_allowed": round(grp["def_epa_pass_allowed"].mean(), 3),
                "def_epa_rush_allowed": round(grp["def_epa_rush_allowed"].mean(), 3),
            }
        )

    if not results:
        return empty_splits_frame()

    out = pd.DataFrame(results)
    # net EPA/play differential - the single number most useful for Step 4.
    # Derived from the already-rounded component means, as in the original.
    out["net_epa_play"] = (
        (out["off_epa_pass"] + out["off_epa_rush"])
        - (out["def_epa_pass_allowed"] + out["def_epa_rush_allowed"])
    ).round(3)
    return out.sort_values("net_epa_play", ascending=False).reset_index(drop=True)


def build_epa_splits(
    pbp: pd.DataFrame,
    season: int,
    week: int,
    n: int = 5,
    run_timestamp_utc: Optional[str] = None,
    strict: bool = True,
) -> pd.DataFrame:
    """Full EPA-splits pipeline with metadata columns attached.

    Parameters
    ----------
    pbp:
        Filtered play-by-play (regular season, pass/run, EPA present).
    season, week:
        The analysis target. Only games with ``week < week`` are used.
    n:
        Trailing game count (client default 5).
    strict:
        When ``True`` an empty result raises :class:`InsufficientDataError`.
        ``weekly_runner`` sets this to ``False`` for week 1, where the
        client's single-season method legitimately has no prior games.
    """
    game_epa = team_game_epa(pbp)
    splits = last_n_splits(game_epa, through_week=week, n=n)

    if splits.empty:
        message = (
            f"No EPA splits could be produced for season={season}, week={week}. "
            "The client's method uses only completed games from the same "
            "season, so week 1 has no prior games to average."
        )
        if strict:
            raise InsufficientDataError(message)
        log.warning(message)
        return empty_splits_frame(with_metadata=True)

    splits = splits.copy()
    splits.insert(0, "run_timestamp_utc", run_timestamp_utc or "")
    splits.insert(0, "analysis_week", int(week))
    splits.insert(0, "season", int(season))

    log.info(
        "EPA calculation complete: %d teams, trailing n=%d, through week %d.",
        len(splits),
        n,
        week - 1,
    )
    return splits[list(EPA_SPLIT_COLUMNS)]


def require_columns(df: pd.DataFrame, required: Sequence[str], source: str) -> None:
    """Fail loudly (and usefully) when nflverse renames or drops a field."""
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise RuntimeError(
            f"{source} is missing required columns: {', '.join(missing)}. "
            "The nflverse schema may have changed - check the nflreadpy "
            "release notes before rerunning."
        )
