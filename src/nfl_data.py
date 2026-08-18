"""The single boundary between nflverse data and the rest of the project.

``nflreadpy`` returns Polars frames; every other module in this project
works in pandas, exactly like the client's original scripts. Converting in
one place (implementation guide section 8) keeps the migration low-risk:
the aggregation code below this boundary is unchanged pandas.

Responsibilities:

* retry genuinely transient download failures (network / HTTP), never
  deterministic ones;
* validate that the nflverse schema still contains the fields we need;
* narrow the 372-column play-by-play frame to the columns actually used,
  which cuts memory by well over an order of magnitude;
* apply the client's original row filter in pandas so the ``NaN`` versus
  ``null`` semantics match the original behaviour exactly.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import pandas as pd

from .epa import require_columns
from .logging_utils import get_logger
from .net import enable_system_trust_store
from .retry import with_retries

log = get_logger("data")

#: Columns the project actually consumes out of play-by-play.
PBP_COLUMNS: tuple = (
    "game_id",
    "season",
    "season_type",
    "week",
    "play_type",
    "epa",
    "posteam",
    "defteam",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
)

#: Columns the project consumes out of the schedule.
SCHEDULE_COLUMNS: tuple = (
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "weekday",
    "gametime",
    "away_team",
    "home_team",
    "away_score",
    "home_score",
    "result",
)

# nflreadpy surfaces network problems as ConnectionError; urllib/OS issues
# arrive as OSError or TimeoutError. Everything else is deterministic and
# must not be retried (implementation guide 20.1).
TRANSIENT_ERRORS: tuple = (ConnectionError, TimeoutError, OSError)


class NflDataError(RuntimeError):
    """Raised when nflverse data cannot be loaded or fails validation."""


def _nflreadpy():
    """Import nflreadpy lazily so unit tests can run without the network."""
    enable_system_trust_store()
    try:
        import nflreadpy
    except ImportError as exc:  # pragma: no cover
        raise NflDataError(
            "nflreadpy is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return nflreadpy


def _normalise_seasons(seasons: int | Iterable[int]) -> list:
    if isinstance(seasons, int):
        return [seasons]
    values = sorted({int(s) for s in seasons})
    if not values:
        raise ValueError("At least one season must be requested.")
    return values


def _select(frame, columns: Sequence[str], source: str):
    """Narrow a Polars frame to ``columns``, failing loudly if any are gone."""
    available = set(frame.columns)
    missing = sorted(set(columns) - available)
    if missing:
        raise NflDataError(
            f"{source} is missing required columns: {', '.join(missing)}. "
            "The nflverse schema may have changed - check the nflreadpy "
            "release notes before rerunning."
        )
    return frame.select(list(columns))


def filter_plays(df: pd.DataFrame) -> pd.DataFrame:
    """The client's original play filter, unchanged.

    Regular-season pass/run plays that carry a valid EPA value. Applied in
    pandas (not Polars) because Polars treats float ``NaN`` and ``null`` as
    distinct, whereas ``notna()`` excludes both - and the original relied on
    the pandas meaning.
    """
    return df[
        (df["season_type"] == "REG")
        & (df["play_type"].isin(["pass", "run"]))
        & (df["epa"].notna())
    ].copy()


def load_pbp(
    seasons: int | Iterable[int],
    apply_filter: bool = True,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load play-by-play for ``seasons`` as a pandas DataFrame."""
    season_list = _normalise_seasons(seasons)
    wanted = tuple(columns) if columns else PBP_COLUMNS

    log.info("PBP download started for seasons %s.", season_list)

    def _download():
        nfl = _nflreadpy()
        return nfl.load_pbp(season_list)

    frame = with_retries(
        _download,
        description=f"nflreadpy.load_pbp({season_list})",
        retry_on=TRANSIENT_ERRORS,
    )

    df = _select(frame, wanted, "NFL play-by-play data").to_pandas()

    if df.empty:
        raise NflDataError(
            f"nflverse returned no play-by-play rows for seasons {season_list}. "
            "The data for this season may not be published yet."
        )

    df = _coerce_week_season(df)

    raw_rows = len(df)
    if apply_filter:
        df = filter_plays(df)
        if df.empty:
            raise NflDataError(
                f"No regular-season pass/run plays with a valid EPA were found "
                f"for seasons {season_list} (downloaded {raw_rows} raw rows)."
            )

    log.info(
        "PBP download completed: %d rows retained from %d for seasons %s.",
        len(df),
        raw_rows,
        season_list,
    )
    return df


def max_available_pbp_season() -> Optional[int]:
    """Newest season nflreadpy will serve play-by-play for.

    ``load_pbp`` raises ``ValueError`` for a season it has no data for, so
    an out-of-season run (target 2026 week 1 while only 2025 exists) has to
    know the ceiling rather than discover it as a crash.
    """
    try:
        nfl = _nflreadpy()
        return int(nfl.get_current_season())
    except Exception as exc:  # pragma: no cover - depends on upstream
        log.debug("Could not determine the newest PBP season (%s).", exc)
        return None


def load_pbp_available(
    seasons: int | Iterable[int],
    apply_filter: bool = True,
    columns: Optional[Sequence[str]] = None,
) -> tuple:
    """Load play-by-play for whichever requested seasons actually exist.

    Returns ``(dataframe, loaded_seasons)``. Seasons with no published data
    are skipped with a log line instead of aborting the run - this is what
    lets a pre-season run still produce fair lines from last season's
    trailing history, exactly as the client's original script did.
    """
    requested = _normalise_seasons(seasons)
    cap = max_available_pbp_season()

    usable = [s for s in requested if cap is None or s <= cap]
    skipped = [s for s in requested if cap is not None and s > cap]
    if skipped:
        log.info(
            "Seasons %s have no play-by-play published yet; using %s.",
            skipped,
            usable,
        )
    if not usable:
        raise NflDataError(
            f"None of the requested seasons {requested} have play-by-play "
            "data published yet."
        )

    while usable:
        try:
            return load_pbp(usable, apply_filter=apply_filter, columns=columns), list(usable)
        except (ValueError, NflDataError) as exc:
            if len(usable) == 1:
                raise
            dropped = usable.pop()
            log.warning(
                "Play-by-play for %s is unavailable (%s); retrying without it.",
                dropped,
                exc,
            )

    raise NflDataError(f"No play-by-play data could be loaded for {requested}.")


def load_schedules(seasons: int | Iterable[int]) -> pd.DataFrame:
    """Load schedule rows for ``seasons`` as a pandas DataFrame."""
    season_list = _normalise_seasons(seasons)

    def _download():
        nfl = _nflreadpy()
        return nfl.load_schedules(season_list)

    frame = with_retries(
        _download,
        description=f"nflreadpy.load_schedules({season_list})",
        retry_on=TRANSIENT_ERRORS,
    )

    df = _select(frame, SCHEDULE_COLUMNS, "NFL schedule data").to_pandas()
    df = df[df["season"].isin(season_list)].copy()

    if df.empty:
        raise NflDataError(
            f"nflverse returned no schedule rows for seasons {season_list}. "
            "Next season's schedule is usually published in May."
        )

    df = _coerce_week_season(df)
    require_columns(df, SCHEDULE_COLUMNS, "NFL schedule data")
    log.info("Schedule loaded: %d rows for seasons %s.", len(df), season_list)
    return df


def load_team_codes() -> set:
    """Current nflverse team abbreviations, used to sanity-check matchups."""

    def _download():
        nfl = _nflreadpy()
        return nfl.load_teams()

    try:
        frame = with_retries(
            _download,
            description="nflreadpy.load_teams()",
            retry_on=TRANSIENT_ERRORS,
        )
    except Exception as exc:
        # Team validation is a convenience check, not a hard dependency:
        # the schedule itself is the authoritative source of team codes
        # (implementation guide 31.3).
        log.warning("Could not load the team list (%s). Skipping code check.", exc)
        return set()

    df = frame.to_pandas()
    for candidate in ("team_abbr", "team", "abbr"):
        if candidate in df.columns:
            return set(df[candidate].dropna().astype(str))
    return set()


def _coerce_week_season(df: pd.DataFrame) -> pd.DataFrame:
    """Make ``season``/``week`` plain Python ints.

    nflverse ships these as int32. Keeping them as int64 avoids surprises in
    comparisons and, importantly, makes the EPA ``weeks`` column read
    ``"1, 2, 3"`` rather than ``"1.0, 2.0, 3.0"``.
    """
    for column in ("season", "week"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("int64")
    return df
