"""Automatic season / week detection and matchup extraction.

Why this is not a one-liner
---------------------------
``nflreadpy.get_current_season()`` answers "which season are we in?" using
the Thursday-after-Labor-Day rule, so during the summer it returns the
season that just *finished*. Verified on 2026-08-18, it returns ``2025``
even though the next games to play are 2026 week 1. The implementation
guide (11 / 31.1) calls this out explicitly: do not trust a single helper
across all calendar dates.

So detection is driven by the schedule itself. The target week is the
first regular-season week that still contains a game whose kickoff is in
the future. That rule is robust in ways a results-based rule is not:

* on Friday evening it still selects the current week even though
  Thursday Night Football has already been played;
* it rolls forward the moment the last game of a week kicks off;
* a postponed or cancelled game that never receives a result cannot wedge
  detection on a stale week.

Kickoff times are built from the schedule's ``gameday`` + ``gametime``,
which nflverse publishes in US Eastern time, and converted to UTC.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Optional, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from .logging_utils import get_logger

log = get_logger("schedule")

EASTERN = ZoneInfo("America/New_York")
PACIFIC = ZoneInfo("America/Los_Angeles")

REGULAR_SEASON = "REG"
MAX_REGULAR_WEEK = 22

STATUS_UPCOMING = "upcoming"
STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETED = "completed"

#: Value written into the projection ``status`` column for games that are
#: not eligible for a new fair line. Mirrors ``fair_line.STATUS_COMPLETED``.
PROJECTION_STATUS_COMPLETED = "already_completed"


class ScheduleError(RuntimeError):
    """Raised when the schedule cannot support the requested analysis."""


# ---------------------------------------------------------------------
# Kickoff timing
# ---------------------------------------------------------------------

def _kickoff_utc(gameday, gametime) -> Optional[pd.Timestamp]:
    """Combine nflverse ``gameday`` + ``gametime`` (US Eastern) into UTC."""
    if gameday is None or (isinstance(gameday, float) and pd.isna(gameday)):
        return None
    try:
        date_part = pd.to_datetime(str(gameday)[:10], format="%Y-%m-%d")
    except (ValueError, TypeError):
        return None

    time_text = "" if gametime is None else str(gametime).strip()
    if not time_text or time_text.lower() in {"nan", "none", "nat"}:
        # No published kickoff yet: treat the game as late in its own day so
        # the week stays 'upcoming' for the whole of that date.
        hour, minute = 23, 59
    else:
        try:
            hour, minute = (int(part) for part in time_text.split(":")[:2])
        except (ValueError, TypeError):
            hour, minute = 23, 59

    local = datetime(
        date_part.year, date_part.month, date_part.day, hour, minute, tzinfo=EASTERN
    )
    return pd.Timestamp(local.astimezone(timezone.utc))


def annotate_schedule(
    schedule: pd.DataFrame,
    now_utc: Optional[datetime] = None,
) -> pd.DataFrame:
    """Add ``kickoff_utc`` and a ``schedule_status`` column."""
    now = now_utc or datetime.now(timezone.utc)
    now_ts = pd.Timestamp(now)

    df = schedule.copy()
    df["kickoff_utc"] = [
        _kickoff_utc(gameday, gametime)
        for gameday, gametime in zip(df["gameday"], df["gametime"])
    ]

    has_result = df["result"].notna()
    if "home_score" in df.columns:
        has_result = has_result | df["home_score"].notna()

    kicked_off = df["kickoff_utc"].map(
        lambda ts: ts is not None and ts <= now_ts
    )

    df["schedule_status"] = STATUS_UPCOMING
    df.loc[kicked_off, "schedule_status"] = STATUS_IN_PROGRESS
    df.loc[has_result, "schedule_status"] = STATUS_COMPLETED
    return df


# ---------------------------------------------------------------------
# Season / week detection
# ---------------------------------------------------------------------

def calendar_season_guess(now_utc: Optional[datetime] = None) -> int:
    """The NFL season a given date belongs to.

    An NFL season is named for the year it starts in and runs into
    February, so anything before roughly March belongs to the previous
    season's year. August onward clearly belongs to the coming season.
    """
    now = now_utc or datetime.now(timezone.utc)
    return now.year if now.month >= 8 else now.year - 1


def _first_pending_week(
    annotated: pd.DataFrame,
    now_utc: datetime,
) -> Optional[int]:
    """Lowest REG week that still has a game kicking off in the future."""
    now_ts = pd.Timestamp(now_utc)
    reg = annotated[annotated["game_type"] == REGULAR_SEASON]
    if reg.empty:
        return None

    future = reg[
        reg["kickoff_utc"].map(lambda ts: ts is not None and ts > now_ts)
    ]
    if future.empty:
        return None
    return int(future["week"].min())


def detect_target_season_and_week(
    now_utc: Optional[datetime] = None,
    schedule_loader: Optional[Callable[[Sequence[int]], pd.DataFrame]] = None,
) -> tuple:
    """Return ``(season, week, info)`` for the week that should be analysed.

    ``info`` carries diagnostic detail for the logs and the run summary.
    """
    if schedule_loader is None:
        from .nfl_data import load_schedules as schedule_loader  # local import

    now = now_utc or datetime.now(timezone.utc)
    guess = calendar_season_guess(now)

    # Try the calendar-derived season, then the previous one (covers a run
    # in early August before the new schedule is published), then the next.
    for season in (guess, guess - 1):
        try:
            schedule = schedule_loader([season])
        except Exception as exc:
            log.warning("Schedule for %s is unavailable (%s).", season, exc)
            continue

        annotated = annotate_schedule(schedule, now_utc=now)
        week = _first_pending_week(annotated, now)

        if week is not None:
            info = {
                "detected_by": "schedule_first_pending_week",
                "season": season,
                "week": week,
                "reason": (
                    f"{season} week {week} is the first regular-season week "
                    "with a kickoff still in the future"
                ),
            }
            log.info("Target detected: %s week %s.", season, week)
            return season, week, info

        # Every regular-season game of this season has kicked off.
        # If the following season is published, week 1 of it is next.
        next_season = season + 1
        try:
            next_schedule = schedule_loader([next_season])
        except Exception:
            next_schedule = None

        if next_schedule is not None and not next_schedule.empty:
            next_annotated = annotate_schedule(next_schedule, now_utc=now)
            next_week = _first_pending_week(next_annotated, now)
            if next_week is not None:
                info = {
                    "detected_by": "next_season_rollover",
                    "season": next_season,
                    "week": next_week,
                    "reason": (
                        f"{season} regular season is complete; "
                        f"{next_season} week {next_week} is next"
                    ),
                }
                log.info(
                    "Season %s is complete. Target detected: %s week %s.",
                    season,
                    next_season,
                    next_week,
                )
                return next_season, next_week, info

        log.warning(
            "Season %s appears complete and %s is not published yet.",
            season,
            next_season,
        )

    raise ScheduleError(
        "Could not determine the target NFL season and week from the "
        f"schedule around {now.date()}. Rerun with explicit "
        "--season and --week values, or check that nflverse schedule data "
        "is reachable."
    )


def resolve_season_week(
    season: Optional[int] = None,
    week: Optional[int] = None,
    now_utc: Optional[datetime] = None,
    schedule_loader: Optional[Callable[[Sequence[int]], pd.DataFrame]] = None,
) -> tuple:
    """Honour explicit overrides, otherwise detect automatically."""
    if season is not None and week is not None:
        info = {"detected_by": "manual_override", "season": season, "week": week,
                "reason": "explicit --season/--week arguments"}
        log.info("Using manual override: %s week %s.", season, week)
        return int(season), int(week), info

    if (season is None) != (week is None):
        raise ScheduleError(
            "--season and --week must be supplied together (or neither, to "
            "use automatic detection)."
        )

    return detect_target_season_and_week(
        now_utc=now_utc, schedule_loader=schedule_loader
    )


def validate_week_exists(schedule: pd.DataFrame, season: int, week: int) -> None:
    """Fail early when a requested week has no regular-season games."""
    reg = schedule[
        (schedule["season"] == season) & (schedule["game_type"] == REGULAR_SEASON)
    ]
    if reg.empty:
        raise ScheduleError(
            f"No regular-season schedule rows exist for {season}."
        )
    weeks = sorted(int(w) for w in reg["week"].unique())
    if int(week) not in weeks:
        raise ScheduleError(
            f"Week {week} does not exist in the {season} regular season. "
            f"Available weeks: {weeks[0]}-{weeks[-1]}."
        )


# ---------------------------------------------------------------------
# Matchups
# ---------------------------------------------------------------------

def extract_matchups(
    schedule: pd.DataFrame,
    season: int,
    week: int,
    include_completed: bool = False,
    now_utc: Optional[datetime] = None,
) -> list:
    """Return one dict per scheduled regular-season game for the week.

    Every game in the week is returned so the client can see the full
    slate. Games that have already kicked off carry
    ``status='already_completed'``, which stops the fair-line stage from
    presenting a finished Thursday game as an upcoming pick
    (implementation guide 12.1 / 31.9). Setting ``include_completed``
    projects them anyway, which is useful for historical backfills.
    """
    validate_week_exists(schedule, season, week)

    annotated = annotate_schedule(schedule, now_utc=now_utc)
    rows = annotated[
        (annotated["season"] == season)
        & (annotated["week"] == week)
        & (annotated["game_type"] == REGULAR_SEASON)
    ].copy()

    if rows.empty:
        raise ScheduleError(
            f"No regular-season games found for {season} week {week}."
        )

    rows = rows.sort_values(["kickoff_utc", "game_id"], na_position="last")

    seen = set()
    matchups = []
    for _, row in rows.iterrows():
        key = (row["away_team"], row["home_team"])
        if key in seen:
            log.warning(
                "Duplicate schedule row for %s@%s in %s week %s; skipping.",
                row["away_team"],
                row["home_team"],
                season,
                week,
            )
            continue
        seen.add(key)

        if pd.isna(row["away_team"]) or pd.isna(row["home_team"]):
            log.warning("Schedule row %s has a missing team code; skipping.",
                        row.get("game_id"))
            continue
        if row["away_team"] == row["home_team"]:
            log.warning(
                "Schedule row %s lists the same team home and away; skipping.",
                row.get("game_id"),
            )
            continue

        already_played = row["schedule_status"] != STATUS_UPCOMING
        matchups.append(
            {
                "game_id": row.get("game_id"),
                "season": int(row["season"]),
                "week": int(row["week"]),
                "game_date": str(row.get("gameday") or ""),
                "game_time_et": str(row.get("gametime") or ""),
                "weekday": str(row.get("weekday") or ""),
                "away_team": row["away_team"],
                "home_team": row["home_team"],
                "kickoff_utc": row["kickoff_utc"],
                "schedule_status": row["schedule_status"],
                "status": (
                    PROJECTION_STATUS_COMPLETED
                    if already_played and not include_completed
                    else None
                ),
            }
        )

    if not matchups:
        raise ScheduleError(
            f"No usable matchups could be extracted for {season} week {week}."
        )

    counts = {}
    for matchup in matchups:
        counts[matchup["schedule_status"]] = (
            counts.get(matchup["schedule_status"], 0) + 1
        )
    log.info(
        "Found %d scheduled games for %s week %s (%s).",
        len(matchups),
        season,
        week,
        ", ".join(f"{v} {k}" for k, v in sorted(counts.items())),
    )
    return matchups


def parse_manual_matchups(text: str) -> list:
    """Parse the original ``'KC@BUF,SF@LA'`` debugging format."""
    matchups = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.count("@") != 1:
            raise ScheduleError(
                f"Could not parse matchup {chunk!r}. Expected AWAY@HOME, "
                "for example 'KC@BUF'."
            )
        away, home = (part.strip().upper() for part in chunk.split("@"))
        if not away or not home:
            raise ScheduleError(f"Could not parse matchup {chunk!r}.")
        if away == home:
            raise ScheduleError(
                f"Matchup {chunk!r} lists the same team on both sides."
            )
        matchups.append(
            {
                "game_id": None,
                "away_team": away,
                "home_team": home,
                "game_date": "",
                "schedule_status": STATUS_UPCOMING,
                "status": None,
            }
        )
    if not matchups:
        raise ScheduleError("No matchups could be parsed from --matchups.")
    return matchups


def check_team_codes(matchups: Iterable[dict], known_teams: set) -> list:
    """Return warnings for team codes nflverse does not recognise."""
    if not known_teams:
        return []
    warnings = []
    for matchup in matchups:
        for side in ("away_team", "home_team"):
            code = matchup.get(side)
            if code and code not in known_teams:
                warnings.append(
                    f"Team code {code!r} in "
                    f"{matchup.get('away_team')}@{matchup.get('home_team')} "
                    "is not in the current nflverse team list."
                )
    for warning in warnings:
        log.warning("%s", warning)
    return warnings


def next_friday_run_local(now_utc: Optional[datetime] = None) -> datetime:
    """The next scheduled Friday 18:00 America/Los_Angeles run (for logs)."""
    now = (now_utc or datetime.now(timezone.utc)).astimezone(PACIFIC)
    days_ahead = (4 - now.weekday()) % 7  # Monday=0 ... Friday=4
    candidate = now.replace(hour=18, minute=0, second=0, microsecond=0) + timedelta(
        days=days_ahead
    )
    if candidate <= now:
        candidate += timedelta(days=7)
    return candidate
