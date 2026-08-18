"""Season/week detection and matchup extraction."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from conftest import make_schedule
from src.schedule import (
    PROJECTION_STATUS_COMPLETED,
    STATUS_COMPLETED,
    STATUS_UPCOMING,
    ScheduleError,
    annotate_schedule,
    calendar_season_guess,
    check_team_codes,
    detect_target_season_and_week,
    extract_matchups,
    next_friday_run_local,
    parse_manual_matchups,
    resolve_season_week,
    validate_week_exists,
)


def loader_for(*frames):
    """Return a schedule_loader serving the given frames by season."""
    combined = pd.concat(frames, ignore_index=True)

    def _load(seasons):
        subset = combined[combined["season"].isin(list(seasons))].copy()
        if subset.empty:
            raise RuntimeError(f"no schedule for {seasons}")
        return subset

    return _load


class TestCalendarSeasonGuess:
    @pytest.mark.parametrize(
        "when,expected",
        [
            (datetime(2026, 8, 18, tzinfo=timezone.utc), 2026),
            (datetime(2026, 9, 15, tzinfo=timezone.utc), 2026),
            (datetime(2026, 12, 25, tzinfo=timezone.utc), 2026),
            (datetime(2027, 1, 10, tzinfo=timezone.utc), 2026),
            (datetime(2027, 2, 8, tzinfo=timezone.utc), 2026),
            (datetime(2027, 5, 1, tzinfo=timezone.utc), 2026),
        ],
    )
    def test_february_still_belongs_to_the_previous_season(self, when, expected):
        assert calendar_season_guess(when) == expected


class TestAnnotateSchedule:
    def test_kickoff_converted_from_eastern_to_utc(self):
        schedule = make_schedule(weeks=1)

        annotated = annotate_schedule(
            schedule, now_utc=datetime(2026, 9, 1, tzinfo=timezone.utc)
        )

        # 2026-09-10 20:15 EDT (UTC-4) -> 2026-09-11 00:15 UTC
        thursday = annotated[annotated["weekday"] == "Thursday"].iloc[0]
        assert thursday["kickoff_utc"] == pd.Timestamp(
            "2026-09-11 00:15", tz="UTC"
        )

    def test_games_with_a_result_are_completed(self):
        schedule = make_schedule(weeks=2, completed_through_week=1)

        annotated = annotate_schedule(
            schedule, now_utc=datetime(2026, 9, 1, tzinfo=timezone.utc)
        )

        week1 = annotated[annotated["week"] == 1]
        assert (week1["schedule_status"] == STATUS_COMPLETED).all()

    def test_future_games_are_upcoming(self):
        schedule = make_schedule(weeks=2)

        annotated = annotate_schedule(
            schedule, now_utc=datetime(2026, 9, 1, tzinfo=timezone.utc)
        )

        assert (annotated["schedule_status"] == STATUS_UPCOMING).all()


class TestDetection:
    def test_picks_the_first_week_with_a_future_kickoff(self):
        schedule = make_schedule(season=2026, weeks=4)
        loader = loader_for(schedule)

        season, week, info = detect_target_season_and_week(
            now_utc=datetime(2026, 9, 1, tzinfo=timezone.utc),
            schedule_loader=loader,
        )

        assert (season, week) == (2026, 1)
        assert info["detected_by"] == "schedule_first_pending_week"

    def test_friday_after_thursday_night_still_selects_current_week(self):
        """The scheduled run is Friday 18:00 PT - TNF has already happened."""
        schedule = make_schedule(
            season=2026, weeks=4, completed_through_week=1,
            completed_games_in_current_week=1,
        )
        loader = loader_for(schedule)
        # Friday of week 2: Thursday's game is done, Sunday's is not.
        friday = datetime(2026, 9, 18, 22, 0, tzinfo=timezone.utc)

        season, week, _ = detect_target_season_and_week(
            now_utc=friday, schedule_loader=loader
        )

        assert (season, week) == (2026, 2)

    def test_rolls_over_to_the_next_season_when_one_is_complete(self):
        finished = make_schedule(
            season=2026, weeks=2, completed_through_week=2,
            start_date="2026-09-10",
        )
        upcoming = make_schedule(season=2027, weeks=2, start_date="2027-09-09")
        loader = loader_for(finished, upcoming)

        season, week, info = detect_target_season_and_week(
            now_utc=datetime(2027, 3, 1, tzinfo=timezone.utc),
            schedule_loader=loader,
        )

        assert (season, week) == (2027, 1)
        assert info["detected_by"] == "next_season_rollover"

    def test_preseason_gap_selects_the_upcoming_season_not_the_finished_one(self):
        """The trap nflreadpy.get_current_season() falls into in August."""
        finished = make_schedule(
            season=2025, weeks=2, completed_through_week=2,
            start_date="2025-09-04",
        )
        upcoming = make_schedule(season=2026, weeks=2, start_date="2026-09-10")
        loader = loader_for(finished, upcoming)

        season, week, _ = detect_target_season_and_week(
            now_utc=datetime(2026, 8, 18, tzinfo=timezone.utc),
            schedule_loader=loader,
        )

        assert (season, week) == (2026, 1)

    def test_a_cancelled_game_without_a_result_cannot_wedge_detection(self):
        schedule = make_schedule(season=2026, weeks=3, completed_through_week=3)
        # Week 1 Thursday never gets a result (postponed and abandoned).
        schedule.loc[
            (schedule["week"] == 1) & (schedule["weekday"] == "Thursday"),
            ["result", "home_score", "away_score"],
        ] = None
        loader = loader_for(schedule, make_schedule(season=2027, weeks=1,
                                                   start_date="2027-09-09"))

        season, week, _ = detect_target_season_and_week(
            now_utc=datetime(2026, 10, 20, tzinfo=timezone.utc),
            schedule_loader=loader,
        )

        assert (season, week) == (2027, 1)

    def test_raises_when_nothing_can_be_determined(self):
        def loader(seasons):
            raise RuntimeError("network down")

        with pytest.raises(ScheduleError):
            detect_target_season_and_week(
                now_utc=datetime(2026, 9, 1, tzinfo=timezone.utc),
                schedule_loader=loader,
            )


class TestResolveSeasonWeek:
    def test_manual_override_wins(self):
        season, week, info = resolve_season_week(season=2025, week=6)

        assert (season, week) == (2025, 6)
        assert info["detected_by"] == "manual_override"

    def test_half_an_override_is_rejected(self):
        with pytest.raises(ScheduleError):
            resolve_season_week(season=2025, week=None)


class TestValidateWeekExists:
    def test_accepts_a_real_week(self):
        validate_week_exists(make_schedule(season=2026, weeks=4), 2026, 3)

    def test_rejects_a_week_beyond_the_season(self):
        with pytest.raises(ScheduleError) as exc:
            validate_week_exists(make_schedule(season=2026, weeks=4), 2026, 9)

        assert "1-4" in str(exc.value)


class TestExtractMatchups:
    def test_home_and_away_are_not_reversed(self):
        schedule = make_schedule(season=2026, weeks=1)

        matchups = extract_matchups(
            schedule, 2026, 1, now_utc=datetime(2026, 9, 1, tzinfo=timezone.utc)
        )

        for matchup in matchups:
            row = schedule[
                (schedule["away_team"] == matchup["away_team"])
                & (schedule["home_team"] == matchup["home_team"])
            ]
            assert len(row) == 1

    def test_completed_games_are_flagged_not_projected(self):
        schedule = make_schedule(
            season=2026, weeks=2, completed_through_week=1,
            completed_games_in_current_week=1,
        )
        friday = datetime(2026, 9, 18, 22, 0, tzinfo=timezone.utc)

        matchups = extract_matchups(schedule, 2026, 2, now_utc=friday)

        completed = [m for m in matchups if m["status"] == PROJECTION_STATUS_COMPLETED]
        upcoming = [m for m in matchups if m["status"] is None]
        assert len(completed) == 1
        assert len(upcoming) == 1

    def test_include_completed_projects_everything(self):
        schedule = make_schedule(
            season=2026, weeks=2, completed_through_week=2,
        )

        matchups = extract_matchups(
            schedule, 2026, 2, include_completed=True,
            now_utc=datetime(2026, 10, 1, tzinfo=timezone.utc),
        )

        assert all(m["status"] is None for m in matchups)

    def test_no_duplicate_matchups(self):
        schedule = make_schedule(season=2026, weeks=1)
        doubled = pd.concat([schedule, schedule], ignore_index=True)

        matchups = extract_matchups(
            doubled, 2026, 1, now_utc=datetime(2026, 9, 1, tzinfo=timezone.utc)
        )

        pairs = [(m["away_team"], m["home_team"]) for m in matchups]
        assert len(pairs) == len(set(pairs))

    def test_same_team_both_sides_is_skipped(self):
        schedule = make_schedule(season=2026, weeks=1)
        schedule.loc[0, "away_team"] = schedule.loc[0, "home_team"]

        matchups = extract_matchups(
            schedule, 2026, 1, now_utc=datetime(2026, 9, 1, tzinfo=timezone.utc)
        )

        assert all(m["away_team"] != m["home_team"] for m in matchups)

    def test_missing_week_raises(self):
        with pytest.raises(ScheduleError):
            extract_matchups(make_schedule(season=2026, weeks=2), 2026, 5)


class TestManualMatchups:
    def test_parses_the_original_format(self):
        matchups = parse_manual_matchups("KC@BUF,SF@LA")

        assert [(m["away_team"], m["home_team"]) for m in matchups] == [
            ("KC", "BUF"),
            ("SF", "LA"),
        ]

    def test_tolerates_whitespace_and_case(self):
        matchups = parse_manual_matchups(" kc@buf , sf@la ")

        assert matchups[0]["away_team"] == "KC"
        assert matchups[1]["home_team"] == "LA"

    @pytest.mark.parametrize("text", ["KCBUF", "KC@BUF@LA", "KC@KC", "@BUF", ""])
    def test_rejects_malformed_input(self, text):
        with pytest.raises(ScheduleError):
            parse_manual_matchups(text)


class TestTeamCodeCheck:
    def test_flags_unknown_codes(self):
        matchups = [{"away_team": "KC", "home_team": "ZZZ"}]

        warnings = check_team_codes(matchups, {"KC", "BUF"})

        assert len(warnings) == 1
        assert "ZZZ" in warnings[0]

    def test_no_warnings_when_team_list_unavailable(self):
        assert check_team_codes([{"away_team": "ZZ", "home_team": "YY"}], set()) == []


class TestNextFridayRun:
    def test_is_always_a_future_friday_at_six_pm_pacific(self):
        for day in range(1, 29):
            when = datetime(2026, 9, day, 12, 0, tzinfo=timezone.utc)
            nxt = next_friday_run_local(when)
            assert nxt.weekday() == 4
            assert (nxt.hour, nxt.minute) == (18, 0)
            assert nxt > when
