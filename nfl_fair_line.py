"""
NFL Fair-Line Model  (nflreadpy edition)
-----------------------------------------
Trains a simple regression mapping each team's trailing-5-game net EPA/play
differential to actual game margins, then uses it to convert this week's
EPA into a projected point spread you can compare against the market.

This is the migrated version of the client's original script. The
regression methodology is unchanged - see ``original/nfl_fair_line.py`` for
the untouched reference copy and ``src/fair_line.py`` for the shared
implementation.

What changed in the migration:
  * data now comes from ``nflreadpy`` instead of ``nfl_data_py``;
  * the model lives at ``models/fair_line_model.pkl`` (a project-relative
    path) with a ``models/model_metadata.json`` provenance sidecar;
  * training writes atomically, so a failed retrain cannot destroy the
    last working model;
  * the trailing-game ``n`` recorded at training time is enforced at
    projection time instead of being silently ignored;
  * ``project`` returns a pandas DataFrame and writes a CSV, rather than
    only printing;
  * ``--matchups`` is now optional: without it the week's real schedule is
    used automatically.

The sign convention is unchanged:
    fair_spread_home = -predicted_home_margin
    negative -> home team favoured;  positive -> home team underdog.

USAGE:
    Train + evaluate the model on historical seasons:
        python nfl_fair_line.py train --seasons 2021 2022 2023 2024

    Project spreads for a week using the real schedule:
        python nfl_fair_line.py project --season 2025 --week 6

    Project specific matchups (debugging override):
        python nfl_fair_line.py project --season 2025 --week 6 \
            --matchups "KC@BUF,SF@LA"
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.fair_line import (
    DEFAULT_N,
    MIN_TRAILING_GAMES,
    TrainingError,
    format_fit_report,
    format_projection_table,
    project_matchups,
)
from src.logging_utils import get_logger, setup_logging
from src.model_manager import ModelError, load_model, train_and_save
from src.nfl_data import NflDataError, load_pbp
from src.paths import OUTPUT_DIR, ensure_dirs
from src.schedule import ScheduleError, extract_matchups, parse_manual_matchups

log = get_logger("fair_line_cli")


def do_train(seasons, n: int, target_season=None, target_week=None) -> int:
    seasons = sorted({int(s) for s in seasons})
    target_season = target_season or max(seasons)
    # Training over completed seasons only: allow every week of the last one.
    target_week = target_week or 99

    log.info("Building training set from %s...", seasons)
    pbp = load_pbp(seasons)

    metadata = train_and_save(
        pbp,
        seasons=seasons,
        target_season=target_season,
        target_week=target_week,
        n=n,
    )

    print()
    print(
        format_fit_report(
            {
                "intercept": metadata["intercept"],
                "coefficient": metadata["coefficient"],
                "n_test": metadata["holdout_rows"],
                "mae": metadata["mae"],
                "r2": metadata["r2"],
            }
        )
    )
    print(f"\nSaved model version {metadata['model_version']}.")
    return 0


def do_project(
    season: int,
    week: int,
    n: int,
    matchups_arg: str | None,
    out: str | None,
    include_completed: bool = False,
) -> int:
    saved = load_model(requested_n=n)
    model = saved["model"]
    model_n = saved["n"]
    metadata = saved.get("metadata") or {}

    if matchups_arg:
        matchups = parse_manual_matchups(matchups_arg)
        log.info("Using %d manually supplied matchups.", len(matchups))
    else:
        from src.nfl_data import load_schedules

        schedule = load_schedules(season)
        matchups = extract_matchups(
            schedule, season, week, include_completed=include_completed
        )
        log.info("Detected %d scheduled games for %s week %s.", len(matchups), season, week)

    # Pull enough history to compute trailing splits into this week
    # (previous season + current season, as the original did).
    pbp = load_pbp(list(range(season - 1, season + 1)))

    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    df = project_matchups(
        pbp,
        matchups=matchups,
        model=model,
        season=season,
        week=week,
        n=model_n,
        model_version=metadata.get("model_version"),
        model_trained_at=metadata.get("trained_at_utc"),
        run_timestamp_utc=run_timestamp,
    )

    ensure_dirs()
    if out:
        out_path = Path(out)
        if not out_path.is_absolute():
            out_path = Path.cwd() / out_path
    else:
        out_path = OUTPUT_DIR / f"{season}_week_{week:02d}_fair_lines.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(format_projection_table(df))
    log.info("Saved %d projection rows to %s", len(df), out_path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fair-line regression model for NFL spreads."
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train the model on historical seasons.")
    p_train.add_argument("--seasons", type=int, nargs="+", required=True)
    p_train.add_argument("--n", type=int, default=DEFAULT_N)

    p_proj = sub.add_parser("project", help="Project spreads for upcoming games.")
    p_proj.add_argument("--season", type=int, required=True)
    p_proj.add_argument("--week", type=int, required=True)
    p_proj.add_argument(
        "--matchups",
        type=str,
        default=None,
        help="Optional debugging override: comma-separated AWAY@HOME, "
        "e.g. 'KC@BUF,SF@LA'. Omit to use the real schedule.",
    )
    p_proj.add_argument("--n", type=int, default=None)
    p_proj.add_argument("--out", type=str, default=None)
    p_proj.add_argument(
        "--include-completed",
        action="store_true",
        help="Project games that have already kicked off. Needed to "
        "reproduce a historical week, since every game in it is finished.",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    try:
        if args.command == "train":
            return do_train(args.seasons, args.n)
        return do_project(
            args.season,
            args.week,
            args.n,
            args.matchups,
            args.out,
            include_completed=args.include_completed,
        )
    except ModelError as exc:
        log.error("Model problem: %s", exc)
        return 1
    except TrainingError as exc:
        log.error("Training failed: %s", exc)
        return 1
    except ScheduleError as exc:
        log.error("Schedule problem: %s", exc)
        return 1
    except NflDataError as exc:
        log.error("NFL data could not be loaded: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
