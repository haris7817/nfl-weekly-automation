"""
NFL Last-5-Game EPA Splits  (nflreadpy edition)
-----------------------------------------------
Pulls play-by-play data via nflreadpy, computes each team's offensive and
defensive EPA/play (pass and rush split) over their last 5 games, and
outputs a clean table you can use for Step 1 of the weekly workflow.

This is the migrated version of the client's original script. The EPA
mathematics are unchanged - see ``original/nfl_epa_splits.py`` for the
untouched reference copy and ``src/epa.py`` for the shared implementation.

What changed in the migration:
  * data now comes from ``nflreadpy`` instead of ``nfl_data_py``
    (Polars is converted to pandas at the loading boundary);
  * required-column and empty-dataset validation were added;
  * output paths are resolved relative to the project, not the shell's
    current working directory;
  * season / analysis week / run timestamp metadata columns were added.

SETUP (run once):
    pip install -r requirements.txt

USAGE:
    python nfl_epa_splits.py --season 2025 --week 6
    (pulls all games through week 5, computes each team's last-5-game splits
    heading into week 6)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.epa import InsufficientDataError, build_epa_splits
from src.logging_utils import get_logger, setup_logging
from src.nfl_data import NflDataError, load_pbp
from src.paths import OUTPUT_DIR, ensure_dirs

log = get_logger("epa_splits")


def run(season: int, week: int, n: int = 5, out: str | None = None) -> Path:
    """Compute EPA splits for ``season``/``week`` and write them to CSV."""
    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    log.info("Pulling %s play-by-play data...", season)
    pbp = load_pbp(season)

    log.info("Aggregating team-game EPA and computing last-%d splits...", n)
    splits = build_epa_splits(
        pbp,
        season=season,
        week=week,
        n=n,
        run_timestamp_utc=run_timestamp,
    )

    ensure_dirs()
    if out:
        out_path = Path(out)
        if not out_path.is_absolute():
            out_path = Path.cwd() / out_path
    else:
        out_path = OUTPUT_DIR / f"{season}_week_{week:02d}_epa_splits.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(out_path, index=False)

    log.info("Saved %d teams to %s", len(splits), out_path)
    print()
    print(splits.to_string(index=False))
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Last-5-game EPA splits for all NFL teams."
    )
    parser.add_argument(
        "--season", type=int, required=True, help="Season year, e.g. 2025"
    )
    parser.add_argument(
        "--week",
        type=int,
        required=True,
        help="Upcoming week you're analyzing (uses games BEFORE this week)",
    )
    parser.add_argument(
        "--n", type=int, default=5, help="Number of trailing games (default 5)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path (default: outputs/<season>_week_<NN>_epa_splits.csv)",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.week < 1 or args.week > 22:
        log.error("--week must be between 1 and 22 (got %s).", args.week)
        return 2
    if args.n < 1:
        log.error("--n must be at least 1 (got %s).", args.n)
        return 2

    try:
        run(args.season, args.week, args.n, args.out)
    except InsufficientDataError as exc:
        log.error("%s", exc)
        return 1
    except NflDataError as exc:
        log.error("NFL data could not be loaded: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
