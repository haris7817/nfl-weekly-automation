"""
NFL Weekly Analytics - production entry point.
===============================================

One command runs the whole workflow:

    python weekly_runner.py

Everything else is an override:

    python weekly_runner.py --season 2025 --week 6      historical rerun
    python weekly_runner.py --retrain                   force retraining
    python weekly_runner.py --skip-google               local only
    python weekly_runner.py --dry-run                   compute, write nothing
    python weekly_runner.py --include-completed         project finished games

Order of operations is deliberate (implementation guide 20.2): the
analytics results are validated and written to disk *before* any Google
call. If Drive or Sheets then fails, the run exits non-zero so the failure
is visible, but the CSVs still exist locally and as GitHub artifacts.

Exit codes
    0  success
    1  the analytics pipeline failed; nothing was published
    2  invalid arguments
    3  analytics succeeded and were saved locally, but Google publishing
       failed
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import Config
from src.epa import InsufficientDataError, build_epa_splits, empty_splits_frame
from src.fair_line import (
    FAIR_LINE_COLUMNS,
    STATUS_PROJECTED,
    TrainingError,
    format_projection_table,
    project_matchups,
)
from src.logging_utils import get_logger, setup_logging
from src.model_manager import (
    ModelError,
    ModelNotFoundError,
    load_model,
    metadata_rows,
    read_metadata,
    save_model_atomically,
    should_retrain,
    train_model,
    training_seasons,
)
from src.nfl_data import (
    NflDataError,
    load_pbp_available,
    load_schedules,
    load_team_codes,
)
from src.paths import (
    MODEL_METADATA_PATH,
    MODEL_PATH,
    ensure_dirs,
    week_output_dir,
)
from src.schedule import (
    ScheduleError,
    check_team_codes,
    extract_matchups,
    next_friday_run_local,
    resolve_season_week,
)
from src.validation import (
    ValidationError,
    log_warnings,
    validate_epa_splits,
    validate_fair_lines,
)

log = get_logger("runner")

EXIT_OK = 0
EXIT_ANALYTICS_FAILED = 1
EXIT_BAD_ARGS = 2
EXIT_GOOGLE_FAILED = 3


# ---------------------------------------------------------------------
# Google helpers (import lazily so --skip-google needs no Google packages)
# ---------------------------------------------------------------------

def _restore_model(config: Config):
    """Pull the stored model from Drive so retraining survives cloud runs."""
    from src.google_auth import build_drive_service
    from src.google_drive import restore_model_from_drive

    service = build_drive_service(config)
    restored = restore_model_from_drive(
        service, config.google_drive_folder_id, MODEL_PATH, MODEL_METADATA_PATH
    )
    return service, restored


def _publish(config: Config, drive_service, summary: dict, paths: list,
             epa_df: pd.DataFrame, predictions_df: pd.DataFrame,
             model_metadata, retrained: bool) -> dict:
    """Upload CSVs, persist the model and update every Sheet tab."""
    from src.google_auth import build_drive_service, build_sheets_service
    from src.google_drive import persist_model_to_drive, upload_week_outputs
    from src.google_sheets import sync_all

    drive = drive_service or build_drive_service(config)

    uploaded = upload_week_outputs(
        drive,
        config.google_drive_folder_id,
        summary["season"],
        summary["week"],
        paths,
    )

    if retrained:
        persist_model_to_drive(
            drive, config.google_drive_folder_id, MODEL_PATH, MODEL_METADATA_PATH
        )

    sheets = build_sheets_service(config)
    run_log_row = [
        summary["run_timestamp_utc"],
        summary["season"],
        summary["week"],
        summary["status"],
        summary["matchups"],
        summary["projections"],
        "yes" if retrained else "no",
        summary.get("notes", ""),
    ]
    history = sync_all(
        sheets,
        config.google_sheet_id,
        epa_df=epa_df,
        predictions_df=predictions_df,
        model_rows=metadata_rows(model_metadata),
        run_log_row=run_log_row,
    )
    return {"drive_files": uploaded, "sheet_history": history}


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def run(args, config: Config) -> int:
    run_started = datetime.now(timezone.utc)
    run_timestamp = run_started.strftime("%Y-%m-%dT%H:%M:%SZ")

    log.info("=" * 70)
    log.info("NFL weekly workflow started at %s", run_timestamp)
    log.info("%s", config.describe())
    log.info(
        "Next scheduled Friday run: %s",
        next_friday_run_local(run_started).isoformat(),
    )

    # --- 1. target season / week -------------------------------------
    season, week, detection = resolve_season_week(args.season, args.week)
    log.info("Target: %s Week %s (%s).", season, week, detection["reason"])

    # --- 2. schedule + matchups --------------------------------------
    schedule = load_schedules(season)
    matchups = extract_matchups(
        schedule,
        season,
        week,
        include_completed=config.include_completed_week_games,
    )
    check_team_codes(matchups, load_team_codes())

    upcoming = sum(1 for m in matchups if m["status"] is None)
    log.info("%d of %d games are eligible for a new projection.", upcoming, len(matchups))
    if upcoming == 0:
        log.warning(
            "Every game in %s week %s has already kicked off. Add "
            "--include-completed to generate projections anyway (useful for "
            "historical reruns).",
            season,
            week,
        )

    # --- 3. play-by-play ---------------------------------------------
    # Projections need the previous season too, so early-season trailing
    # history can reach back, exactly as the client's original script did.
    pbp, loaded_seasons = load_pbp_available([season - 1, season])

    # --- 4. EPA splits ------------------------------------------------
    epa_pbp = pbp[pbp["season"] == season]
    if epa_pbp.empty:
        log.warning(
            "No %s play-by-play exists yet, so EPA splits cannot be "
            "computed. This is expected before week 2 of a season.",
            season,
        )
        epa_df = empty_splits_frame(with_metadata=True)
    else:
        epa_df = build_epa_splits(
            epa_pbp,
            season=season,
            week=week,
            n=config.trailing_games_n,
            run_timestamp_utc=run_timestamp,
            strict=False,
        )

    # --- 5. model: restore, decide, retrain ---------------------------
    drive_service = None
    if not config.skip_google and config.google_drive_folder_id:
        try:
            drive_service, _ = _restore_model(config)
        except Exception as exc:
            log.warning(
                "Could not restore the model from Drive (%s). Falling back to "
                "the local model directory.",
                exc,
            )

    retrain, reason = should_retrain(
        season=season,
        week=week,
        retrain_every_weeks=config.retrain_every_weeks,
        force=config.force_retrain,
    )
    log.info("Retraining %s: %s.", "required" if retrain else "not required", reason)

    model_metadata = read_metadata()
    retrained = False
    model = None
    model_n = config.trailing_games_n

    if retrain:
        seasons = training_seasons(season, config.training_start_season)
        train_pbp, train_seasons = load_pbp_available(seasons)
        try:
            model, model_metadata = train_model(
                train_pbp,
                seasons=train_seasons,
                target_season=season,
                target_week=week,
                n=config.trailing_games_n,
                min_games=config.min_trailing_games,
            )
            retrained = True
            log.info(
                "Model metrics: MAE=%.2f points, R^2=%.3f, coefficient=%.2f, "
                "intercept=%.2f.",
                model_metadata["mae"],
                model_metadata["r2"],
                model_metadata["coefficient"],
                model_metadata["intercept"],
            )
            if config.dry_run:
                log.info(
                    "--dry-run: the new model was fitted in memory and is "
                    "used for these projections, but was NOT saved."
                )
            else:
                save_model_atomically(
                    model, n=config.trailing_games_n, metadata=model_metadata
                )
        except TrainingError as exc:
            log.error("Retraining failed: %s", exc)
            model = None
            if not MODEL_PATH.is_file():
                raise
            log.warning(
                "Keeping the previous production model; this run will use it."
            )

    if model is None:
        saved = load_model(requested_n=config.trailing_games_n)
        model = saved["model"]
        model_n = saved["n"]
        model_metadata = saved.get("metadata") or model_metadata or read_metadata() or {}

    log.info(
        "Model in use: version %s, trained %s, n=%s.",
        (model_metadata or {}).get("model_version", "?"),
        (model_metadata or {}).get("trained_at_utc", "?"),
        model_n,
    )

    # --- 6. fair-line projections -------------------------------------
    predictions_df = project_matchups(
        pbp,
        matchups=matchups,
        model=model,
        season=season,
        week=week,
        n=model_n,
        min_games=config.min_trailing_games,
        model_version=model_metadata.get("model_version"),
        model_trained_at=model_metadata.get("trained_at_utc"),
        run_timestamp_utc=run_timestamp,
    )

    # --- 7. validation -------------------------------------------------
    log_warnings(
        validate_epa_splits(epa_df, season, week, allow_empty=True),
        "epa_splits",
    )
    log_warnings(
        validate_fair_lines(predictions_df, season, week, expected_games=len(matchups)),
        "fair_lines",
    )

    projections = int((predictions_df["status"] == STATUS_PROJECTED).sum())

    summary = {
        "run_timestamp_utc": run_timestamp,
        "season": season,
        "week": week,
        "detection": detection,
        "pbp_seasons_loaded": loaded_seasons,
        "matchups": len(matchups),
        "projections": projections,
        "epa_teams": int(len(epa_df)),
        "retrained": retrained,
        "model_version": model_metadata.get("model_version"),
        "model_trained_at_utc": model_metadata.get("trained_at_utc"),
        "trailing_games_n": model_n,
        "status": "success",
        "notes": detection["reason"],
    }

    print(format_projection_table(predictions_df))

    # --- 8. local outputs (always before Google) -----------------------
    if config.dry_run:
        log.info("--dry-run: no files written, no Google calls made.")
        log.info("Workflow completed successfully (dry run).")
        return EXIT_OK

    ensure_dirs()
    out_dir = week_output_dir(season, week)
    epa_path = out_dir / "epa_splits.csv"
    fair_path = out_dir / "fair_lines.csv"
    summary_path = out_dir / "run_summary.json"

    epa_df.to_csv(epa_path, index=False)
    predictions_df.to_csv(fair_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=str)

    paths = [epa_path, fair_path, summary_path]
    for path in paths:
        log.info("Local output: %s", path)

    # --- 9. Google publishing ------------------------------------------
    if config.skip_google:
        log.info("--skip-google: Drive upload and Sheet update were skipped.")
        log.info("Weekly NFL workflow completed successfully.")
        return EXIT_OK

    missing = config.missing_google_settings()
    if missing:
        log.error(
            "Google publishing is not configured (missing: %s). The analytics "
            "outputs above were saved locally. Configure the GitHub secrets, "
            "or rerun with --skip-google.",
            ", ".join(missing),
        )
        return EXIT_GOOGLE_FAILED

    try:
        published = _publish(
            config,
            drive_service,
            summary,
            paths,
            epa_df,
            predictions_df,
            model_metadata,
            retrained,
        )
        summary["google"] = published
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, default=str)
    except Exception as exc:
        log.error("Google publishing failed: %s", exc)
        log.error(
            "The analytics results are intact at %s and are also uploaded as "
            "GitHub Actions artifacts. Fix the Google problem and rerun; the "
            "Sheet update is idempotent, so rerunning is safe.",
            out_dir,
        )
        log.debug("%s", traceback.format_exc())
        return EXIT_GOOGLE_FAILED

    log.info("Weekly NFL workflow completed successfully.")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full weekly NFL analytics workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--season", type=int, default=None,
                        help="Override the detected season (use with --week).")
    parser.add_argument("--week", type=int, default=None,
                        help="Override the detected week (use with --season).")
    parser.add_argument("--retrain", action="store_true",
                        help="Force model retraining before projecting.")
    parser.add_argument("--skip-google", action="store_true",
                        help="Run analytics only; no Drive upload or Sheet update.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute everything but write nothing anywhere.")
    parser.add_argument("--include-completed", action="store_true",
                        help="Project games that have already kicked off "
                             "(needed for historical reruns).")
    parser.add_argument("--log-level", type=str, default=None,
                        help="DEBUG, INFO, WARNING or ERROR.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    config = Config.from_env()
    config.skip_google = args.skip_google
    config.dry_run = args.dry_run
    config.force_retrain = args.retrain
    if args.include_completed:
        config.include_completed_week_games = True
    if args.log_level:
        config.log_level = args.log_level.upper()

    setup_logging(config.log_level)

    if (args.season is None) != (args.week is None):
        log.error("--season and --week must be used together.")
        return EXIT_BAD_ARGS
    if args.week is not None and not (1 <= args.week <= 22):
        log.error("--week must be between 1 and 22 (got %s).", args.week)
        return EXIT_BAD_ARGS

    try:
        return run(args, config)
    except (ScheduleError, NflDataError, ValidationError, InsufficientDataError,
            ModelError, ModelNotFoundError, TrainingError) as exc:
        log.error("Workflow failed: %s", exc)
        return EXIT_ANALYTICS_FAILED
    except KeyboardInterrupt:
        log.error("Interrupted by the user.")
        return EXIT_ANALYTICS_FAILED
    except Exception as exc:  # unexpected - show the traceback for debugging
        log.error("Unexpected failure: %s: %s", type(exc).__name__, exc)
        log.error("%s", traceback.format_exc())
        return EXIT_ANALYTICS_FAILED


if __name__ == "__main__":
    sys.exit(main())
