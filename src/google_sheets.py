"""Google Sheet synchronisation.

Tabs (implementation guide 24):

======================  ==========================================
Latest Predictions      replaced with this week's fair lines
Prediction History      upserted by (season, week, away, home)
EPA Splits              replaced with the latest splits
Model Info              replaced with current model provenance
Run Log                 one appended row per run
======================  ==========================================

Every update is idempotent (25): rerunning the same week overwrites that
week's history rows rather than appending a second copy, so the client can
safely trigger a manual run twice without corrupting the history tab.
"""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from .fair_line import FAIR_LINE_COLUMNS
from .google_api import execute
from .logging_utils import get_logger

log = get_logger("google.sheets")

TAB_LATEST = "Latest Predictions"
TAB_HISTORY = "Prediction History"
TAB_EPA = "EPA Splits"
TAB_MODEL = "Model Info"
TAB_RUN_LOG = "Run Log"

ALL_TABS = (TAB_LATEST, TAB_HISTORY, TAB_EPA, TAB_MODEL, TAB_RUN_LOG)

#: Columns that identify a unique projection row in Prediction History.
HISTORY_KEY = ("season", "week", "away_team", "home_team")

PREDICTION_HEADERS = [
    "Run Timestamp",
    "Season",
    "Week",
    "Game Date",
    "Away Team",
    "Home Team",
    "Away Trailing Net EPA",
    "Home Trailing Net EPA",
    "EPA Difference",
    "Predicted Home Margin",
    "Fair Spread (Home)",
    "Model n",
    "Model Version",
    "Model Trained At",
    "Status",
    "Notes",
]

EPA_HEADERS = [
    "Season",
    "Analysis Week",
    "Run Timestamp",
    "Team",
    "Games Included",
    "Weeks",
    "Off EPA Pass",
    "Off EPA Rush",
    "Def EPA Pass Allowed",
    "Def EPA Rush Allowed",
    "Net EPA Play",
]

RUN_LOG_HEADERS = [
    "Run Timestamp",
    "Season",
    "Week",
    "Status",
    "Matchups",
    "Projections",
    "Retrained",
    "Notes",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _cell(value):
    """Convert a pandas value into something the Sheets API accepts."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


def _frame_to_values(df: pd.DataFrame, columns: Sequence[str]) -> list:
    return [[_cell(row[column]) for column in columns] for _, row in df.iterrows()]


def get_spreadsheet(service, spreadsheet_id: str) -> dict:
    return execute(
        service.spreadsheets().get(spreadsheetId=spreadsheet_id),
        "Google Sheets spreadsheet lookup",
    )


def ensure_tabs(service, spreadsheet_id: str, titles: Sequence[str] = ALL_TABS) -> None:
    """Create any missing tab. Existing tabs and their data are untouched."""
    meta = get_spreadsheet(service, spreadsheet_id)
    existing = {sheet["properties"]["title"] for sheet in meta.get("sheets", [])}
    missing = [title for title in titles if title not in existing]
    if not missing:
        return

    execute(
        service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={
                "requests": [
                    {"addSheet": {"properties": {"title": title}}}
                    for title in missing
                ]
            },
        ),
        f"Google Sheets tab creation ({', '.join(missing)})",
    )
    log.info("Created missing Sheet tab(s): %s.", ", ".join(missing))


def replace_tab(
    service,
    spreadsheet_id: str,
    title: str,
    header: Sequence[str],
    rows: Sequence[Sequence],
) -> None:
    """Clear a tab and write header + rows."""
    execute(
        service.spreadsheets().values().clear(
            spreadsheetId=spreadsheet_id, range=f"'{title}'"
        ),
        f"Google Sheets clear of {title!r}",
    )
    body = {"values": [list(header)] + [list(row) for row in rows]}
    execute(
        service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=f"'{title}'!A1",
            valueInputOption="RAW",
            body=body,
        ),
        f"Google Sheets update of {title!r}",
    )
    log.info("Sheet tab %r replaced with %d row(s).", title, len(rows))


def read_tab(service, spreadsheet_id: str, title: str) -> list:
    response = execute(
        service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id, range=f"'{title}'"
        ),
        f"Google Sheets read of {title!r}",
    )
    return response.get("values", [])


# ---------------------------------------------------------------------
# Tab writers
# ---------------------------------------------------------------------

def update_latest_predictions(service, spreadsheet_id: str, df: pd.DataFrame) -> None:
    replace_tab(
        service,
        spreadsheet_id,
        TAB_LATEST,
        PREDICTION_HEADERS,
        _frame_to_values(df, FAIR_LINE_COLUMNS),
    )


def update_epa_splits(service, spreadsheet_id: str, df: pd.DataFrame) -> None:
    columns = [
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
    ]
    available = [column for column in columns if column in df.columns]
    replace_tab(
        service,
        spreadsheet_id,
        TAB_EPA,
        EPA_HEADERS[: len(available)],
        _frame_to_values(df, available),
    )


def update_model_info(service, spreadsheet_id: str, rows: Sequence[Sequence]) -> None:
    replace_tab(
        service,
        spreadsheet_id,
        TAB_MODEL,
        ["Field", "Value"],
        [[_cell(a), _cell(b)] for a, b in rows],
    )


def upsert_prediction_history(
    service,
    spreadsheet_id: str,
    df: pd.DataFrame,
) -> dict:
    """Merge this week's rows into Prediction History without duplicating.

    Existing rows whose (season, week, away, home) key matches a row in
    ``df`` are replaced in place; everything else is preserved. A rerun of
    the same week therefore refreshes that week rather than appending a
    second set (implementation guide 25 / 31.7).
    """
    existing = read_tab(service, spreadsheet_id, TAB_HISTORY)

    new_values = _frame_to_values(df, FAIR_LINE_COLUMNS)
    key_positions = [FAIR_LINE_COLUMNS.index(column) for column in HISTORY_KEY]

    def key_of(row: Sequence) -> tuple:
        return tuple(str(row[position]) if position < len(row) else "" for position in key_positions)

    incoming = {key_of(row): row for row in new_values}

    kept = []
    replaced = 0
    if existing:
        body_rows = existing[1:] if existing[0] and existing[0][0] == PREDICTION_HEADERS[0] else existing
        for row in body_rows:
            if not any(str(cell).strip() for cell in row):
                continue
            padded = list(row) + [""] * (len(FAIR_LINE_COLUMNS) - len(row))
            if key_of(padded) in incoming:
                replaced += 1
                continue
            kept.append(padded[: len(FAIR_LINE_COLUMNS)])

    merged = kept + new_values

    def sort_key(row):
        try:
            return (int(row[1]), int(row[2]), str(row[4]), str(row[5]))
        except (ValueError, TypeError, IndexError):
            return (0, 0, "", "")

    merged.sort(key=sort_key)

    replace_tab(
        service, spreadsheet_id, TAB_HISTORY, PREDICTION_HEADERS, merged
    )
    log.info(
        "Prediction History upsert: %d existing row(s) refreshed, "
        "%d new row(s), %d total.",
        replaced,
        len(new_values) - replaced if len(new_values) > replaced else 0,
        len(merged),
    )
    return {"replaced": replaced, "total": len(merged)}


def append_run_log(service, spreadsheet_id: str, row: Sequence) -> None:
    """Append one run-level row, creating the header if the tab is empty."""
    existing = read_tab(service, spreadsheet_id, TAB_RUN_LOG)
    if not existing:
        execute(
            service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=f"'{TAB_RUN_LOG}'!A1",
                valueInputOption="RAW",
                body={"values": [RUN_LOG_HEADERS]},
            ),
            "Google Sheets run-log header",
        )

    execute(
        service.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id,
            range=f"'{TAB_RUN_LOG}'!A1",
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": [[_cell(cell) for cell in row]]},
        ),
        "Google Sheets run-log append",
    )
    log.info("Run Log row appended.")


def create_spreadsheet(service, title: str) -> str:
    """Create the client's spreadsheet with all tabs (one-time setup)."""
    body = {
        "properties": {"title": title},
        "sheets": [{"properties": {"title": tab}} for tab in ALL_TABS],
    }
    created = execute(
        service.spreadsheets().create(body=body, fields="spreadsheetId"),
        f"Google Sheets creation of {title!r}",
    )
    return created["spreadsheetId"]


def sync_all(
    service,
    spreadsheet_id: str,
    epa_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    model_rows: Sequence[Sequence],
    run_log_row: Optional[Sequence] = None,
) -> dict:
    """Update every tab in the documented order."""
    ensure_tabs(service, spreadsheet_id)
    update_latest_predictions(service, spreadsheet_id, predictions_df)
    history = upsert_prediction_history(service, spreadsheet_id, predictions_df)
    if not epa_df.empty:
        update_epa_splits(service, spreadsheet_id, epa_df)
    else:
        log.warning("EPA splits are empty; leaving the EPA Splits tab as it was.")
    update_model_info(service, spreadsheet_id, model_rows)
    if run_log_row is not None:
        append_run_log(service, spreadsheet_id, run_log_row)
    log.info("Sheet update success.")
    return history
