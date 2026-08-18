# NFL Weekly Automation Project — Complete End-to-End Implementation Guide

## 1. Project Goal

Build a reliable, mostly hands-free weekly NFL analytics workflow around the client's two existing Python scripts.

The finished system must:

1. Migrate both scripts from `nfl_data_py` to `nflreadpy`.
2. Preserve the client's existing EPA and linear-regression logic unless a change is explicitly approved.
3. Detect the correct NFL season/week automatically.
4. Detect that week's NFL matchups automatically.
5. Generate weekly EPA results.
6. Generate fair-line/regression projections.
7. Save outputs as CSV files.
8. Upload/archive the CSV files in Google Drive.
9. Create and maintain a new Google Sheet containing the results.
10. Run automatically every Friday at 6:00 PM Pacific Time even when the client's PC is off.
11. Allow a manual cloud run at any time.
12. Periodically retrain the regression model using newer completed NFL data.
13. Add logging, validation, retries, and useful failure messages.
14. Complete local and cloud end-to-end testing.
15. Leave the client with ownership/access and clear operating instructions.

---

## 2. Confirmed Client Requirements

The client has already confirmed:

- Existing code: two working Python scripts.
- Data migration: change from `nfl_data_py` to `nflreadpy`.
- Output: CSV **and** Google Sheets.
- CSV location: Google cloud folder, preferably Google Drive.
- Google Sheet: create a **new** Sheet.
- Execution: must run even if the client's PC is off.
- Schedule: every **Friday at 6:00 PM Pacific Time**.
- NFL season/week: detect automatically.
- Weekly matchups: detect automatically.
- Model: periodically retrain using newer data.
- Development approach: most coding/troubleshooting can be done message-by-message; client will be available for Google authorization/final setup.
- Cloud approach: a private GitHub repository + GitHub Actions is the recommended lightweight scheduler.
- Manual-run option: include it so the client can trigger a run without waiting for Friday.

### Important scheduling note

Use the IANA timezone:

```text
America/Los_Angeles
```

rather than a hard-coded UTC offset or the literal abbreviation `PST`.

This automatically handles both Pacific Standard Time and Pacific Daylight Time.

---

# 3. Existing Source Files

Original client files:

```text
nfl_epa_splits.py
nfl_fair_line.py
```

Never edit the only copy of the originals.

Create permanent backups before making any changes.

Recommended structure:

```text
nfl-weekly-automation/
│
├── original/
│   ├── nfl_epa_splits.py
│   └── nfl_fair_line.py
│
├── src/
│   ├── __init__.py
│   ├── nfl_data.py
│   ├── epa.py
│   ├── fair_line.py
│   ├── schedule.py
│   ├── google_drive.py
│   ├── google_sheets.py
│   ├── model_manager.py
│   └── logging_utils.py
│
├── nfl_epa_splits.py
├── nfl_fair_line.py
├── weekly_runner.py
│
├── models/
│   ├── .gitkeep
│   └── model_metadata.json
│
├── outputs/
│   └── .gitkeep
│
├── logs/
│   └── .gitkeep
│
├── tests/
│   ├── test_epa.py
│   ├── test_schedule.py
│   ├── test_fair_line.py
│   └── test_runner.py
│
├── scripts/
│   └── google_auth_setup.py
│
├── .github/
│   └── workflows/
│       └── weekly_nfl.yml
│
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```

A simpler structure is acceptable, but keep these responsibilities separate.

---

# 4. What the Existing Scripts Currently Do

## 4.1 `nfl_epa_splits.py`

Current behavior:

- Downloads NFL play-by-play data.
- Keeps regular-season pass/run plays with a valid EPA.
- Calculates offensive pass EPA.
- Calculates offensive rush EPA.
- Calculates defensive pass EPA allowed.
- Calculates defensive rush EPA allowed.
- Uses the last N games before the requested week.
- Defaults to N = 5.
- Generates `epa_splits.csv`.

The existing script is already close to automation-ready.

The main changes needed are:

- replace the old data library;
- improve path handling;
- guard empty datasets;
- return structured data so the master runner can reuse it;
- preserve CSV generation;
- optionally add metadata columns such as season/week/run timestamp.

## 4.2 `nfl_fair_line.py`

Current behavior:

- Downloads historical play-by-play data.
- Builds team-game offensive/defensive EPA.
- Calculates trailing net EPA.
- Builds historical training rows.
- Uses one feature: home EPA minus away EPA.
- Target: home score minus away score.
- Trains `LinearRegression`.
- Evaluates with MAE and R².
- Saves the model as `fair_line_model.pkl`.
- Loads the model for projections.
- Accepts manually entered `season`, `week`, and `matchups`.
- Prints the fair-line results to the terminal.

Main changes needed:

- migrate data library;
- remove dependency on manual matchup entry for normal production runs;
- return projections as a DataFrame instead of only printing;
- save fair-line results to CSV;
- use stable model/output paths;
- validate model metadata;
- add retraining policy;
- add logging/error handling.

---

# 5. Important Existing-Code Observations

These should be understood before changing the model.

## 5.1 The two scripts are not actually chained together

`nfl_fair_line.py` calculates its own EPA values. It does not read `epa_splits.csv`.

That is not necessarily wrong.

The final solution should avoid unnecessary duplicate code by moving common data-loading/EPA helpers into shared modules, but do not change the mathematical behavior merely for refactoring.

## 5.2 The EPA split script and fair-line script use related but not identical EPA calculations

The EPA split script separately averages pass and rush EPA and combines them into a `net_epa_play` value.

The fair-line script calculates an overall offensive EPA mean and defensive EPA allowed mean across eligible plays.

Therefore, the values are not guaranteed to be mathematically identical.

Do **not** silently rewrite the client's model to force them to match.

Preserve current behavior first.

Any model-methodology improvement should be treated as a separate improvement and explained to the client.

## 5.3 The current saved model stores `n`, but projection can use another `n`

Training stores:

```python
{
    "model": model,
    "n": n
}
```

but projection should validate/use the saved training configuration rather than allowing an accidental mismatch.

Recommended:

```python
saved_n = saved["n"]

if requested_n is not None and requested_n != saved_n:
    raise ValueError(
        f"Model was trained with n={saved_n}, "
        f"but projection requested n={requested_n}"
    )
```

or simply use `saved_n` by default.

## 5.4 Relative paths can fail in cloud/task environments

Avoid:

```python
MODEL_PATH = "fair_line_model.pkl"
```

Use a path based on the project directory:

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "fair_line_model.pkl"
```

Apply the same pattern to:

- outputs;
- logs;
- temporary files;
- local Google token setup files.

## 5.5 Fair-line projections need structured output

Instead of only:

```python
print(...)
```

build rows like:

```python
rows.append({
    "season": season,
    "week": week,
    "away_team": away,
    "home_team": home,
    "away_epa": away_epa,
    "home_epa": home_epa,
    "epa_diff": diff,
    "predicted_home_margin": pred_margin,
    "fair_spread_home": -pred_margin,
})
```

Then:

```python
return pd.DataFrame(rows)
```

This DataFrame becomes the source for:

- terminal display;
- CSV;
- Google Sheets;
- tests.

---

# 6. Development Strategy

Do not build everything at once.

Use this order:

```text
Phase 1  Backup + local environment
Phase 2  nflreadpy migration
Phase 3  Validate EPA script
Phase 4  Validate fair-line training/projection
Phase 5  Automatic season/week/matchups
Phase 6  Structured CSV output
Phase 7  Model management/retraining
Phase 8  Build weekly_runner.py
Phase 9  Logging/retries/error handling
Phase 10 Google Drive/Sheets
Phase 11 GitHub Actions cloud execution
Phase 12 Full end-to-end tests
Phase 13 Client authorization/session
Phase 14 Production handoff
```

Do not move to the next major phase until the previous one passes its acceptance tests.

---

# 7. Phase 1 — Local Project Setup

## 7.1 Create the project

```bash
mkdir nfl-weekly-automation
cd nfl-weekly-automation
```

Copy the client's originals into:

```text
original/
```

Then make working copies at the project root or under `src/`.

## 7.2 Initialize Git

```bash
git init
git branch -M main
```

Create a private GitHub repository when ready.

During development it can live under the developer's GitHub account.

Before final handoff:

- invite the client as collaborator; or preferably
- transfer the finished private repository to the client's GitHub account if he creates one.

The automation should ultimately be under client control.

## 7.3 Python version

Use a currently supported Python version that works with all dependencies.

A practical target is Python 3.11 or 3.12 unless dependency testing shows a reason to choose otherwise.

Record the actual tested version in `README.md`.

## 7.4 Create virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade packaging tools:

```bash
python -m pip install --upgrade pip
```

## 7.5 Initial dependencies

Install:

```bash
pip install nflreadpy pandas numpy pyarrow scikit-learn
pip install google-api-python-client google-auth google-auth-oauthlib google-auth-httplib2
pip install pytest python-dotenv
```

Do not finalize exact version pins until the solution has been tested.

After testing, pin the known-good versions in `requirements.txt`.

Example structure:

```text
nflreadpy==<tested-version>
pandas==<tested-version>
numpy==<tested-version>
pyarrow==<tested-version>
scikit-learn==<tested-version>
google-api-python-client==<tested-version>
google-auth==<tested-version>
google-auth-oauthlib==<tested-version>
google-auth-httplib2==<tested-version>
python-dotenv==<tested-version>
```

Keep `pytest` as a development dependency if desired.

---

# 8. Phase 2 — Migrate `nfl_data_py` to `nflreadpy`

The official `nflreadpy` package returns Polars DataFrames.

The existing client code is pandas-based.

The lowest-risk migration is:

1. load with `nflreadpy`;
2. convert to pandas;
3. preserve existing pandas aggregation logic.

Example:

```python
import nflreadpy as nfl

def load_pbp(seasons):
    pbp = nfl.load_pbp(seasons)
    df = pbp.to_pandas()
    return df
```

Then retain the existing filters.

Example:

```python
def load_pbp(seasons):
    pbp = nfl.load_pbp(seasons)
    df = pbp.to_pandas()

    df = df[
        (df["season_type"] == "REG")
        & (df["play_type"].isin(["pass", "run"]))
        & (df["epa"].notna())
    ].copy()

    return df
```

## 8.1 Do not optimize to Polars during migration

Do not rewrite all calculations into Polars in the first implementation.

Reason:

- it increases risk;
- it changes too much at once;
- it makes output comparison harder.

First make behavior equivalent.

Optimization can happen later.

## 8.2 Validate data schema

Immediately inspect:

```python
print(df.columns.tolist())
print(df.shape)
print(df[["season", "week", "game_id"]].head())
```

Verify required fields are present:

```text
season
season_type
week
game_id
play_type
epa
posteam
defteam
home_team
away_team
home_score
away_score
```

If nflverse changes a field name, fail with a useful message instead of silently producing incorrect output.

Recommended helper:

```python
def require_columns(df, required):
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise RuntimeError(
            "NFL data schema is missing required columns: "
            + ", ".join(missing)
        )
```

---

# 9. Phase 3 — Validate the Migrated EPA Script

Do this **before** changing its calculation logic.

Historical test:

```bash
python nfl_epa_splits.py --season 2025 --week 6
```

Confirm:

- PBP downloads;
- only regular-season pass/run plays remain;
- teams are grouped;
- trailing games are selected correctly;
- `games_included` looks reasonable;
- week list is correct;
- no duplicated teams;
- `epa_splits.csv` is created;
- numeric columns are numeric;
- output is not empty.

## 9.1 Add empty-data protection

The current production version should not assume rows exist.

Example:

```python
if game_epa.empty:
    raise RuntimeError("No qualifying NFL play-by-play data was available.")

if splits.empty:
    raise RuntimeError(
        f"No EPA splits could be produced for season={season}, week={week}."
    )
```

## 9.2 Add metadata columns

Useful optional columns:

```text
season
analysis_week
run_timestamp_utc
team
games_included
weeks
off_epa_pass
off_epa_rush
def_epa_pass_allowed
def_epa_rush_allowed
net_epa_play
```

Do not remove existing columns unnecessarily.

---

# 10. Phase 4 — Validate the Migrated Fair-Line Script

First test training with behavior as close to the original as possible.

Example:

```bash
python nfl_fair_line.py train --seasons 2021 2022 2023 2024
```

Confirm:

- historical PBP loads;
- training rows are created;
- `epa_diff` has numeric values;
- `actual_margin` has numeric values;
- model trains;
- MAE prints;
- R² prints;
- coefficient prints;
- intercept prints;
- model pickle is created.

Then test projection using a known historical week/matchup.

Example:

```bash
python nfl_fair_line.py project \
  --season 2025 \
  --week 6 \
  --matchups "KC@BUF,SF@LA"
```

Use real valid team codes for the historical dataset being tested.

## 10.1 Preserve original model initially

Do not change:

```text
feature = home trailing net EPA - away trailing net EPA
target  = home score - away score
model   = LinearRegression
```

until the automation is validated.

## 10.2 Optional future model improvement — not required for initial delivery

The current random train/test split is simple.

A later analytics enhancement could use:

- chronological holdout;
- season-based holdout;
- rolling validation.

This can reduce the risk of overly optimistic evaluation from random temporal mixing.

Do **not** make this an unannounced scope change.

---

# 11. Phase 5 — Automatic Season and Week Detection

`nflreadpy` provides current-season/current-week utilities.

However, production logic should handle preseason/offseason edge cases.

Important: before the regular season starts, a helper based on regular-season start date can identify the previous NFL season.

Therefore, do not blindly trust a single helper for all calendar dates.

## 11.1 Recommended logic

Create:

```python
def detect_target_season_and_week():
    ...
```

Strategy:

1. Use the current calendar date.
2. Try the current calendar year's schedule.
3. Filter to regular-season games.
4. Identify the next regular-season game/week that has not been completed.
5. If current-year schedule is unavailable, fall back to `nflreadpy` current-season/week helpers.
6. Validate that the resulting week exists in the schedule.
7. Return `(season, week)`.

During the actual NFL regular season, `nflreadpy.get_current_week()` can be used as the primary helper.

## 11.2 Historical override must remain available

The production runner should support:

```bash
python weekly_runner.py --season 2025 --week 6
```

This is extremely useful for:

- testing;
- debugging;
- reruns;
- reproducing client results.

Automatic detection should be the default.

---

# 12. Phase 6 — Automatically Detect Matchups

Load schedule data for the target season.

Concept:

```python
schedule = nfl.load_schedules([season]).to_pandas()
```

Then filter:

```text
season == target season
week   == target week
regular season only
```

Build matchup rows containing at least:

```text
season
week
away_team
home_team
game_date
game_time (if available)
status/result
```

## 12.1 Friday-run behavior

Because the scheduled run is Friday evening Pacific Time, Thursday Night Football may already be complete.

Recommended behavior:

- detect the **full week's schedule**;
- label games as upcoming/completed when possible;
- generate production projections for games that are still upcoming;
- do not accidentally treat a completed Thursday game as a new upcoming prediction.

Make this behavior configurable.

Example:

```text
INCLUDE_COMPLETED_WEEK_GAMES=false
```

## 12.2 Validate team codes

Before projecting:

```python
if away not in known_teams:
    ...
if home not in known_teams:
    ...
```

Return a clear error/warning instead of failing during EPA lookup.

## 12.3 No manual matchup input required in production

The client should not need:

```text
KC@BUF,SF@LA,...
```

every Friday.

Keep manual `--matchups` only as an optional debugging override if useful.

---

# 13. Phase 7 — Structured Fair-Line Output

Refactor projection so it returns a DataFrame.

Recommended columns:

```text
run_timestamp_utc
season
week
game_date
away_team
home_team
away_trailing_net_epa
home_trailing_net_epa
epa_diff
predicted_home_margin
fair_spread_home
model_n
model_trained_at
model_version
status
```

Optional:

```text
notes
```

Example notes:

```text
insufficient history
missing schedule row
projection generated
```

## 13.1 Sign convention

Keep the existing convention clear.

If:

```text
fair_spread_home = -predicted_home_margin
```

then:

```text
negative value = home team favored
positive value = home team underdog
```

Document this in the Sheet and README.

---

# 14. Phase 8 — Output File Design

Recommended local output structure:

```text
outputs/
└── 2026/
    ├── week_01/
    │   ├── epa_splits.csv
    │   ├── fair_lines.csv
    │   └── run_summary.json
    └── week_02/
        ├── epa_splits.csv
        ├── fair_lines.csv
        └── run_summary.json
```

Recommended filenames can also include dates:

```text
2026_week_01_epa_splits.csv
2026_week_01_fair_lines.csv
```

Do not overwrite historical weekly files unless intentionally rerunning the same week.

For reruns choose one policy and document it:

### Policy A — overwrite same week

Good for a clean archive.

### Policy B — timestamp every run

Example:

```text
2026_week_01_fair_lines_2026-09-11T180000-0700.csv
```

Good for audit history.

Recommended compromise:

- Drive folder keeps one canonical weekly file;
- optional `runs/` subfolder keeps timestamped versions.

---

# 15. Phase 9 — Model Storage and Metadata

Create:

```text
models/fair_line_model.pkl
models/model_metadata.json
```

Metadata example:

```json
{
  "model_version": 1,
  "trained_at_utc": "2026-09-01T00:00:00Z",
  "trailing_games_n": 5,
  "training_seasons": [2021, 2022, 2023, 2024, 2025],
  "training_last_season": 2025,
  "training_last_week": 18,
  "coefficient": 12.34,
  "intercept": 1.75,
  "mae": 10.2,
  "r2": 0.08
}
```

Do not rely on the pickle alone.

Metadata makes debugging and client transparency much easier.

---

# 16. Phase 10 — Periodic Retraining

Do not necessarily retrain every Friday.

Recommended default:

```text
RETRAIN_EVERY_WEEKS=4
```

Example production policy:

- initial deployment: train/retrain;
- regular run: project weekly;
- every fourth week: retrain before projection;
- manual `--retrain` always available.

Possible trigger:

```python
if args.retrain:
    should_retrain = True
elif model_missing:
    should_retrain = True
elif weeks_since_retrain >= RETRAIN_EVERY_WEEKS:
    should_retrain = True
```

## 16.1 Avoid future-data leakage

When retraining for an upcoming Week N:

Use only games completed before the projection cutoff.

Recommended training filter for the current season:

```text
game_week < target_week
```

Historical completed seasons can use all completed regular-season games.

This keeps the model consistent with information available at projection time.

## 16.2 Training seasons

Make configurable.

Example:

```text
TRAINING_START_SEASON=2021
```

At 2026 Week 10, training might use:

```text
2021
2022
2023
2024
2025
2026 completed weeks before target week
```

## 16.3 Save model only after successful training

Use a temporary file then replace the old model.

Do not destroy the previous working model if training fails.

Concept:

```text
fair_line_model.tmp.pkl
        ↓ successful serialization
fair_line_model.pkl
```

---

# 17. Phase 11 — Build `weekly_runner.py`

This becomes the single production entry point.

The client/cloud should normally run only:

```bash
python weekly_runner.py
```

Recommended CLI:

```bash
python weekly_runner.py
python weekly_runner.py --season 2025 --week 6
python weekly_runner.py --retrain
python weekly_runner.py --skip-google
python weekly_runner.py --dry-run
```

## 17.1 Runner flow

```text
START
  ↓
load configuration
  ↓
initialize logging
  ↓
detect target season/week
  ↓
load schedule
  ↓
detect matchups
  ↓
load required NFL PBP history
  ↓
calculate EPA splits
  ↓
check model
  ↓
retrain if required
  ↓
generate fair-line projections
  ↓
validate outputs
  ↓
save local CSV/JSON
  ↓
upload CSV files to Google Drive
  ↓
update Google Sheet
  ↓
write success summary
  ↓
END
```

## 17.2 Pseudocode

```python
def main():
    config = load_config()
    logger = setup_logging()

    season, week = resolve_season_week(config)

    schedule = load_target_schedule(season, week)
    matchups = extract_matchups(schedule)

    epa_df = build_epa_splits(
        season=season,
        week=week,
        n=config.trailing_games
    )

    should_retrain = model_manager.should_retrain(
        season=season,
        week=week,
        force=config.force_retrain
    )

    if should_retrain:
        model_manager.train_and_save(
            target_season=season,
            target_week=week
        )

    predictions_df = create_fair_lines(
        season=season,
        week=week,
        matchups=matchups
    )

    validate_outputs(epa_df, predictions_df)

    paths = save_local_outputs(
        season,
        week,
        epa_df,
        predictions_df
    )

    if not config.skip_google:
        upload_csvs_to_drive(paths)
        update_google_sheet(epa_df, predictions_df)

    logger.info("Weekly NFL workflow completed successfully.")
```

---

# 18. Phase 12 — Configuration

Use environment variables for infrastructure/configuration.

Example `.env.example`:

```env
TRAILING_GAMES_N=5
TRAINING_START_SEASON=2021
RETRAIN_EVERY_WEEKS=4

GOOGLE_DRIVE_FOLDER_ID=
GOOGLE_SHEET_ID=

GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GOOGLE_REFRESH_TOKEN=
```

Do not commit real secrets.

Other useful configuration:

```env
INCLUDE_COMPLETED_WEEK_GAMES=false
LOG_LEVEL=INFO
```

---

# 19. Phase 13 — Logging

Use Python's standard `logging`.

Log to console for GitHub Actions.

Optionally also log to:

```text
logs/nfl_weekly.log
```

Each run should clearly show:

```text
workflow started
target season detected
target week detected
schedule loaded
number of matchups found
PBP download started/completed
EPA calculation completed
model found/missing
retraining started/completed/skipped
model metrics
number of fair-line projections
local CSV paths
Drive upload success/failure
Sheet update success/failure
workflow completed
```

Use timestamps in UTC.

Example:

```text
2026-09-11T01:00:04Z INFO Target: 2026 Week 2
2026-09-11T01:00:05Z INFO Found 15 scheduled games
2026-09-11T01:00:19Z INFO EPA calculation complete
2026-09-11T01:00:20Z INFO Model retraining not required
2026-09-11T01:00:21Z INFO Generated 14 upcoming projections
```

Never log:

- OAuth refresh tokens;
- client secrets;
- credential JSON;
- GitHub secrets.

---

# 20. Phase 14 — Error Handling and Retries

Cloud automation must not fail silently.

Handle at least:

```text
NFL data download/network error
required NFL column missing
schedule empty
no target week found
no matchups found
insufficient EPA history
model file missing
model metadata mismatch
model unpickle failure
training failure
CSV write failure
Google authentication failure
Google Drive upload failure
Google Sheets update failure
```

## 20.1 Retry transient data/API calls

Recommended pattern:

```text
attempt 1
wait 2 seconds
attempt 2
wait 5 seconds
attempt 3
fail with useful error
```

Do not retry deterministic validation errors.

## 20.2 Save local results before Google calls

Order:

```text
calculate
validate
save local CSV
then upload/update Google
```

If Google fails, the analytics output still exists and can be recovered from the cloud job.

## 20.3 GitHub artifact fallback

Optionally upload generated CSVs as GitHub Actions artifacts on every run.

This provides a backup when Google is temporarily unavailable.

---

# 21. Phase 15 — Google Authentication Architecture

This part matters because the job must run unattended while the client's PC is off.

## Recommended for a normal personal Google Drive

Use Google OAuth 2.0 authorization for the client's Google account.

One-time session:

1. create/select Google Cloud project;
2. enable Drive API;
3. enable Sheets API;
4. configure OAuth consent;
5. create OAuth client;
6. client authorizes access;
7. capture a refresh token;
8. store required credential values as GitHub Actions secrets;
9. automation uses the refresh token to obtain access tokens in future.

The client never gives you his Google password.

## Why not rely blindly on a service account for personal My Drive uploads?

A service account does not have normal personal Drive storage ownership.

For unattended creation/upload of CSV files into a normal user's My Drive, user OAuth is the safer general solution.

A service account is a good alternative if:

- the client uses a Google Workspace **Shared Drive**; or
- the workflow only needs access to explicitly shared existing documents and the required file-creation behavior has been verified.

For this client's "new CSV files into Google Drive" requirement, OAuth on the client's Google account is the default recommendation.

---

# 22. Google OAuth Setup — Client Session

Do this only when the code-side workflow is ready.

## 22.1 Google Cloud

Using the client's Google account, preferably create the Google Cloud project under his ownership.

Suggested project name:

```text
NFL Weekly Automation
```

Enable:

```text
Google Drive API
Google Sheets API
```

## 22.2 OAuth client

Create an OAuth client appropriate for the one-time authorization helper.

Do not expose the client secret publicly.

## 22.3 One-time authorization helper

Create:

```text
scripts/google_auth_setup.py
```

Its job:

- start OAuth authorization;
- open/provide Google authorization URL;
- client signs in;
- client grants requested access;
- obtain/store refresh token locally for setup;
- output the values needed to add to GitHub Secrets.

After GitHub secrets are configured:

- remove local token files if not required;
- never commit them.

## 22.4 Use least-required scopes

Do not request broad Google access unnecessarily.

The app only needs enough access to:

- create/update files it manages in Drive;
- create/update the required spreadsheet.

Choose the narrowest scopes that still reliably satisfy the workflow.

---

# 23. Phase 16 — Google Drive Folder

During client setup create a folder such as:

```text
NFL Weekly Model
```

Recommended layout:

```text
NFL Weekly Model/
│
├── CSV Archive/
│   └── 2026/
│       ├── Week 01/
│       │   ├── epa_splits.csv
│       │   └── fair_lines.csv
│       └── Week 02/
│           ├── epa_splits.csv
│           └── fair_lines.csv
│
└── NFL Weekly Analytics (Google Sheet)
```

Store the root/archive folder ID in:

```text
GOOGLE_DRIVE_FOLDER_ID
```

The code should search for/create the season/week subfolder as needed.

Avoid creating duplicate folders on reruns.

Recommended helper:

```python
get_or_create_folder(name, parent_id)
```

---

# 24. Phase 17 — Google Sheet Design

Create one new Google Spreadsheet for the client.

Suggested name:

```text
NFL Weekly Analytics
```

Recommended tabs:

## Tab 1 — `Latest Predictions`

Replace contents each successful run.

Columns:

```text
Season
Week
Game Date
Away Team
Home Team
Away Trailing Net EPA
Home Trailing Net EPA
EPA Difference
Predicted Home Margin
Fair Spread (Home)
Model Version
Run Timestamp
Status
```

## Tab 2 — `Prediction History`

Append/upsert rows each week.

Use a unique key such as:

```text
season + week + away_team + home_team
```

On a rerun, either:

- update existing week rows; or
- append with a run timestamp.

Choose one documented policy.

Recommended: update same week's canonical rows while preserving timestamped CSV archives.

## Tab 3 — `EPA Splits`

Replace with latest EPA split results.

Suggested columns:

```text
Season
Analysis Week
Team
Games Included
Weeks
Off EPA Pass
Off EPA Rush
Def EPA Pass Allowed
Def EPA Rush Allowed
Net EPA Play
Run Timestamp
```

## Tab 4 — `Model Info`

Useful fields:

```text
Model Version
Last Trained
Trailing Games
Training Start Season
Training Through Season
Training Through Week
Coefficient
Intercept
MAE
R²
```

## Optional Tab 5 — `Run Log`

Store compact run-level status:

```text
Run Timestamp
Season
Week
Status
Matchups
Predictions
Retrained
Notes
```

Do not dump secrets or full error traces into the Sheet.

---

# 25. Google Sheets Update Rules

Make updates idempotent.

A repeated cloud run for the same week should not create accidental duplicate history rows unless timestamped history is intentionally desired.

Recommended:

```text
Latest Predictions → replace
EPA Splits         → replace
Model Info         → replace
Prediction History → upsert by unique game key
Run Log            → append one run-level row
```

Use batch updates where possible.

---

# 26. Phase 18 — GitHub Repository

Recommended:

```text
private repository
```

Never commit:

```text
.env
credentials.json
token.json
OAuth refresh token
Google client secret
service-account JSON key
logs containing secrets
```

Recommended `.gitignore`:

```gitignore
.venv/
__pycache__/
*.pyc

.env
credentials.json
token.json
google_credentials.json

logs/*.log

outputs/
!outputs/.gitkeep

models/*.tmp
```

Decide whether the production model pickle is:

1. tracked in the private repository;
2. recreated by workflow;
3. stored as a GitHub artifact;
4. stored in Drive.

For this small project, a tracked model can be acceptable in a private repository if size is small and the client is comfortable with it.

A cleaner option is:

- train on first run if model missing;
- persist/restore the model through an intentional storage strategy;
- keep metadata separately.

Important: GitHub-hosted runners are ephemeral, so a newly retrained model must be persisted somewhere if the next run is expected to reuse it. Do not assume a file written during one Action run will still exist next Friday.

Recommended production options:

- upload the model + metadata to a protected Google Drive model folder and download it at the start of each run; or
- version the small model file in the private repository through a controlled workflow; or
- use another persistent cloud store.

For this project, storing the model and metadata in a dedicated private Google Drive folder is a simple fit because Google integration already exists.

---

# 27. Phase 19 — GitHub Actions Workflow

Create:

```text
.github/workflows/weekly_nfl.yml
```

It must support:

```text
scheduled run
manual workflow_dispatch run
```

Conceptual workflow:

```yaml
name: NFL Weekly Automation

on:
  workflow_dispatch:

  schedule:
    - cron: "0 18 * * 5"
      timezone: "America/Los_Angeles"

jobs:
  run-weekly-model:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run weekly workflow
        env:
          GOOGLE_DRIVE_FOLDER_ID: ${{ secrets.GOOGLE_DRIVE_FOLDER_ID }}
          GOOGLE_SHEET_ID: ${{ secrets.GOOGLE_SHEET_ID }}
          GOOGLE_CLIENT_ID: ${{ secrets.GOOGLE_CLIENT_ID }}
          GOOGLE_CLIENT_SECRET: ${{ secrets.GOOGLE_CLIENT_SECRET }}
          GOOGLE_REFRESH_TOKEN: ${{ secrets.GOOGLE_REFRESH_TOKEN }}
        run: |
          python weekly_runner.py
```

Verify the exact current GitHub Actions syntax in the GitHub documentation when implementing.

## 27.1 Why `America/Los_Angeles`

The client asked for:

```text
Friday 6:00 PM Pacific
```

A timezone-aware schedule handles daylight-saving changes.

Do not convert once to a fixed UTC hour and forget about DST.

## 27.2 Scheduled job timing

Cloud schedulers can start a little later than the exact minute during platform load.

The requirement should be treated as a scheduled target, not real-time financial-exchange timing.

---

# 28. GitHub Actions Secrets

At minimum:

```text
GOOGLE_CLIENT_ID
GOOGLE_CLIENT_SECRET
GOOGLE_REFRESH_TOKEN
GOOGLE_DRIVE_FOLDER_ID
GOOGLE_SHEET_ID
```

Never print them.

Never paste them into source files.

Never put them in `requirements.txt`, README, or screenshots.

If using a single serialized credentials secret instead, document exactly how it is reconstructed in the workflow.

---

# 29. Phase 20 — Manual Cloud Run

The workflow must include:

```text
workflow_dispatch
```

This allows the client/developer to:

- run immediately;
- test after code changes;
- rerun after a temporary API issue;
- regenerate a week.

Manual runs should use the same production code path as scheduled runs.

Do not maintain two separate automation implementations.

---

# 30. Phase 21 — Testing Plan

Testing must happen at multiple levels.

## 30.1 Unit tests

At minimum:

### EPA

- filters only pass/run;
- excludes missing EPA;
- trailing N games works;
- week cutoff excludes target week;
- empty input handled.

### Schedule

- target season/week detected;
- matchups extracted;
- home/away not reversed;
- completed vs upcoming games handled;
- duplicate games not produced.

### Fair line

- insufficient history handled;
- model loads;
- stored `n` validated;
- sign convention correct;
- projection output columns stable.

### Model retraining

- missing model triggers train;
- forced retrain works;
- 4-week rule works;
- training failure does not destroy old model;
- cloud persistence/restoration of model works.

### Google

Mock API calls where practical.

Do not require live Google API access for every unit test.

## 30.2 Historical integration test

Pick a known completed week.

Example:

```bash
python weekly_runner.py \
  --season 2025 \
  --week 6 \
  --skip-google
```

Confirm:

```text
correct week
reasonable number of games
EPA output exists
fair-line output exists
no future-week data is included
CSV columns correct
```

## 30.3 Local Google integration test

With temporary/test authorization:

```text
upload one CSV
update test Sheet
rerun
verify no duplicate folder chaos
verify Sheet tabs update correctly
```

## 30.4 GitHub manual-run test

Trigger `workflow_dispatch`.

Confirm:

```text
runner starts
Python installs
NFL data downloads
model loads/retrains
CSV generated
Drive upload succeeds
Sheet update succeeds
workflow exits 0
```

## 30.5 Production-like test

Run using the automatic season/week logic.

Confirm the detected season/week is exactly what the client expects.

This is especially important before Week 1/preseason.

---

# 31. Critical Edge Cases

Do not skip these.

## 31.1 Preseason/offseason

Automatic current-season helpers can have semantics that differ from "upcoming NFL season".

Validate using schedule data.

## 31.2 Week 1 / early season

The fair-line script currently supports using previous-season games for trailing history during early season.

Preserve this behavior unless client asks otherwise.

## 31.3 Team code changes

NFL data can use specific abbreviations.

Always use the schedule dataset's own team codes rather than manually maintaining a fragile list when possible.

## 31.4 Missing/late PBP update

If nflverse data is temporarily delayed:

- retry;
- log the latest data status;
- fail safely;
- do not upload an empty "successful" Sheet.

## 31.5 No sufficient history

Do not invent a fair line.

Output:

```text
status = insufficient_history
```

and leave projection numeric fields blank/NA.

## 31.6 Google outage/authentication error

Keep locally generated output.

Make GitHub job fail clearly after preserving artifacts.

## 31.7 Rerun same week

Do not accidentally duplicate permanent historical rows.

## 31.8 Model file corrupted

Try validation when loading.

If invalid:

- log failure;
- retrain from source data if allowed;
- never silently use a corrupted object.

## 31.9 Friday after Thursday Night Football

Decide whether the Sheet should display all Week N games or only still-upcoming games.

Recommended:

- preserve all schedule rows with status;
- only label a fair-line row as a live upcoming projection if the game has not already completed;
- never accidentally use completed same-week games as if they were future games.

## 31.10 GitHub runner persistence

Every GitHub-hosted run starts in a fresh environment.

Do not rely on local `models/` or `outputs/` from last week's runner unless they are restored from persistent storage.

---

# 32. Output Validation Before Upload

Before publishing:

```python
assert not epa_df.empty
assert "team" in epa_df.columns
assert epa_df["team"].is_unique

assert not predictions_df.empty or all_games_unprojectable_is_expected
assert {"away_team", "home_team", "fair_spread_home"}.issubset(
    predictions_df.columns
)
```

Also detect:

```text
duplicate matchup rows
NaN team names
same home and away team
invalid week
season mismatch
unexpected zero-game schedule
```

Do not upload clearly broken data.

---

# 33. Model Quality Guardrails

Because the client's existing model is intentionally simple, automation should not claim it predicts betting outcomes with certainty.

Preserve the intended interpretation:

```text
transparent fair-line baseline
```

not:

```text
guaranteed betting signal
```

Model outputs should remain descriptive/analytical.

For retraining, record MAE/R² so large unexpected degradation can be detected.

Optional quality guard:

```python
if not np.isfinite(mae):
    reject_new_model()
```

Optional future guard:

```text
if new MAE is catastrophically worse than previous model:
    keep previous production model
    log warning
```

Define a threshold only after observing real model behavior.

---

# 34. Security Checklist

Before pushing:

- [ ] originals preserved
- [ ] `.env` ignored
- [ ] OAuth tokens ignored
- [ ] credential JSON ignored
- [ ] no password in source
- [ ] no secret in Git history
- [ ] no secret in logs
- [ ] GitHub repository private
- [ ] GitHub Secrets configured
- [ ] Google permissions limited to required account/resources
- [ ] local setup credentials removed when no longer needed
- [ ] client owns/controls final Google resources
- [ ] client can revoke Google access later

If a secret is accidentally committed:

1. revoke/rotate it immediately;
2. remove it from Git history;
3. create a new secret;
4. update GitHub Secrets.

Deleting the file in a later commit is not enough.

---

# 35. Client Session — What Must Be Ready Beforehand

Before contacting David for the session, complete as much as possible:

- [ ] `nflreadpy` migration done
- [ ] EPA script tested
- [ ] fair-line training tested
- [ ] fair-line projection tested
- [ ] automatic season/week detection implemented
- [ ] automatic matchup detection implemented
- [ ] fair-line CSV implemented
- [ ] `weekly_runner.py` implemented
- [ ] periodic retraining implemented
- [ ] model persistence strategy implemented
- [ ] local logging implemented
- [ ] Google integration code prepared
- [ ] GitHub Actions YAML prepared
- [ ] local non-Google end-to-end run passes

The client session should mainly be for:

```text
Google authorization
Google Drive folder
new Google Sheet
GitHub secrets
final cloud run
client access/handoff
```

Do not spend the client's session understanding the original scripts.

---

# 36. Client Session — Exact Checklist

## Google

- [ ] client signs into chosen Google account
- [ ] Google Cloud project created/selected
- [ ] Drive API enabled
- [ ] Sheets API enabled
- [ ] OAuth client created
- [ ] client completes OAuth consent
- [ ] refresh token generated
- [ ] Drive root/archive folder created
- [ ] new Google Sheet created
- [ ] Sheet tabs initialized
- [ ] folder ID recorded
- [ ] spreadsheet ID recorded

## GitHub

- [ ] private repository ready
- [ ] code pushed
- [ ] required GitHub Secrets entered
- [ ] scheduled workflow enabled
- [ ] manual run triggered

## Validation

- [ ] GitHub job successful
- [ ] correct season/week detected
- [ ] correct matchups detected
- [ ] EPA CSV uploaded
- [ ] fair-line CSV uploaded
- [ ] Google Sheet updated
- [ ] model info visible
- [ ] logs understandable

## Client access

- [ ] invite client to repository if he has GitHub
- [ ] explain manual-run button
- [ ] explain Drive folder
- [ ] explain Sheet tabs
- [ ] explain Friday schedule
- [ ] explain retraining frequency

---

# 37. Definition of Done

The project is complete only when all of these are true:

## Code

- [ ] both scripts migrated to `nflreadpy`
- [ ] original calculations preserved unless approved
- [ ] no required manual week entry
- [ ] no required manual matchup entry
- [ ] fair-line projections returned as structured data
- [ ] CSVs generated successfully
- [ ] model metadata stored
- [ ] periodic retraining works
- [ ] manual retraining works
- [ ] errors logged clearly

## Cloud

- [ ] workflow runs without client's PC
- [ ] Friday 6 PM Pacific schedule configured
- [ ] DST handled with `America/Los_Angeles`
- [ ] manual GitHub run works
- [ ] required secrets stored securely
- [ ] retrained model survives across cloud runs via persistent storage

## Google

- [ ] CSVs reach Google Drive
- [ ] dated archive structure works
- [ ] new Google Sheet exists
- [ ] Latest Predictions updates
- [ ] Prediction History updates
- [ ] EPA Splits updates
- [ ] Model Info updates
- [ ] no unwanted duplicate rows/folders

## Reliability

- [ ] historical test passes
- [ ] current automatic test passes
- [ ] rerun same week behaves correctly
- [ ] Google failure is visible/recoverable
- [ ] no empty broken output is published
- [ ] client can understand how to rerun/check results

## Handoff

- [ ] client has Google ownership
- [ ] repository access/ownership arranged
- [ ] final instructions delivered
- [ ] no client password collected
- [ ] no secret left in chat/source/history unnecessarily

---

# 38. Recommended README for the Final Repository

The final `README.md` should include:

## Project

```text
NFL Weekly Analytics Automation
```

## What it does

One short explanation.

## Automatic schedule

```text
Friday at 6:00 PM America/Los_Angeles
```

## Outputs

```text
Google Drive CSV archive
Google Sheet
GitHub run logs
```

## Manual local run

```bash
python weekly_runner.py
```

## Historical run

```bash
python weekly_runner.py --season 2025 --week 6 --skip-google
```

## Force retrain

```bash
python weekly_runner.py --retrain
```

## Manual GitHub run

Explain:

```text
Actions
→ NFL Weekly Automation
→ Run workflow
```

## Sheet tabs

Explain:

```text
Latest Predictions
Prediction History
EPA Splits
Model Info
```

## Fair spread convention

Clearly explain negative/positive home spread convention.

## Troubleshooting

Include:

```text
NFL data unavailable
Google authorization expired
GitHub Action failed
model retraining failed
```

---

# 39. Suggested Implementation Commit Order

Use small commits.

```text
1. chore: preserve original client scripts
2. chore: create project structure and dependencies
3. refactor: migrate PBP loading to nflreadpy
4. test: validate EPA workflow
5. refactor: migrate fair-line data loading to nflreadpy
6. feat: return fair-line projections as dataframe
7. feat: automatic season and week resolution
8. feat: automatic weekly matchup detection
9. feat: add CSV archive output
10. feat: add model metadata and retraining policy
11. feat: add model cloud persistence
12. feat: add weekly runner
13. feat: add logging retries and validation
14. feat: add Google Drive upload
15. feat: add Google Sheets synchronization
16. ci: add manual GitHub Actions workflow
17. ci: add Friday Pacific schedule
18. test: complete end-to-end cloud validation
19. docs: add client README and handoff guide
```

This makes debugging/reverting much easier.

---

# 40. Recommended First Work Session

Do these steps now and nothing else until they pass:

## Step 1

Create project + backup originals.

## Step 2

Create virtual environment.

## Step 3

Install `nflreadpy` and current dependencies.

## Step 4

Migrate only the `load_pbp()` function in `nfl_epa_splits.py`.

## Step 5

Run:

```bash
python nfl_epa_splits.py --season 2025 --week 6
```

## Step 6

Confirm valid `epa_splits.csv`.

## Step 7

Migrate `nfl_fair_line.py`.

## Step 8

Train:

```bash
python nfl_fair_line.py train --seasons 2021 2022 2023 2024
```

## Step 9

Test historical projection.

Only after these pass should automatic week/matchups be implemented.

---

# 41. Work Sequence for Claude Max / Codex

AI coding tools should receive one bounded task at a time.

Do not ask:

```text
"Build the entire project."
```

Prefer:

```text
Task 1:
Migrate nfl_epa_splits.py from nfl_data_py to nflreadpy.
Preserve all existing EPA calculations and CLI behavior.
nflreadpy returns Polars; convert to pandas at the loading boundary.
Add required-column validation and empty-data handling.
Do not change the model/math.
Return the complete updated file plus a short test command.
```

Then test it.

Next:

```text
Task 2:
Migrate nfl_fair_line.py to nflreadpy.
Preserve training logic and existing CLI.
Use project-relative model paths.
Refactor projection to return a pandas DataFrame and optionally save CSV.
Validate saved trailing-games n.
Do not change regression methodology.
```

Then test it.

Then:

```text
Task 3:
Implement automatic season/week and schedule matchup detection using nflreadpy.
Make automatic mode the default, but retain explicit season/week overrides for tests.
Handle preseason/offseason carefully by validating against the loaded regular-season schedule.
```

Continue in this pattern.

Every AI-generated code change must be run and checked.

Do not trust generated code without execution.

---

# 42. Final AI Coding Checklist

For every Claude/Codex change:

- [ ] read current file first
- [ ] state what must remain unchanged
- [ ] make one focused change
- [ ] run formatter/linter if configured
- [ ] run relevant test
- [ ] inspect output
- [ ] compare against prior behavior
- [ ] commit only after pass

AI should help write code.

Human/developer must verify:

```text
data correctness
credentials
cloud behavior
output semantics
client expectations
```

---

# 43. Things Not to Change Without Client Approval

Do not silently change:

- the regression algorithm;
- trailing-game count from default 5;
- EPA mathematical definition;
- fair-spread sign convention;
- training target;
- treatment of home-field intercept;
- the meaning of historical outputs;
- schedule day/time;
- Google account ownership;
- CSV archival policy after client starts relying on it.

If a modeling issue is discovered, complete the requested automation first where safe, then propose the improvement separately.

---

# 44. Nice-to-Have Enhancements After Delivery

These are not required for the current agreed scope unless added deliberately.

Possible future improvements:

```text
email notification after each run
Slack/Discord notification
market spread ingestion
model-vs-market edge calculation
dashboard
additional regression features
weather/injury inputs
time-based cross validation
model comparison
weekly performance tracking
backtesting
automatic charts
data freshness alert
failure notification
```

Keep them out of the initial delivery unless needed to satisfy the agreed requirements.

---

# 45. Final Production Architecture

```text
                 GitHub Actions
          Friday 6:00 PM Pacific
       + manual workflow_dispatch
                       │
                       ▼
                weekly_runner.py
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
  NFL Schedule      NFL PBP       Model Manager
   nflreadpy       nflreadpy      load/retrain
        │              │              │
        └───────┬──────┴──────┬───────┘
                ▼             ▼
          EPA Splits      Fair Lines
                │             │
                └──────┬──────┘
                       ▼
                Local Validation
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   CSV Archive    Google Sheet   Model Store
   Google Drive   Latest/History Google Drive
                  EPA/Model Info or other store
```

---

# 46. Final Handoff Message Content

When everything is complete, tell the client clearly:

```text
- nflreadpy migration is complete
- week/matchups are automatic
- cloud job is active
- PC does not need to be on
- schedule is Friday 6 PM Pacific
- CSV archive location
- Google Sheet location
- model retraining schedule
- how to manually run the workflow
- how to view GitHub logs
- what happens if a run fails
- how to contact you for future modifications
```

Do not simply say:

```text
"Done."
```

Show the client the actual working flow.

---

# 47. Final Acceptance Test to Run With David

During/after the agreed session:

1. Trigger a GitHub manual run.
2. Watch the workflow logs.
3. Confirm automatically detected season.
4. Confirm automatically detected week.
5. Confirm schedule/matchups.
6. Confirm EPA generation.
7. Confirm fair-line generation.
8. Confirm model state/retraining decision.
9. Confirm model can be restored/persisted across cloud runs.
10. Confirm CSV files appear in Google Drive.
11. Open the new Google Sheet.
12. Confirm Latest Predictions.
13. Confirm EPA Splits.
14. Confirm Model Info.
15. Confirm historical/archive behavior.
16. Explain Friday automatic run.
17. Show the manual Run Workflow button.
18. Confirm client has access.
19. Confirm no password/secret was shared insecurely.
20. Record final successful run timestamp.
21. Deliver.

---

# 48. End-to-End Master Checklist

## Preparation

- [ ] client scripts received
- [ ] requirements confirmed
- [ ] originals backed up
- [ ] private repository created
- [ ] virtual environment created
- [ ] dependencies installed

## Migration

- [ ] EPA script uses `nflreadpy`
- [ ] fair-line script uses `nflreadpy`
- [ ] Polars converted to pandas at boundary
- [ ] schema validated
- [ ] original calculations preserved

## EPA

- [ ] historical test passes
- [ ] CSV created
- [ ] empty-data protection added
- [ ] metadata added where useful

## Model

- [ ] historical training passes
- [ ] pickle uses safe project path
- [ ] metadata file created
- [ ] stored `n` validated
- [ ] fair-line DataFrame returned
- [ ] fair-line CSV created
- [ ] model persistence between GitHub runs solved

## Automation logic

- [ ] season auto-detected
- [ ] week auto-detected
- [ ] preseason edge case handled
- [ ] schedule loaded
- [ ] matchups automatic
- [ ] completed/upcoming status handled
- [ ] manual overrides retained

## Retraining

- [ ] model missing → train
- [ ] forced retrain works
- [ ] periodic retraining works
- [ ] future-week leakage avoided
- [ ] old model protected on training failure
- [ ] new model persisted to cloud storage

## Runner

- [ ] one-command execution
- [ ] logging
- [ ] retries
- [ ] validation
- [ ] local output first
- [ ] Google second
- [ ] non-zero exit on true failure

## Google

- [ ] Google Cloud project
- [ ] Drive API
- [ ] Sheets API
- [ ] OAuth authorization
- [ ] refresh token secured
- [ ] Drive folder
- [ ] Sheet created
- [ ] Latest Predictions
- [ ] Prediction History
- [ ] EPA Splits
- [ ] Model Info
- [ ] rerun behavior verified

## GitHub

- [ ] private repo
- [ ] no credentials committed
- [ ] Actions secrets
- [ ] workflow_dispatch
- [ ] Friday schedule
- [ ] `America/Los_Angeles`
- [ ] manual cloud run succeeds
- [ ] scheduled workflow enabled

## Final

- [ ] current automatic run succeeds
- [ ] Drive contains CSVs
- [ ] Sheet contains expected data
- [ ] GitHub logs clean
- [ ] client access provided
- [ ] README delivered
- [ ] client understands manual run
- [ ] client understands retraining
- [ ] client understands failure check
- [ ] final ownership confirmed

---

# 49. Recommended Immediate Next Action

The very first coding task is:

```text
Migrate nfl_epa_splits.py from nfl_data_py to nflreadpy,
without changing its EPA calculations,
then run it against a known historical season/week
and verify that epa_splits.csv is generated correctly.
```

After that passes:

```text
migrate nfl_fair_line.py
→ validate training
→ validate projection
→ automatic season/week
→ automatic matchups
→ structured CSV
→ model manager
→ cloud model persistence
→ weekly_runner.py
→ Google
→ GitHub Actions
→ client authorization
→ final test
```

Follow this order and the project stays easy to debug and safe to hand over.

---

# 50. Official Documentation Areas to Verify During Implementation

Because packages and cloud platforms evolve, verify the final implementation against the current official documentation for:

- `nflreadpy` installation, `load_pbp()`, `load_schedules()`, `get_current_season()`, and `get_current_week()`;
- GitHub Actions scheduled workflows, timezone support, secrets, and `workflow_dispatch`;
- Google Drive API OAuth/scopes, file uploads, folder creation, and refresh-token handling;
- Google Sheets API spreadsheet creation and value/batch updates.

Do not depend on an old blog/tutorial when official documentation conflicts with it.

---

# 51. Source-Specific Notes From the Client's Original Code

Keep these points visible while implementing:

### `nfl_epa_splits.py`

- Input currently requires `--season` and `--week`.
- Default trailing games: 5.
- Default output filename: `epa_splits.csv`.
- It calculates pass/rush offense and defense separately.
- It only uses games where `week < through_week`.
- It already writes CSV.

### `nfl_fair_line.py`

- Model: `LinearRegression`.
- Feature: home trailing net EPA minus away trailing net EPA.
- Target: home score minus away score.
- Training currently uses `train_test_split(..., random_state=42)`.
- Model is stored in `fair_line_model.pkl`.
- Saved pickle contains the model and trailing-game `n`.
- Projection currently loads two seasons (`season - 1` and `season`) so early-season history can include the previous year.
- Projection currently requires manual matchup strings such as `AWAY@HOME`.
- Projection currently prints results but does not save a projection CSV.

Those behaviors form the baseline that the automation should preserve unless a change is intentionally approved.

---

# 52. Final Success State

A successful final Friday workflow should look like this:

```text
Friday 6:00 PM Pacific
        ↓
GitHub Action starts while client's PC may be off
        ↓
Credentials loaded securely
        ↓
Current/upcoming NFL regular-season week resolved
        ↓
Weekly schedule loaded
        ↓
Remaining/upcoming matchups identified
        ↓
Latest nflverse PBP loaded through nflreadpy
        ↓
EPA splits calculated
        ↓
Existing model restored from persistent storage
        ↓
Model retrained if policy says it is due
        ↓
New model + metadata persisted safely
        ↓
Fair-line projections generated
        ↓
Output validation passes
        ↓
Local CSVs created
        ↓
CSV archive uploaded to Google Drive
        ↓
Google Sheet tabs updated idempotently
        ↓
GitHub run ends successfully with readable logs
        ↓
Client can open Drive/Sheet and see the new week's results
```

That is the target state for delivery.
