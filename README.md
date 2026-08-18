# NFL Weekly Analytics Automation

Automated weekly NFL analytics: every Friday at **6:00 PM Pacific**, a cloud job
works out which NFL week is next, pulls the latest play-by-play data, computes
EPA splits, generates fair-line projections, archives the CSVs to Google Drive
and refreshes a Google Sheet.

**It runs in GitHub's cloud, so your PC does not need to be on.**

The full project specification this system was built against lives at
[`docs/NFL_Weekly_Automation_End_to_End_Implementation_Guide.md`](docs/NFL_Weekly_Automation_End_to_End_Implementation_Guide.md).
This README covers day-to-day operation; the guide covers the requirements,
design decisions and acceptance criteria behind it.

---

## Table of contents

- [What it produces](#what-it-produces)
- [How it works](#how-it-works)
- [Fair spread sign convention](#fair-spread-sign-convention)
- [Installation](#installation)
- [Running it](#running-it)
- [The Google Sheet](#the-google-sheet)
- [Google Drive layout](#google-drive-layout)
- [Model and retraining](#model-and-retraining)
- [Configuration](#configuration)
- [Google setup (one time)](#google-setup-one-time)
- [GitHub Actions setup](#github-actions-setup)
- [Logging](#logging)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Security notes](#security-notes)
- [What changed from the original scripts](#what-changed-from-the-original-scripts)
- [Known limitations](#known-limitations)

---

## What it produces

Every run writes three files per week:

| File | Contents |
|---|---|
| `epa_splits.csv` | Each team's trailing-5-game offensive and defensive EPA per play, split pass/rush, plus a `net_epa_play` differential |
| `fair_lines.csv` | One row per scheduled game with the model's projected home margin and fair spread |
| `run_summary.json` | What the run detected and did — season, week, matchup count, model version, retraining decision |

They land in three places: locally under `outputs/`, in Google Drive, and as
downloadable GitHub Actions artifacts.

---

## How it works

```
             GitHub Actions
      Friday 18:00 America/Los_Angeles
        + manual "Run workflow" button
                    |
                    v
            weekly_runner.py
                    |
     +--------------+--------------+
     v              v              v
 NFL schedule    NFL PBP      Model store
  nflreadpy      nflreadpy    (Google Drive)
     |              |              |
     +------+-------+------+-------+
            v              v
      EPA splits      Fair lines
            |              |
            +------+-------+
                   v
            Output validation
                   |
        +----------+----------+
        v          v          v
   CSV archive  Google     Model +
   Google Drive  Sheet     metadata
```

The runner is the single production entry point. Manual runs and scheduled runs
use exactly the same code path.

### Automatic season and week detection

The target week is **the first regular-season week that still has a game
kicking off in the future**.

This matters more than it sounds. `nflreadpy.get_current_season()` uses a
Thursday-after-Labor-Day rule, so during the summer it reports the season that
just *finished* — verified on 2026-08-18, it returns `2025` even though the next
games to be played are 2026 Week 1. Detection is therefore driven by the
schedule itself, which also means:

- on Friday evening it still selects the current week, even though Thursday
  Night Football has already been played;
- it rolls forward once the last game of a week has kicked off;
- a postponed game that never receives a final score cannot wedge detection on
  a stale week.

You can always override it with `--season` and `--week`.

---

## Fair spread sign convention

This is unchanged from the original script and is worth stating plainly:

```
fair_spread_home = -(predicted home margin)
```

| Value | Meaning |
|---|---|
| `fair_spread_home = -6.5` | **Home team favoured** by 6.5 points |
| `fair_spread_home = +3.0` | **Home team underdog** by 3.0 points |

Read it as your model's fair line for the home team, to compare directly
against the market number. It is a transparent baseline, not a bet signal.

---

## Installation

Tested on **Python 3.13.7**. The GitHub Actions workflow pins the same version.

```powershell
git clone <your-private-repo-url>
cd nfl-weekly-automation

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

python -m pip install --upgrade pip
pip install -r requirements.txt
```

For local development, also install the test dependencies:

```powershell
pip install -r requirements-dev.txt
```

Optional but recommended locally — cache nflverse downloads on disk so repeat
runs take seconds instead of minutes:

```powershell
$env:NFLREADPY_CACHE_MODE = "filesystem"
```

---

## Running it

Everything below works from the project root.

### The normal run

```powershell
python weekly_runner.py
```

Detects the season and week, does everything, publishes to Google.

### Common variations

```powershell
# Analytics only - no Google calls at all
python weekly_runner.py --skip-google

# Reproduce a specific historical week
python weekly_runner.py --season 2025 --week 6 --skip-google --include-completed

# Force the model to retrain now
python weekly_runner.py --retrain

# Compute everything, write nothing anywhere
python weekly_runner.py --dry-run --skip-google
```

| Flag | Effect |
|---|---|
| `--season` / `--week` | Override detection. Must be used together. |
| `--retrain` | Force retraining before projecting. |
| `--skip-google` | Skip Drive upload and Sheet update. |
| `--dry-run` | Run the full pipeline but write no files and make no Google calls. |
| `--include-completed` | Project games that already kicked off. **Required for historical reruns**, because every game in a past week is finished. |
| `--log-level` | `DEBUG`, `INFO`, `WARNING`, `ERROR`. |

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | The analytics pipeline failed; nothing was published |
| `2` | Invalid arguments |
| `3` | **Analytics succeeded and were saved locally**, but Google publishing failed |

Code `3` is deliberately distinct: it tells you the numbers are fine and only
the publishing step needs attention. The CSVs are on disk and in the run's
artifacts, and rerunning is safe because Sheet updates are idempotent.

### The two original scripts

Both still run standalone, which is useful for spot checks:

```powershell
python nfl_epa_splits.py --season 2025 --week 6

python nfl_fair_line.py train --seasons 2021 2022 2023 2024
python nfl_fair_line.py project --season 2025 --week 6 --include-completed
python nfl_fair_line.py project --season 2025 --week 6 --matchups "KC@BUF,SF@LA"
```

The untouched originals are preserved in [`original/`](original/) for reference.

---

## The Google Sheet

Named **NFL Weekly Analytics**, with five tabs:

| Tab | Update rule |
|---|---|
| **Latest Predictions** | Replaced with the newest week's fair lines |
| **Prediction History** | Upserted on `season + week + away_team + home_team` |
| **EPA Splits** | Replaced with the latest EPA table |
| **Model Info** | Replaced with current model provenance and metrics |
| **Run Log** | One appended row per run |

All updates are **idempotent**. Running the same week twice refreshes that
week's history rows rather than appending a duplicate set, so you can safely
hit "Run workflow" more than once.

---

## Google Drive layout

```
NFL Weekly Model/                 <- GOOGLE_DRIVE_FOLDER_ID
├── CSV Archive/
│   └── 2026/
│       ├── Week 01/
│       │   ├── epa_splits.csv
│       │   ├── fair_lines.csv
│       │   └── run_summary.json
│       └── Week 02/
│           └── ...
├── Model/
│   ├── fair_line_model.pkl
│   └── model_metadata.json
└── NFL Weekly Analytics          (the Google Sheet)
```

Folders are looked up by name before being created, and a file with the same
name in the same folder is updated in place rather than duplicated — so reruns
do not litter the archive.

---

## Model and retraining

The model is the original one, unchanged:

| Aspect | Value |
|---|---|
| Feature | `home trailing net EPA − away trailing net EPA` |
| Target | `home_score − away_score` |
| Estimator | `sklearn.linear_model.LinearRegression` |
| Trailing window | 5 games (minimum 3 required) |
| Evaluation | `train_test_split(test_size=0.2, random_state=42)`, scored by MAE and R² |

For context, Vegas closing lines run about 10–10.5 MAE points against actual
margins. Beating that is not the goal; a transparent baseline is.

### When it retrains

Automatically when **any** of these is true:

- no model exists yet;
- the stored model is corrupt or cannot be loaded;
- a **new season** has started (a full completed season of new data is now
  available);
- `RETRAIN_EVERY_WEEKS` (default 4) weeks have passed since the last training;
- you passed `--retrain`.

### No future-data leakage

When retraining ahead of Week N, current-season games from Week N onward are
excluded from the training rows. Earlier completed seasons are used in full.
The run log states this explicitly:

```
Leakage guard: 1165 of 1359 completed games kept (excluded season 2025 week >= 6).
```

### Safety

Training writes to a temporary file, reads it back to confirm it loads, and
only then replaces the live model. **A failed retrain can never destroy the
last working model.** The trailing-game `n` is recorded at training time and
enforced at projection time, so a model trained with `n=5` cannot silently be
used to project with `n=7`.

### Model persistence

GitHub runners are ephemeral — anything written during one run is gone by the
next. The model and its metadata are therefore stored in the `Model/` folder in
Drive, downloaded at the start of each run and re-uploaded after retraining.

---

## Configuration

Copy `.env.example` to `.env` for local use. In the cloud these are GitHub
Actions secrets and variables, never a file.

| Variable | Default | Purpose |
|---|---|---|
| `TRAILING_GAMES_N` | `5` | Trailing games for EPA and the model feature |
| `MIN_TRAILING_GAMES` | `3` | Minimum history before a team can be projected |
| `TRAINING_START_SEASON` | `2021` | Oldest season used for training |
| `RETRAIN_EVERY_WEEKS` | `4` | Retraining cadence |
| `INCLUDE_COMPLETED_WEEK_GAMES` | `false` | Project games that already kicked off |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `GOOGLE_DRIVE_FOLDER_ID` | — | Root Drive folder |
| `GOOGLE_SHEET_ID` | — | The spreadsheet |
| `GOOGLE_CLIENT_ID` | — | **Secret** |
| `GOOGLE_CLIENT_SECRET` | — | **Secret** |
| `GOOGLE_REFRESH_TOKEN` | — | **Secret** |
| `NFLREADPY_CACHE_MODE` | memory | Set to `filesystem` locally to cache downloads |
| `USE_SYSTEM_CERT_STORE` | auto | OS certificate store for TLS (see troubleshooting) |

Changing `TRAILING_GAMES_N` is a modelling change, not a setting tweak — it
alters the numbers the client relies on. Retrain after changing it.

---

## Google setup (one time)

Done once, with the account owner present. **Nobody ever needs the client's
Google password** — sign-in happens on Google's own page.

1. In [Google Cloud Console](https://console.cloud.google.com/), create a
   project (suggested name: *NFL Weekly Automation*) **under the client's
   account**, so the client owns it.
2. Enable the **Google Drive API** and the **Google Sheets API**.
3. Configure the OAuth consent screen. While it is in "Testing", add the
   client's address as a test user.
4. Create an **OAuth client ID** of type **Desktop app** and download the JSON.
5. Run the helper:

   ```powershell
   python scripts/google_auth_setup.py --client-secrets client_secret.json
   ```

   It opens Google's consent screen, captures a refresh token, creates the
   Drive folder and the spreadsheet with all five tabs, and prints the five
   values to store as GitHub secrets.
6. Paste those five values into GitHub Actions secrets.
7. **Delete `client_secret.json` from the machine.**

### Why the setup helper creates the folder and the Sheet

The automation requests the narrow `drive.file` scope, which grants access
**only to files the app itself created**. It can never read the rest of the
client's Drive. The trade-off is that a folder created by hand in the browser
would be invisible to it — hence the helper creates them via the API.

If an existing hand-made folder must be used, rerun the helper with
`--scope-drive-full --drive-folder-id <id>`, which requests broader access.
Prefer the default.

---

## GitHub Actions setup

Use a **private** repository.

Add these under *Settings → Secrets and variables → Actions*:

| Secret | From |
|---|---|
| `GOOGLE_CLIENT_ID` | setup helper output |
| `GOOGLE_CLIENT_SECRET` | setup helper output |
| `GOOGLE_REFRESH_TOKEN` | setup helper output |
| `GOOGLE_DRIVE_FOLDER_ID` | setup helper output |
| `GOOGLE_SHEET_ID` | setup helper output |

Optional repository **variables** (not secrets) override analytics defaults:
`TRAILING_GAMES_N`, `TRAINING_START_SEASON`, `RETRAIN_EVERY_WEEKS`, `LOG_LEVEL`.

### Running it manually

*Actions → **NFL Weekly Automation** → **Run workflow***

The manual form accepts optional season/week overrides and toggles for
retraining, including completed games, and skipping Google.

### How the Friday 6 PM Pacific schedule works

GitHub Actions supports an IANA `timezone` field on cron schedules (added in
the late-March 2026 Actions update), so the workflow states the requirement
directly and daylight-saving changes are handled by GitHub:

```yaml
schedule:
  - cron: "0 18 * * 5"
    timezone: "America/Los_Angeles"
```

GitHub may start scheduled jobs late under platform load — during peak windows
by 15 minutes to 2+ hours. Treat 6:00 PM as a target, not a hard deadline.

---

## Logging

Logs go to the console (visible live in the Actions run) and to
`logs/nfl_weekly.log` locally. **Timestamps are UTC.**

```
2026-08-18T10:01:24Z INFO  nfl.schedule: Target detected: 2026 week 1.
2026-08-18T10:01:24Z INFO  nfl.schedule: Found 16 scheduled games for 2026 week 1 (16 upcoming).
2026-08-18T10:01:24Z INFO  nfl.data:     PBP download completed: 168277 rows retained.
2026-08-18T10:01:24Z INFO  nfl.fair_line: Leakage guard: 1359 of 1359 completed games kept.
2026-08-18T10:01:24Z INFO  nfl.runner:   Model metrics: MAE=9.97 points, R^2=0.129.
2026-08-18T10:01:24Z INFO  nfl.fair_line: Generated 16 projections from 16 scheduled games.
```

Credentials are never logged. Drive and spreadsheet IDs appear only in
redacted form (`1a2b...<len=44>`).

---

## Testing

```powershell
pytest -q
```

164 tests, all on synthetic data — **no network access and no Google
credentials required**, so they are safe to run anywhere, including on pull
requests. They cover play filtering, the trailing-window rules, target-week
exclusion, season/week detection (including the preseason trap and the
Friday-after-TNF case), home/away direction, matchup deduplication, the spread
sign convention, model loading and `n` validation, all four retraining
triggers, failed-training preservation, output validation, Sheet idempotency,
and full runner flows.

For a real end-to-end check against live nflverse data:

```powershell
python weekly_runner.py --season 2025 --week 6 --skip-google --include-completed
```

---

## Troubleshooting

### `CERTIFICATE_VERIFY_FAILED` on a local Windows machine

Consumer antivirus (AVG, Avast, Kaspersky, ESET) and corporate proxies
intercept HTTPS and re-sign it with a locally generated root CA. That CA is in
the Windows certificate store, so browsers work, but Python's bundled `certifi`
list does not include it.

The project handles this automatically by verifying against the **operating
system** trust store on Windows (via `truststore`) — certificate checking stays
fully enabled. To force the certifi bundle instead, set
`USE_SYSTEM_CERT_STORE=false`. This does not affect the Linux cloud runner.

### `Season must be between 1999 and <year>`

nflverse has not published play-by-play for that season yet. This is normal
before a season starts; the runner detects it and continues using the seasons
that do exist.

### EPA splits are empty in Week 1

Expected. The EPA splits method uses completed games **from the same season
only**, and in Week 1 there are none. Fair-line projections still work, because
they reach back into the previous season for trailing history — that is the
original script's behaviour, deliberately preserved. EPA splits fill in from
Week 2.

### Every game shows `already_completed` and there are no projections

You are rerunning a week whose games have all finished. Add
`--include-completed`.

### `invalid_grant` from Google

The refresh token has been revoked or expired. Rerun
`scripts/google_auth_setup.py` and update `GOOGLE_REFRESH_TOKEN`.

### Google returns 404 for the folder or spreadsheet

With the `drive.file` scope the app can only reach files it created. Confirm
`GOOGLE_DRIVE_FOLDER_ID` and `GOOGLE_SHEET_ID` are the values the setup helper
printed, not IDs of items created by hand in the browser.

### The scheduled run did not fire

Scheduled runs can start well after the scheduled minute under platform load —
check the *Actions* tab before assuming it was missed. Note that GitHub
disables scheduled workflows in repositories with no activity for 60 days; a
commit or a manual run re-enables them.

### The workflow failed but I need the numbers

Open the run page and download the `nfl-outputs-*` artifact. Artifacts are
uploaded even when the job fails.

---

## Security notes

- The repository must stay **private**.
- Never commit `.env`, `credentials.json`, `client_secret*.json`, `token.json`,
  refresh tokens or service-account keys. All are in `.gitignore`, and the test
  workflow fails the build if such a file is ever tracked.
- The client's Google password is never requested, seen, or stored — sign-in
  happens on Google's own page.
- Scopes are the narrowest that work: `drive.file` (only files this app
  created) and `spreadsheets`.
- Secrets are never printed; IDs are redacted in logs.
- The client can revoke access at any time at
  [myaccount.google.com/permissions](https://myaccount.google.com/permissions).

**If a secret is ever committed:** revoke and rotate it immediately in Google
Cloud Console, purge it from git history, create a new one, and update the
GitHub secret. Deleting it in a later commit is *not* enough.

---

## What changed from the original scripts

The EPA and regression mathematics are **unchanged**. This was verified by
running the original `nfl_data_py` code and the migrated code side by side:

- **EPA splits: identical.** All 32 teams, every column, byte for byte.
- **Fair line: identical.** Same games selected, same target values. The fitted
  coefficients differ by about `5e-06`, entirely because the original called
  `import_pbp_data(downcast=True)`, which stored EPA as `float32`. Casting the
  migrated data to `float32` reproduces the original output exactly
  (max difference `0.000e+00`). The effect on a fair spread is
  **~0.000001 points**.

What changed is the infrastructure around the maths:

| Area | Before | After |
|---|---|---|
| Data source | `nfl_data_py` | `nflreadpy` (Polars → pandas at the boundary) |
| Season/week | Typed in by hand | Detected from the schedule |
| Matchups | Typed in as `KC@BUF,SF@LA` | Read from the schedule |
| Fair-line output | Printed to the terminal | DataFrame → CSV → Drive → Sheet |
| Model path | `fair_line_model.pkl` in the working directory | `models/` resolved from the project root |
| Model provenance | None | `model_metadata.json` with metrics and training range |
| Trailing `n` | Stored but not enforced | Enforced at projection time |
| Retraining | Manual | Automatic policy, leakage-guarded |
| Training speed | O(rows) lookup per game | Pre-grouped index — **~117× faster**, arithmetically identical |
| Failure handling | Exception | Retries, validation, clear messages, distinct exit codes |

---

## Known limitations

These are honest constraints, not defects:

1. **EPA splits are empty in Week 1** of any season — the client's method uses
   same-season completed games only. Fair lines are unaffected.
2. **The reported MAE/R² depend on training-row order**, because the original
   uses a random 80/20 split with a fixed seed rather than a chronological
   holdout. This is the original methodology, preserved deliberately.
3. **Offseason runs** target the upcoming season's Week 1 and will regenerate
   the same projections each Friday. History rows are upserted, so nothing
   duplicates, but the numbers will not move until games are played.
4. **The model is intentionally simple** — one feature plus a home-field
   intercept. It is a transparent fair-line baseline for comparison against the
   market, not a betting signal.

### Possible future enhancements (not in the agreed scope)

Chronological or season-based holdout validation instead of a random split;
market-line ingestion and edge calculation; additional regression features
(rest, weather, injuries); email or Slack notification on failure; weekly
performance tracking and backtesting.

Each would be a deliberate, separately agreed change — see the implementation
guide's note on not silently altering the client's model.
