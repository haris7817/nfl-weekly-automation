"""NFL Weekly Automation - shared package.

Modules
-------
paths          Project-relative filesystem locations (cloud-safe).
config         Environment-driven configuration.
logging_utils  UTC logging setup for local + GitHub Actions.
nfl_data       nflreadpy loading boundary (Polars -> pandas) with retries.
epa            EPA splits (client's original math, preserved).
fair_line      Trailing net EPA, training set, structured projections.
schedule       Automatic season/week detection and matchup extraction.
model_manager  Model persistence, metadata and retraining policy.
validation     Output sanity checks performed before publishing.
google_auth    OAuth credential construction from environment variables.
google_drive   Drive archive + model store.
google_sheets  Google Sheet synchronisation.
"""

__version__ = "1.0.0"
