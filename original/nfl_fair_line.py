"""
NFL Fair-Line Model - Step 4 of the weekly workflow
----------------------------------------------------
Trains a simple regression mapping each team's trailing-5-game net EPA/play
differential to actual game margins, then uses it to convert this week's
EPA splits into a projected point spread you can compare against the market.

This is intentionally simple (one feature: EPA differential, plus home
field). The point is a defensible, transparent baseline - not a black box.
Treat its output as your fair line to compare against the market, not as
a bet signal on its own (see Section 3, Step 5 of the framework).

SETUP: same as nfl_epa_splits.py
    pip install nfl_data_py appdirs --break-system-packages --no-deps
    pip install pandas numpy pyarrow scikit-learn --break-system-packages

USAGE:
    Train + evaluate the model on historical seasons:
        python nfl_fair_line.py train --seasons 2021 2022 2023 2024

    Project spreads for an upcoming week using a trained model:
        python nfl_fair_line.py project --season 2025 --week 6 \
            --matchups "KC@BUF,SF@LA"
"""

import argparse
import pickle
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

MODEL_PATH = "fair_line_model.pkl"


# ---------------------------------------------------------------------
# Shared aggregation logic (same as nfl_epa_splits.py)
# ---------------------------------------------------------------------

def load_pbp(seasons):
    df = nfl.import_pbp_data(seasons, downcast=True, cache=False)
    df = df[
        (df["season_type"] == "REG")
        & (df["play_type"].isin(["pass", "run"]))
        & (df["epa"].notna())
    ].copy()
    return df


def team_game_epa(df):
    off = (
        df.groupby(["posteam", "season", "week", "game_id"])["epa"]
        .mean()
        .reset_index()
        .rename(columns={"posteam": "team", "epa": "off_epa"})
    )
    deff = (
        df.groupby(["defteam", "season", "week", "game_id"])["epa"]
        .mean()
        .reset_index()
        .rename(columns={"defteam": "team", "epa": "def_epa_allowed"})
    )
    merged = off.merge(deff, on=["team", "season", "week", "game_id"], how="outer")
    merged["net_epa"] = merged["off_epa"] - merged["def_epa_allowed"]
    return merged.sort_values(["team", "season", "week"])


def trailing_net_epa(game_epa, team, season, week, n=5):
    """Trailing n-game net EPA for `team`, using only games strictly before `week` in `season`."""
    hist = game_epa[
        (game_epa["team"] == team)
        & (
            (game_epa["season"] < season)
            | ((game_epa["season"] == season) & (game_epa["week"] < week))
        )
    ].sort_values(["season", "week"]).tail(n)
    if len(hist) < 3:  # require a minimum sample (early season = less reliable)
        return None
    return hist["net_epa"].mean()


def game_scores(df):
    """One row per game with home/away teams, scores, season, week."""
    return (
        df[["game_id", "season", "week", "home_team", "away_team", "home_score", "away_score"]]
        .drop_duplicates("game_id")
        .dropna(subset=["home_score", "away_score"])
    )


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def build_training_set(seasons, n=5):
    pbp = load_pbp(seasons)
    ge = team_game_epa(pbp)
    games = game_scores(pbp)

    rows = []
    for _, g in games.iterrows():
        home_epa = trailing_net_epa(ge, g["home_team"], g["season"], g["week"], n)
        away_epa = trailing_net_epa(ge, g["away_team"], g["season"], g["week"], n)
        if home_epa is None or away_epa is None:
            continue
        rows.append({
            "season": g["season"], "week": g["week"],
            "home_team": g["home_team"], "away_team": g["away_team"],
            "epa_diff": home_epa - away_epa,   # feature
            "actual_margin": g["home_score"] - g["away_score"],  # target (home perspective)
        })
    return pd.DataFrame(rows)


def train(seasons, n=5):
    print(f"Building training set from {seasons}...")
    data = build_training_set(seasons, n)
    print(f"{len(data)} games with sufficient trailing history.\n")

    X = data[["epa_diff"]].values
    y = data["actual_margin"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("=== Model fit ===")
    print(f"Predicted margin = {model.intercept_:.2f} + {model.coef_[0]:.2f} * (home_epa - away_epa)")
    print(f"  Intercept ({model.intercept_:.2f}) is roughly the home-field-advantage term in points.")
    print(f"\nHold-out test set (n={len(y_test)}):")
    print(f"  Mean Absolute Error: {mae:.2f} points")
    print(f"  R^2: {r2:.3f}")
    print(
        "\nContext: Vegas closing lines typically have an MAE of roughly "
        "10-10.5 points against actual margins league-wide, since final "
        "scores are inherently noisy. Your model beating that bar isn't "
        "realistic or the goal -- the goal is a transparent baseline to "
        "compare against the market number, not to replace it."
    )

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "n": n}, f)
    print(f"\nSaved model to {MODEL_PATH}")

    return model


# ---------------------------------------------------------------------
# Projecting upcoming games
# ---------------------------------------------------------------------

def project(season, week, matchups, n=5):
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]

    # pull enough history to compute trailing splits into this week
    seasons_needed = list(range(season - 1, season + 1))
    pbp = load_pbp(seasons_needed)
    ge = team_game_epa(pbp)

    print(f"\n{'Matchup':<12}{'Fair Spread (home)':<22}{'Note'}")
    print("-" * 60)
    for m in matchups:
        away, home = m.split("@")
        home_epa = trailing_net_epa(ge, home, season, week, n)
        away_epa = trailing_net_epa(ge, away, season, week, n)
        if home_epa is None or away_epa is None:
            print(f"{away}@{home:<8}{'insufficient history':<22}(need >=3 games this/last season)")
            continue
        diff = home_epa - away_epa
        pred_margin = model.intercept_ + model.coef_[0] * diff
        # Convention: negative = home favored by that many points
        print(f"{away}@{home:<8}{-pred_margin:+.1f} {'':<17}home_epa={home_epa:.3f}, away_epa={away_epa:.3f}")

    print(
        "\nRead as: your model's projected home-team spread. Compare directly "
        "to the market spread for the same game (Step 5 of the framework)."
    )


# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fair-line regression model for NFL spreads.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train the model on historical seasons.")
    p_train.add_argument("--seasons", type=int, nargs="+", required=True)
    p_train.add_argument("--n", type=int, default=5)

    p_proj = sub.add_parser("project", help="Project spreads for upcoming games.")
    p_proj.add_argument("--season", type=int, required=True)
    p_proj.add_argument("--week", type=int, required=True)
    p_proj.add_argument("--matchups", type=str, required=True,
                         help="Comma-separated AWAY@HOME, e.g. 'KC@BUF,SF@LA'")
    p_proj.add_argument("--n", type=int, default=5)

    args = parser.parse_args()

    if args.command == "train":
        train(args.seasons, args.n)
    elif args.command == "project":
        matchups = args.matchups.split(",")
        project(args.season, args.week, matchups, args.n)


if __name__ == "__main__":
    main()
