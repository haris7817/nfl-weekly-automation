"""
NFL Last-5-Game EPA Splits
--------------------------
Pulls play-by-play data via nfl_data_py, computes each team's offensive
and defensive EPA/play (pass and rush split) over their last 5 games,
and outputs a clean table you can use for Step 1 of the weekly workflow.

SETUP (run once):
    pip install nfl_data_py appdirs --break-system-packages --no-deps
    pip install pandas numpy pyarrow --break-system-packages

Note: nfl_data_py's listed requirements (pandas<2.0, numpy<2.0) are outdated
and unnecessarily strict. The package works fine with modern pandas/numpy,
so install it with --no-deps to avoid a broken dependency resolution.

USAGE:
    python nfl_epa_splits.py --season 2025 --week 6
    (pulls all games through week 5, computes each team's last-5-game splits
    heading into week 6)
"""

import argparse
import pandas as pd
import nfl_data_py as nfl


def load_pbp(season: int) -> pd.DataFrame:
    """Pull play-by-play data for the given season."""
    df = nfl.import_pbp_data([season], downcast=True, cache=False)
    # keep regular-season offensive plays with a valid EPA value
    df = df[
        (df["season_type"] == "REG")
        & (df["play_type"].isin(["pass", "run"]))
        & (df["epa"].notna())
    ].copy()
    return df


def team_game_epa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse play-by-play into one row per team per game, with offensive
    and defensive EPA/play split by pass and rush.
    """
    # --- offense: EPA a team generated on plays where they had the ball ---
    off = (
        df.groupby(["posteam", "week", "game_id", "play_type"])["epa"]
        .mean()
        .unstack("play_type")
        .rename(columns={"pass": "off_epa_pass", "run": "off_epa_rush"})
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    # --- defense: EPA a team allowed on plays where they were on defense ---
    deff = (
        df.groupby(["defteam", "week", "game_id", "play_type"])["epa"]
        .mean()
        .unstack("play_type")
        .rename(columns={"pass": "def_epa_pass_allowed", "run": "def_epa_rush_allowed"})
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    merged = off.merge(deff, on=["team", "week", "game_id"], how="outer")
    return merged.sort_values(["team", "week"])


def last_n_splits(game_epa: pd.DataFrame, through_week: int, n: int = 5) -> pd.DataFrame:
    """
    For each team, average their last n games' EPA splits, using only
    games completed before `through_week`.
    """
    results = []
    for team, grp in game_epa[game_epa["week"] < through_week].groupby("team"):
        grp = grp.sort_values("week").tail(n)
        if grp.empty:
            continue
        results.append({
            "team": team,
            "games_included": len(grp),
            "weeks": ", ".join(str(w) for w in grp["week"]),
            "off_epa_pass": round(grp["off_epa_pass"].mean(), 3),
            "off_epa_rush": round(grp["off_epa_rush"].mean(), 3),
            "def_epa_pass_allowed": round(grp["def_epa_pass_allowed"].mean(), 3),
            "def_epa_rush_allowed": round(grp["def_epa_rush_allowed"].mean(), 3),
        })
    out = pd.DataFrame(results)
    # net EPA/play differential - the single number most useful for Step 4
    out["net_epa_play"] = (
        (out["off_epa_pass"] + out["off_epa_rush"])
        - (out["def_epa_pass_allowed"] + out["def_epa_rush_allowed"])
    ).round(3)
    return out.sort_values("net_epa_play", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Last-5-game EPA splits for all NFL teams.")
    parser.add_argument("--season", type=int, required=True, help="Season year, e.g. 2025")
    parser.add_argument("--week", type=int, required=True,
                         help="Upcoming week you're analyzing (uses games BEFORE this week)")
    parser.add_argument("--n", type=int, default=5, help="Number of trailing games (default 5)")
    parser.add_argument("--out", type=str, default="epa_splits.csv", help="Output CSV filename")
    args = parser.parse_args()

    print(f"Pulling {args.season} play-by-play data...")
    pbp = load_pbp(args.season)

    print("Aggregating team-game EPA...")
    game_epa = team_game_epa(pbp)

    print(f"Computing last {args.n} game splits heading into week {args.week}...")
    splits = last_n_splits(game_epa, through_week=args.week, n=args.n)

    splits.to_csv(args.out, index=False)
    print(f"\nSaved {len(splits)} teams to {args.out}\n")
    print(splits.to_string(index=False))


if __name__ == "__main__":
    main()
