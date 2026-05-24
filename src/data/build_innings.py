"""Aggregate ball-by-ball data into player-season statistics with phase splits."""

from __future__ import annotations

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Allow `python src/data/parse_cricsheet.py` from any working directory
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.constants import CRICSHEET_BALLS, PROCESSED_DIR


def _safe_rate(num: pd.Series, den: pd.Series, mult: float = 1.0) -> pd.Series:
    return (num / den.replace(0, np.nan) * mult).astype(float)


def load_balls() -> pd.DataFrame:
    if not CRICSHEET_BALLS.exists():
        raise FileNotFoundError(f"Run parse_cricsheet first: {CRICSHEET_BALLS}")
    return pd.read_parquet(CRICSHEET_BALLS)


def build_batting_innings(balls: pd.DataFrame) -> pd.DataFrame:
    legal = balls[balls["legal_ball"] == 1].copy()
    #Find aggregate batting stats per player per innings
    g = legal.groupby(
        ["match_id", "competition", "season", "innings", "batter", "batter_id"],
        dropna=False,
    ).agg(
        runs=("runs_batter", "sum"),
        balls=("legal_ball", "sum"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum"),
        batting_team=("batting_team", "first"),
    ).reset_index()

    #Find aggregate batting stats per player per innings per phase (separate aggregates for powerplay, middle, and death)
    phase = (
        legal.groupby(
            ["match_id", "competition", "season", "innings", "batter", "batter_id", "phase"],
            dropna=False,
        )
        .agg(pp_runs=("runs_batter", "sum"), pp_balls=("legal_ball", "sum"))
        .reset_index()
    )

    for p, prefix in [("powerplay", "pp"), ("middle", "mid"), ("death", "death")]:
        sub = phase[phase["phase"] == p][
            ["match_id", "competition", "season", "innings", "batter", "batter_id", "pp_runs", "pp_balls"]
        ].rename(columns={"pp_runs": f"{prefix}_runs", "pp_balls": f"{prefix}_balls"})
        g = g.merge(sub, on=["match_id", "competition", "season", "innings", "batter", "batter_id"], how="left")

    g["strike_rate"] = _safe_rate(g["runs"], g["balls"], 100.0)
    for prefix in ("pp", "mid", "death"):
        g[f"{prefix}_sr"] = _safe_rate(g[f"{prefix}_runs"], g[f"{prefix}_balls"], 100.0)

    g["player_key"] = g["batter_id"].fillna(g["batter"])
    return g


def build_bowling_innings(balls: pd.DataFrame) -> pd.DataFrame:
    g = balls.groupby(
        ["match_id", "competition", "season", "innings", "bowler", "bowler_id"],
        dropna=False,
    ).agg(
        runs_conceded=("runs_total", "sum"),
        balls_bowled=("legal_ball", "sum"),
        wickets=("is_wicket", "sum"),
        wides=("is_wide", "sum"),
        noballs=("is_noball", "sum"),
        bowling_team=("bowling_team", "first"),
    ).reset_index()

    phase = (
        balls.groupby(
            ["match_id", "competition", "season", "innings", "bowler", "bowler_id", "phase"],
            dropna=False,
        )
        .agg(runs=("runs_total", "sum"), balls=("legal_ball", "sum"))
        .reset_index()
    )

    for p, prefix in [("powerplay", "pp"), ("middle", "mid"), ("death", "death")]:
        sub = phase[phase["phase"] == p][
            ["match_id", "competition", "season", "innings", "bowler", "bowler_id", "runs", "balls"]
        ].rename(columns={"runs": f"{prefix}_runs", "balls": f"{prefix}_balls"})
        g = g.merge(sub, on=["match_id", "competition", "season", "innings", "bowler", "bowler_id"], how="left")

    g["economy"] = _safe_rate(g["runs_conceded"], g["balls_bowled"], 6.0)
    for prefix in ("pp", "mid", "death"):
        g[f"{prefix}_economy"] = _safe_rate(g[f"{prefix}_runs"], g[f"{prefix}_balls"], 6.0)

    g["player_key"] = g["bowler_id"].fillna(g["bowler"])
    return g


def aggregate_player_season_batting(innings: pd.DataFrame) -> pd.DataFrame:
    g = innings.groupby(["player_key", "batter", "batter_id", "competition", "season"], dropna=False).agg(
        bat_innings=("match_id", "nunique"),
        runs=("runs", "sum"),
        balls=("balls", "sum"),
        fours=("fours", "sum"),
        sixes=("sixes", "sum"),
        pp_runs=("pp_runs", "sum"),
        pp_balls=("pp_balls", "sum"),
        mid_runs=("mid_runs", "sum"),
        mid_balls=("mid_balls", "sum"),
        death_runs=("death_runs", "sum"),
        death_balls=("death_balls", "sum"),
    ).reset_index()

    g["strike_rate"] = _safe_rate(g["runs"], g["balls"], 100.0)
    g["runs_per_innings"] = _safe_rate(g["runs"], g["bat_innings"], 1.0)
    g["boundary_pct"] = _safe_rate(g["fours"] + g["sixes"], g["balls"], 100.0)
    g["pp_sr"] = _safe_rate(g["pp_runs"], g["pp_balls"], 100.0)
    g["mid_sr"] = _safe_rate(g["mid_runs"], g["mid_balls"], 100.0)
    g["death_sr"] = _safe_rate(g["death_runs"], g["death_balls"], 100.0)
    g["role"] = "bat"
    return g


def aggregate_player_season_bowling(innings: pd.DataFrame) -> pd.DataFrame:
    g = innings.groupby(["player_key", "bowler", "bowler_id", "competition", "season"], dropna=False).agg(
        bowl_innings=("match_id", "nunique"),
        runs_conceded=("runs_conceded", "sum"),
        balls_bowled=("balls_bowled", "sum"),
        wickets=("wickets", "sum"),
        pp_runs=("pp_runs", "sum"),
        pp_balls=("pp_balls", "sum"),
        mid_runs=("mid_runs", "sum"),
        mid_balls=("mid_balls", "sum"),
        death_runs=("death_runs", "sum"),
        death_balls=("death_balls", "sum"),
        wides=("wides", "sum"),
        noballs=("noballs", "sum"),
    ).reset_index()

    g["economy"] = _safe_rate(g["runs_conceded"], g["balls_bowled"], 6.0)
    g["bowling_sr"] = _safe_rate(g["balls_bowled"], g["wickets"], 1.0)
    g["wickets_per_innings"] = _safe_rate(g["wickets"], g["bowl_innings"], 1.0)
    g["pp_economy"] = _safe_rate(g["pp_runs"], g["pp_balls"], 6.0)
    g["mid_economy"] = _safe_rate(g["mid_runs"], g["mid_balls"], 6.0)
    g["death_economy"] = _safe_rate(g["death_runs"], g["death_balls"], 6.0)
    g["role"] = "bowl"
    g["wide_rate"] = _safe_rate(g["wides"], g["balls_bowled"], 6.0)
    g["noball_rate"] = _safe_rate(g["noballs"], g["balls_bowled"], 6.0)
    return g


def build_player_season_stats(balls: pd.DataFrame) -> pd.DataFrame:
    bat_inn   = build_batting_innings(balls)
    bowl_inn  = build_bowling_innings(balls)
    bat_season  = aggregate_player_season_batting(bat_inn)
    bowl_season = aggregate_player_season_bowling(bowl_inn)

    bat_season = bat_season.rename(columns={
        "batter":      "player_name",
        "batter_id":   "player_id",
        "pp_runs":     "bat_pp_runs",
        "pp_balls":    "bat_pp_balls",
        "mid_runs":    "bat_mid_runs",
        "mid_balls":   "bat_mid_balls",
        "death_runs":  "bat_death_runs",
        "death_balls": "bat_death_balls",
        "pp_sr":       "bat_pp_sr",
        "mid_sr":      "bat_mid_sr",
        "death_sr":    "bat_death_sr",
    })

    bowl_season = bowl_season.rename(columns={
        "bowler":      "player_name",
        "bowler_id":   "player_id",
        "pp_runs":     "bowl_pp_runs",
        "pp_balls":    "bowl_pp_balls",
        "mid_runs":    "bowl_mid_runs",
        "mid_balls":   "bowl_mid_balls",
        "death_runs":  "bowl_death_runs",
        "death_balls": "bowl_death_balls",
        "pp_economy":  "bowl_pp_economy",
        "mid_economy": "bowl_mid_economy",
        "death_economy": "bowl_death_economy",
    })

    combined = pd.concat([bat_season, bowl_season], ignore_index=True)
    return combined


def main() -> None:
    balls = load_balls()
    stats = build_player_season_stats(balls)
    print(stats.columns)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "player_season_raw.parquet"
    stats.to_parquet(out, index=False)
    print(f"Saved {len(stats)} player-season rows to {out}")
    print(stats.groupby(["competition", "role"]).size())


if __name__ == "__main__":
    main()
