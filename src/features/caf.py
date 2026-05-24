"""Layer 1: Competition Adjustment Factor (CAF) — domestic → IPL translation multipliers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.constants import (
    MIN_IPL_BAT_INNINGS,
    MIN_IPL_BOWL_INNINGS,
    MIN_SMA_BAT_INNINGS,
    MIN_SMA_BOWL_INNINGS,
    PROCESSED_DIR,
)

CAF_PATH = PROCESSED_DIR / "caf_factors.json"
CAF_CLIP = (0.5, 2.0)

BAT_STATS = ["strike_rate", "runs_per_innings", "boundary_pct", "pp_sr", "mid_sr", "death_sr"]
BOWL_STATS = ["economy", "bowling_sr", "wickets_per_innings", "pp_economy", "mid_economy", "death_economy"]


def _career_stats(df: pd.DataFrame, role: str) -> pd.DataFrame:
    sub = df[df["role"] == role].copy()
    if sub.empty:
        return sub

    if role == "bat":
        agg = sub.groupby("player_key", as_index=False).agg(
            bat_innings=("bat_innings", "sum"),
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
        )
        agg["strike_rate"] = np.where(agg["balls"] > 0, agg["runs"] / agg["balls"] * 100, np.nan)
        agg["runs_per_innings"] = np.where(agg["bat_innings"] > 0, agg["runs"] / agg["bat_innings"], np.nan)
        agg["boundary_pct"] = np.where(
            agg["balls"] > 0, (agg["fours"] + agg["sixes"]) / agg["balls"] * 100, np.nan
        )
        for p in ("pp", "mid", "death"):
            r, b = f"{p}_runs", f"{p}_balls"
            agg[f"{p}_sr"] = np.where(agg[b] > 0, agg[r] / agg[b] * 100, np.nan)
    else:
        agg = sub.groupby("player_key", as_index=False).agg(
            bowl_innings=("bowl_innings", "sum"),
            runs_conceded=("runs_conceded", "sum"),
            balls_bowled=("balls_bowled", "sum"),
            wickets=("wickets", "sum"),
            pp_runs=("pp_runs", "sum"),
            pp_balls=("pp_balls", "sum"),
            mid_runs=("mid_runs", "sum"),
            mid_balls=("mid_balls", "sum"),
            death_runs=("death_runs", "sum"),
            death_balls=("death_balls", "sum"),
        )
        agg["economy"] = np.where(
            agg["balls_bowled"] > 0, agg["runs_conceded"] / agg["balls_bowled"] * 6, np.nan
        )
        agg["bowling_sr"] = np.where(agg["wickets"] > 0, agg["balls_bowled"] / agg["wickets"], np.nan)
        agg["wickets_per_innings"] = np.where(
            agg["bowl_innings"] > 0, agg["wickets"] / agg["bowl_innings"], np.nan
        )
        for p in ("pp", "mid", "death"):
            r, b = f"{p}_runs", f"{p}_balls"
            agg[f"{p}_economy"] = np.where(agg[b] > 0, agg[r] / agg[b] * 6, np.nan)

    return agg


def _eligible_players(career_sma: pd.DataFrame, career_ipl: pd.DataFrame, role: str) -> pd.Index:
    if role == "bat":
        sma_ok = career_sma["bat_innings"] >= MIN_SMA_BAT_INNINGS
        ipl_ok = career_ipl["bat_innings"] >= MIN_IPL_BAT_INNINGS
    else:
        sma_ok = career_sma["bowl_innings"] >= MIN_SMA_BOWL_INNINGS
        ipl_ok = career_ipl["bowl_innings"] >= MIN_IPL_BOWL_INNINGS

    sma_keys = set(career_sma.loc[sma_ok, "player_key"])
    ipl_keys = set(career_ipl.loc[ipl_ok, "player_key"])
    return pd.Index(list(sma_keys & ipl_keys))


def estimate_caf_factors(player_season: pd.DataFrame) -> dict:
    sma = player_season[player_season["competition"] == "SMA"]
    ipl = player_season[player_season["competition"] == "IPL"]

    factors: dict = {"bat": {}, "bowl": {}}

    for role, stats in [("bat", BAT_STATS), ("bowl", BOWL_STATS)]:
        c_sma = _career_stats(sma, role).set_index("player_key")
        c_ipl = _career_stats(ipl, role).set_index("player_key")
        paired = _eligible_players(c_sma.reset_index(), c_ipl.reset_index(), role)

        for stat in stats:
            if stat not in c_sma.columns or stat not in c_ipl.columns:
                continue
            ratios = []
            for pk in paired:
                dom = c_sma.loc[pk, stat] if pk in c_sma.index else np.nan
                ipl_v = c_ipl.loc[pk, stat] if pk in c_ipl.index else np.nan
                if pd.notna(dom) and pd.notna(ipl_v) and dom > 0:
                    r = float(ipl_v / dom)
                    if CAF_CLIP[0] <= r <= CAF_CLIP[1]:
                        ratios.append(r)
            if ratios:
                factors[role][stat] = float(np.median(ratios))
            else:
                factors[role][stat] = 1.0

    return factors


def apply_caf(player_season: pd.DataFrame, factors: dict) -> pd.DataFrame:
    out = player_season.copy()
    for role, stats in [("bat", BAT_STATS), ("bowl", BOWL_STATS)]:
        mask = out["role"] == role
        for stat in stats:
            caf = factors.get(role, {}).get(stat, 1.0)
            if stat in out.columns:
                out.loc[mask, f"caf_{stat}"] = out.loc[mask, stat] * caf
    return out


def save_caf_factors(factors: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(CAF_PATH, "w", encoding="utf-8") as f:
        json.dump(factors, f, indent=2)


def load_caf_factors() -> dict:
    with open(CAF_PATH, encoding="utf-8") as f:
        return json.load(f)
