"""Build supervised SMA-to-IPL training pairs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.constants import PROCESSED_DIR
from src.features.impact_score import add_shrunk_impact_scores, add_tier_labels

FEATURES_PATH = PROCESSED_DIR / "player_features.parquet"
TRAINING_PATH = PROCESSED_DIR / "ml_training_pairs.parquet"
THRESHOLDS_PATH = PROCESSED_DIR / "tier_thresholds.json"
MIN_TARGET_IPL_BAT_INNINGS = 5
MIN_TARGET_IPL_BOWL_INNINGS = 5

BAT_FEATURES = [
    "bat_innings",
    "runs",
    "balls",
    "fours",
    "sixes",
    "strike_rate",
    "runs_per_innings",
    "boundary_pct",
    "bat_pp_sr",
    "bat_mid_sr",
    "bat_death_sr",
    "caf_strike_rate",
    "caf_runs_per_innings",
    "caf_boundary_pct",
    "caf_bat_pp_sr",
    "caf_bat_mid_sr",
    "caf_bat_death_sr",
    "shrunk_strike_rate",
    "shrunk_runs_per_innings",
    "shrunk_boundary_pct",
    "shrunk_bat_pp_sr",
    "shrunk_bat_mid_sr",
    "shrunk_bat_death_sr",
]

BOWL_FEATURES = [
    "bowl_innings",
    "runs_conceded",
    "balls_bowled",
    "wickets",
    "economy",
    "bowling_sr",
    "wickets_per_innings",
    "bowl_pp_economy",
    "bowl_mid_economy",
    "bowl_death_economy",
    "wide_rate",
    "noball_rate",
    "caf_economy",
    "caf_bowling_sr",
    "caf_wickets_per_innings",
    "caf_bowl_pp_economy",
    "caf_bowl_mid_economy",
    "caf_bowl_death_economy",
    "shrunk_economy",
    "shrunk_bowling_sr",
    "shrunk_wickets_per_innings",
    "shrunk_bowl_pp_economy",
    "shrunk_bowl_mid_economy",
    "shrunk_bowl_death_economy",
]


def _weighted_mean(df: pd.DataFrame, col: str, weight_col: str) -> float:
    valid = df[[col, weight_col]].dropna()
    valid = valid[valid[weight_col] > 0]
    if valid.empty:
        return np.nan
    return float(np.average(valid[col], weights=valid[weight_col]))


def _aggregate_role_rows(rows: pd.DataFrame, role: str) -> dict:
    out = {
        "seasons": int(rows["season"].nunique()),
        "start_season": int(rows["season"].min()),
        "end_season": int(rows["season"].max()),
    }

    if role == "bat":
        count_cols = [
            "bat_innings",
            "runs",
            "balls",
            "fours",
            "sixes",
            "bat_pp_runs",
            "bat_pp_balls",
            "bat_mid_runs",
            "bat_mid_balls",
            "bat_death_runs",
            "bat_death_balls",
        ]
        for col in count_cols:
            out[col] = float(rows[col].sum(skipna=True)) if col in rows else np.nan

        balls = out.get("balls", np.nan)
        innings = out.get("bat_innings", np.nan)
        out["strike_rate"] = out["runs"] / balls * 100 if balls and balls > 0 else np.nan
        out["runs_per_innings"] = out["runs"] / innings if innings and innings > 0 else np.nan
        out["boundary_pct"] = (out["fours"] + out["sixes"]) / balls * 100 if balls and balls > 0 else np.nan
        for phase in ("pp", "mid", "death"):
            runs = out.get(f"bat_{phase}_runs", np.nan)
            phase_balls = out.get(f"bat_{phase}_balls", np.nan)
            out[f"bat_{phase}_sr"] = runs / phase_balls * 100 if phase_balls and phase_balls > 0 else np.nan

        for col in [c for c in BAT_FEATURES if c.startswith(("caf_", "shrunk_"))]:
            if col in rows:
                out[col] = _weighted_mean(rows, col, "bat_innings")
    else:
        count_cols = [
            "bowl_innings",
            "runs_conceded",
            "balls_bowled",
            "wickets",
            "bowl_pp_runs",
            "bowl_pp_balls",
            "bowl_mid_runs",
            "bowl_mid_balls",
            "bowl_death_runs",
            "bowl_death_balls",
            "wides",
            "noballs",
        ]
        for col in count_cols:
            out[col] = float(rows[col].sum(skipna=True)) if col in rows else np.nan

        balls = out.get("balls_bowled", np.nan)
        innings = out.get("bowl_innings", np.nan)
        wickets = out.get("wickets", np.nan)
        out["economy"] = out["runs_conceded"] / balls * 6 if balls and balls > 0 else np.nan
        out["bowling_sr"] = balls / wickets if wickets and wickets > 0 else np.nan
        out["wickets_per_innings"] = wickets / innings if innings and innings > 0 else np.nan
        out["wide_rate"] = out["wides"] / balls * 6 if balls and balls > 0 else np.nan
        out["noball_rate"] = out["noballs"] / balls * 6 if balls and balls > 0 else np.nan
        for phase in ("pp", "mid", "death"):
            runs = out.get(f"bowl_{phase}_runs", np.nan)
            phase_balls = out.get(f"bowl_{phase}_balls", np.nan)
            out[f"bowl_{phase}_economy"] = runs / phase_balls * 6 if phase_balls and phase_balls > 0 else np.nan

        for col in [c for c in BOWL_FEATURES if c.startswith(("caf_", "shrunk_"))]:
            if col in rows:
                out[col] = _weighted_mean(rows, col, "bowl_innings")

    return out


def _feature_prefix(row: dict) -> dict:
    return {f"sma_{key}": value for key, value in row.items()}


def build_training_table(features: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Build temporal player-season rows: prior SMA history -> target IPL season."""
    rows = []
    thresholds: dict = {}

    for role in ("bat", "bowl"):
        role_df = features[features["role"] == role].copy()
        for player_key, player_rows in role_df.groupby("player_key", dropna=False):
            ipl_rows = player_rows[player_rows["competition"] == "IPL"]
            if ipl_rows.empty:
                continue

            for target_season in sorted(ipl_rows["season"].dropna().astype(int).unique()):
                target_rows = ipl_rows[ipl_rows["season"] == target_season]
                if target_rows.empty:
                    continue

                if role == "bat":
                    target_sample = target_rows["bat_innings"].sum(skipna=True)
                    if target_sample < MIN_TARGET_IPL_BAT_INNINGS:
                        continue
                else:
                    target_sample = target_rows["bowl_innings"].sum(skipna=True)
                    if target_sample < MIN_TARGET_IPL_BOWL_INNINGS:
                        continue

                sma_rows = player_rows[
                    (player_rows["competition"] == "SMA") & (player_rows["season"] < target_season)
                ]
                if sma_rows.empty:
                    continue

                label_stats = _aggregate_role_rows(target_rows, role)
                feature_stats = _aggregate_role_rows(sma_rows, role)
                player_name = (
                    sma_rows["player_name"].dropna().iloc[-1]
                    if sma_rows["player_name"].notna().any()
                    else None
                )
                player_id = (
                    sma_rows["player_id"].dropna().iloc[-1]
                    if sma_rows["player_id"].notna().any()
                    else player_key
                )

                row = {
                    "player_key": player_key,
                    "player_id": player_id,
                    "player_name": player_name,
                    "role": role,
                    "target_ipl_season": target_season,
                    "target_ipl_sample": float(target_sample),
                    "pair_type": "temporal_player_season",
                }
                row.update(_feature_prefix(feature_stats))
                row.update({f"ipl_{key}": value for key, value in label_stats.items()})
                rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table, thresholds

    label_cols = [c for c in table.columns if c.startswith("ipl_")]
    labels = table[["player_key", "role", *label_cols]].copy()
    labels = labels.rename(columns={c: c.removeprefix("ipl_") for c in label_cols})
    labels = add_shrunk_impact_scores(labels)
    labels, thresholds = add_tier_labels(labels)

    table["y_impact_raw"] = labels["ipl_impact_raw"].values
    table["y_impact_reliability"] = labels["ipl_impact_reliability"].values
    table["y_impact"] = labels["ipl_impact_shrunk"].values
    table["tier"] = labels["tier"].values
    table = table[table["y_impact"].notna() & table["tier"].notna()].reset_index(drop=True)
    return table, thresholds


def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing features table: {FEATURES_PATH}. Run run_features.py first.")

    features = pd.read_parquet(FEATURES_PATH)
    table, thresholds = build_training_table(features)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(TRAINING_PATH, index=False)
    with open(THRESHOLDS_PATH, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)
    print(f"Saved {len(table)} training rows to {TRAINING_PATH}")
    if not table.empty:
        print(table.groupby(["role", "tier"]).size())


if __name__ == "__main__":
    main()
