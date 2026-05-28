"""Observed IPL-only impact labels for supervised training."""

from __future__ import annotations

import numpy as np
import pandas as pd


BAT_IMPACT_WEIGHTS = {
    "strike_rate": 0.25,
    "runs_per_innings": 0.25,
    "boundary_pct": 0.15,
    "bat_pp_sr": 0.10,
    "bat_mid_sr": 0.10,
    "bat_death_sr": 0.15,
}

BOWL_IMPACT_WEIGHTS = {
    "economy": -0.25,
    "bowling_sr": -0.20,
    "wickets_per_innings": 0.25,
    "bowl_pp_economy": -0.15,
    "bowl_death_economy": -0.15,
}


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def add_impact_scores(labels: pd.DataFrame) -> pd.DataFrame:
    """Add role-specific observed IPL impact scores to label rows."""
    out = labels.copy()
    out["ipl_impact"] = np.nan

    for role, weights in [("bat", BAT_IMPACT_WEIGHTS), ("bowl", BOWL_IMPACT_WEIGHTS)]:
        mask = out["role"] == role
        if not mask.any():
            continue

        score = pd.Series(0.0, index=out.loc[mask].index)
        used_weight = 0.0
        for col, weight in weights.items():
            if col not in out.columns:
                continue
            values = out.loc[mask, col]
            valid = values.notna()
            if valid.sum() < 2:
                continue
            score = score.add(_zscore(values).fillna(0.0) * weight, fill_value=0.0)
            used_weight += abs(weight)

        if used_weight > 0:
            out.loc[mask, "ipl_impact"] = score / used_weight
    return out


def add_tier_labels(labels: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Assign fixed percentile tiers within each role from observed IPL impact."""
    out = labels.copy()
    out["tier"] = pd.NA
    thresholds: dict = {}

    for role in sorted(out["role"].dropna().unique()):
        mask = (out["role"] == role) & out["ipl_impact"].notna()
        values = out.loc[mask, "ipl_impact"]
        if len(values) < 4:
            continue

        q25, q60, q85 = np.nanquantile(values, [0.25, 0.60, 0.85])
        thresholds[role] = {
            "bust_max": float(q25),
            "marginal_max": float(q60),
            "solid_max": float(q85),
        }

        out.loc[mask & (out["ipl_impact"] <= q25), "tier"] = "bust"
        out.loc[mask & (out["ipl_impact"] > q25) & (out["ipl_impact"] <= q60), "tier"] = "marginal"
        out.loc[mask & (out["ipl_impact"] > q60) & (out["ipl_impact"] <= q85), "tier"] = "solid"
        out.loc[mask & (out["ipl_impact"] > q85), "tier"] = "star"

    return out, thresholds
