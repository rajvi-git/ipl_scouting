"""Observed IPL-only impact labels for supervised training."""

from __future__ import annotations

import numpy as np
import pandas as pd


BAT_IMPACT_WEIGHTS = {
    "strike_rate": 0.40,
    "runs_per_innings": 0.40,
    "boundary_pct": 0.20,
}

BOWL_IMPACT_WEIGHTS = {
    "economy": -0.40,
    "wickets_per_innings": 0.35,
    "bowling_sr": -0.25,
}

IMPACT_SHRINKAGE_K = {
    "bat": 10.0,
    "bowl": 10.0,
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


def add_shrunk_impact_scores(labels: pd.DataFrame) -> pd.DataFrame:
    """Add raw, reliability, and sample-shrunk IPL impact labels."""
    out = add_impact_scores(labels).rename(columns={"ipl_impact": "ipl_impact_raw"})
    out["ipl_impact_reliability"] = np.nan
    out["ipl_impact_shrunk"] = np.nan
    out["ipl_impact"] = np.nan

    for role in sorted(out["role"].dropna().unique()):
        mask = (out["role"] == role) & out["ipl_impact_raw"].notna()
        if not mask.any():
            continue

        sample_col = "bat_innings" if role == "bat" else "bowl_innings"
        if sample_col not in out.columns:
            sample = pd.Series(0.0, index=out.loc[mask].index)
        else:
            sample = out.loc[mask, sample_col].fillna(0).astype(float)

        k = IMPACT_SHRINKAGE_K.get(role, 10.0)
        reliability = sample / (sample + k)
        role_mean = out.loc[mask, "ipl_impact_raw"].mean()
        shrunk = reliability * out.loc[mask, "ipl_impact_raw"] + (1.0 - reliability) * role_mean

        out.loc[mask, "ipl_impact_reliability"] = reliability
        out.loc[mask, "ipl_impact_shrunk"] = shrunk
        out.loc[mask, "ipl_impact"] = shrunk

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
