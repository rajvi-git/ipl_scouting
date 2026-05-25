"""Layer 4: Bayesian shrinkage toward competition-specific league means."""

from __future__ import annotations

import pandas as pd

from src.data.constants import SHRINKAGE_K

from src.features.caf import BAT_STATS, BOWL_STATS


def _league_means(df: pd.DataFrame, role: str, stats: list[str], prefix: str) -> dict:
    sub = df[(df["role"] == role) & (df["competition"] == "IPL")]
    means = {}
    for stat in stats:
        col = f"{prefix}{stat}"
        if col in sub.columns:
            means[stat] = sub[col].mean()
        elif stat in sub.columns:
            means[stat] = sub[stat].mean()
    return means


def apply_shrinkage(
    df: pd.DataFrame,
    value_prefix: str = "caf_",
    out_prefix: str = "shrunk_",
    k: float = SHRINKAGE_K,
) -> pd.DataFrame:
    """Shrink CAF-adjusted stats toward IPL league means (same competition row uses IPL mean)."""
    out = df.copy()

    for role, stats, inn_col in [
        ("bat", BAT_STATS, "bat_innings"),
        ("bowl", BOWL_STATS, "bowl_innings"),
    ]:
        means = _league_means(df, role, stats, value_prefix)
        mask = out["role"] == role
        n = out.loc[mask, inn_col].fillna(0).astype(float)

        for stat in stats:
            val_col = f"{value_prefix}{stat}"
            if val_col not in out.columns:
                continue
            mu = means.get(stat)
            if mu is None or pd.isna(mu):
                mu = out.loc[mask, val_col].mean()
            shrunk_col = f"{out_prefix}{stat}"
            out.loc[mask, shrunk_col] = (n * out.loc[mask, val_col] + k * mu) / (n + k)

    return out
