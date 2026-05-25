"""Run full feature engineering: parse → aggregates → CAF → shrinkage."""

from __future__ import annotations

import pandas as pd

from src.data.build_innings import build_player_season_stats, load_balls
from src.data.constants import PROCESSED_DIR
from src.data.parse_cricsheet import main as parse_main
from src.features.caf import apply_caf, estimate_caf_factors, save_caf_factors
from src.features.shrinkage import apply_shrinkage


def run_feature_engineering(skip_parse: bool = False) -> pd.DataFrame:
    if not skip_parse:
        print("=== Step 1: Parse Cricsheet JSON ===")
        parse_main()
    else:
        print("=== Step 1: Skipped parse (using existing parquet) ===")

    print("=== Step 2: Build player-season aggregates + phase splits ===")
    balls = load_balls()
    raw = build_player_season_stats(balls)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(PROCESSED_DIR / "player_season_raw.parquet", index=False)

    print("=== Step 3: Layer 1 — CAF ===")
    factors = estimate_caf_factors(raw)
    save_caf_factors(factors)
    factor_values = factors.get("factors", factors)
    print(f"  CAF factors saved ({sum(len(v) for v in factor_values.values())} multipliers)")
    with_caf = apply_caf(raw, factors)

    print("=== Step 4: Layer 4 — Bayesian shrinkage (toward IPL means) ===")
    featured = apply_shrinkage(with_caf, value_prefix="caf_", out_prefix="shrunk_")

    out_path = PROCESSED_DIR / "player_features.parquet"
    featured.to_parquet(out_path, index=False)
    print(f"=== Done: {len(featured)} rows -> {out_path} ===")

    summary = featured.groupby(["competition", "role"]).size().unstack(fill_value=0)
    print(summary)
    return featured


def main() -> None:
    run_feature_engineering(skip_parse=False)


if __name__ == "__main__":
    main()
