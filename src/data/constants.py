from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"

# Kaggle (optional QA) — separate from Cricsheet interim files
KAGGLE_MATCHES = INTERIM_DIR / "kaggle_matches.parquet"
KAGGLE_DELIVERIES = INTERIM_DIR / "kaggle_deliveries.parquet"
# Legacy names written by early load_kaggle.py runs
KAGGLE_MATCHES_LEGACY = INTERIM_DIR / "matches.parquet"
KAGGLE_DELIVERIES_LEGACY = INTERIM_DIR / "deliveries.parquet"

# Cricsheet (primary pipeline for feature engineering)
CRICSHEET_BALLS = INTERIM_DIR / "cricsheet_balls.parquet"
CRICSHEET_MATCHES = INTERIM_DIR / "cricsheet_matches.parquet"

CRICSHEET_DIRS = {
    "IPL": RAW_DIR / "cricsheet_ipl",
    "SMA": RAW_DIR / "cricsheet_sma",
}

# CRICSHEET_DIRS = {
#     "IPL": RAW_DIR / "cricsheet_ipl_single",
# }

def resolve_kaggle_matches_path() -> Path:
    if KAGGLE_MATCHES.exists():
        return KAGGLE_MATCHES
    if KAGGLE_MATCHES_LEGACY.exists():
        return KAGGLE_MATCHES_LEGACY
    return KAGGLE_MATCHES


def resolve_kaggle_deliveries_path() -> Path:
    if KAGGLE_DELIVERIES.exists():
        return KAGGLE_DELIVERIES
    if KAGGLE_DELIVERIES_LEGACY.exists():
        return KAGGLE_DELIVERIES_LEGACY
    return KAGGLE_DELIVERIES

MIN_IPL_BAT_INNINGS = 10
MIN_SMA_BAT_INNINGS = 15
MIN_IPL_BOWL_INNINGS = 10
MIN_SMA_BOWL_INNINGS = 15
SHRINKAGE_K = 20
