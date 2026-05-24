import pandas as pd

from src.data.constants import (
    INTERIM_DIR,
    KAGGLE_DELIVERIES,
    KAGGLE_MATCHES,
    RAW_DIR,
)

KAGGLE_RAW = RAW_DIR / "kaggle_ipl"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)


def load_matches():
    df = pd.read_csv(KAGGLE_RAW / "matches.csv")
    df.columns = [col.lower().strip().replace(" ", "_") for col in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["season"] = df["season"].astype(str)
    return df


def load_deliveries():
    df = pd.read_csv(KAGGLE_RAW / "deliveries.csv")
    df.columns = [col.lower().strip().replace(" ", "_") for col in df.columns]
    required_columns = {
        "match_id", "inning", "over", "ball", "batter", "bowler",
        "batting_team", "bowling_team", "total_runs",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return df


def main():
    matches = load_matches()
    deliveries = load_deliveries()
    matches.to_parquet(KAGGLE_MATCHES, index=False)
    deliveries.to_parquet(KAGGLE_DELIVERIES, index=False)
    print(f"Matches: {len(matches)} rows → {KAGGLE_MATCHES}")
    print(f"Deliveries: {len(deliveries)} rows → {KAGGLE_DELIVERIES}")


if __name__ == "__main__":
    main()
