"""Parse Cricsheet JSON files into unified balls and matches parquet tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow `python src/data/parse_cricsheet.py` from any working directory
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.data.constants import CRICSHEET_BALLS, CRICSHEET_DIRS, CRICSHEET_MATCHES, INTERIM_DIR


def _phase(over: int) -> str:
    if over <= 5:
        return "powerplay"
    if over <= 14:
        return "middle"
    return "death"


def _season_year(dates: list) -> int | None:
    if not dates:
        return None
    return int(str(dates[0])[:4])


def _competition_code(event_name: str | None, folder_code: str) -> str:
    if event_name and "Indian Premier League" in event_name:
        return "IPL"
    if event_name and "Syed Mushtaq Ali" in event_name:
        return "SMA"
    return folder_code


def _parse_delivery(
    delivery: dict,
    match_id: str,
    competition: str,
    season: int | None,
    match_date: str | None,
    innings: int,
    batting_team: str,
    bowling_team: str,
    over: int,
    ball: int,
    registry: dict,
) -> dict:
    extras = delivery.get("extras") or {}
    is_wide = "wides" in extras
    is_noball = "noballs" in extras
    runs = delivery.get("runs") or {}
    batter = delivery.get("batter")
    bowler = delivery.get("bowler")
    wickets = delivery.get("wickets") or []
    is_wicket = int(len(wickets) > 0)
    dismissal_kind = wickets[0].get("kind") if wickets else None
    player_out = wickets[0].get("player_out") if wickets else None

    legal_ball = 0 if is_wide or is_noball else 1

    return {
        "match_id": match_id,
        "competition": competition,
        "season": season,
        "match_date": match_date,
        "innings": innings,
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "over": over,
        "ball": ball,
        "phase": _phase(over),
        "batter": batter,
        "bowler": bowler,
        "non_striker": delivery.get("non_striker"),
        "batter_id": registry.get(batter) if batter else None,
        "bowler_id": registry.get(bowler) if bowler else None,
        "runs_batter": runs.get("batter", 0),
        "runs_extras": runs.get("extras", 0),
        "runs_total": runs.get("total", 0),
        "is_wide": int(is_wide),
        "is_noball": int(is_noball),
        "legal_ball": legal_ball,
        "is_wicket": is_wicket,
        "dismissal_kind": dismissal_kind,
        "player_out": player_out,
        "is_four": int(runs.get("batter", 0) == 4),
        "is_six": int(runs.get("batter", 0) == 6),
    }


def parse_match_file(path: Path, folder_code: str) -> tuple[list[dict], dict]:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)

    info = doc.get("info") or {}
    match_id = path.stem
    dates = info.get("dates") or []
    match_date = dates[0] if dates else None
    season = _season_year(dates)
    event = info.get("event") or {}
    event_name = event.get("name")
    competition = _competition_code(event_name, folder_code)
    registry = (info.get("registry") or {}).get("people") or {}
    teams = info.get("teams") or []

    match_row = {
        "match_id": match_id,
        "competition": competition,
        "season": season,
        "match_date": match_date,
        "venue": info.get("venue"),
        "city": info.get("city"),
        "match_type": info.get("match_type"),
        "gender": info.get("gender"),
        "team1": teams[0] if len(teams) > 0 else None,
        "team2": teams[1] if len(teams) > 1 else None,
        "event_name": event_name,
    }

    balls: list[dict] = []
    for inn_idx, innings_block in enumerate(doc.get("innings") or [], start=1):
        batting_team = innings_block.get("team")
        bowling_team = teams[1] if teams[0] == batting_team else teams[0] if len(teams) > 1 else None
        for over_block in innings_block.get("overs") or []:
            over = over_block.get("over", 0)
            for ball_idx, delivery in enumerate(over_block.get("deliveries") or [], start=1):
                balls.append(
                    _parse_delivery(
                        delivery,
                        match_id,
                        competition,
                        season,
                        match_date,
                        inn_idx,
                        batting_team,
                        bowling_team,
                        over,
                        ball_idx,
                        registry,
                    )
                )

    return balls, match_row


def parse_competition_folder(folder: Path, folder_code: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_balls: list[dict] = []
    all_matches: list[dict] = []

    paths = sorted(folder.glob("*.json"))
    for path in paths:
        if path.name.upper() == "README.TXT":
            continue
        try:
            balls, match_row = parse_match_file(path, folder_code)
            all_balls.extend(balls)
            all_matches.append(match_row)
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            print(f"Skip {path.name}: {exc}")

    balls_df = pd.DataFrame(all_balls)
    matches_df = pd.DataFrame(all_matches)
    if not matches_df.empty and "match_date" in matches_df.columns:
        matches_df["match_date"] = pd.to_datetime(matches_df["match_date"], errors="coerce")
    return balls_df, matches_df


def parse_all_cricsheet() -> tuple[pd.DataFrame, pd.DataFrame]:
    ball_frames = []
    match_frames = []
    for folder_code, folder in CRICSHEET_DIRS.items():
        if not folder.exists():
            print(f"Missing folder: {folder}")
            continue
        print(f"Parsing {folder_code} from {folder}...")
        balls_df, matches_df = parse_competition_folder(folder, folder_code)
        print(f"  {len(matches_df)} matches, {len(balls_df)} balls")
        ball_frames.append(balls_df)
        match_frames.append(matches_df)

    balls = pd.concat(ball_frames, ignore_index=True) if ball_frames else pd.DataFrame()
    matches = pd.concat(match_frames, ignore_index=True) if match_frames else pd.DataFrame()
    return balls, matches


def save_parquet(balls: pd.DataFrame, matches: pd.DataFrame) -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    balls.to_parquet(CRICSHEET_BALLS, index=False)
    matches.to_parquet(CRICSHEET_MATCHES, index=False)


def main() -> None:
    balls, matches = parse_all_cricsheet()
    save_parquet(balls, matches)
    print(f"Saved {len(balls)} balls, {len(matches)} matches to {INTERIM_DIR}")


if __name__ == "__main__":
    main()
