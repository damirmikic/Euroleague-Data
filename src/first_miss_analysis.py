import sys
import sqlite3
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parser import process_season_raw

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("first_miss_analysis")

SHOT_TYPES_ALL = {"2FGM", "2FGA", "3FGM", "3FGA", "FTM", "FTA"}
MISSED_SHOT_TYPES = {"2FGA", "3FGA", "FTA"}
FG_MISSED_TYPES = {"2FGA", "3FGA"}

def analyze_game_first_miss(season: str, game_code: int, plays: pd.DataFrame, game_meta: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Analyzes a single game to find the player who missed the first shot and calculates their full match stats.
    """
    if plays.empty:
        return None

    # Sort plays chronologically
    sorted_plays = plays.sort_values(by=["quarter_num", "play_number", "minute"]).reset_index(drop=True)

    # 1. Identify all shot events
    shot_plays = sorted_plays[sorted_plays["play_type"].isin(SHOT_TYPES_ALL)].copy()
    if shot_plays.empty:
        return None

    first_shot_event = shot_plays.iloc[0]
    first_shot_type = first_shot_event["play_type"]
    first_shot_player = first_shot_event["player_name"]
    first_shot_player_id = first_shot_event["player_id"]
    first_shot_is_miss = first_shot_type in MISSED_SHOT_TYPES

    # 2. Find the FIRST MISSED SHOT in the entire match
    missed_shot_plays = sorted_plays[sorted_plays["play_type"].isin(MISSED_SHOT_TYPES)].copy()
    if missed_shot_plays.empty:
        # Extremely rare: 100% shooting game
        return None

    first_miss_event = missed_shot_plays.iloc[0]
    target_player_id = first_miss_event["player_id"]
    target_player_name = first_miss_event["player_name"]
    target_team = first_miss_event["team_code"]
    target_team_name = first_miss_event["team_name"]

    if not target_player_id or not target_player_name:
        return None

    # 3. Calculate target player's total match stats
    player_plays = sorted_plays[sorted_plays["player_id"] == target_player_id]

    fg2m = len(player_plays[player_plays["play_type"] == "2FGM"])
    fg2a = len(player_plays[player_plays["play_type"] == "2FGA"]) + fg2m
    fg3m = len(player_plays[player_plays["play_type"] == "3FGM"])
    fg3a = len(player_plays[player_plays["play_type"] == "3FGA"]) + fg3m
    ftm = len(player_plays[player_plays["play_type"] == "FTM"])
    fta = len(player_plays[player_plays["play_type"] == "FTA"]) + ftm

    total_pts = (fg2m * 2) + (fg3m * 3) + ftm
    total_fgm = fg2m + fg3m
    total_fga = fg2a + fg3a
    fg_pct = round((total_fgm / total_fga) * 100, 1) if total_fga > 0 else 0.0

    reb_def = len(player_plays[player_plays["play_type"] == "D"])
    reb_off = len(player_plays[player_plays["play_type"] == "O"])
    assists = len(player_plays[player_plays["play_type"] == "AS"])
    steals = len(player_plays[player_plays["play_type"] == "ST"])
    turnovers = len(player_plays[player_plays["play_type"] == "TO"])
    blocks = len(player_plays[player_plays["play_type"] == "BLK"])
    fouls = len(player_plays[player_plays["play_type"] == "CM"])

    # Game context
    team_a = game_meta.get("team_a_name") if game_meta else sorted_plays["team_name"].dropna().unique()[0] if not sorted_plays["team_name"].dropna().empty else None
    team_b = game_meta.get("team_b_name") if game_meta else None
    final_score_a = game_meta.get("final_score_a") if game_meta else None
    final_score_b = game_meta.get("final_score_b") if game_meta else None

    return {
        "season": season,
        "game_code": game_code,
        "matchup": f"{team_a} vs {team_b}" if team_a and team_b else f"Game {game_code}",
        "player_id": target_player_id,
        "player_name": target_player_name,
        "player_team": target_team_name or target_team,
        "first_miss_quarter": first_miss_event["quarter_name"],
        "first_miss_minute": first_miss_event["minute"],
        "first_miss_clock": first_miss_event["marker_time"],
        "first_miss_type": first_miss_event["play_type"],
        "first_miss_desc": first_miss_event["play_info"],
        "was_first_shot_of_game": (first_miss_event["play_number"] == first_shot_event["play_number"]),
        "total_points_scored": total_pts,
        "fg2_made": fg2m,
        "fg2_attempted": fg2a,
        "fg3_made": fg3m,
        "fg3_attempted": fg3a,
        "ft_made": ftm,
        "ft_attempted": fta,
        "total_fg_made": total_fgm,
        "total_fg_attempted": total_fga,
        "fg_percentage": fg_pct,
        "total_rebounds": reb_def + reb_off,
        "assists": assists,
        "turnovers": turnovers,
        "steals": steals,
        "fouls_committed": fouls,
        "final_score_a": final_score_a,
        "final_score_b": final_score_b
    }

def run_first_miss_analysis(
    plays_df: Optional[pd.DataFrame] = None,
    games_df: Optional[pd.DataFrame] = None,
    output_dir: str = "data/processed"
) -> pd.DataFrame:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if plays_df is None or plays_df.empty:
        # Load from processed files
        plays_parquet = out_path / "euroleague_plays.parquet"
        games_parquet = out_path / "euroleague_games.parquet"
        plays_csv = out_path / "euroleague_plays.csv"
        games_csv = out_path / "euroleague_games.csv"

        if plays_parquet.exists():
            logger.info(f"Loading plays from {plays_parquet}...")
            plays_df = pd.read_parquet(plays_parquet)
        elif plays_csv.exists():
            logger.info(f"Loading plays from {plays_csv}...")
            plays_df = pd.read_csv(plays_csv)
        else:
            raise FileNotFoundError("Plays dataset not found. Run pipeline first.")

        if games_parquet.exists():
            games_df = pd.read_parquet(games_parquet)
        elif games_csv.exists():
            games_df = pd.read_csv(games_csv)

    results = []
    grouped = plays_df.groupby(["season", "game_code"])

    logger.info(f"Analyzing {len(grouped)} games for first-missed-shot player points...")

    # Build games metadata lookup
    games_lookup = {}
    if games_df is not None and not games_df.empty:
        for _, row in games_df.iterrows():
            games_lookup[(row["season"], row["game_code"])] = row.to_dict()

    for (season, game_code), game_plays in grouped:
        game_meta = games_lookup.get((season, game_code))
        res = analyze_game_first_miss(season, game_code, game_plays, game_meta)
        if res:
            results.append(res)

    analysis_df = pd.DataFrame(results)

    # Sort results
    analysis_df = analysis_df.sort_values(by=["season", "game_code"]).reset_index(drop=True)

    # Save to CSV, Parquet, and SQLite
    analysis_csv = out_path / "first_missed_shot_analysis.csv"
    analysis_parquet = out_path / "first_missed_shot_analysis.parquet"
    db_file = out_path / "euroleague.db"

    analysis_df.to_csv(analysis_csv, index=False, encoding="utf-8")
    analysis_df.to_parquet(analysis_parquet, index=False)

    try:
        conn = sqlite3.connect(db_file)
        analysis_df.to_sql("first_missed_shot_analysis", conn, if_exists="replace", index=False)
        conn.close()
    except Exception as e:
        logger.warning(f"Could not write to SQLite: {e}")

    logger.info(f"Analysis complete! Saved {len(analysis_df)} game records to {analysis_csv}")
    return analysis_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Euroleague First Missed Shot Calculation")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    args = parser.parse_args()

    run_first_miss_analysis(output_dir=args.output_dir)
