import sys
import sqlite3
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("last_points_analysis")

SCORING_PLAY_TYPES = {"2FGM", "3FGM", "FTM"}

def analyze_game_last_points(season: str, game_code: int, plays: pd.DataFrame, game_meta: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Analyzes a single game to find the player who scored the last point(s) and calculates their full match stats.
    """
    if plays.empty:
        return None

    # Sort plays chronologically
    sorted_plays = plays.sort_values(by=["quarter_num", "play_number", "minute"]).reset_index(drop=True)

    # Filter scoring plays
    scoring_plays = sorted_plays[
        (sorted_plays["play_type"].isin(SCORING_PLAY_TYPES)) | 
        (sorted_plays["points_scored_a"] > 0) | 
        (sorted_plays["points_scored_b"] > 0)
    ].copy()

    if scoring_plays.empty:
        return None

    # Last scoring play of the match
    last_score_event = scoring_plays.iloc[-1]
    target_player_id = last_score_event["player_id"]
    target_player_name = last_score_event["player_name"]
    target_team = last_score_event["team_code"]
    target_team_name = last_score_event["team_name"]

    if not target_player_id or not target_player_name:
        # If last scoring play didn't have player attached, check reverse until player found
        for i in range(len(scoring_plays) - 1, -1, -1):
            cand = scoring_plays.iloc[i]
            if cand["player_id"] and cand["player_name"]:
                last_score_event = cand
                target_player_id = cand["player_id"]
                target_player_name = cand["player_name"]
                target_team = cand["team_code"]
                target_team_name = cand["team_name"]
                break

    if not target_player_id or not target_player_name:
        return None

    # Points scored on that final shot
    pts_on_final_shot = 0
    if last_score_event["play_type"] == "3FGM":
        pts_on_final_shot = 3
    elif last_score_event["play_type"] == "2FGM":
        pts_on_final_shot = 2
    elif last_score_event["play_type"] == "FTM":
        pts_on_final_shot = 1
    else:
        pts_on_final_shot = max(int(last_score_event.get("points_scored_a", 0) or 0), int(last_score_event.get("points_scored_b", 0) or 0))

    # Calculate target player's total match stats
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
    team_a = game_meta.get("team_a_name") if game_meta else None
    team_b = game_meta.get("team_b_name") if game_meta else None
    final_score_a = game_meta.get("final_score_a") if game_meta else None
    final_score_b = game_meta.get("final_score_b") if game_meta else None

    # Check if final score was in clutch time (<= 30s remaining in 4th quarter or OT)
    seconds_remaining = last_score_event["seconds_remaining_in_quarter"]
    is_clutch_final_shot = (last_score_event["quarter_num"] >= 4 and seconds_remaining is not None and seconds_remaining <= 30)

    return {
        "season": season,
        "game_code": game_code,
        "matchup": f"{team_a} vs {team_b}" if team_a and team_b else f"Game {game_code}",
        "player_id": target_player_id,
        "player_name": target_player_name,
        "player_team": target_team_name or target_team,
        "last_score_quarter": last_score_event["quarter_name"],
        "last_score_minute": last_score_event["minute"],
        "last_score_clock": last_score_event["marker_time"],
        "last_score_seconds_left": seconds_remaining,
        "last_score_type": last_score_event["play_type"],
        "last_score_points": pts_on_final_shot,
        "last_score_desc": last_score_event["play_info"],
        "is_clutch_final_shot": is_clutch_final_shot,
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

def run_last_points_analysis(
    plays_df: Optional[pd.DataFrame] = None,
    games_df: Optional[pd.DataFrame] = None,
    output_dir: str = "data/processed"
) -> pd.DataFrame:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if plays_df is None or plays_df.empty:
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

    logger.info(f"Analyzing {len(grouped)} games for last-points player stats...")

    games_lookup = {}
    if games_df is not None and not games_df.empty:
        for _, row in games_df.iterrows():
            games_lookup[(row["season"], row["game_code"])] = row.to_dict()

    for (season, game_code), game_plays in grouped:
        game_meta = games_lookup.get((season, game_code))
        res = analyze_game_last_points(season, game_code, game_plays, game_meta)
        if res:
            results.append(res)

    analysis_df = pd.DataFrame(results)
    analysis_df = analysis_df.sort_values(by=["season", "game_code"]).reset_index(drop=True)

    # Save outputs
    analysis_csv = out_path / "last_points_scored_analysis.csv"
    analysis_parquet = out_path / "last_points_scored_analysis.parquet"
    db_file = out_path / "euroleague.db"

    analysis_df.to_csv(analysis_csv, index=False, encoding="utf-8")
    analysis_df.to_parquet(analysis_parquet, index=False)

    try:
        conn = sqlite3.connect(db_file)
        analysis_df.to_sql("last_points_scored_analysis", conn, if_exists="replace", index=False)
        conn.close()
    except Exception as e:
        logger.warning(f"Could not write to SQLite: {e}")

    logger.info(f"Last points analysis complete! Saved {len(analysis_df)} records to {analysis_csv}")
    return analysis_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Euroleague Last Points Scored Analysis")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    args = parser.parse_args()

    run_last_points_analysis(output_dir=args.output_dir)
