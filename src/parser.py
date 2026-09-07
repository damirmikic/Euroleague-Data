import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd

logger = logging.getLogger("euroleague_parser")

QUARTER_KEYS = [
    ("FirstQuarter", 1, "Q1"),
    ("SecondQuarter", 2, "Q2"),
    ("ThirdQuarter", 3, "Q3"),
    ("ForthQuarter", 4, "Q4"),
    ("ExtraTime", 5, "OT"),
]

def clean_str(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None

def parse_markertime(mtime: Optional[str]) -> Optional[int]:
    """Converts MM:SS marker time string to total seconds remaining in quarter."""
    if not mtime or not isinstance(mtime, str):
        return None
    parts = mtime.strip().split(":")
    if len(parts) == 2:
        try:
            return int(parts[0]) * 60 + int(parts[1])
        except ValueError:
            return None
    return None

def parse_game_json(season: str, gamecode: int, data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Parses a single game JSON into game metadata and structured play-by-play events.
    """
    team_a = clean_str(data.get("TeamA"))
    team_b = clean_str(data.get("TeamB"))
    code_team_a = clean_str(data.get("CodeTeamA"))
    code_team_b = clean_str(data.get("CodeTeamB"))
    actual_quarter = data.get("ActualQuarter")
    live_status = str(data.get("Live", "False")).lower() == "true"

    all_plays = []
    curr_score_a = 0
    curr_score_b = 0

    for q_key, q_num, q_name in QUARTER_KEYS:
        quarter_data = data.get(q_key, [])
        if not isinstance(quarter_data, list):
            continue

        for play in quarter_data:
            if not isinstance(play, dict):
                continue

            num_play = play.get("NUMBEROFPLAY")
            play_type = clean_str(play.get("PLAYTYPE"))
            play_info = clean_str(play.get("PLAYINFO"))
            player_id = clean_str(play.get("PLAYER_ID"))
            player_name = clean_str(play.get("PLAYER"))
            team_code = clean_str(play.get("CODETEAM"))
            team_name = clean_str(play.get("TEAM"))
            dorsal = clean_str(play.get("DORSAL"))
            minute = play.get("MINUTE")
            marker_time = clean_str(play.get("MARKERTIME"))
            raw_pts_a = play.get("POINTS_A")
            raw_pts_b = play.get("POINTS_B")
            raw_type = play.get("TYPE")
            dor = play.get("DOR")
            comment = clean_str(play.get("COMMENT"))

            pts_a_val = int(raw_pts_a) if raw_pts_a is not None and str(raw_pts_a).isdigit() else None
            pts_b_val = int(raw_pts_b) if raw_pts_b is not None and str(raw_pts_b).isdigit() else None

            # Calculate points scored in this play
            pts_scored_a = 0
            pts_scored_b = 0

            if pts_a_val is not None and pts_a_val > curr_score_a:
                pts_scored_a = pts_a_val - curr_score_a
                curr_score_a = pts_a_val

            if pts_b_val is not None and pts_b_val > curr_score_b:
                pts_scored_b = pts_b_val - curr_score_b
                curr_score_b = pts_b_val

            is_scoring = (pts_scored_a > 0 or pts_scored_b > 0)
            lead_a = curr_score_a - curr_score_b
            seconds_remaining = parse_markertime(marker_time)

            # Calculate elapsed seconds in game
            quarter_offset_seconds = (q_num - 1) * 600 if q_num <= 4 else (2400 + (q_num - 5) * 300)
            quarter_length = 600 if q_num <= 4 else 300
            if seconds_remaining is not None:
                elapsed_in_game = quarter_offset_seconds + (quarter_length - seconds_remaining)
            elif minute is not None:
                elapsed_in_game = minute * 60
            else:
                elapsed_in_game = None

            play_row = {
                "season": season,
                "game_code": gamecode,
                "quarter_num": q_num,
                "quarter_name": q_name,
                "play_number": num_play,
                "play_type": play_type,
                "play_info": play_info,
                "team_code": team_code,
                "team_name": team_name,
                "player_id": player_id,
                "player_name": player_name,
                "dorsal": dorsal,
                "minute": minute,
                "marker_time": marker_time,
                "seconds_remaining_in_quarter": seconds_remaining,
                "elapsed_seconds_in_game": elapsed_in_game,
                "current_score_a": curr_score_a,
                "current_score_b": curr_score_b,
                "points_scored_a": pts_scored_a,
                "points_scored_b": pts_scored_b,
                "score_lead_a": lead_a,
                "is_scoring_play": is_scoring,
                "raw_type": raw_type,
                "comment": comment
            }
            all_plays.append(play_row)

    game_row = {
        "season": season,
        "game_code": gamecode,
        "team_a_code": code_team_a,
        "team_a_name": team_a,
        "team_b_code": code_team_b,
        "team_b_name": team_b,
        "actual_quarter": int(actual_quarter) if actual_quarter and str(actual_quarter).isdigit() else len([k for k, _, _ in QUARTER_KEYS if data.get(k)]),
        "final_score_a": curr_score_a,
        "final_score_b": curr_score_b,
        "total_plays": len(all_plays),
        "is_live": live_status
    }

    return game_row, all_plays

def process_season_raw(raw_dir: Path, season: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads all cached raw JSON files for a season and returns (games_df, plays_df).
    """
    season_dir = raw_dir / season
    if not season_dir.exists():
        logger.warning(f"Directory {season_dir} does not exist.")
        return pd.DataFrame(), pd.DataFrame()

    games_list = []
    plays_list = []

    json_files = sorted(
        season_dir.glob("game_*.json"),
        key=lambda p: int(re.search(r"game_(\d+)\.json", p.name).group(1)) if re.search(r"game_(\d+)\.json", p.name) else 0
    )

    for fpath in json_files:
        try:
            match = re.search(r"game_(\d+)\.json", fpath.name)
            if not match:
                continue
            gamecode = int(match.group(1))

            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict) or data.get("empty", False) or not data.get("TeamA"):
                continue

            game_meta, plays = parse_game_json(season, gamecode, data)
            games_list.append(game_meta)
            plays_list.extend(plays)
        except Exception as e:
            logger.error(f"Error parsing {fpath}: {e}")

    games_df = pd.DataFrame(games_list)
    plays_df = pd.DataFrame(plays_list)
    return games_df, plays_df
