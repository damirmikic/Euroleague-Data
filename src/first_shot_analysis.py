import sys
import sqlite3
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("first_shot_analysis")

SHOT_TYPES_ALL = {"2FGM", "2FGA", "3FGM", "3FGA", "FTM", "FTA"}
MADE_SHOT_TYPES = {"2FGM", "3FGM", "FTM"}

# play_type -> (shot category, point value of the shot)
SHOT_CATEGORY = {
    "2FGM": ("2P", 2),
    "2FGA": ("2P", 2),
    "3FGM": ("3P", 3),
    "3FGA": ("3P", 3),
    "FTM": ("FT", 1),
    "FTA": ("FT", 1),
}

# Columns left empty when the first shot of the match was missed
PLAYER_STAT_COLUMNS = [
    "total_points_scored",
    "fg2_made", "fg2_attempted", "fg2_percentage",
    "fg3_made", "fg3_attempted", "fg3_percentage",
    "ft_made", "ft_attempted", "ft_percentage",
    "total_fg_made", "total_fg_attempted", "fg_percentage",
]

def analyze_game_first_shot(season: str, game_code: int, plays: pd.DataFrame, game_meta: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Analyzes a single game to find the very first shot attempt of the match: whether it went in,
    what kind of shot it was (2P / 3P / FT) and who took it.

    If that first shot was MADE, the shooter's full match scoring and shooting stats are attached.
    If it was MISSED, every player stat column is left empty.
    """
    if plays.empty:
        return None

    # Sort plays chronologically
    sorted_plays = plays.sort_values(by=["quarter_num", "play_number", "minute"]).reset_index(drop=True)

    # 1. The first shot attempt of the match (made or missed)
    shot_plays = sorted_plays[sorted_plays["play_type"].isin(SHOT_TYPES_ALL)]
    if shot_plays.empty:
        return None

    first_shot = shot_plays.iloc[0]
    shot_type = first_shot["play_type"]
    shooter_id = first_shot["player_id"]
    shooter_name = first_shot["player_name"]

    if not shooter_id or not shooter_name:
        return None

    shot_category, shot_value = SHOT_CATEGORY[shot_type]
    was_made = shot_type in MADE_SHOT_TYPES

    # Game context
    team_a = game_meta.get("team_a_name") if game_meta else None
    team_b = game_meta.get("team_b_name") if game_meta else None
    final_score_a = game_meta.get("final_score_a") if game_meta else None
    final_score_b = game_meta.get("final_score_b") if game_meta else None

    record: Dict[str, Any] = {
        "season": season,
        "game_code": game_code,
        "matchup": f"{team_a} vs {team_b}" if team_a and team_b else f"Game {game_code}",
        "first_shot_quarter": first_shot["quarter_name"],
        "first_shot_minute": first_shot["minute"],
        "first_shot_clock": first_shot["marker_time"],
        "first_shot_type": shot_type,
        "first_shot_category": shot_category,
        "first_shot_value": shot_value,
        "first_shot_result": "IN" if was_made else "OUT",
        "first_shot_made": was_made,
        "first_shot_desc": first_shot["play_info"],
        "player_id": shooter_id,
        "player_name": shooter_name,
        "player_team": first_shot["team_name"] or first_shot["team_code"],
        "player_team_code": first_shot["team_code"],
        "final_score_a": final_score_a,
        "final_score_b": final_score_b,
    }

    # 2. Shooter's full match stats - only when the first shot was scored
    if not was_made:
        for col in PLAYER_STAT_COLUMNS:
            record[col] = None
        return record

    player_plays = sorted_plays[sorted_plays["player_id"] == shooter_id]
    counts = player_plays["play_type"].value_counts()

    fg2m = int(counts.get("2FGM", 0))
    fg2a = int(counts.get("2FGA", 0)) + fg2m
    fg3m = int(counts.get("3FGM", 0))
    fg3a = int(counts.get("3FGA", 0)) + fg3m
    ftm = int(counts.get("FTM", 0))
    fta = int(counts.get("FTA", 0)) + ftm

    total_fgm = fg2m + fg3m
    total_fga = fg2a + fg3a

    record.update({
        "total_points_scored": (fg2m * 2) + (fg3m * 3) + ftm,
        "fg2_made": fg2m,
        "fg2_attempted": fg2a,
        "fg2_percentage": round((fg2m / fg2a) * 100, 1) if fg2a > 0 else 0.0,
        "fg3_made": fg3m,
        "fg3_attempted": fg3a,
        "fg3_percentage": round((fg3m / fg3a) * 100, 1) if fg3a > 0 else 0.0,
        "ft_made": ftm,
        "ft_attempted": fta,
        "ft_percentage": round((ftm / fta) * 100, 1) if fta > 0 else 0.0,
        "total_fg_made": total_fgm,
        "total_fg_attempted": total_fga,
        "fg_percentage": round((total_fgm / total_fga) * 100, 1) if total_fga > 0 else 0.0,
    })

    return record

def build_summary(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Builds the IN/OUT and 2P/3P/FT percentage breakdown, per season and overall."""
    rows = []

    def summarize(label: str, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        made = df["first_shot_made"]
        cat = df["first_shot_category"]
        scored = df[made]

        def made_pct(c: str) -> float:
            sub = df[cat == c]
            return round(sub["first_shot_made"].mean() * 100, 1) if len(sub) else 0.0

        return {
            "scope": label,
            "games": n,
            "first_shot_in": int(made.sum()),
            "first_shot_out": int((~made).sum()),
            "first_shot_in_pct": round(made.mean() * 100, 1) if n else 0.0,
            "first_shot_out_pct": round((~made).mean() * 100, 1) if n else 0.0,
            "shot_2p": int((cat == "2P").sum()),
            "shot_3p": int((cat == "3P").sum()),
            "shot_ft": int((cat == "FT").sum()),
            "shot_2p_pct": round((cat == "2P").mean() * 100, 1) if n else 0.0,
            "shot_3p_pct": round((cat == "3P").mean() * 100, 1) if n else 0.0,
            "shot_ft_pct": round((cat == "FT").mean() * 100, 1) if n else 0.0,
            "in_pct_when_2p": made_pct("2P"),
            "in_pct_when_3p": made_pct("3P"),
            "in_pct_when_ft": made_pct("FT"),
            "avg_pts_scorer": round(scored["total_points_scored"].mean(), 2) if len(scored) else 0.0,
            "median_pts_scorer": float(scored["total_points_scored"].median()) if len(scored) else 0.0,
            "avg_fg_pct_scorer": round(scored["fg_percentage"].mean(), 1) if len(scored) else 0.0,
        }

    for season in sorted(analysis_df["season"].unique()):
        rows.append(summarize(season, analysis_df[analysis_df["season"] == season]))
    rows.append(summarize("ALL", analysis_df))

    return pd.DataFrame(rows)

def print_summary(analysis_df: pd.DataFrame, summary_df: pd.DataFrame):
    print("=" * 80)
    print(f" EUROLEAGUE: FIRST SHOT OF THE MATCH ({len(analysis_df)} Games)")
    print("=" * 80)

    for _, r in summary_df.iterrows():
        print(f"\n--- {r['scope']} ({r['games']} Games) ---")
        print(f"- First Shot IN (scored):    {r['first_shot_in']:4d}  ({r['first_shot_in_pct']}%)")
        print(f"- First Shot OUT (missed):   {r['first_shot_out']:4d}  ({r['first_shot_out_pct']}%)")
        print(f"- First Shot Was 2PT:        {r['shot_2p']:4d}  ({r['shot_2p_pct']}%)  -> made {r['in_pct_when_2p']}%")
        print(f"- First Shot Was 3PT:        {r['shot_3p']:4d}  ({r['shot_3p_pct']}%)  -> made {r['in_pct_when_3p']}%")
        print(f"- First Shot Was FT:         {r['shot_ft']:4d}  ({r['shot_ft_pct']}%)  -> made {r['in_pct_when_ft']}%")
        print(f"- Scorer Avg Match Points:   {r['avg_pts_scorer']} PTS (median {r['median_pts_scorer']})")
        print(f"- Scorer Avg FG%:            {r['avg_fg_pct_scorer']}%")

    scored = analysis_df[analysis_df["first_shot_made"]]
    if not scored.empty:
        print("\n" + "-" * 80)
        print(" TOP 10 MATCH SCORING PERFORMANCES BY THE FIRST-SHOT SCORER")
        print("-" * 80)
        for _, row in scored.sort_values(by="total_points_scored", ascending=False).head(10).iterrows():
            print(f"- [{row['season']}-G{int(row['game_code']):03d}] {row['player_name']:<24} ({row['player_team_code']}): "
                  f"{int(row['total_points_scored']):2d} PTS | FG: {int(row['total_fg_made'])}/{int(row['total_fg_attempted'])} "
                  f"({row['fg_percentage']}%) | 1st Shot: {row['first_shot_category']} at {row['first_shot_clock']}")

def run_first_shot_analysis(
    plays_df: Optional[pd.DataFrame] = None,
    games_df: Optional[pd.DataFrame] = None,
    output_dir: str = "data/processed"
):
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

    logger.info(f"Analyzing {len(grouped)} games for first-shot outcome and shooter stats...")

    games_lookup = {}
    if games_df is not None and not games_df.empty:
        for _, row in games_df.iterrows():
            games_lookup[(row["season"], row["game_code"])] = row.to_dict()

    for (season, game_code), game_plays in grouped:
        game_meta = games_lookup.get((season, game_code))
        res = analyze_game_first_shot(season, game_code, game_plays, game_meta)
        if res:
            results.append(res)

    analysis_df = pd.DataFrame(results)
    analysis_df = analysis_df.sort_values(by=["season", "game_code"]).reset_index(drop=True)

    # Nullable dtypes keep missed-shot rows blank instead of coercing counts to 0.0
    for col in PLAYER_STAT_COLUMNS:
        if col.endswith("_percentage"):
            analysis_df[col] = analysis_df[col].astype("Float64")
        else:
            analysis_df[col] = analysis_df[col].astype("Int64")

    summary_df = build_summary(analysis_df)

    # Save outputs
    analysis_csv = out_path / "first_shot_analysis.csv"
    analysis_parquet = out_path / "first_shot_analysis.parquet"
    summary_csv = out_path / "first_shot_summary.csv"
    db_file = out_path / "euroleague.db"

    analysis_df.to_csv(analysis_csv, index=False, encoding="utf-8")
    analysis_df.to_parquet(analysis_parquet, index=False)
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")

    try:
        conn = sqlite3.connect(db_file)
        analysis_df.to_sql("first_shot_analysis", conn, if_exists="replace", index=False)
        summary_df.to_sql("first_shot_summary", conn, if_exists="replace", index=False)
        conn.close()
    except Exception as e:
        logger.warning(f"Could not write to SQLite: {e}")

    logger.info(f"First shot analysis complete! Saved {len(analysis_df)} game records to {analysis_csv}")
    return analysis_df, summary_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Euroleague First Shot of the Match Analysis")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Skip printing the console summary")
    args = parser.parse_args()

    analysis_df, summary_df = run_first_shot_analysis(output_dir=args.output_dir)
    if not args.quiet:
        print_summary(analysis_df, summary_df)
