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
logger = logging.getLogger("first_last_shot_analysis")

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

# Per-shooter stat columns, left empty when that end's shot was missed
PLAYER_STAT_SUFFIXES = [
    "total_points_scored",
    "fg2_made", "fg2_attempted", "fg2_percentage",
    "fg3_made", "fg3_attempted", "fg3_percentage",
    "ft_made", "ft_attempted", "ft_percentage",
    "total_fg_made", "total_fg_attempted", "fg_percentage",
]

PLAYER_STAT_COLUMNS = [f"{side}_{col}" for side in ("first", "last") for col in PLAYER_STAT_SUFFIXES]

# Combined outcome buckets
OUTCOME_BOTH_IN = "BOTH_SCORED"
OUTCOME_BOTH_OUT = "BOTH_MISSED"
OUTCOME_FIRST_ONLY = "FIRST_ONLY"
OUTCOME_LAST_ONLY = "LAST_ONLY"


def _shooter_stats(sorted_plays: pd.DataFrame, shooter_id: str) -> Dict[str, Any]:
    """Full match shooting line for one player, same definition as first_shot_analysis."""
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

    return {
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
    }


def _shot_record(side: str, shot: pd.Series, sorted_plays: pd.DataFrame) -> Dict[str, Any]:
    """Builds the `{side}_*` block for one shot: context, result and (if made) the shooter's match line."""
    shot_type = shot["play_type"]
    shot_category, shot_value = SHOT_CATEGORY[shot_type]
    was_made = shot_type in MADE_SHOT_TYPES

    block: Dict[str, Any] = {
        f"{side}_shot_quarter": shot["quarter_name"],
        f"{side}_shot_minute": shot["minute"],
        f"{side}_shot_clock": shot["marker_time"],
        f"{side}_shot_type": shot_type,
        f"{side}_shot_category": shot_category,
        f"{side}_shot_value": shot_value,
        f"{side}_shot_result": "IN" if was_made else "OUT",
        f"{side}_shot_made": was_made,
        f"{side}_shot_desc": shot["play_info"],
        f"{side}_player_id": shot["player_id"],
        f"{side}_player_name": shot["player_name"],
        f"{side}_player_team": shot["team_name"] or shot["team_code"],
        f"{side}_player_team_code": shot["team_code"],
    }

    if was_made:
        for col, val in _shooter_stats(sorted_plays, shot["player_id"]).items():
            block[f"{side}_{col}"] = val
    else:
        for col in PLAYER_STAT_SUFFIXES:
            block[f"{side}_{col}"] = None

    return block


def analyze_game_first_last_shot(season: str, game_code: int, plays: pd.DataFrame, game_meta: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Analyzes a single game to find both the very FIRST shot attempt and the very LAST shot
    attempt of the match: whether each went in, what kind of shot it was (2P / 3P / FT)
    and who took it.

    For each of the two shots: if it was MADE, that shooter's full match scoring and shooting
    stats are attached; if it was MISSED, that shooter's stat columns are left empty.

    The two shots are then classified into a combined outcome:
      BOTH_SCORED / FIRST_ONLY / LAST_ONLY / BOTH_MISSED
    """
    if plays.empty:
        return None

    # Sort plays chronologically
    sorted_plays = plays.sort_values(by=["quarter_num", "play_number", "minute"]).reset_index(drop=True)

    # Every shot attempt of the match (made or missed)
    shot_plays = sorted_plays[sorted_plays["play_type"].isin(SHOT_TYPES_ALL)]
    if shot_plays.empty:
        return None

    # Both ends must be attributable to a player, otherwise the shooter line is meaningless
    named_shots = shot_plays[shot_plays["player_id"].notna() & shot_plays["player_name"].notna()]
    named_shots = named_shots[(named_shots["player_id"] != "") & (named_shots["player_name"] != "")]
    if named_shots.empty:
        return None

    first_shot = named_shots.iloc[0]
    last_shot = named_shots.iloc[-1]

    # Game context
    team_a = game_meta.get("team_a_name") if game_meta else None
    team_b = game_meta.get("team_b_name") if game_meta else None

    record: Dict[str, Any] = {
        "season": season,
        "game_code": game_code,
        "matchup": f"{team_a} vs {team_b}" if team_a and team_b else f"Game {game_code}",
        "final_score_a": game_meta.get("final_score_a") if game_meta else None,
        "final_score_b": game_meta.get("final_score_b") if game_meta else None,
    }

    record.update(_shot_record("first", first_shot, sorted_plays))
    record.update(_shot_record("last", last_shot, sorted_plays))

    first_made = record["first_shot_made"]
    last_made = record["last_shot_made"]

    if first_made and last_made:
        outcome = OUTCOME_BOTH_IN
    elif first_made:
        outcome = OUTCOME_FIRST_ONLY
    elif last_made:
        outcome = OUTCOME_LAST_ONLY
    else:
        outcome = OUTCOME_BOTH_OUT

    record["shots_scored"] = int(first_made) + int(last_made)
    record["outcome"] = outcome
    record["both_scored"] = outcome == OUTCOME_BOTH_IN
    record["both_missed"] = outcome == OUTCOME_BOTH_OUT
    record["partial_scored"] = outcome in (OUTCOME_FIRST_ONLY, OUTCOME_LAST_ONLY)
    # Degenerate guard: a match whose first and last attributable shot are the same play
    record["same_shot"] = bool(
        first_shot["play_number"] == last_shot["play_number"]
        and first_shot["quarter_num"] == last_shot["quarter_num"]
    )
    record["same_player"] = bool(first_shot["player_id"] == last_shot["player_id"])

    return record


def build_summary(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Combined first+last outcome breakdown (both scored / partial / both missed), per season and overall."""
    rows = []

    def summarize(label: str, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        first_made = df["first_shot_made"]
        last_made = df["last_shot_made"]
        outcome = df["outcome"]

        def cnt(o: str) -> int:
            return int((outcome == o).sum())

        def pct(c: int) -> float:
            return round((c / n) * 100, 1) if n else 0.0

        both_in = cnt(OUTCOME_BOTH_IN)
        first_only = cnt(OUTCOME_FIRST_ONLY)
        last_only = cnt(OUTCOME_LAST_ONLY)
        both_out = cnt(OUTCOME_BOTH_OUT)
        partial = first_only + last_only

        def cat_share(side: str, c: str) -> float:
            col = df[f"{side}_shot_category"]
            return round((col == c).mean() * 100, 1) if n else 0.0

        def made_pct_when(side: str, c: str) -> float:
            sub = df[df[f"{side}_shot_category"] == c]
            return round(sub[f"{side}_shot_made"].mean() * 100, 1) if len(sub) else 0.0

        return {
            "scope": label,
            "games": n,
            # --- combined outcome ---
            "both_scored": both_in,
            "both_scored_pct": pct(both_in),
            "partial_scored": partial,
            "partial_scored_pct": pct(partial),
            "first_only_scored": first_only,
            "first_only_scored_pct": pct(first_only),
            "last_only_scored": last_only,
            "last_only_scored_pct": pct(last_only),
            "both_missed": both_out,
            "both_missed_pct": pct(both_out),
            "avg_shots_scored": round(df["shots_scored"].mean(), 2) if n else 0.0,
            # --- each end on its own ---
            "first_shot_in": int(first_made.sum()),
            "first_shot_in_pct": round(first_made.mean() * 100, 1) if n else 0.0,
            "last_shot_in": int(last_made.sum()),
            "last_shot_in_pct": round(last_made.mean() * 100, 1) if n else 0.0,
            # --- shot type mix at each end ---
            "first_2p_pct": cat_share("first", "2P"),
            "first_3p_pct": cat_share("first", "3P"),
            "first_ft_pct": cat_share("first", "FT"),
            "last_2p_pct": cat_share("last", "2P"),
            "last_3p_pct": cat_share("last", "3P"),
            "last_ft_pct": cat_share("last", "FT"),
            "first_in_pct_when_2p": made_pct_when("first", "2P"),
            "first_in_pct_when_3p": made_pct_when("first", "3P"),
            "first_in_pct_when_ft": made_pct_when("first", "FT"),
            "last_in_pct_when_2p": made_pct_when("last", "2P"),
            "last_in_pct_when_3p": made_pct_when("last", "3P"),
            "last_in_pct_when_ft": made_pct_when("last", "FT"),
            # --- independence check: product of the two marginals vs observed both-scored ---
            "expected_both_scored_pct": round(first_made.mean() * last_made.mean() * 100, 1) if n else 0.0,
            "same_player_both_ends": int(df["same_player"].sum()),
        }

    for season in sorted(analysis_df["season"].unique()):
        rows.append(summarize(season, analysis_df[analysis_df["season"] == season]))
    rows.append(summarize("ALL", analysis_df))

    return pd.DataFrame(rows)


def print_summary(analysis_df: pd.DataFrame, summary_df: pd.DataFrame):
    print("=" * 84)
    print(f" EUROLEAGUE: FIRST & LAST SHOT OF THE MATCH ({len(analysis_df)} Games)")
    print("=" * 84)

    for _, r in summary_df.iterrows():
        print(f"\n--- {r['scope']} ({r['games']} Games) ---")
        print(f"- BOTH SCORED:               {r['both_scored']:4d}  ({r['both_scored_pct']}%)")
        print(f"- PARTIAL (exactly one in):  {r['partial_scored']:4d}  ({r['partial_scored_pct']}%)")
        print(f"    - first IN, last OUT:    {r['first_only_scored']:4d}  ({r['first_only_scored_pct']}%)")
        print(f"    - first OUT, last IN:    {r['last_only_scored']:4d}  ({r['last_only_scored_pct']}%)")
        print(f"- BOTH MISSED:               {r['both_missed']:4d}  ({r['both_missed_pct']}%)")
        print(f"- Avg shots scored (0-2):    {r['avg_shots_scored']}")
        print(f"- First Shot IN:             {r['first_shot_in']:4d}  ({r['first_shot_in_pct']}%)"
              f"   mix 2P {r['first_2p_pct']}% / 3P {r['first_3p_pct']}% / FT {r['first_ft_pct']}%")
        print(f"- Last Shot IN:              {r['last_shot_in']:4d}  ({r['last_shot_in_pct']}%)"
              f"   mix 2P {r['last_2p_pct']}% / 3P {r['last_3p_pct']}% / FT {r['last_ft_pct']}%")
        print(f"- Both-scored if independent: {r['expected_both_scored_pct']}%  (observed {r['both_scored_pct']}%)")
        print(f"- Same player took both ends: {r['same_player_both_ends']}")

    print("\n" + "-" * 84)
    print(" OUTCOME MATRIX (ALL SEASONS)  rows = first shot, cols = last shot")
    print("-" * 84)
    matrix = pd.crosstab(
        analysis_df["first_shot_result"].rename("first"),
        analysis_df["last_shot_result"].rename("last"),
    )
    print(matrix.to_string())

    both = analysis_df[analysis_df["both_scored"]]
    if not both.empty:
        print("\n" + "-" * 84)
        print(" TOP 10 COMBINED SCORING (FIRST-SHOT SCORER + LAST-SHOT SCORER)")
        print("-" * 84)
        combined = both["first_total_points_scored"] + both["last_total_points_scored"]
        top = both.assign(combined_pts=combined).sort_values(by="combined_pts", ascending=False).head(10)
        for _, row in top.iterrows():
            print(f"- [{row['season']}-G{int(row['game_code']):03d}] {int(row['combined_pts']):3d} PTS combined | "
                  f"first: {row['first_player_name']} ({row['first_player_team_code']}) "
                  f"{int(row['first_total_points_scored']):2d} | "
                  f"last: {row['last_player_name']} ({row['last_player_team_code']}) "
                  f"{int(row['last_total_points_scored']):2d}")


def run_first_last_shot_analysis(
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

    logger.info(f"Analyzing {len(grouped)} games for first & last shot outcome and shooter stats...")

    games_lookup = {}
    if games_df is not None and not games_df.empty:
        for _, row in games_df.iterrows():
            games_lookup[(row["season"], row["game_code"])] = row.to_dict()

    for (season, game_code), game_plays in grouped:
        game_meta = games_lookup.get((season, game_code))
        res = analyze_game_first_last_shot(season, game_code, game_plays, game_meta)
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
    analysis_csv = out_path / "first_last_shot_analysis.csv"
    analysis_parquet = out_path / "first_last_shot_analysis.parquet"
    summary_csv = out_path / "first_last_shot_summary.csv"
    db_file = out_path / "euroleague.db"

    analysis_df.to_csv(analysis_csv, index=False, encoding="utf-8")
    analysis_df.to_parquet(analysis_parquet, index=False)
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")

    try:
        conn = sqlite3.connect(db_file)
        analysis_df.to_sql("first_last_shot_analysis", conn, if_exists="replace", index=False)
        summary_df.to_sql("first_last_shot_summary", conn, if_exists="replace", index=False)
        conn.close()
    except Exception as e:
        logger.warning(f"Could not write to SQLite: {e}")

    logger.info(f"First & last shot analysis complete! Saved {len(analysis_df)} game records to {analysis_csv}")
    return analysis_df, summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Euroleague First & Last Shot of the Match Analysis")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Skip printing the console summary")
    args = parser.parse_args()

    analysis_df, summary_df = run_first_last_shot_analysis(output_dir=args.output_dir)
    if not args.quiet:
        print_summary(analysis_df, summary_df)
