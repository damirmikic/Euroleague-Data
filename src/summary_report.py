import sys
import io
from pathlib import Path
import pandas as pd

# Configure UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def print_first_miss_summary():
    csv_path = Path("data/processed/first_missed_shot_analysis.csv")
    if not csv_path.exists():
        print("First missed shot analysis file not found.")
        return

    df = pd.read_csv(csv_path)
    print("=" * 80)
    print(f" EUROLEAGUE: PLAYER WHO MISSED FIRST SHOT IN MATCH ({len(df)} Games)")
    print("=" * 80)
    
    for season in sorted(df["season"].unique()):
        sdf = df[df["season"] == season]
        pts = sdf["total_points_scored"]
        print(f"\n--- Season {season} ({len(sdf)} Games) ---")
        print(f"• Average Points Scored:      {pts.mean():.2f} PTS")
        print(f"• Median Points Scored:       {pts.median():.1f} PTS")
        print(f"• Standard Deviation:         {pts.std():.2f} PTS")
        print(f"• Min / Max Points:           {pts.min()} / {pts.max()} PTS")
        print(f"• Over 9.5 Points:            {(pts > 9.5).mean() * 100:.1f}%")
        print(f"• Over 12.5 Points:           {(pts > 12.5).mean() * 100:.1f}%")
        print(f"• Over 15.5 Points:           {(pts > 15.5).mean() * 100:.1f}%")
        print(f"• First Miss Was 2PT Attempt: {(sdf['first_miss_type'] == '2FGA').mean() * 100:.1f}%")
        print(f"• First Miss Was 3PT Attempt: {(sdf['first_miss_type'] == '3FGA').mean() * 100:.1f}%")

    print("\n" + "-" * 80)
    print(" TOP 10 HIGHEST SCORING PERFORMANCES AFTER MISSING FIRST SHOT")
    print("-" * 80)
    top10 = df.sort_values(by="total_points_scored", ascending=False).head(10)
    for _, row in top10.iterrows():
        print(f"• [{row['season']}-G{row['game_code']:03d}] {row['player_name']:<24} ({row['player_team']}): {row['total_points_scored']:2d} PTS | FG: {row['total_fg_made']}/{row['total_fg_attempted']} ({row['fg_percentage']}%) | 1st Miss: {row['first_miss_type']} at {row['first_miss_clock']}")

def print_last_points_summary():
    csv_path = Path("data/processed/last_points_scored_analysis.csv")
    if not csv_path.exists():
        print("Last points scored analysis file not found.")
        return

    df = pd.read_csv(csv_path)
    print("\n" + "=" * 80)
    print(f" EUROLEAGUE: PLAYER WHO SCORED LAST POINTS IN MATCH ({len(df)} Games)")
    print("=" * 80)
    
    for season in sorted(df["season"].unique()):
        sdf = df[df["season"] == season]
        pts = sdf["total_points_scored"]
        ft_pct = (sdf["last_score_type"] == "FTM").mean() * 100
        fg2_pct = (sdf["last_score_type"] == "2FGM").mean() * 100
        fg3_pct = (sdf["last_score_type"] == "3FGM").mean() * 100
        clutch_pct = sdf["is_clutch_final_shot"].mean() * 100

        print(f"\n--- Season {season} ({len(sdf)} Games) ---")
        print(f"• Average Total Points Scored:  {pts.mean():.2f} PTS")
        print(f"• Median Total Points Scored:   {pts.median():.1f} PTS")
        print(f"• Over 9.5 Points:              {(pts > 9.5).mean() * 100:.1f}%")
        print(f"• Over 14.5 Points:             {(pts > 14.5).mean() * 100:.1f}%")
        print(f"• Final Score Shot Type:        Free Throw: {ft_pct:.1f}% | 2-Pointer: {fg2_pct:.1f}% | 3-Pointer: {fg3_pct:.1f}%")
        print(f"• Final Score in Clutch (<=30s): {clutch_pct:.1f}% of games")

    print("\n" + "-" * 80)
    print(" TOP 10 PLAYERS WITH MOST MATCH-CLOSING / FINAL-POINTS SCORED")
    print("-" * 80)
    top_players = df.groupby(["player_id", "player_name", "player_team"]).agg(
        final_scores_count=("game_code", "count"),
        avg_game_pts=("total_points_scored", "mean")
    ).sort_values(by="final_scores_count", ascending=False).head(10).reset_index()

    for _, row in top_players.iterrows():
        print(f"• {row['player_name']:<24} ({row['player_team']}): {row['final_scores_count']} times scored final points (Avg {row['avg_game_pts']:.1f} PTS/game)")

def main():
    print_first_miss_summary()
    print_last_points_summary()

if __name__ == "__main__":
    main()
