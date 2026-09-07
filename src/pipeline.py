import sys
import argparse
import logging
import sqlite3
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scraper import EuroleagueScraper
from src.parser import process_season_raw

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("euroleague_pipeline")

def save_sqlite(db_path: Path, games_df: pd.DataFrame, plays_df: pd.DataFrame):
    """Saves DataFrames into SQLite database with optimized indexes."""
    logger.info(f"Writing to SQLite database: {db_path}")
    conn = sqlite3.connect(db_path)
    
    games_df.to_sql("games", conn, if_exists="replace", index=False)
    plays_df.to_sql("plays", conn, if_exists="replace", index=False)

    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_season_code ON games(season, game_code);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_plays_season_code ON plays(season, game_code);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_plays_player ON plays(season, player_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_plays_team ON plays(season, team_code);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_plays_playtype ON plays(play_type);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_plays_elapsed ON plays(season, game_code, elapsed_seconds_in_game);")
    
    conn.commit()
    conn.close()
    logger.info("SQLite indexes created successfully.")

def run_pipeline(
    seasons: list = None,
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    skip_scrape: bool = False,
    force_scrape: bool = False,
    delay: float = 0.35,
    max_games: int = 350
):
    if seasons is None:
        seasons = ["E2024", "E2025"]

    raw_path = Path(raw_dir)
    proc_path = Path(processed_dir)
    proc_path.mkdir(parents=True, exist_ok=True)

    if not skip_scrape:
        scraper = EuroleagueScraper(raw_dir=raw_dir, request_delay=delay)
        for season in seasons:
            logger.info(f"=== Scraping season: {season} ===")
            scraper.fetch_season(season, max_games=max_games, force=force_scrape)

    all_games = []
    all_plays = []

    for season in seasons:
        logger.info(f"=== Parsing season: {season} ===")
        games_df, plays_df = process_season_raw(raw_path, season)
        logger.info(f"[{season}] Parsed {len(games_df)} games and {len(plays_df)} play events.")
        
        if not games_df.empty:
            all_games.append(games_df)
            # Save season-specific CSVs
            games_df.to_csv(proc_path / f"{season}_games.csv", index=False, encoding="utf-8")
        if not plays_df.empty:
            all_plays.append(plays_df)
            plays_df.to_csv(proc_path / f"{season}_plays.csv", index=False, encoding="utf-8")

    if not all_games or not all_plays:
        logger.warning("No games or plays found to compile combined dataset.")
        return

    combined_games = pd.concat(all_games, ignore_index=True)
    combined_plays = pd.concat(all_plays, ignore_index=True)

    logger.info(f"Combined total: {len(combined_games)} games, {len(combined_plays)} play records across {seasons}.")

    # Export Master CSV
    combined_games.to_csv(proc_path / "euroleague_games.csv", index=False, encoding="utf-8")
    combined_plays.to_csv(proc_path / "euroleague_plays.csv", index=False, encoding="utf-8")
    logger.info(f"Exported master CSV files to {proc_path}")

    # Export Master Parquet
    try:
        combined_games.to_parquet(proc_path / "euroleague_games.parquet", index=False)
        combined_plays.to_parquet(proc_path / "euroleague_plays.parquet", index=False)
        logger.info(f"Exported Parquet files to {proc_path}")
    except Exception as e:
        logger.warning(f"Failed exporting Parquet: {e}")

    # Export SQLite DB
    db_file = proc_path / "euroleague.db"
    save_sqlite(db_file, combined_games, combined_plays)

    logger.info("=== Dataset generation complete! ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Euroleague Play-by-Play Data Pipeline")
    parser.add_argument("--seasons", nargs="+", default=["E2024", "E2025"], help="Seasons to process (e.g. E2024 E2025)")
    parser.add_argument("--skip-scrape", action="store_true", help="Skip scraping and process cached files only")
    parser.add_argument("--force-scrape", action="store_true", help="Force re-download of already cached games")
    parser.add_argument("--delay", type=float, default=0.35, help="Delay in seconds between requests (default: 0.35)")
    parser.add_argument("--max-games", type=int, default=350, help="Max gamecode to probe per season (default: 350)")

    args = parser.parse_args()
    run_pipeline(
        seasons=args.seasons,
        skip_scrape=args.skip_scrape,
        force_scrape=args.force_scrape,
        delay=args.delay,
        max_games=args.max_games
    )
